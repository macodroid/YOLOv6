import argparse
import os
import time
from queue import Queue, Empty
from threading import Event, Thread

import cv2
import numpy as np

from radar import Radar
from utils import get_calibration_params, compute_camera_calibration, get_transform_matrix_with_criterion
from yolov6.core.inferer import Inferer
from yolov6.utils.events import LOGGER

TIMEOUT = 20


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Yolov6 3d speed measurement', add_help=add_help)
    parser.add_argument('--weights', type=str, default='weights/yolov6s.pt', help='model path(s) for inference.')
    parser.add_argument('--source', type=str, default='data/images', help='the source path, e.g. image-file/dir.')
    parser.add_argument('--yaml', type=str, default='data/bcs.yaml', help='data yaml file.')
    parser.add_argument('--yolo-img-size', nargs='+', type=int, default=[352, 640],
                        help='the image-size(h,w) in inference size.')
    parser.add_argument('--img-size', nargs='+', type=int, default=[960, 540],
                        help='The image size (h,w) for inference.')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='NMS IoU threshold for inference.')
    parser.add_argument('--half', action='store_true',
                        help='whether to use FP16 half-precision inference.')
    parser.add_argument('--camera-calib-file', type=str, default='data/camera_calibration.yaml',
                        help='camera calibration file.')
    parser.add_argument('--output_path', default=None, help='Path to output video')
    parser.add_argument('--vid_path', type=str, help='Path to video')
    parser.add_argument('--road-mask-path', type=str, help='Path to road mask')
    parser.add_argument('--process-offline', type=bool, default=False,
                        help='Process video offline, output will be save just in the json file')
    parser.add_argument('--video-fps', type=int, default=50, help='Video FPS')
    parser.add_argument('--test-name', type=str, default='yolov6_3d_qarepvgg_23', help='Test name')
    parser.add_argument('--result-dir', type=str, default='', help='Result directory')
    parser.add_argument('--batch-size-processing', type=int, default=32, help='Batch size for processing')
    args = parser.parse_args()
    LOGGER.info(args)
    return args


def batch_test_video(inferer: Inferer,
                     camera_calibration_file: str,
                     video_path: str,
                     road_mask_path: str,
                     img_size: tuple,
                     result_dir: str,
                     test_name: str,
                     batch_size_processing: int = 32,
                     video_fps: int = 50,
                     ):
    avg_fps = []
    im_w, im_h = img_size
    camera_calibration = get_calibration_params(camera_calibration_file)
    vp1, vp2, vp3, pp, road_plane, focal = compute_camera_calibration(
        [camera_calibration['vp1'], camera_calibration['vp2']], camera_calibration['pp']
    )
    vp1 = vp1[:-1] / vp1[-1]
    vp2 = vp2[:-1] / vp2[-1]
    vp3 = vp3[:-1] / vp3[-1]

    cap = cv2.VideoCapture(video_path)
    road_mask = cv2.imread(road_mask_path, cv2.IMREAD_GRAYSCALE)

    M, IM = get_transform_matrix_with_criterion(vp3, vp2, road_mask, im_w, im_h, constraint=0.8,
                                                vp_top=None)

    vp0_t = np.array([vp1], dtype="float32")
    vp0_t = np.array([vp0_t])
    vp0_t = cv2.perspectiveTransform(vp0_t, M)
    vp0_t = vp0_t[0][0]

    warp_perspective_lambda = lambda image, transform_matrix, img_size: cv2.warpPerspective(
        image,
        transform_matrix,
        (img_size[0], img_size[1]),
        borderMode=cv2.BORDER_CONSTANT,
    )
    radar = Radar(transform_matrix=M,
                  inv_transform_matrix=IM,
                  vanishing_points=[vp1, vp2, vp3],
                  vp0_t=vp0_t,
                  image_size=img_size,
                  projector=None,
                  video_fps=video_fps,
                  result_path=result_dir,
                  result_name=test_name,
                  camera_calib_structure=camera_calibration)

    q_frames = Queue(batch_size_processing)
    q_images = Queue(batch_size_processing)
    q_predict = Queue(batch_size_processing)
    e_stop = Event()

    def read_frames():
        while cap.isOpened() and not e_stop.is_set():
            images = []
            frames = []
            for _ in range(batch_size_processing):
                ret, frame = cap.read()
                if not ret or frame is None:
                    cap.release()
                    if len(images) > 0:
                        q_images.put(images)
                        q_frames.put(frames)
                    q_images.put(None)
                    q_frames.put(None)
                    break
                frames.append(frame)
                image = cv2.bitwise_and(frame, frame, mask=road_mask)
                t_image = warp_perspective_lambda(image, M, (im_w, im_h))
                images.append(t_image)
            q_images.put(images)
            q_frames.put(frames)

    def predict():
        while not e_stop.is_set():
            try:
                images = q_images.get(timeout=TIMEOUT)
                if images is None:
                    e_stop.set()
                    break
            except Empty:
                e_stop.set()
                break
            gpu_time = time.time()
            images = np.stack(images, axis=0)
            bbox_2d, fub = inferer.simple_inference(images, 0.65, 0.65, None, None, 1000)
            q_predict.put((bbox_2d, fub))
            gpu_finish_time = (time.time() - gpu_time)
            avg_fps.append(batch_size_processing / gpu_finish_time)
            # print("GPU FPS: {}".format(batch_size_processing / gpu_finish_time))

    def process():
        while not e_stop.is_set():
            try:
                frames = q_frames.get(timeout=TIMEOUT)
                bbox_2d, fub = q_predict.get(timeout=TIMEOUT)
                if frames is None:
                    e_stop.set()
                    break
            except Empty:
                e_stop.set()
                break
            for i, (frame, box, f) in enumerate(zip(frames, bbox_2d, fub)):
                image_b = radar.process_frame(box, f, frame)
                # cv2.imshow('frame', image_b)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     e_stop.set()

    reader = Thread(target=read_frames)
    predictor = Thread(target=predict)
    processor = Thread(target=process)

    reader.start()
    predictor.start()
    processor.start()

    reader.join()
    predictor.join()
    processor.join()
    mean_fps = int(np.mean(avg_fps))
    with open(os.path.join(result_dir, 'avg_fps_' + test_name + '.txt'), 'a') as f:
        f.write("Average GPU time:" + str(mean_fps) + ", for " + video_path + "\n")
    print("Average GPU time:", mean_fps, ", for ", video_path)


if __name__ == "__main__":
    args = get_args_parser()
    vid_list = []
    calib_list = []
    store_results_list = []
    road_mask_list = []
    video_path = "/home/maco/Documents/BrnoCompSpeed/dataset"
    results_path = "/home/maco/Documents/BrnoCompSpeed/results"

    for i in range(4, 7):
        dir_list = ['session{}_center'.format(i), 'session{}_left'.format(i), 'session{}_right'.format(i)]
        vid_list.extend([os.path.join(video_path, d, 'video.avi') for d in dir_list])
        road_mask_list.extend([os.path.join(video_path, d, 'video_mask.png') for d in dir_list])
        calib_list.extend(
            [os.path.join(results_path, d, 'system_SochorCVIU_Edgelets_BBScale_Reg.json') for d in dir_list])
        store_results_list.extend([os.path.join(results_path, d) for d in dir_list])

    inferer = Inferer(source=None,
                      webcam=None,
                      webcam_addr=None,
                      weights=args.weights,
                      device="0",
                      yaml=args.yaml,
                      img_size=args.yolo_img_size,
                      half=args.half)

    for vid_path, calib_path, store_results_path, mask_path in zip(vid_list, calib_list, store_results_list,
                                                                   road_mask_list):
        start_processing = time.time()
        print("Processing: {}".format(vid_path))
        batch_test_video(inferer,
                         calib_path,
                         vid_path,
                         mask_path,
                         args.img_size,
                         store_results_path,
                         args.test_name,
                         args.batch_size_processing)
        print("Finished. Processing time: {}".format(time.time() - start_processing))
