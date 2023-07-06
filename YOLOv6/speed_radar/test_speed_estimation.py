import argparse
import time

import cv2
import numpy as np

from transform_3D_utils.radar import Radar
from trt_inferer import TrtInferer
from utils import get_world_coordinates_on_road_plane, get_transform_matrix_with_criterion, \
    get_calibration_params, compute_camera_calibration
from YOLOv6.yolov6.core.inferer import Inferer
from YOLOv6.yolov6.utils.events import LOGGER


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Yolov6 3d speed measurement', add_help=add_help)
    parser.add_argument('--weights', type=str, default='weights/yolov6s.pt', help='model path(s) for inference.')
    parser.add_argument('--trt-model', type=str, default='', help='model path(s) for inference.')
    parser.add_argument('--source', type=str, default='data/images', help='the source path, e.g. image-file/dir.')
    parser.add_argument('--yaml', type=str, default='../data/bcs.yaml', help='data yaml file.')
    parser.add_argument('--yolo-img-size', nargs='+', type=int, default=[544, 960],
                        help='the image-size(h,w) in inference size.')
    parser.add_argument('--img-size', nargs='+', type=int, default=[960, 540],
                        help='The image size (h,w) for inference.')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='NMS IoU threshold for inference.')
    parser.add_argument('--half', default=True, action='store_true',
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
    args = parser.parse_args()
    LOGGER.info(args)
    return args


def process_video(trt_inferer, video_path: str, road_mask: np.ndarray,
                  transform_matrix: np.ndarray, inv_transform_matrix: np.ndarray, img_size: tuple,
                  inferer: Inferer, vp0_t: np.ndarray, vp1: np.ndarray,
                  vp2: np.ndarray, vp3: np.ndarray, projector, offline: bool, video_fps: int, result_dir: str,
                  test_name: str, camera_calib_structure: dict):
    cap = cv2.VideoCapture(video_path)
    radar = Radar(transform_matrix=transform_matrix,
                  inv_transform_matrix=inv_transform_matrix,
                  vanishing_points=[vp1, vp2, vp3],
                  vp0_t=vp0_t,
                  image_size=img_size,
                  projector=projector,
                  video_fps=video_fps,
                  result_path=result_dir,
                  result_name=test_name,
                  camera_calib_structure=camera_calib_structure)
    if not cap.isOpened():
        print("Error opening video stream or file")
    frame_count = 0
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.bitwise_and(frame, frame, mask=road_mask)
        t_image = cv2.warpPerspective(
            image,
            transform_matrix,
            (img_size[0], img_size[1]),
            borderMode=cv2.BORDER_CONSTANT,
        )
        bbox2d, fub = trt_inferer.infer(t_image)
        # bbox2d, fub = inferer.simplified_inference(t_image, 0.65, 0.4, None, None, 1000)
        if offline:
            radar.process_frame_offline(bbox2d, fub)
        else:
            if bbox2d.shape[0] == 0:
                # cv2.imshow('frame', frame)
                continue
            else:
                img_with_3d_bb = radar.process_frame(bbox2d, fub, frame)
                cv2.imshow('frame', img_with_3d_bb)
            # Increment the frame counter
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Calculate and print the FPS
    fps = frame_count / elapsed_time
    print(f'FPS: {fps}')

def main(args):
    inferer = Inferer(source=None,
                      webcam=None,
                      webcam_addr=None,
                      weights=args.weights,
                      device="0",
                      yaml=args.yaml,
                      img_size=args.yolo_img_size,
                      half=args.half)

    camera_calibration = get_calibration_params(args.camera_calib_file)
    scale = camera_calibration['scale']
    vp1, vp2, vp3, pp, road_plane, focal = compute_camera_calibration(
        [camera_calibration['vp1'], camera_calibration['vp2']], camera_calibration['pp']
    )
    vp1 = vp1[:-1] / vp1[-1]
    vp2 = vp2[:-1] / vp2[-1]
    vp3 = vp3[:-1] / vp3[-1]
    road_mask = cv2.imread(args.road_mask_path, cv2.IMREAD_GRAYSCALE)
    projector = lambda x: scale * get_world_coordinates_on_road_plane(x, focal, road_plane, pp)

    M, IM = get_transform_matrix_with_criterion(
        vp3,
        vp2,
        road_mask,
        args.img_size[0],
        args.img_size[1],
    )
    vp0_t = np.array([vp1], dtype="float32")

    vp0_t = np.array([vp0_t])
    vp0_t = cv2.perspectiveTransform(vp0_t, M)
    vp0_t = vp0_t[0][0]

    trt_inferer = TrtInferer(trt_engine_path=args.trt_model, image_size=args.yolo_img_size, stride=32, half=args.half)

    process_video(trt_inferer, args.vid_path, road_mask, M, IM, args.img_size, inferer, vp0_t, vp1, vp2, vp3, projector,
                  offline=args.process_offline, video_fps=args.video_fps, result_dir=args.result_dir,
                  test_name=args.test_name, camera_calib_structure=camera_calibration)


if __name__ == "__main__":
    args = get_args_parser()
    main(args)
