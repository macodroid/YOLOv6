import os

import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms

from transform_3D_utils.utils import (
    get_calibration_points,
    compute_camera_calibration,
    get_transform_matrix_with_criterion,
    blob_boxer,
    convert_to_yolo_annotation_format,
)


class Boxer:
    def __init__(
        self,
        video_path,
        saved_path_images,
        saved_path_annotations,
        mask_path,
        calibration_file,
        model,
        image_width=960,
        image_height=540,
        frame_interval=25,
        pair_of_vanishing_points="23",
    ):
        self.video_path = video_path
        self.saved_path_images = saved_path_images
        self.saved_path_annotations = saved_path_annotations
        self.mask_path = mask_path
        self.calibration_file = calibration_file
        self.model = model
        self.image_width = image_width
        self.image_height = image_height
        self.frame_interval = frame_interval
        self.pair_of_vanishing_points = pair_of_vanishing_points

    def get_vanishing_points(self):
        calibration_vp1, calibration_vp2, calibration_pp = get_calibration_points(
            self.calibration_file
        )
        vp1, vp2, vp3, _, _, _ = compute_camera_calibration(
            [calibration_vp1, calibration_vp2], calibration_pp
        )
        # Converting to homogenous coordinates
        vp1 = vp1[:-1] / vp1[-1]
        vp2 = vp2[:-1] / vp2[-1]
        vp3 = vp3[:-1] / vp3[-1]
        return vp1, vp2, vp3

    def process_video(self):
        frame_counter = 0
        (
            vanishing_point_1,
            vanishing_point_2,
            vanishing_point_3,
        ) = self.get_vanishing_points()
        road_mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)

        # We are just using vanishing point 2 and 3.
        M, IM = get_transform_matrix_with_criterion(
            vanishing_point_3,
            vanishing_point_2,
            road_mask,
            self.image_width,
            self.image_height,
        )
        vp0_t = np.array([vanishing_point_1], dtype="float32")

        vp0_t = np.array([vp0_t])
        vp0_t = cv2.perspectiveTransform(vp0_t, M)
        vp0_t = vp0_t[0][0]

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error opening video stream or file")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_counter % self.frame_interval == 0:
                frame = cv2.bitwise_and(frame, frame, mask=road_mask)
                t_image = cv2.warpPerspective(
                    frame,
                    M,
                    (self.image_width, self.image_height),
                    borderMode=cv2.BORDER_CONSTANT,
                )
                try:
                    cv2.imwrite(
                        f"{self.saved_path_images}/{frame_counter}.jpg", t_image
                    )
                except:
                    print("Error saving image")

                transform = transforms.Compose([transforms.ToTensor()])
                t_image = transform(t_image)
                # Transfer to GPU
                t_image = t_image.to(device)
                # Predict
                result = model([t_image])
                result = result[0]

                for indx, label in enumerate(result["labels"]):
                    if label in [3, 4, 6, 8] and result["scores"][indx] > 0.65:
                        mask = result["masks"][indx, :, :].detach().cpu().numpy()
                        x_min, x_max, y_min, y_max, cc = blob_boxer(
                            mask, vp0_t, self.image_height
                        )
                        (
                            x_center,
                            y_center,
                            bb_width,
                            bb_height,
                        ) = convert_to_yolo_annotation_format(
                            x_min,
                            x_max,
                            y_min,
                            y_max,
                            self.image_width,
                            self.image_height,
                        )
                        if os.path.isfile(
                            f"{self.saved_path_annotations}/{frame_counter}.txt"
                        ):
                            with open(
                                f"{self.saved_path_annotations}/{frame_counter}.txt",
                                "a",
                            ) as f:
                                f.write(
                                    f"{label} {x_center} {y_center} {bb_width} {bb_height} {cc}\n"
                                )
                        else:
                            with open(
                                f"{self.saved_path_annotations}/{frame_counter}.txt",
                                "w",
                            ) as f:
                                f.write(
                                    f"{label} {x_center} {y_center} {bb_width} {bb_height} {cc}\n"
                                )
            frame_counter += 1


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights="COCO_V1", progress=True
    )
    model.eval()
    model.to(device)

    dataset_names = []
    try:
        os.mkdir(f"dataset_annotations")
    except:
        print("Directory already exists")

    for i in range(4):
        dataset_names.append(f"session{i}_center")
        dataset_names.append(f"session{i}_left")
        dataset_names.append(f"session{i}_right")

    for dataset_name in dataset_names:
        print("Starting with dataset: ", dataset_name)
        saved_images_path = f"../dataset_annotations/{dataset_name}/images"
        saved_annotation_path = f"../dataset_annotations/{dataset_name}/annotations"
        video_path: str = f"../dataset/{dataset_name}/video.avi"
        mask_path: str = f"../dataset/{dataset_name}/video_mask.png"
        vanishing_point_file: str = (
            f"../dataset/{dataset_name}/system_SochorCVIU_ManualCalib_ManualScale.json"
        )

        try:
            os.mkdir(f"dataset_annotations/{dataset_name}")
            os.mkdir(saved_images_path)
            os.mkdir(saved_annotation_path)
        except FileExistsError:
            print("Directory already exists")

        boxer = Boxer(
            video_path=video_path,
            saved_path_images=saved_images_path,
            saved_path_annotations=saved_annotation_path,
            mask_path=mask_path,
            calibration_file=vanishing_point_file,
            model=model,
        )
        boxer.process_video()
        print("dataset: ", dataset_name, "finished")
