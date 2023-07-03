import os
import pickle
import shutil
import cv2


def convert_to_yolo_annotation_format(
    x_min, x_max, y_min, y_max, image_width, image_height
):
    x_center = ((x_min + x_max) / 2) / image_width
    y_center = ((y_min + y_max) / 2) / image_height
    bbox_width = (x_max - x_min) / image_width
    bbox_height = (y_max - y_min) / image_height
    return (
        x_center,
        y_center,
        bbox_width,
        bbox_height,
    )


def parse_boxcars(dataset_path, images_path, result_path):
    with open(dataset_path, "rb") as f:
        ds = pickle.load(f, encoding="latin-1", fix_imports=True)
    for sample in ds["samples"]:
        for i_id, instance in enumerate(sample["instances"]):
            filename = os.path.join(images_path, instance["filename"])
            if filename is None:
                continue
            img_name = instance["filename"].split("/")[-1]
            img_height, img_width = cv2.imread(filename).shape[:2]

            shutil.copyfile(
                filename, os.path.join(root_result_path, "images", img_name)
            )
            cc = (instance["bb_in"]["y_min"] - instance["bb_out"]["y_min"]) / (
                instance["bb_out"]["y_max"] - instance["bb_out"]["y_min"]
            )
            (
                x_center,
                y_center,
                bbox_width,
                bbox_height,
            ) = convert_to_yolo_annotation_format(
                instance["bb_out"]["x_min"],
                instance["bb_out"]["x_max"],
                instance["bb_out"]["y_min"],
                instance["bb_out"]["y_max"],
                img_width,
                img_height,
            )
            annotation_name = img_name.split(".")[0]
            with open(
                f"{result_path}/{annotation_name}.txt",
                "w",
            ) as f:
                f.write(f"3 {x_center} {y_center} {bbox_width} {bbox_height} {cc}\n")


if __name__ == "__main__":
    root_images_path = "/home/k/kocur15/data/BoxCars116k/images_warped23"
    annotation_file_path = "/home/k/kocur15/data/BoxCars116k/dataset_warped23.pkl"
    root_result_path = "/home/m/macko70/projects/masters/dataset/BoxCars116k"

    try:
        os.makedirs(os.path.join(root_result_path, "images"))
        os.makedirs(os.path.join(root_result_path, "labels"))
    except FileExistsError:
        os.rmdir(os.path.join(root_result_path, "images"))
        os.rmdir(os.path.join(root_result_path, "labels"))
        os.makedirs(os.path.join(root_result_path, "images"))
        os.makedirs(os.path.join(root_result_path, "labels"))

    parse_boxcars(
        annotation_file_path, root_images_path, os.path.join(root_result_path, "labels")
    )
