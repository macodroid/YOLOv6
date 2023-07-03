import os
import shutil


def copy_files(dateset_name, split_type, images, annotations):
    for img, ann in zip(images, annotations):
        shutil.copy(
            f"cleaned_dataset/{dateset_name}/annotations/{ann}",
            f"train_val/labels/{split_type}/{ann}",
        )
        shutil.copy(
            f"cleaned_dataset/{dateset_name}/images/{img}",
            f"train_val/images/{split_type}/{img}",
        )


if __name__ == "__main__":
    try:
        # create images directory
        os.mkdir("train_val/images/train")
        os.mkdir("train_val/images/val")
        # create labels directory
        os.mkdir("train_val/labels/train")
        os.mkdir("train_val/labels/val")
    except FileExistsError:
        print("Directory already exists")

    dataset_names = []
    for i in range(7):
        dataset_names.append(f"session{i}_center")
        dataset_names.append(f"session{i}_left")
        dataset_names.append(f"session{i}_right")

    for ds in dataset_names:
        annotations = os.listdir(f"cleaned_dataset/{ds}/annotations")
        images = os.listdir(f"cleaned_dataset/{ds}/images")
        sorted_annotations = sorted(annotations)
        sorted_images = sorted(images)
        train_split_ratio = 0.8
        val_split_ratio = 0.2
        train_split = int(len(sorted_annotations) * train_split_ratio)
        val_split = int(len(sorted_annotations) * val_split_ratio)
        # val split
        val_split_images = sorted_images[:val_split]
        val_split_annotations = sorted_annotations[:val_split]
        try:
            copy_files(ds, "val", val_split_images, val_split_annotations)
        except Exception as e:
            print(e)
        # train split
        train_split_images = sorted_images[val_split:]
        train_split_annotations = sorted_annotations[val_split:]
        try:
            copy_files(ds, "train", train_split_images, train_split_annotations)
        except Exception as e:
            print(e)
