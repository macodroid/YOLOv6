import os
import shutil


if __name__ == "__main__":
    root_path_for_data = "/home/m/macko70/projects/masters/dataset/speed_estimation_dataset"
    boxcar_images = "/home/m/macko70/projects/masters/dataset/BoxCars116k/images"
    boxcar_labels = "/home/m/macko70/projects/masters/dataset/BoxCars116k/labels"
    try:
        # create images directory
        os.mkdir(f"{root_path_for_data}/images/train")
        os.mkdir(f"{root_path_for_data}/images/val")
        # create labels directory
        os.mkdir(f"{root_path_for_data}/labels/train")
        os.mkdir(f"{root_path_for_data}/labels/val")
    except FileExistsError:
        print("Directory already exists")

    train_split = 0.8
    val_split = 0.2

    images = os.listdir(boxcar_images)
    labels = os.listdir(boxcar_labels)
    filenames = [image.split(".")[0] for image in images]
    num_files = len(filenames)

    train_num = int(num_files * train_split)
    train_file = filenames[:train_num]
    val_file = filenames[train_num:]

    for i, t_file in enumerate(train_file):
        shutil.copyfile(
            f"{boxcar_images}/{t_file}.png",
            f"{root_path_for_data}/images/train/{t_file}.png",
        )
        shutil.copyfile(
            f"{boxcar_labels}/{t_file}.txt",
            f"{root_path_for_data}/labels/train/{t_file}.txt",
        )
        
    print("Train done!")

    for i, v_file in enumerate(val_file):
        shutil.copyfile(
            f"{boxcar_images}/{v_file}.png",
            f"{root_path_for_data}/images/val/{v_file}.png",
        )
        shutil.copyfile(
            f"{boxcar_labels}/{v_file}.txt",
            f"{root_path_for_data}/labels/val/{v_file}.txt",
        )
        
    print("Val done!")
