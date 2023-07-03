import os
import shutil

# KISS (Keep is simple stupid :D)
def get_prefix(session):
    if session == "session0_center":
        return "s0c_"
    elif session == "session0_left":
        return "s0l_"
    elif session == "session0_right":
        return "s0r_"
    elif session == "session1_center":
        return "s1c_"
    elif session == "session1_left":
        return "s1l_"
    elif session == "session1_right":
        return "s1r_"
    elif session == "session2_center":
        return "s2c_"
    elif session == "session2_left":
        return "s2l_"
    elif session == "session2_right":
        return "s2r_"
    elif session == "session3_center":
        return "s3c_"
    elif session == "session3_left":
        return "s3l_"
    elif session == "session3_right":
        return "s3r_"
    elif session == "session4_center":
        return "s4c_"
    elif session == "session4_left":
        return "s4l_"
    elif session == "session4_right":
        return "s4r_"
    elif session == "session5_center":
        return "s5c_"
    elif session == "session5_left":
        return "s5l_"
    elif session == "session5_right":
        return "s5r_"
    elif session == "session6_center":
        return "s6c_"
    elif session == "session6_left":
        return "s6l_"
    elif session == "session6_right":
        return "s6r_"


if __name__ == "__main__":
    root_data_dir = "dataset_annotations/"
    root_cleaned_data_dir = "cleaned_dataset/"
    dataset_names = []

    for i in range(7):
        dataset_names.append(f"session{i}_center")
        dataset_names.append(f"session{i}_left")
        dataset_names.append(f"session{i}_right")

    for dataset_name in dataset_names:
        os.mkdir(f"{root_cleaned_data_dir}/{dataset_name}")
        os.mkdir(f"{root_cleaned_data_dir}/{dataset_name}/images")
        os.mkdir(f"{root_cleaned_data_dir}/{dataset_name}/annotations")
        annotations = os.listdir(root_data_dir + dataset_name + "/annotations")
        ann = [a.split(".")[0] for a in annotations]
        images = os.listdir(root_data_dir + dataset_name + "/images")
        imgs = [i.split(".")[0] for i in images]
        result = [existing for existing in imgs if existing in ann]
        prefix = get_prefix(dataset_name)
        for i in result:
            source_txt_path = (
                root_data_dir + dataset_name + "/annotations/" + i + ".txt"
            )
            destination_txt_path = (
                root_cleaned_data_dir
                + dataset_name
                + "/annotations/"
                + prefix
                + i
                + ".txt"
            )
            source_img_path = root_data_dir + dataset_name + "/images/" + i + ".jpg"
            destination_img_path = (
                root_cleaned_data_dir + dataset_name + "/images/" + prefix + i + ".jpg"
            )
            shutil.copy(source_txt_path, destination_txt_path)
            shutil.copy(source_img_path, destination_img_path)
