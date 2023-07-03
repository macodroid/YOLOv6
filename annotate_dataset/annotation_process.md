# Data annotation

To train YOLOv6_transform_3D model for detecting cars in 2D boxes with addition parameter.  
We used two dataset:

1. [BrnoCompSpeed](https://github.com/JakubSochor/BrnoCompSpeed)
2. [BoxCars](https://github.com/JakubSochor/BoxCars) - fine-tuning dataset

## BrnoCompSpeed

First, download dataset accordingly to
the [BrnoCompSpeed Download](https://github.com/JakubSochor/BrnoCompSpeed#download).  
Downloaded dataset need to be stored in root directory of this project in **dataset** directory or the paths in the
scripts need to be changed [``anotator_bcs.py, clean_dataset_bcs.py, train_val_split.py``].
Step to run annotation and cleaning scripts:

1. Run ``python3 anotator_bcs.py`` to annotate dataset.
2. Run ``python3 clean_dataset_bcs.py`` to clean dataset.
3. Run ``python3 train_val_split.py`` to split dataset into train and validation set.

## BoxCars

First, download dataset accordingly to the [BoxCars Download](https://github.com/JakubSochor/BoxCars).
Downloaded dataset need to be stored in root directory of this project in **dataset** directory or the paths in the
scripts need to be changed [``anotator_boxcar116.py, create_split_boxcar116.py``].
Step to run annotation and cleaning scripts:

1. Run ``python3 anotator_boxcar116.py`` to annotate dataset.
2. Run ``python3 create_split_boxcar116.py`` to split dataset into train and validation set.
