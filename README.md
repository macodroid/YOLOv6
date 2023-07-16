# YOLOv6 Transform 3D

This is fork of original YOLOv6 repository.
The models predict 2D bouding box with additional parameters. This change was according to the
paper [Detection of 3D Bounding Boxes of Vehicles Using Perspective Transformation for Accurate Speed Measurement](https://arxiv.org/pdf/2003.13137.pdf).

## Dataset

To obtain dataset BrnoCompSpeed on which You can try to run it
You need to contact authors of dataset as it isn't public.
Contact information can be found [here](https://github.com/JakubSochor/BrnoCompSpeed).

## Requirements

To train models and run whole pipeline You need to install requirements:

``pip install -r requirements.txt``  
 
Nvidia driver: **535.54.03**  
CUDA version: **12.2**

## Training

To train our modified YOLOv6 model You need to prepare your dataset to the YOLO format.  
Example of directories and files structure of YOLO format:

```
custom_dataset
├── images
│   ├── train
│   │   ├── train0.jpg
│   │   └── train1.jpg
│   └── val
│       ├── val0.jpg
│       └── val1.jpg
└── labels
    ├── train
    │   ├── train0.txt
    │   └── train1.txt
    └── val
        ├── val0.txt
        └── val1.txt
```

Example of annotation file in YOLO format:

```txt
# class_id center_x center_y bbox_width bbox_height
0 0.300926 0.617063 0.601852 0.765873
# this second annotation is here is there are for example two dogs in one picture. (But this is not case in this dataset)
1 0.575 0.319531 0.4 0.551562
```  

After creating dataset You need to create file ``custom.yaml`` in ``./data/`` directory.

``
python ./yolov6/train.py <args>
``  
All arguments can be found in ``./yolov6/train.py`` file.
In training process best model is saved in ``./runs/train/<experiment_name>/weights/best.pt``.

## Testing speed estimation pipeline

After obtaining dataset You need to place dataset in format as:

```
    ├── BrnoCompSpeed
    │    ├── dataset
    │    │      ├── season0_center
    │    │      │     ├── video.avi
    │    │      │     └── video_mask
    │    │      ...
    │    │      └── season6_right
    │    │            ├── video.avi
    │    │            └── video_mask
    │    │              
    │    └── results
    │          ├── season0_center
    │          │     └── system_SochorCVIU_Edgelets_BBScale_Reg.json # Calibration file, without this pipele will not work 
    │          ...
    │          └── season6_right
    │                └── system_SochorCVIU_Edgelets_BBScale_Reg.json # Calibration file       
```

All path need to be change in ``batch_test_speed_estimation.py`` and if running
TensorRT ``tensorrt_batch_test_speed_estimation.py``.

To run pipeline You can simply run:
``python batch_test_speed_estimation.py --weights=<path_to_the_model> --test-name=<name_of_the_test_file> \\  
--batch-size-processing=32 --half --show-video=True``

If You want to change input size look for all arguments in ``batch_test_speed_estimation.py``.

### Quantization and TensorRT

To be able to run quantization and TensorRT You need to install CUDA
Toolkit [here](https://developer.nvidia.com/cuda-downloads) and of course You need to have compatible hardware for
running quantization with TensorRT.
To learn more about TensorRT and quantization look
at [this](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/)
blog post.

#### Quantization

1. `best.pt` is the model obtained from training need to be exported to ONNX format:
   ``python ./deploy/ONNX/export_onnx.py --weights=<path_to_the_model> --img-size=<size_of_input_image> --batch-size=<batch_size>``
   All arguments can be found in ``./deploy/ONNX/export_onnx.py`` file.
2. After exporting model to ONNX format You need to run quantization:
   ``python ./tools/quantization/tensorrt/post_training/onnx_to_tensorrt.py --onnx-model=<path_to_the_model> --explicit-batch --vv --fp16 --int8 --calibration-data=<path_to_calibration_data> --calibration-batch-size=<batch_size>``
   All arguments can be found in ``./tools/quantization/tensorrt/post_training/onnx_to_tensorrt.py`` file.
3. To run pipeline on video utilizing TensorRT with operation precision INT8:
   ``python tensorrt_batch_test_speed_estimation.py --trt-model=<path_to_the_model> --test-name=<name_of_the_test_file> \\``

All models can be found in [here](https://github.com/macodroid/YOLOv6)

## Evaluation of speed estimation pipeline for BrnoCompSpeed dataset

To evaluate speed estimation You need to use ``eval.py``
from [BrnoCompSpeed/code](https://github.com/JakubSochor/BrnoCompSpeed/tree/master/code) repository.