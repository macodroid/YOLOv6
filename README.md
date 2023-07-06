# YOLOv6 Transform 3D
This is fork of original YOLOv6 repository.
So models predict 2D bouding box with additional parameters. This change was according to the paper [Detection of 3D Bounding Boxes of Vehicles Using Perspective Transformation for Accurate Speed Measurement](https://arxiv.org/pdf/2003.13137.pdf).

To run whole pipeline You need to install requirements:

``pip install -r requirements.txt``

To obtain dataset BrnoCompSpeed on which You can try to run it 
You need to contact authors of dataset as it is not public.
Contact information can be found [here](https://github.com/JakubSochor/BrnoCompSpeed).
Or second option for obtaining dataset is using FMFI UK saturn server.

After obtaining dataset You need to place dataset in format as:
```
    ├── BrnoCompSpeed                    # Test files (alternatively `spec` or `tests`)
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
All path need to be change in ``batch_test_speed_estimation.py`` and if running TensorRT ``tensorrt_batch_test_speed_estimation.py``.
All trained and exported model can be found in directory ``models``.

To run pipeline You can simply run:
``python batch_test_speed_estimation.py --weights=<path_to_the_model> --test-name=<name_of_the_test_file> \\  
--batch-size-processing=32 --half  --show-video=True``

If You want to change input size look for all arguments in ``batch_test_speed_estimation.py``.

To run pipeline on video utilizing TensorRT with operation precision INT8:
``python tensorrt_batch_test_speed_estimation.py --trt-model=<path_to_the_model> --test-name=<name_of_the_test_file> \\``

All models can be found in [here](https://github.com/macodroid/YOLOv6)