echo "Running Nano model..."
python batch_test_speed_estimation.py --weights=/home/maco/Documents/repos/YOLOv6/models/qa_nano/yolov6_qa_nano_transform_3d.pt --test-name=yolov6_nano_qa_b32_480_270_fp16 --half --batch-size-processing=32
echo "Nano finished..."

echo "Running Nano distill model..."
python batch_test_speed_estimation.py --weights=/home/maco/Documents/repos/YOLOv6/models/qa_nano_distill/yolov6_qa_nano_distill_transform_3d.pt --test-name=yolov6_nano_qa_distill_b32_480_270_fp16 --half --batch-size-processing=32
echo "Nano distill finished..."

echo "Running small model..."
python batch_test_speed_estimation.py --weights=/home/maco/Documents/repos/YOLOv6/models/qa_small/yolov6_qa_small_3d_transform.pt --test-name=yolov6_small_qa_b32_480_270_fp16 --half --batch-size-processing=32
echo "Small finished..."

echo "Running small model..."
python batch_test_speed_estimation.py --weights=/home/maco/Documents/repos/YOLOv6/models/qa_small_distill/weights/best_ckpt.pt --test-name=yolov6_small_qa_distill_b32_480_270_fp16 --half --batch-size-processing=32
echo "Small finished..."

echo "Running medium model..."
python batch_test_speed_estimation.py --weights=/home/maco/Documents/repos/YOLOv6/models/qa_medium/yolov6_qa_medium_3d_transform.pt --test-name=yolov6_medium_qa_b32_480_270_fp16 --half --batch-size-processing=32
echo "Medium finished..."

echo "Running medium model..."
python batch_test_speed_estimation.py --weights=/home/maco/Documents/repos/YOLOv6/models/qa_medium/yolov6_qa_medium_3d_transform.pt --test-name=yolov6_medium_qa_b32_480_270_fp16 --half
echo "Medium finished..."

echo "Running Large model..."
python batch_test_speed_estimation.py --weights=/home/maco/Documents/repos/YOLOv6/models/large/yolov6_large_3d_transform.pt --test-name=yolov6_large_b32_480_270_fp16 --half --batch-size-processing=32
echo "Large finished..."
