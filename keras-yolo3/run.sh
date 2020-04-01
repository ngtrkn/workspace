# train
#python train_yolo.py --object /mnt/sda1/code/dev/heligate/maru_debug/Infordio-OCR/model_data/custom_object_marucheck.txt --label /mnt/sda1/code/dev/data/mark/train_marucheck_750.txt --output model_data/yolov3 -r gray
python train_yolo.py --model /mnt/sda1/code/github/workspace/keras-yolo3/model_data/pretrain/mc_logsep046-loss52.870-val_loss77.223.h5 --object /mnt/sda1/code/dev/heligate/maru_debug/Infordio-OCR/model_data/custom_object_marucheck.txt --label /mnt/sda1/code/dev/data/mark/train_marucheck_750.txt --output model_data/yolov3 -r gray
