# How to use docker?
sudo docker images
sudo docker run --rm --runtime=nvidia -it --gpus all -v /media/itx4090/系统1/code:/home/ yolo_v5:1_1 bash
conda activate test

# training cmd for small human detection
python train.py --img 640 --epochs 1000 --data dataset_human.yaml --weights yolov5l.pt --optimizer Adam --hyp data/hyps/hyp.scratch-high_site.yaml

python train.py --img 1280 --epochs 1000 --data dataset_human.yaml --weights yolov5l6.pt --optimizer Adam --hyp data/hyps/hyp.scratch-high_site.yaml

# training cmd for 333 system based on dataset-333-v1

## test1
python train.py --img 1280 --epochs 1000 --data dataset_333_v1.yaml --weights yolov5l6.pt --optimizer Adam --hyp data/hyps/hyp.scratch-high_site.yaml

## test2
python train.py --img 1280 --epochs 100 --data data/dataset_333_v1.yaml --weights yolov5l6.pt --optimizer Adam --hyp data/hyps/hyp.scratch-high.yaml

# training cmd for 333 system based on dataset-333-v3_1
python train.py --img 1280 --epochs 1000 --data dataset_333.yaml --weights yolov5l6.pt --optimizer Adam --hyp data/hyps/hyp.scratch-high.yaml

# training cmd for 333 system based on dataset-333-v4_1
python train.py --img 1280 --epochs 1000 --batch-size 1 --data dataset_333.yaml --weights yolov5l6.pt --optimizer Adam --hyp data/hyps/hyp.scratch-high_site.yaml