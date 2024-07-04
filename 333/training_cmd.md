# How to use docker?
sudo docker images
sudo docker run --rm --runtime=nvidia -it --gpus all -v /media/itx4090/系统1/code:/home/ yolo_v5:1_1 bash
conda activate test

sudo docker run --rm -it  -v /home/haochen/HKCRC:/home/ yolo_v5:1_2 bash

# training cmd for small human detection
python train.py --img 640 --epochs 1000 --data dataset_human.yaml --weights yolov5l.pt --optimizer Adam --hyp data/hyps/hyp.scratch-high_site.yaml

python train.py --img 1280 --epochs 1000 --data dataset_human.yaml --weights yolov5l6.pt --optimizer Adam --hyp data/hyps/hyp.scratch-high_site.yaml

# training cmd for 333 system based on dataset-333-v1

## test1
python train.py --img 1280 --epochs 1000 --data dataset_333_v1.yaml --weights yolov5l6.pt --optimizer Adam --hyp data/hyps/hyp.scratch-high_site.yaml

## test2
python train.py --img 1280 --epochs 100 --data data/dataset_333_v1.yaml --weights yolov5l6.pt --optimizer Adam --hyp data/hyps/hyp.scratch-high.yaml
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      98/99      6.28G    0.02033    0.01094   0.001074          7       1280: 1
                 Class     Images  Instances          P          R      mAP50   
                   all         28        140      0.848      0.902       0.86       0.56

# training cmd for 333 system based on dataset-333-v3_1
python train.py --img 1280 --epochs 1000 --data data/dataset_333_v3.yaml --weights yolov5l6.pt --optimizer Adam --hyp data/hyps/hyp.scratch-high.yaml

# training cmd for 333 system based on dataset-333-v4_1
python train.py --img 1280 --epochs 1000 --batch-size 1 --data data/dataset_333_v4.yaml --weights yolov5l6.pt --optimizer Adam --hyp data/hyps/hyp.scratch-high_site.yaml

# training cmd for 333 system based on dataset-333-v4_2
python train.py --img 1280 --epochs 1000 --batch-size 1 --data data/dataset_333_v4.yaml --weights '' --cfg models/hub/yolov5l6.yaml --optimizer Adam --hyp data/hyps/hyp.scratch-high.yaml

                 Class     Images  Instances          P          R      mAP50   mAP50-95: 
                   all        100        627      0.696      0.586      0.662       0.37
                  hook        100         99      0.611      0.505      0.567      0.185
                   mic        100        100      0.532       0.78      0.781      0.543
             mic_frame        100        100      0.988      0.797      0.961      0.637
                people        100        328      0.652      0.262      0.338      0.116
