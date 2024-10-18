#!/bin/bash


#python train.py --model_name DSFNet --gpus 0 --load_model ./checkpoints/model_best1.pth --num_epochs 75 --batch_size 6 --lr 0.000177 --val_intervals 5  --test_large_size True --datasetname rsdata --data_dir  ./data/RsCarData/

nohup python train.py --model_name DSFNet --gpus 0 --load_model ./checkpoints/model_75.pth --resume True --num_epochs 100 --batch_size 6 --lr 0.000177 --val_intervals 5  --test_large_size True --datasetname rsdata --data_dir  ./data/RsCarData/ > output.log 2>&1 &
