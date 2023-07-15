#!/bin/bash

##################################
### RUN NAIVE BAYES EXPERIMENT ###
##################################
python main.py --mode train --dataset train --model naive-bayes
python main.py \
  --mode evaluate \
  --dataset test \
  --model naive-bayes \
  --model-path 'models/nbayes_pipeline.pkl'

#######################################
### RUN TEXT-DAVINCI-002 EXPERIMENT ###
#######################################
python main.py \
  --mode evaluate \
  --dataset test \
  --model text-davinci-002

#####################################
### RUN NEURAL NETWORK EXPERIMENT ###
#####################################
python main.py --mode preprocess --dataset train
python main.py --mode embed --dataset train

python main.py \
  --mode train \
  --dataset train \
  --model nn \
  --epochs 20 \
  --batch-size 64 \
  --learning-rate 0.001

python main.py --mode preprocess --dataset test
python main.py --mode embed --dataset test

python main.py \
  --mode evaluate \
  --dataset test \
  --model nn \
  --model-path 'models/nn_2023-07-05 13:18:24.pkl'
