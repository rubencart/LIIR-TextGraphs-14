#!/bin/bash

# example usage:
# bash run_scripts/run_inference_hyperparams.sh ./config/v2.1/chains_hyper_test.json 2 20 L

# adjust these lines to your own set-up
export PYTHONPATH="$PYTHONPATH":/export/home2/NoCsBack/hci/rubenc/textgraphs
export PYTHONPATH="$PYTHONPATH":/export/home1/NoCsBack/hci/rubenc/textgraphs
source ~/.bashrc
conda activate genv

CONFIG_PATH=$1
export CUDA_VISIBLE_DEVICES=$2
NUM_EVAL=$3
PARAM=$4

python3.7 -u hyperparams/test_hyperparams.py --config_path "$CONFIG_PATH" --num_eval "$NUM_EVAL" --param "$PARAM"
