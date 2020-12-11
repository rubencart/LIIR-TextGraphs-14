#!/bin/bash

# example usage:
# bash run_scripts/run_single_fact_ranker.sh ./config/v2.1/single_fact_ranknet_1.json 2,3

# adjust these lines to your own set-up
export PYTHONPATH="$PYTHONPATH":/export/home1/NoCsBack/hci/rubenc/textgraphs
source /export/home1/NoCsBack/hci/rubenc/miniconda3/etc/profile.d/conda.sh
source /export/home2/NoCsBack/hci/rubenc/miniconda3/etc/profile.d/conda.sh
conda activate genv

CONFIG_PATH=$1
export CUDA_VISIBLE_DEVICES=$2

python3.7 -u rankers/single_fact_ranker.py --config_path "$CONFIG_PATH"
