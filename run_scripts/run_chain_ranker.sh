#!/bin/bash

# example usage:
# bash run_scripts/run_chain_ranker.sh ./config/chains.json 0,1,2

# adjust these lines to your own set-up
export PYTHONPATH="$PYTHONPATH":/export/home1/NoCsBack/hci/rubenc/textgraphs
source /export/home1/NoCsBack/hci/rubenc/miniconda3/etc/profile.d/conda.sh
source /export/home2/NoCsBack/hci/rubenc/miniconda3/etc/profile.d/conda.sh
conda activate genv

CONFIG_PATH=$1
export CUDA_VISIBLE_DEVICES=$2

python3.7 -u rankers/chain_ranker.py --config_path "$CONFIG_PATH"
