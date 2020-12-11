# LIIR-KULeuven at TextGraphs-14

This repository contains the implementation for our submission to the TextGraphs-14 shared task on Multi-Hop Inference for Explanation Regeneration.
If you run into troubles, or if you have questions, contact Ruben Cartuyvels at first . last @ kuleuven . be.

We used Das et al.'s implementation for the single-fact and path-rerank baseline results in our paper: https://github.com/ameyagodbole/multihop_inference_explanation_regeneration.
However, we implemented training (with a pairwise loss) and inference for the single-fact baseline ourselves as well.

Link to workshop web page: https://sites.google.com/view/textgraphs2020

Link to competition github page: https://github.com/cognitiveailab/tg2020task

Link to 2019 competition summary: https://www.aclweb.org/anthology/D19-5309.pdf

Link to WorldTree v2 dataset paper: https://www.aclweb.org/anthology/2020.lrec-1.671.pdf

If you use this work please cite:
```
@inproceedings{cartuyvels2020autoregressive,
    title     = {Autoregressive Reasoning over Chains of Facts with Transformers},
    author    = {Cartuyvels, Ruben and Spinks, Graham and Moens, Marie-Francine},
    booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
    year      = {2020},
    publisher = {Association for Computational Linguistics},
}
```

## Installation

<!--
To install wmd on Mac OS:
`CFLAGS=-stdlib=libc++ pip install wmd==1.3.1`.
-->

Requirements:
- (mini)conda
- PyTorch
- transformers
- see environment.yml

```
git clone --recurse-submodules git@github.com:rubencart/textgraphs.git
cd textgraphs
conda env create -f environment.yml
cd tg2020task/
mkdir dataset_v2
make dataset
unzip tg2020task-dataset.zip -d dataset_v2/
cd ..
```

## Running

ARCF training or inference:
```
conda activate genv
export PYTHONPATH="$PYTHONPATH":<path_to_textgraphs_dir>
CUDA_VISIBLE_DEVICES=2 python3.7 -u rankers/chain_ranker.py --config_path ./config/v2.1/chains_ranknet_1.json
```

Single-fact baseline training and inference (we used Das et al.'s implementation for results in the paper):
```
conda activate genv
export PYTHONPATH="$PYTHONPATH":<path_to_textgraphs_dir>
CUDA_VISIBLE_DEVICES=2 python3.7 -u rankers/single_fact_ranker.py --config_path ./config/v2.1/single_fact_xent.json
```

### Settings

Settings can be chosen in `.json` files in `config/`.
That folder contains all `.json` files leading to the results in the paper, except for the results obtained with Das et al.'s implementation. 
An overview of the meaning of the settings is given below.

Notably, if you are using LR decay, and lr_decay_per_epoch = false (so the LR decreases after every batch), you need to
set approx_num_steps to the approximate number of steps in an epoch.
You could first do a run with lr_decay_per_epoch = true, check the amount of steps, set approx_num_steps, and use lr_decay_per_epoch = false afterwards.
The reason is that the number of batches in an epoch is not constant. When use_all_negatives = false, there is 1 batch per 
visible positive fact per question (the same question can occur multiple times in an epoch, max 2^powerset_complete_combs times,
but this is fixed, known in advance, and saved in a `.bin` in train_chains_file). Since the amount of visible positive facts
depends on the gold facts in the prefix, and since the gold facts in the prefix are sampled during training, the exact number of batches is
not known in advance. 
It is not feasible to train for every question with every subset of corresponding gold facts as prefix, so we sample the subset.
The total number of steps usually varies only with 1% or so.
We know this is annoying and bad engineering, sorry.
Alternatively, you could just use lr_decay_per_epoch = true, performance was comparable. Or count on ADAM's built-in
LR decay and set lr_decay = false.

ARCF (`chains_ranknet_1.json`):
```
{
  "seed": 41,           # random seed for reproducibility
  "do_train": true,     # whether to train, validate or test
  "do_eval": false,
  "do_test": true,
  "evaluate_during_training": true,

  "task": "20",                         # 19 or 20 data
  "algo": "chains",                     # 'chains' = ARCF, 'single-fact' = single-fact baseline
  "2020v2": true,                       # true for updated WorldTree v2 data (v2.1 instead of v2)
  "fact_path": "./tg2020task/dataset_v2/tables",    # paths to questions and fact tables
  "qa_dir": "./tg2020task/dataset_v2/",
  "train_qa_file": "questions.train.tsv",
  "val_qa_file": "questions.dev.tsv",
  "test_qa_file": "questions.test.tsv",
  "lvl_col_name": "arcset",

  "data_dir": "./data",         # folder where dataset files will be cached for later reuse
  "output_dir": "./output",     # checkpoints, results and settings will be saved in an automatically 
                                # generated subfolder of output_dir with a timestamp
  "cache_dir": "./cache",       # pretrained model weights are saved here by huggingface's transformers

  "answer_choices": "correct",  # remove incorrect multiple choice answers from the questions
  "mark_correct_in_qa": true,   # if answer_choices == 'all', mark correct answer with '(correct)'

  "model_type": "roberta",      # which transformer architecture is used
  "model_name_or_path": "distilroberta-base",   # huggingface model_name or path to checkpoint to be loaded
  "_model_name_or_path": "./output/roberta_chains_2020-05-18_11h11/checkpoint-4_31639/",
  "config_name": "",
  "tokenizer_name": "",
  "init_counters_from_checkpoint": true,    # initialize training counters from checkpoint
  "init_gear_from_checkpoint": true,        # initialize training gear (optimizer, LR schedule,...) from checkpoint

  "no_lower_case": false,               # change if using cased transformer model
  "gradient_accumulation_steps": 1,
  "learning_rate": 2e-05,
  "weight_decay": 0.01,
  "adam_epsilon": 1e-08,
  "max_grad_norm": 1.0,
  "num_train_epochs": 4,
  "max_steps": -1,
  "warmup_steps": 0,
  
  # set to the approx. number of steps in an epoch, but epoch length is not exactly constant,
  # so for LR decay to work correctly when lr_decay_per_epoch=false, the code needs to know this
  # check bert_ranker.py
  "approx_num_steps": 26510,            
  "lr_decay_per_epoch": false,          # if true, decay the LR every epoch, else, decay after every batch
  "lr_decay": true,                     # if false, keep lr constant (overwrites other lr decay settings).
                                        # otherwise, decay linearly starting from learning_rate, until 0
  # log intermediate results every X steps set to the approx. number of steps in an epoch
  "logging_steps": 26510,               
  "save_steps": 26510,              # save checkpoint every X steps
  "eval_steps": 26510,              # run on dev set every X steps (result is only logged if current step % logging_steps == 0)
  "eval_all_checkpoints": false,
  "no_cuda": false,
  "overwrite_output_dir": true,
  "overwrite_cache": false,            # set to true to recompute datasets

  "fp16": false,
  "fp16_opt_level": "O1",
  
  # file where cached training samples are saved to and loaded from
  "train_chains_file": "./data/20-v2-train-chains-2-combs.bin", 
  # setting that determines amount of training samples generated per question in 1 epoch, see train_chain_dataset.py
  "powerset_complete_combs": 2,
  "nearest_k_visible": 180,                 # neighborhood size
  # when using pointwise losses (XENT), batches might contain samples from different questions if this is set to true
  # else, no effect
  "single_question_per_batch": false,  
  # For pairwise/contrastive losses (ranknet, nce,...), there is 1 positive sample per batch, and the batch is
  # filled with negatives until per_gpu_train_tokens_per_batch is reached. All positives are always used. If
  # this setting is set to true, all negative samples from the current neighborhood are used, possibly reusing the positives.
  "use_all_negatives": false,               
  # if > 0, replace fraction of positive samples in the prefix by randomly sampled negatives 
  "condition_on_negs_rate": 0.0,        
  # sample a rate between 0 and condition_on_negs_rate for every sample
  "sample_condition_on_negs_rate": true,
  "init_condition_on_negs_rate": 0.0,
  "max_condition_on_negs_rate": 0.0,
  # linear or steps, check chain_ranker.py. Determines condition on negs schedule together with above 2 settings 
  "anneal_cond_on_negs_schedule": null,

  "mark_answer_in_qa": true,                # mark answer with (answer) in question
  "mark_expl": true,                        # start appended list of facts with (explanation)
  "answer_behind_expl": false,              # put the answer behind the list of facts
  
  # compute neighborhood distances with mode 'tf_idf', 'wmd' (word mover's distance), or 'sent_embed'
  "distance_mode": "tf_idf",
  "distance_func": "cosine",                            # cosine or euclidean
  "_distance_mode": "sent_embed",
  "embedder_name_or_path": "deepset/sentence_bert",     # sentence embedding model to use (huggingface id or path)
  "embedder_aggregate_func": "avg_pool",
  "overwrite_embeddings": false,
  "distance_wo_fill": false,                            # compute neighborhood distances ignoring filler words
  "compute_overlap": false,                             # compute lexical overlap
  # sample number of facts in prefix of training sample from categorical distribution with frequencies of number of gold facts of 
  # questions in training dataset as probabilities (so trained more often with <5 fact sequences than with 15-20 fact sequences) 
  "categorical_expl_length": false,                     
  
  # if false, use last iteration's scores to rank scored but unselected facts second (S2). Else, use average over iterations.
  "average_scores_over_partials": false,
  # rank rest with tf_idf distance from concatenation of question and selected facts (R3)
  "rank_rest": "tf_idf",
  "max_expl_length": 8,
  "min_expl_length": 1,
  # stop iterating when p without newly appended facts is ranked highest (> than p with newly appended facts)
  "predict_stop_expl": true,
  # the above only happens when the difference > stop_delta
  "stop_delta": 0.0,
  # rank selected facts first (otherwise all scored facts would just be ranked by their S2 scores, regardless of
  # whether they had been selected or not)
  "use_partial_expl_in_ranking": true,
  # do beam search when beam_size > 1
  "beam_size": 1,
  # beam search settings, see beam_eval_dataset.py
  "average_scores_over_beams": true,
  "average_scores_over_beams_intermed": false,
  "beam_score_relative_size":  false,
  "average_scores_weighted": false,
  "beam_score_acc": "last",
  "eval_use_logprobs": false,
  "beam_fact_selection": "sample",
  "beam_decode_top_p": 0.3,

  # which loss to use, see loss_relay.py and loss_functions.py
  "loss": "ranknet",
  "lambdaloss_scheme": "ndcgLoss2",

  # the number of tokens allowed in a training/eval batch
  "per_gpu_train_tokens_per_batch": 5000,
  "per_gpu_eval_tokens_per_batch": 24000,
  # number of train/eval workers to use (leave num_eval_workers = 1 unless you checked whether the code can 
  # correctly handle >1 workers, not so obvious because we use PyTorch IterableDatasets)
  "num_train_workers": 4,
  "num_eval_workers": 1
}
```

Single-fact (`single_fact_ranknet_1.json`):
```
{
  "seed": 42,
  "do_eval": false,
  "do_test": false,
  "do_train": true,

  # settings to choose positive/negative up/downsampling. Hasn't been used since long, so you should check 
  # the implementations yourself in single_fact_dataset.py
  "train_neg_sample_rate": 0.9,
  "downsample_negatives": false,
  "upsample_positives": false,

  "data_dir": "./data",
  "output_dir": "./output",
  "cache_dir": "./cache",

  "task": "20",
  "algo": "single-fact",

  "2020v2": true,
  "fact_path": "./tg2020task/dataset_v2/tables",
  "qa_dir": "./tg2020task/dataset_v2/",
  "train_qa_file": "questions.train.tsv",
  "val_qa_file": "questions.dev.tsv",
  "test_qa_file": "questions.test.tsv",
  "lvl_col_name": "arcset",

  "train_pair_file": "./data/20-v2-train-qa-fact-all-pairs-df-new.bin",
  "dev_pair_file": "./data/20-v2-dev-qa-fact-all-pairs-df-new.bin",
  "test_pair_file": "./data/20-v2-test-qa-fact-all-pairs-df-new.bin",

  "max_seq_length": 72,               # max sequence length to allow (Das et al. use 72 and 140) 
  "answer_choices": "correct",

  "mark_correct_in_qa": true,
  "mark_answer_in_qa": true,

  "model_type": "roberta",
  "model_name_or_path": "distilroberta-base",
  "_model_name_or_path": "./outputs/bert_rerank_correctchoices_unweighted_v2_42/",
  "config_name": "",
  "tokenizer_name": "",

  "loss": "ranknet",
  "evaluate_during_training": true,
  "no_lower_case": false,

  "learning_rate": 2e-5,
  "weight_decay": 0.0,
  "adam_epsilon": 1e-08,
  "max_grad_norm": 1.0,
  "max_steps": -1,
  "lr_decay_per_epoch": false,
  "lr_decay": true,
  "approx_num_steps": 12695,

  "logging_steps": 12695,
  "eval_steps": 12695,
  "save_steps": 12695,
  "num_train_epochs": 5,

  "no_cuda": false,
  "overwrite_output_dir": true,
  "overwrite_cache": false,

  "per_gpu_train_tokens_per_batch": 5000,
  "per_gpu_train_batch_size": 0,
  # number of SAMPLES in an eval batch. This is different from ARCF, which uses number of TOKENS in a batch!
  "per_gpu_eval_batch_size": 1200,
  "per_gpu_eval_tokens_per_batch": 0,
  "num_train_workers": 4,
  "num_eval_workers": 1
}
```


## Pretrained models

We provide two trained models, you can download them [here](https://drive.google.com/drive/folders/1_WRXVqULJG0YYHM8AYFjNDCCj1IRFyda?usp=sharing). 
The file `ARCF-RankNet.zip` contains a file `pytorch_model.bin`
with the parameters of our best performing model;
the `distilroberta-base` model that got a MAP score of 0.5815 on the 2020 test set and that was trained with RankNet.

The file `Single-Fact-Das-et-al.zip` has a file with the parameters of the `distilroberta-base` model we trained on the
2020 data using the implementation of Das et al. ([link](https://github.com/ameyagodbole/multihop_inference_explanation_regeneration)), 
which got a MAP of 0.4992 on the 2020 test set. 

If you're interested in trained versions of other models mentioned in the paper (like the one trained with the NCE loss),
please contact us.

## Acknowledgements

Thanks to Peter Jansen and Dmitry Ustalov for organizing the competition!
Also thanks to the Huggingface people for their transformers package, and to the PyTorch developers.

The research leading to this paper and these results received funding from the Research Foundation 
Flanders (FWO) under Grant Agreement No. G078618N and from
the European Research Council (ERC) under Grant Agreement No. 788506.
We would also like to thank the Flemish Supercomputer Center (VSC) for providing their hardware.

If you use this work please cite:
```
@inproceedings{cartuyvels2020autoregressive,
    title     = {Autoregressive Reasoning over Chains of Facts with Transformers},
    author    = {Cartuyvels, Ruben and Spinks, Graham and Moens, Marie-Francine},
    booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
    year      = {2020},
    publisher = {Association for Computational Linguistics},
}
```
Cite the workshop report:
```
@inproceedings{jansen2020textgraphs,
    author    = {Jansen, Peter and Ustalov, Dmitry},
    title     = {{TextGraphs~2020 Shared Task on Multi-Hop Inference for Explanation Regeneration}},
    year      = {2020},
    booktitle = {Proceedings of the Graph-based Methods for Natural Language Processing (TextGraphs)},
    publisher = {Association for Computational Linguistics},
    isbn      = {978-1-952148-42-2},
}
```
