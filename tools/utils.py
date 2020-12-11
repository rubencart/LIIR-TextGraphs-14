import argparse
import json
import logging
import os
import random
import shutil
import sys
from argparse import Namespace
from datetime import datetime
from itertools import chain, combinations

import torch
import numpy as np

from tools.config import Config

logger = logging.getLogger(__name__)


def set_seed(args: Config):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None,)
    return parser.parse_args()


def initialize_logging(to_file=True, config=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%d-%m-%Y %H:%M:%S')
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if to_file:
        fh = logging.FileHandler(os.path.join(config.output_dir, 'output.log'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def write_questions_and_facts_to_file(ds):
    challenge = ds.qa_feats[ds.df_qas.focusnotes == 'Challenge']
    with open('./challenge-qas-facts.txt', 'w') as f:
        for i in range(len(challenge.index)):
            id = challenge.iloc[i].id
            question = challenge.iloc[i].original
            gold_idxs = challenge.iloc[i].gold_facts
            facts = list(ds.df_facts.iloc[gold_idxs].text)
            f.write('Question {}: {}\n'.format(id, question))
            f.writelines('Fact {}: '.format(i) + fact + '\n' for i, fact in enumerate(facts))
            f.write('\n')


def write_questions_and_facts_to_file_2(vds_chains):
    with open('./data/dataset-samples.txt', 'w') as f:
        for index, row in vds_chains.qa_feats.iterrows():
            gold_idxs = row.gold_facts
            facts = list(vds_chains.df_facts.iloc[gold_idxs].text)
            fact_roles = vds_chains.qf_roles.loc[
                (vds_chains.qf_roles.q_idx == index) & (vds_chains.qf_roles.f_idx.isin(gold_idxs))]
            f.write('Question {}: {}\n'.format(row.id, row.original))
            f.writelines('Fact {} - Role {}: '.format(i, fact_roles.loc[fact_roles.f_idx == fact_idx, 'role'].iloc[
                0]) + fact + '\n'
                         for i, (fact_idx, fact) in enumerate(zip(gold_idxs, facts)))
            f.write('\n')


def print_predicted_facts(preds, fact_id, ds):
    for i, f_id in enumerate(preds[fact_id][:15]):
        gold = ds.fact_uid_2_idx[f_id] in ds.qa_feats.iloc[
            ds.qa_uid_2_idx[fact_id]].gold_facts
        print(fact_id)
        print(
            'Fact {} - {}: {}'.format(i, gold,
                                      ds.fact_feats.loc[ds.fact_feats.id == f_id].iloc[0].original)
        )


def get_output_dir(config: Config) -> str:
    output_dir = os.path.join(
        config.output_dir,
        '{}_{}_{}_{}'.format(config.model_type, config.algo, str(datetime.now().strftime('%Y-%m-%d_%Hh%M')), os.getpid())
    )
    if config.debug:
        output_dir += '_debug'
    if config.task == '19':
        output_dir += '_19'
    if config.v2:
        output_dir += '_v2'
    return output_dir


def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def plot(sttl, i, loss_ylim=None, constr_ylim=None):
    from matplotlib import pyplot as plt
    plt.plot([s['loss'] for s in sttl])
    if loss_ylim is not None:
        plt.ylim(top=loss_ylim)
    plt.savefig('data/loss_%i.png' % i)
    plt.close()
    plt.plot([s['other']['lambda'] for s in sttl])
    plt.savefig('data/lambda_%i.png' % i)
    plt.close()
    plt.plot([s['other']['constraint'] for s in sttl])
    if constr_ylim is not None:
        plt.ylim(top=constr_ylim)
    plt.savefig('data/constraint_%i.png' % i)
    plt.close()


def get_maps(file, start):
    with open(file, 'r') as f:
        stats = json.load(f)
    sttl = stats['stat_points'][start:]
    return [s['map'] for s in sttl if s['map'] is not None]


def open_and_plot_stats(file, start, i, ly=None, cy=None):
    with open(file, 'r') as f:
        stats = json.load(f)
    sttl = stats['stat_points'][start:]
    plot(sttl, i, ly, cy)


def remove_debug_folders(root_dir):
    for path in [f.path for f in os.scandir(root_dir) if f.is_dir() and f.path.endswith('debug')]:
        shutil.rmtree(path)


def map_from_txt_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    results = {}
    for line in lines:
        key, value = line.rstrip('\n').split('\t')
        key, value = key.lower(), value.lower()
        old = results.get(key, [])
        results[key] = old + [value]
    from rankers.utils import mean_average_precision_score
    from datasets.factory import DatasetFactory
    from rankers.chain_ranker import set_up_experiment
    ranker, config, tokenizer = set_up_experiment('./config/chains.json')
    vds = DatasetFactory.load_and_cache_dataset(config, tokenizer, config.val_qa_path, valid=True)
    return mean_average_precision_score(vds.gold, results)


def h_to_s(h, m, s):
    return 60 * (m + 60 * h) + s


def s_to_h(s):
    h = s // 3600
    mrest = s % 3600
    m = mrest // 60
    srest = mrest % 60
    return h, m, srest
