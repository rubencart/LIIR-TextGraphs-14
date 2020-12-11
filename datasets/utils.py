import copy
import logging
import math
import random
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger()

EXPL_LENGTHS_TRAIN_20 = [(1, 208),
                         (2, 299),
                         (3, 354),
                         (4, 375),
                         (5, 298),
                         (6, 298),
                         (7, 224),
                         (8, 160),
                         (9, 140),
                         (10, 99),
                         (11, 61),
                         (12, 56),
                         (13, 45),
                         (14, 22),
                         (15, 21),
                         (16, 15),
                         (17, 13),
                         (18, 9),
                         (19, 5),
                         (20, 3),
                         (21, 9),
                         (22, 1)]
FREQS = np.array([c for i, c in EXPL_LENGTHS_TRAIN_20])
EXPL_LENGTH_FREQS = FREQS / sum(FREQS)


def gold_facts_in_n_closest_all(dataset, nearest_k):
    results = {}
    for i, row in dataset.qa_feats.iterrows():
        gold_to_find = set(copy.deepcopy(row.gold_facts))
        visible_facts = set(row.closest[:nearest_k])
        lens_visible = []
        gold_found = set()
        while len(gold_to_find) > 0:
            lens_visible.append(len(visible_facts))
            found = set([fact for fact in gold_to_find if fact in visible_facts])
            if len(found) == 0:
                break
            gold_found = gold_found.union(found)
            gold_to_find -= found
            for fact in found:
                visible_from_fact = set(dataset.fact_feats.iloc[fact].closest[:nearest_k])
                visible_facts = visible_facts.union(visible_from_fact)
        results[i] = {
            'all': set(copy.deepcopy(row.gold_facts)),
            'found': gold_found,
            'not_found': gold_to_find,
            'mean_len_visible': np.mean(lens_visible)
        }
    return results


def gold_facts_in_n_closest_cur(dataset, nearest_k):
    results = {}
    for i, row in dataset.qa_feats.iterrows():
        gold_to_find = set(copy.deepcopy(row.gold_facts))
        visible_facts = set(row.closest[:nearest_k])
        lens_visible = []
        gold_found = set()
        while len(gold_to_find) > 0:
            lens_visible.append(len(visible_facts))
            facts = [fact for fact in gold_to_find if fact in visible_facts]
            if len(facts) == 0:
                break
            selected = facts[0]
            gold_found.add(selected)
            gold_to_find -= {selected}
            visible_facts = dataset.fact_feats.iloc[selected].closest[:nearest_k]
        results[i] = {
            'all': set(copy.deepcopy(row.gold_facts)),
            'found': gold_found,
            'not_found': gold_to_find,
            'mean_len_visible': np.mean(lens_visible)
        }
    return results


def find_nearest_k_for_rate(dataset, target_rate, func=gold_facts_in_n_closest_all, start_at=0):
    k = start_at
    results = func(dataset, k)
    nb_all = sum([len(res['all']) for res in results.values()])
    while True:
        nb_found = sum([len(res['found']) for res in results.values()])
        mean_len_visible = np.mean([res['mean_len_visible'] for res in results.values()])
        rate = nb_found / nb_all
        if rate > target_rate:
            break
        k += 10
        print('Trying k = %s, rate was %s' % (k, rate))
        results = func(dataset, k)
    return k, rate, mean_len_visible


def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n - r)


def nb_combinations(dataset):
    def q_nb_combinations(nb_facts):
        return sum([dataset.nCr(nb_facts, i) for i in range(1, nb_facts)])

    return sum([q_nb_combinations(len(gf)) for gf in dataset.qa_feats.gold_facts])


def nb_samples(dataset):
    combs = [(i, sum([dataset.nCr(i, j) for j in range(0, i + 1)]))
             for i in range(1, 23)]
    lens = [(i, len([row for _, row in dataset.qa_feats.iterrows() if len(row.gold_facts) == i]))
            for i in range(1, 23)]
    tot = [(combs[i][0], combs[i][1] * lens[i][1]) for i in range(22)]
    cum_tot = np.cumsum([count for _, count in tot])
    real_counts = [(i + 1, c + sum(combs[i][1] * lens[j][1] for j in range(i + 1, 22)))
                   for i, c in enumerate(cum_tot)]
    return combs, lens, tot, real_counts


def max_length_of_explanation(dataset):
    """
    make sure that this fits in language model (max seq length - max_position_embeddings)
    >> 19:  (344, 91.55, 21, 734)
    """
    max_length = 0
    lengths = []
    for i, row in dataset.qa_feats.iterrows():
        qa_tok = row.tokenized
        facts = list(dataset.fact_feats.iloc[list(row.gold_facts)].tokenized)
        encoded = qa_tok + [t for fact in facts for t in fact]
        length = len(encoded)
        if length > max_length:
            max_length = length
        lengths.append(length)
    longest_qa = sorted(list(dataset.qa_feats.tokenized),
                        key=lambda x: len(x), reverse=True)[0]
    max_nb_facts = max([len(gf) for gf in dataset.qa_feats.gold_facts])
    longest_facts = sorted(list(dataset.fact_feats.tokenized),
                           key=lambda x: len(x), reverse=True)[:max_nb_facts]
    flattened_longest_facts = [t for fact in longest_facts for t in fact]
    return (max_length, sum(lengths) / len(lengths), max_nb_facts,
            len(longest_qa) + len(flattened_longest_facts))


POINTWISE_LOSSES = ['xent', 'mse', 'xent-2']
BATCHWISE_LOSSES = ['fisher']
NOISE_CONTRASTIVE_LOSSES: List[str] = ['nce', 'ranking-nce', 'binary-nce']
CONTRASTIVE_LOSSES = ['ranknet', 'lambdaloss', 'margin-pairs'] + NOISE_CONTRASTIVE_LOSSES


def should_recompute_lengths(args, dataset, train, valid, infer):
    return (
            (train and (not hasattr(dataset, 'max_length_in_batches_single_q')
                        and not hasattr(dataset, 'max_length_in_batches')))
            or args.nearest_k_visible != dataset.nearest_k_visible
            or (train and (dataset.batch_size != args.train_batch_size or
                           dataset.tokens_per_batch != args.train_tokens_per_batch))
            or ((valid or infer) and (dataset.batch_size != args.eval_batch_size or
                                      dataset.tokens_per_batch != args.eval_tokens_per_batch))
    )


def read_explanations(path: str) -> List[Tuple[str, str]]:
    header = []
    uid = None

    df = pd.read_csv(path, sep='\t', dtype=str)

    for name in df.columns:
        if name.startswith('[SKIP]'):
            if 'UID' in name and not uid:
                uid = name
        else:
            header.append(name)

    if not uid or len(df) == 0:
        warnings.warn('Possibly misformatted file: ' + path)
        return []

    return df.apply(lambda r: (r[uid], ' '.join(str(s) for s in list(r[header]) if not pd.isna(s))), 1).tolist()


def read_explanations_wo_fill(path: str) -> List[Tuple[str, str]]:
    header = []
    uid = None

    df = pd.read_csv(path, sep='\t', dtype=str)

    for name in df.columns:
        if name.startswith('[SKIP]'):
            if 'UID' in name and not uid:
                uid = name
        else:
            if not name.strip().startswith('[FILL]'):       # this is the difference
                header.append(name)

    if not uid or len(df) == 0:
        warnings.warn('Possibly misformatted file: ' + path)
        return []

    return df.apply(lambda r: (r[uid], ' '.join(str(s) for s in list(r[header]) if not pd.isna(s))), 1).tolist()


def worker_init_fn(x):
    # this is BAD, seeds every worker in every epoch again with same seed ==> epochs won't be different.
    # (workers get recreated at start of every epoch)
    # seed = x
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # # if torch.cuda.device_count() > 0:
    # #     torch.cuda.manual_seed_all(seed)
    #
    # rather: (https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097)
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
