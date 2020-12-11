import copy
import logging
import os
from collections import Counter

import pandas as pd
import spacy
import numpy as np
import torch
from tqdm import tqdm
from wmd import WMD

from tools.config import ChainConfig

logger = logging.getLogger()


class SpacyEmbeddings(object):
    def __init__(self, nlp):
        self.nlp = nlp

    def __getitem__(self, item):
        return self.nlp.vocab[item].vector


class WMDCalculator:
    def __init__(self, config: ChainConfig):
        self.config = config
        self.nlp = spacy.load('en_core_web_md')

    def __call__(self, f_df, q_df, mode='train'):
        cached_facts_fn = '20_wmd_closest_facts.pkl'
        cached_qs_fn = '20_wmd_closest_qs_{}.pkl'.format(mode)
        cached_qs_fn += '_debug' if self.config.debug else ''

        if os.path.exists(cached_qs_fn):
            logger.info('Loading question wmds from cached file %s' % cached_qs_fn)
            q_closest = torch.load(cached_qs_fn)
            f_nbow = None
        else:
            q_closest, f_nbow = self.compute_q(f_df, q_df, return_f_nbow=True)
            logger.info('Saving question wmds to file %s' % cached_qs_fn)
            torch.save(q_closest, cached_qs_fn)

        if os.path.exists(cached_facts_fn):
            logger.info('Loading fact wmds from cached file %s' % cached_facts_fn)
            f_closest = torch.load(cached_facts_fn)
        else:
            logger.info('Saving fact wmds to file %s' % cached_facts_fn)
            f_closest = self.compute_f(f_df, f_nbow)
            torch.save(f_closest, cached_facts_fn)

        return f_closest, q_closest

    def compute_f(self, f_df, f_nbow=None):
        logger.info('Computing fact wmds')
        f_nbow = {
            row.Index: self.nbowify(row.Index, row.original) for row in f_df.itertuples()
        } if f_nbow is None else f_nbow

        f_calc = WMD(SpacyEmbeddings(self.nlp), f_nbow, vocabulary_min=1, verbosity=logging.WARNING)
        f_calc.cache_centroids()
        f_closest = pd.Series(
            np.array([i for i, _ in f_calc.nearest_neighbors(idx, k=self.config.nearest_k_visible)])
            for idx in tqdm(f_nbow.keys(), desc='Fact wmd...')
        )
        return f_closest

    def compute_q(self, f_df, q_df, return_f_nbow=False):
        logger.info('Computing question wmds')
        f_nbow = {
            row.Index: self.nbowify(row.Index, row.original) for row in f_df.itertuples()
        }
        nb_facts = len(f_nbow)
        q_nbow = {
            row.Index + nb_facts: self.nbowify(row.Index + nb_facts, row.original) for row in q_df.itertuples()
        }

        merged_fnbow = copy.copy(f_nbow)
        merged_fnbow.update(q_nbow)
        q_calc = WMD(SpacyEmbeddings(self.nlp), merged_fnbow, vocabulary_min=1, verbosity=logging.WARNING)
        q_calc.cache_centroids()
        q_closest = pd.Series(
            np.array([i for i, _ in q_calc.nearest_neighbors(idx, k=self.config.nearest_k_visible) if i < nb_facts])
            for idx in tqdm(q_nbow.keys(), desc='Question wmd...')
        )
        return (q_closest, f_nbow) if return_f_nbow else q_closest

    def nbowify(self, idx, raw_text):
        text = self.nlp(raw_text)
        tokens = [t for t in text if t.is_alpha]    # and not t.is_stop
        words = Counter(t.text for t in tokens)
        orths = {t.text: t.orth for t in tokens}
        sorted_words = sorted(words)
        return (idx, [orths[t] for t in sorted_words],
                np.array([words[t] for t in sorted_words], dtype=np.float32))
