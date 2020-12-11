import itertools
import logging
from abc import ABC
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from torch.utils.data import IterableDataset, DataLoader
from transformers import PreTrainedTokenizer

from datasets.chains.chain_batch import EvalChainTuple
from datasets.nlp import NLP
from datasets.utils import worker_init_fn
from datasets.worldtree_dataset import WorldTreeDataset
from representations.sentence_embeddings import SentenceEmbedder
from representations.tf_idf import TfIdf
from representations.wmd import WMDCalculator
from tools.config import ChainConfig

logger = logging.getLogger()


class ChainDataset(WorldTreeDataset, IterableDataset, ABC):
    def __init__(self,
                 config: ChainConfig,
                 path_to_qas: str,
                 tokenizer: PreTrainedTokenizer,
                 inference: bool = False,
                 validation: bool = False,
                 filter_empty: bool = True,
                 # stop_explanation_id=-1,
                 tokenize: bool = False,
                 ):
        logger.info('...')
        super().__init__(config, path_to_qas, tokenizer, inference,
                         validation, filter_empty, tokenize=tokenize)
        self.nearest_k_visible = config.nearest_k_visible
        logger.info('Using nearest k: %s' % self.nearest_k_visible)
        self.single_question_per_batch = config.single_question_per_batch
        self.config = config
        self.stop_explanation_id = len(self.fact_feats.index)
        self.predict_stop_expl = config.predict_stop_expl
        self.tokens_per_batch = None
        self.pad_token_id = self.tokenizer.pad_token_id

        if config.mark_expl:
            self.tokenized_expl_mark = self.tokenizer.encode(' (explanation) ', add_special_tokens=False)
            self.expl_mark = '(explanation) '
        else:
            self.tokenized_expl_mark = []
            self.expl_mark = ''

        self.tfidf = None
        if self.mode != 'train' and self.config.rank_rest == 'tf_idf':
            self.tfidf = TfIdf(self.config)
            self.tfidf.fit_vectorizer(self.qa_feats.original.append(self.fact_feats.original))

            self.fact_feats['stemmed'] = self.tfidf.stem(self.fact_feats.original)
            self.qa_feats['stemmed'] = self.tfidf.stem(self.qa_feats.original)
            self.transformed_facts = self.tfidf(self.fact_feats.original, fit_vectorizer=False)

        if self.config.distance_mode == 'wmd':
            self.compute_closest_wmd()
        else:
            self.compute_closest()

        if self.config.compute_overlap or self.validation:
            logger.info('Computing overlap...')
            self.compute_overlap()

    def compute_closest(self):
        self.set_columns()

        transform_func = self.transform_tf_idf if self.config.distance_mode == 'tf_idf' else self.transform_embeddings
        dist_func = (
            cosine_distances if self.config.distance_func == 'cosine'
            else lambda x, y: euclidean_distances(x, y, squared=True)
        )
        logger.info('Using representation: %s - distance: %s' % (transform_func.__name__, dist_func.__name__))

        transformed_qs, transformed_facts = transform_func()
        distances_facts = dist_func(transformed_facts, transformed_facts)
        if self.mode != 'train' and self.config.rank_rest == 'tf_idf':
            self.transformed_facts = transformed_facts

        def nearest_facts(cos_distances, fact_idx: int, nearest_k: int) -> np.ndarray:
            return np.argsort(cos_distances[fact_idx])[1:nearest_k]

        logger.info('Finding closest facts for each fact and QA...')
        self.fact_feats['closest'] = pd.Series(
            nearest_facts(distances_facts, i, len(self.fact_feats.index))
            for i in range(len(self.fact_feats.index))
        )

        self.qa_feats['closest'] = self.qa_feats.index.map(
            lambda i: np.argsort(dist_func(
                    transformed_qs[i] if len(transformed_qs[i].shape) > 1 else transformed_qs[i].unsqueeze(0),
                    transformed_facts
                )[0]
            )
        )

    def transform_embeddings(self):
        embedder = SentenceEmbedder(self.config)
        transformed_facts = embedder(self.fact_feats.dist_original, mode=self.mode, sents='facts')
        transformed_qs = embedder(self.qa_feats.dist_original, mode=self.mode, sents='questions')
        return transformed_qs, transformed_facts

    def compute_closest_wmd(self):
        wmd_calculator = WMDCalculator(self.config)
        f_dists, q_dists = wmd_calculator(self.fact_feats, self.qa_feats, mode=self.mode)
        self.fact_feats['closest'] = f_dists
        self.qa_feats['closest'] = q_dists

    def transform_tf_idf(self):
        if self.tfidf is None:
            self.tfidf = TfIdf(self.config)
            self.tfidf.fit_vectorizer(self.qa_feats.dist_original.append(self.fact_feats.dist_original))

        transformed_facts = self.tfidf(self.fact_feats.dist_original, fit_vectorizer=False)
        transformed_qs = self.tfidf(self.qa_feats.dist_original, fit_vectorizer=False)
        return transformed_qs, transformed_facts

    def compute_overlap(self):
        nlp = NLP()
        # nlp = self.nlp

        self.fact_feats['original_wo_punct'] = self.fact_feats.original_wo_fill.apply(
            lambda x: [
                word for word in nlp.tokenize(x) if nlp.interesting(word)
            ]
        )
        self.qa_feats['original_wo_punct'] = self.qa_feats.original.apply(
            lambda x: [
                word for word in nlp.tokenize(x) if nlp.interesting(word)
            ]
        )

        self.fact_feats['nb_overlaps'] = self.fact_feats.original_wo_punct.apply(
            lambda fact: self.fact_feats.original_wo_punct.apply(
                lambda x: len(set(x).intersection(set(fact)))
            ).to_numpy()
        )
        self.fact_feats['closest_overlap'] = self.fact_feats.nb_overlaps.apply(
            lambda overlaps: np.flip(np.argsort(overlaps))
        )

        self.qa_feats['nb_overlaps'] = self.qa_feats.original_wo_punct.apply(
            lambda qa: self.fact_feats.original_wo_punct.apply(
                lambda x: len(set(x).intersection(set(qa)))
            ).to_numpy()
        )
        self.qa_feats['closest_overlap'] = self.qa_feats.nb_overlaps.apply(
            lambda overlaps: np.flip(np.argsort(overlaps))
        )

        # todo also for qs --> nb hops away?
        self.fact_feats['overlap'] = self.fact_feats.original_wo_punct.apply(
            lambda fact: self.fact_feats[
                self.fact_feats.original_wo_punct.apply(
                    lambda x: bool(set(x).intersection(set(fact)))
                )
            ].index.to_numpy()
        )
        self.qa_feats['overlap'] = self.qa_feats.original_wo_punct.apply(
            lambda qa: self.fact_feats[
                self.fact_feats.original_wo_punct.apply(
                    lambda x: bool(set(x).intersection(set(qa)))
                )
            ].index.to_numpy()
        )

    def set_columns(self):
        if self.config.distance_wo_fill:
            self.fact_feats['dist_original'] = self.fact_feats.original_wo_fill.copy()
        else:
            self.fact_feats['dist_original'] = self.fact_feats.original.copy()
        self.qa_feats['dist_original'] = self.qa_feats.original.copy()

    def get_visible_facts(self, qa: pd.Series, facts: pd.DataFrame) -> Tuple[int]:
        from_qa = qa.closest.tolist()[:self.config.nearest_k_visible]
        if len(facts) == 0:
            return tuple(set(from_qa) - set(facts.index))
        if len(facts) == 1:
            from_facts = facts.closest.iloc[0].tolist()[:self.config.nearest_k_visible]
        else:
            from_facts = list(itertools.chain(
                *facts.closest.apply(lambda x: x.tolist()[:self.config.nearest_k_visible])
            ))
        # print(from_facts, from_qa, facts)
        return tuple(set(from_qa).union(set(from_facts)) - set(facts.index))

    def prepare_with_candidate(self, qa_with_facts, vf_index, logprob=None):
        fact: pd.Series = self.fact_feats.iloc[vf_index]

        if self.config.encoding == 'single_candidate':
            assert qa_with_facts.explanation == '', qa_with_facts.explanation
            text, text_pair = qa_with_facts.original, fact.original
        else:
            text, text_pair = qa_with_facts.original, qa_with_facts.explanation + '' + fact.original
        encoded = self.tokenizer.encode_plus(text=text,
                                             text_pair=text_pair,
                                             add_special_tokens=True)
        return {
            'tokenized': encoded['input_ids'],
            'token_type_ids': encoded['token_type_ids'] if self.use_segment_ids else None,
            'attention_mask': encoded['attention_mask'],
            'question_idx': qa_with_facts.question_idx,
            'partial_idx': qa_with_facts.partial_idx if isinstance(qa_with_facts, EvalChainTuple) else -1,
            'cand_fact_idx': vf_index,
            'label': vf_index in qa_with_facts.gold,
            'noise_logprob': logprob,
        }

    # def split_txt_and_txt_pair(self, fact, qa_with_facts):
    #     if self.config.encoding == 'single_candidate':
    #         text, text_pair = qa_with_facts.original, fact.original
    #     else:
    #         text, text_pair = qa_with_facts.original, qa_with_facts.explanation + '' + fact.original
    #     return text, text_pair

    def prepare_without_candidate(self, qa_with_facts, logprob=None):
        if self.config.encoding == 'single_candidate':
            assert qa_with_facts.explanation == '', qa_with_facts.explanation
            encoded = self.tokenizer.encode_plus(text=qa_with_facts.original,
                                                 add_special_tokens=True)
        else:
            encoded = self.tokenizer.encode_plus(text=qa_with_facts.original,
                                                 text_pair=qa_with_facts.explanation,
                                                 add_special_tokens=True)

        return {
            'tokenized': encoded['input_ids'],
            'token_type_ids': encoded['token_type_ids'] if self.use_segment_ids else None,
            'attention_mask': encoded['attention_mask'],
            'question_idx': qa_with_facts.question_idx,
            'partial_idx': qa_with_facts.partial_idx if isinstance(qa_with_facts, EvalChainTuple) else -1,
            'cand_fact_idx': self.stop_explanation_id,
            'label': len(qa_with_facts.gold) == 0,
            'noise_logprob': logprob,
        }

    def set_settings(self, config: ChainConfig, train=True, valid=False, infer=False):
        if train:
            bs, tbs = config.train_batch_size, config.train_tokens_per_batch
        else:
            bs, tbs = config.eval_batch_size, config.eval_tokens_per_batch
        self.batch_size = bs if bs else self.batch_size
        self.tokens_per_batch = tbs if tbs else self.tokens_per_batch
        self.nearest_k_visible = config.nearest_k_visible
        self.predict_stop_expl = config.predict_stop_expl
        self.single_question_per_batch = config.single_question_per_batch
        self.config = config

    def create_dataloader(self, **kwargs):
        # todo Pin memory? If dataset does not fit in memory and is loaded on CPU,
        # in our case, fits in memory: https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/2
        dl = DataLoader(self, batch_size=None, pin_memory=True, worker_init_fn=worker_init_fn, **kwargs)
        return dl
