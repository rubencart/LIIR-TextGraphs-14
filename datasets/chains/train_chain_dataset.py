import itertools
import logging
import math
import os
import random
from abc import ABC
from collections import Generator
from typing import Iterable, Iterator, Tuple

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from datasets.chains.chain_batch import ChainTuple, PartialChainBatch
from datasets.chains.chain_dataset import ChainDataset
from datasets.utils import NOISE_CONTRASTIVE_LOSSES, FREQS
from tools import utils
from tools.config import ChainConfig

logger = logging.getLogger()


class TrainChainDataset(ChainDataset, ABC):
    """
    CAUTION because this dataset uses sampling, calling ds[i] twice with equal i is NOT GUARANTEED TO
        RETURN THE SAME RESULT!!!
    """

    def __init__(self,
                 config: ChainConfig,
                 path_to_qas: str,
                 tokenizer: PreTrainedTokenizer,
                 filter_empty: bool = True
                 ):
        super().__init__(config, path_to_qas, tokenizer, False, False, filter_empty, tokenize=True)
        self.powerset_complete_combs = config.powerset_complete_combs
        self.tokens_per_batch = config.train_tokens_per_batch

        if config.train_chains_file and os.path.exists(config.train_chains_file):
            logger.info('Loading qa-fact explanation combinations from {}'.format(config.train_chains_file))
            self.partial_expls_df = torch.load(config.train_chains_file)
        else:
            logger.info('Exploding all qa-fact explanation combinations...')
            self.partial_expls_df = self.explode_partial_explanations(self.qa_feats,
                                                                      compl=config.powerset_complete_combs)
            if config.train_chains_file:
                logger.info('Saving pos and neg qa-fact combinations to {}'.format(config.train_chains_file))
                torch.save(self.partial_expls_df, config.train_chains_file)

        self.compute_sizes()

    def compute_sizes(self):
        logger.info('Length calculations...')
        self.partial_expls_df['max_p_facts'] = self.partial_expls_df.apply(
            lambda x: (
                # condition for sampling --> potentially longer explanations
                x.partial_expl if not self.sampling_condition(x.partial_expl, q_idx=x.q_idx)
                else self.qa_feats.gold_facts.iloc[x.q_idx]
            ),
            axis=1,
        )
        self.partial_expls_df['max_vis_facts'] = self.partial_expls_df.apply(
            lambda x: set(
                self.qa_feats.closest.iloc[x.q_idx].tolist()[:self.config.nearest_k_visible]
                + [
                    vis for sublist in
                    [
                        self.fact_feats.closest.iloc[f_idx].tolist()[:self.config.nearest_k_visible]
                        for f_idx in x.max_p_facts
                    ]
                    for vis in sublist
                    if vis not in x.max_p_facts and (not self.config.use_all_negatives or vis in x.gold_facts)
                ]
            ),
            axis=1,
        )
        self.partial_expls_df['nb_max_vis_facts'] = self.partial_expls_df.apply(
            lambda x: len(x.max_vis_facts) + (
                1 if self.config.predict_stop_expl and len(x.partial_expl) > 0 else 0
            ),
            axis=1,
        )
        self.qa_feats['tok_len'] = self.qa_feats.tokenized.apply(len)
        self.fact_feats['tok_len'] = self.fact_feats.tokenized.apply(len)
        self.max_length_in_samples = sum(self.partial_expls_df.nb_max_vis_facts)
        logger.info('Max nb of samples in DS = {}'.format(self.max_length_in_samples))
        if not self.config.use_all_negatives:   # one pos per batch
            self.max_length_in_batches_single_q = self.max_length_in_batches = sum(self.partial_expls_df.nb_max_vis_facts)
        else:
            if self.batch_size:
                self.partial_expls_df['max_batches'] = self.partial_expls_df.nb_max_vis_facts.apply(
                    lambda x: x // self.batch_size + 1,
                )
                self.max_length_in_batches = self.max_length_in_samples // self.batch_size + 1
                logger.info('Max nb of batches in DS = {}'.format(self.max_length_in_batches))
            elif self.tokens_per_batch:
                self.partial_expls_df['max_tokens'] = self.partial_expls_df.apply(
                    lambda x: len(x.max_vis_facts) * sum(
                        [self.fact_feats.iloc[f_idx].tok_len for f_idx in x.max_p_facts]
                        + [self.qa_feats.iloc[x.q_idx].tok_len]
                    ) + sum(
                        [self.fact_feats.iloc[f_idx].tok_len for f_idx in x.max_vis_facts]
                    ),
                    axis=1
                )
                self.partial_expls_df['max_batches'] = self.partial_expls_df.max_tokens.apply(
                    lambda x: x // self.tokens_per_batch + 1
                )
                self.max_length_in_batches = sum(self.partial_expls_df.max_tokens) // self.tokens_per_batch + 1
                logger.info('Max most efficient nb of batches in DS = {}'.format(self.max_length_in_batches))
            self.max_length_in_batches_single_q = sum(self.partial_expls_df.max_batches)
        logger.info('Max nb of batches in DS if single question / batch = {}'
                    .format(self.max_length_in_batches_single_q))

    def sampling_condition(self, partial_expl, q_idx=None, gold_facts=None):
        gold = self.qa_feats.iloc[q_idx].gold_facts if gold_facts is None else gold_facts
        return (
                len(gold) > round(self.powerset_complete_combs) and len(partial_expl) != 0
        )

    @staticmethod
    def explode_partial_explanations(qa_feats, compl=5):
        logger.info('Exploding max {} combinations...'.format(2 ** compl * len(qa_feats.index)))
        # powerset of list of gold facts: these are all combinations of 1..K facts drawn from N facts
        # length of powerset == 2^(length of set)
        qa_feats['powerset'] = qa_feats.gold_facts.apply(utils.powerset).apply(list).apply(np.array)
        qas = qa_feats[['gold_facts', 'powerset']].copy()

        # keep only first max_combs from powerset, these are all combs for 1..compl facts
        qas['partial_expl'] = qas.powerset.apply(lambda x: x[:round(2 ** compl)])
        qas = qas.explode('partial_expl')
        qas.partial_expl = qas.partial_expl.apply(np.array)

        # fill remaining with gold facts not yet in partial_expl, (add '-1' as stop token    + [-1] )
        qas['remaining'] = qas.apply(lambda x: np.array(list(set(x.gold_facts) - set(x.partial_expl))),
                                     axis=1)
        qas['q_idx'] = qas.index.copy()
        qas = qas.reset_index(drop=True)

        qas = qas.drop(['powerset', 'gold_facts'], axis=1)
        return qas.copy()

    def __getitem__(self, index: int) -> ChainTuple:
        expl_row = self.partial_expls_df.iloc[index]
        q_idx = expl_row.q_idx
        qa_row = self.qa_feats.iloc[q_idx]
        qa_id = qa_row.id
        qa_txt = qa_row.original

        if self.sampling_condition(expl_row.partial_expl, gold_facts=qa_row.gold_facts):
            # pick partial explanation length, can be [1, len-1]
            # for now, stop condition = if current explanation, without any new facts appended, gets highest score?
            if self.config.categorical_expl_length:
                freqs = FREQS[:len(qa_row.gold_facts)] / sum(FREQS[:len(qa_row.gold_facts)])
                expl_length = int(np.where(np.random.multinomial(n=1, pvals=freqs))[0][0] + 1)
            else:
                expl_length = random.randrange(1, len(qa_row.gold_facts))

            # choose expl_length facts from the gold facts without replacement
            cur_fact_idxs = random.sample(population=set(qa_row.gold_facts.tolist()), k=expl_length)
            remaining_fact_idxs = list(set(qa_row.gold_facts.tolist()) - set(cur_fact_idxs))
        else:
            cur_fact_idxs = expl_row.partial_expl.tolist()
            remaining_fact_idxs = expl_row.remaining.tolist()
            expl_length = len(cur_fact_idxs)

        cur_facts = self.fact_feats.iloc[cur_fact_idxs].copy()
        visible_fact_idxs = set(self.get_visible_facts(qa=qa_row, facts=cur_facts))

        # replace some with negatives to condition on negatives (instead of only on positives)
        neg_rate = self.config.condition_on_negs_rate
        effective_neg_rate = neg_rate * random.random() if self.config.sample_condition_on_negs_rate else neg_rate
        # always at least 1 positive
        from_neg = min(round(effective_neg_rate * expl_length), expl_length - 1)
        if from_neg > 0:
            all_negatives = tuple(visible_fact_idxs - set(qa_row.gold_facts))
            sample_negs = random.sample(population=all_negatives, k=from_neg)

            # replace
            cur_facts.loc[random.sample(cur_fact_idxs, k=from_neg)] = self.fact_feats.iloc[sample_negs].values
            # recompute visible facts, now from viewpoint of just added negs as well
            visible_fact_idxs = self.get_visible_facts(qa=qa_row, facts=cur_facts)

        cur_fact_txt = ''.join(cur_facts.original)
        if cur_fact_txt:
            if self.config.encoding == 'single_candidate':
                cur_fact_txt = self.expl_mark + cur_fact_txt
            if self.config.answer_behind_expl and '(answer)' in qa_txt:
                qa, answer = qa_txt.split('(answer)')
                qa_txt, cur_fact_txt = qa, cur_fact_txt + '(answer)' + answer

        remaining_fact_idxs: Tuple[int] = tuple(set(remaining_fact_idxs).intersection(visible_fact_idxs))

        if self.config.encoding == 'single_candidate':
            original, explanation = qa_txt + cur_fact_txt, ''
        else:
            original, explanation = qa_txt, cur_fact_txt

        return ChainTuple(original=original,
                          explanation=explanation,
                          gold=tuple(remaining_fact_idxs),
                          question_id=str(qa_id),
                          question_idx=q_idx,
                          fact_idxs=cur_fact_idxs,
                          visible_facts=tuple(visible_fact_idxs), )

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        """
        Because of sampling, this length DOES NOT HAVE TO EQUAL the observed length.
        """
        if self.batch_size and not self.single_question_per_batch:
            return self.max_length_in_batches
        else:
            return self.max_length_in_batches_single_q

    def set_settings(self, config: ChainConfig, train=True, valid=False, infer=False):
        super().set_settings(config, train)
        if (
                (train and (not hasattr(self, 'max_length_in_batches_single_q')
                            and not hasattr(self, 'max_length_in_batches')))
                or config.nearest_k_visible != self.nearest_k_visible
                or (train and (self.batch_size != config.train_batch_size or
                               self.tokens_per_batch != config.train_tokens_per_batch))
                or ((valid or infer) and (self.batch_size != config.eval_batch_size or
                                          self.tokens_per_batch != config.eval_tokens_per_batch))
        ):
            self.compute_sizes()


class PointwiseTrainChainDataset(TrainChainDataset):
    # todo check fact with id 'good'
    def __iter__(self):
        """
            with nearest_k_visible = 180, 2500 tokens per batch, more or less
                - bs                        31
                - num pos per batch         0.12
                - num pos per (pos+neg)     0.0046
            => if one pos per batch:
                - num pos per (pos+neg)     1/31 = 0.032 = 7 * 0.0046
                => either many positives RE-used OR many negatives UN-used
        """
        indices = random.sample(list(range(len(self.partial_expls_df.index))), len(self.partial_expls_df.index))
        indices = self.split_among_workers(indices)
        batch = PartialChainBatch(self.pad_token_id)
        for i in indices:
            qa_with_facts: ChainTuple = self[i]
            num_vf = len(qa_with_facts.visible_facts)

            if len(qa_with_facts.fact_idxs) > 0 and self.config.predict_stop_expl:
                # append partial expl without candidate fact, to allow stopping
                batch.append(**self.prepare_without_candidate(qa_with_facts))

                if self.is_batch_full(len(batch.question_idxs), batch.max_length * len(batch.question_idxs)):
                    yield batch.yield_batch(last=num_vf == 0)
                    batch = PartialChainBatch(self.pad_token_id)

            for vf_it, vf_index in enumerate(qa_with_facts.visible_facts):
                batch.append(**self.prepare_with_candidate(qa_with_facts, vf_index))

                if self.is_batch_full(len(batch.question_idxs), batch.max_length * len(batch.question_idxs)):
                    yield batch.yield_batch(last=vf_it == num_vf - 1)
                    batch = PartialChainBatch(self.pad_token_id)

            # loss will be computed per QA-F, for all candidate facts at once, so mixed
            # batches are not allowed (gradient of first half of batch will have nothing to do
            # with results of second half of batch, whose gradient will only be computed when
            # all candidate facts for that QA-F have been processed). Depending on loss function!
            if self.single_question_per_batch and len(batch.question_idxs) > 0:
                yield batch.yield_batch(last=True)
                batch = PartialChainBatch(self.pad_token_id)
        if len(batch.question_idxs) > 0:
            yield batch.yield_batch(last=True)


class ContrastiveTrainChainDataset(TrainChainDataset):
    def __iter__(self) -> Iterator:
        """
            First gather all samples related to a question + partial expls, then batchify with
            positives distributed
        """
        indices = random.sample(list(range(len(self.partial_expls_df.index))), len(self.partial_expls_df.index))
        indices = self.split_among_workers(indices)
        samples = []

        include_lp = self.config.loss in NOISE_CONTRASTIVE_LOSSES and self.mode == 'train'
        logger.info('Categorical expl length: %s' % self.config.categorical_expl_length)
        for i in indices:
            qa_with_facts: ChainTuple = self[i]

            vfs, len_vfs = qa_with_facts.visible_facts, len(qa_with_facts.visible_facts)
            # +1 for stop explanation id
            lp = math.log(1 / (len_vfs + 1)) if include_lp else None

            if len(qa_with_facts.fact_idxs) > 0 and self.config.predict_stop_expl:
                # append partial expl without candidate fact, to allow stopping
                samples.append(self.prepare_without_candidate(qa_with_facts, logprob=lp))

            for vf_it, vf_index in enumerate(vfs):
                samples.append(self.prepare_with_candidate(qa_with_facts, vf_index, logprob=lp))

            yield from self.batchify(samples)
            samples = []

    def batchify(self, samples: Iterable) -> Generator:
        """
            Put 1 positive in batch with rest negatives (until batch full), yield batches cycling through
            pos and negs (until every pos and neg has been used at least once.)
            EDIT:        until every pos has been used exactly once.

            NOTE:   this might not preserve precomputed dataset length! Since samples might be reused

            when only num_pos > 0 as condition: epoch = 20k batches with k=180 and bs=2500
        """
        random.shuffle(samples)
        positives, negatives = [], []
        for sample in samples:
            (positives if bool(sample['label']) else negatives).append(sample)

        num_pos, num_neg = len(positives), len(negatives)
        if num_pos < 1 or num_neg < 1:
            logging.error('Zero negatives or positives - p: %s - n: %s' % (num_pos, num_neg))
            return

        pos_gen, neg_gen = itertools.cycle(positives), itertools.cycle(negatives)
        batch = PartialChainBatch(self.pad_token_id)
        while self.condition(num_pos, num_neg):
            num_pos -= 1
            batch.append(**next(pos_gen))
            while not self.is_batch_full(len(batch.question_idxs), batch.max_length * len(batch.question_idxs)):
                num_neg -= 1
                batch.append(**next(neg_gen))

            # batch.shuffle()
            yield batch.yield_batch()     # or num_neg > 0      last=not (num_pos > 0 or num_neg > 0)
            batch = PartialChainBatch(self.pad_token_id)

    def condition(self, p, n):
        return (p > 0 or n > 0) if self.config.use_all_negatives else (p > 0)
