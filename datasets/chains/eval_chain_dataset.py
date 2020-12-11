import logging
from typing import List, Tuple

import torch
from transformers import PreTrainedTokenizer

from datasets.eval.beam_eval_dataset import SampleBeamEvalDatasetMixin, GreedyBeamEvalDatasetMixin
from datasets.chains.chain_batch import ChainTuple, PartialChainBatch, EvalChainTuple
from datasets.chains.chain_dataset import ChainDataset
from datasets.eval.eval_dataset import EvalDatasetMixin
from tools.config import ChainConfig

logger = logging.getLogger()


class EvalChainMixin(ChainDataset):
    def __init__(self,
                 config: ChainConfig,
                 path_to_qas: str,
                 tokenizer: PreTrainedTokenizer,
                 inference: bool = False,
                 validation: bool = True,
                 filter_empty: bool = True,
                 ):
        super().__init__(config, path_to_qas, tokenizer, inference, validation, filter_empty)
        assert inference or validation

        self.tokens_per_batch = config.eval_tokens_per_batch
        # for scatter: treat stop id like last fact
        self.nb_facts = len(self.fact_feats.index) + 1
        self.reset_partials()

        # prepare for analysis of scores for types of facts
        if self.validation:
            # fact roles
            # TODO there is a fact with role 'role'
            logger.info('Creating qa-fact role DF')
            qf_roles = self.df_qas[['questionid', 'explanation']].copy()
            qf_roles = qf_roles.rename(columns={'questionid': 'q_id'})
            qf_roles['id_roles'] = qf_roles.explanation.apply(lambda x: x.split())
            qf_roles = qf_roles.explode('id_roles')
            # self.qa_fact_roles = self.qa_fact_roles[['q_id', 'id_roles']]
            qf_roles = qf_roles.drop(['explanation'], axis=1)
            qf_roles.id_roles = qf_roles.id_roles.apply(lambda x: x.split('|'))
            qf_roles[['f_id', 'role']] = qf_roles.apply(
                lambda x: (x.id_roles[0], x.id_roles[1]), result_type='expand', axis=1
            )
            # self.qa_fact_roles = self.qa_fact_roles[['q_id', 'f_id', 'role']]
            qf_roles = qf_roles.drop(['id_roles'], axis=1)
            qf_roles['f_idx'] = qf_roles.f_id.apply(lambda x: self.fact_uid_2_idx[x])
            qf_roles['q_idx'] = qf_roles.index.copy()
            self.qf_roles = qf_roles.reset_index(drop=True)

            # nb of hops away
            def calc_hops(hood_func):
                df = self.qa_feats[['id', 'gold_facts']].copy().rename(columns={'gold_facts': 'f_idx',
                                                                                'id': 'q_id'})
                df = df.explode('f_idx')
                df['f_id'] = df.f_idx.apply(lambda x: self.fact_idx_2_uid[x])
                df['q_idx'] = df.index.copy()
                df = df.reset_index(drop=True)
                df['hops'] = -1  # default, unreachable

                for i, row in self.qa_feats.iterrows():
                    gold_to_find = set(row.gold_facts)
                    visible_facts = set(hood_func(row))
                    gold_found = set()
                    hops_away = 0
                    while len(gold_to_find) > 0:
                        hops_away += 1
                        found = gold_to_find.intersection(visible_facts)
                        if len(found) == 0:
                            break
                        gold_found = gold_found.union(found)
                        gold_to_find -= found
                        for fact in found:
                            visible_from_fact = set(hood_func(self.fact_feats.iloc[fact]))
                            visible_facts = visible_facts.union(visible_from_fact)
                            df.loc[(df.q_idx == i) & (df.f_idx == fact), 'hops'] = hops_away
                return df

            def lex_overlap_hood(row):
                return row.overlap

            def nearest_k_hood(row):
                # only this part is not possible for single_fact_dataset
                return row.closest[:config.nearest_k_visible]

            logger.info('Creating k hops away DFs')
            self.qf_lex_hops = calc_hops(hood_func=lex_overlap_hood)
            self.qf_k_hops = calc_hops(hood_func=nearest_k_hood)

    def get_qa_with_facts(self, q_idx: int, fact_idxs: List[int], partial_idx: int = -1) -> EvalChainTuple:
        qa_row = self.qa_feats.iloc[q_idx]
        qa_id = qa_row.id
        # qa_tok = qa_row.tokenized.tolist()
        qa_txt = qa_row.original

        cur_facts = self.fact_feats.iloc[fact_idxs]
        visible_facts = self.get_visible_facts(qa=qa_row, facts=cur_facts)

        # cur_fact_ids = tuple(cur_facts.id)
        # cur_fact_toks = cur_facts.tokenized
        cur_fact_txt = ''.join(cur_facts.original)
        if cur_fact_txt:
            if self.config.encoding == 'single_candidate':
                cur_fact_txt = self.expl_mark + cur_fact_txt
            if self.config.answer_behind_expl and '(answer)' in qa_txt:
                qa, answer = qa_txt.split('(answer)')
                qa_txt, cur_fact_txt = qa, cur_fact_txt + '(answer)' + answer

        # flatten fact lists
        # concat_cur_fact_toks = [t for fact in cur_fact_toks for t in fact.tolist()]

        if self.supply_gold:
            remaining_fact_idxs: Tuple[int] = tuple(set(qa_row.gold_facts.tolist()) - set(fact_idxs))
        else:
            remaining_fact_idxs = tuple()
        # if remaining_fact_idxs:
        #     rem_facts = self.fact_feats.iloc[remaining_fact_idxs]
        #     remaining_fact_ids = tuple(rem_facts.id)
        # else:
        #     remaining_fact_ids = tuple()

        # input_ids = self.tokenizer.encode(text=qa_tok,
        #                                   text_pair=concat_cur_fact_toks if concat_cur_fact_toks else None,
        #                                   add_special_tokens=False)

        if self.config.encoding == 'single_candidate':
            original, explanation = qa_txt + cur_fact_txt, ''
        else:
            original, explanation = qa_txt, cur_fact_txt

        return EvalChainTuple(original=original,
                              explanation=explanation,
                              gold=remaining_fact_idxs,
                              question_id=str(qa_id),
                              question_idx=q_idx,
                              fact_idxs=tuple(fact_idxs),
                              visible_facts=visible_facts,
                              partial_idx=partial_idx,)

    def __iter__(self):
        indices = list(self.partials.loc[~self.partials.finished].index)
        logger.info('Nb of unfinished questions left: %s' % len(indices))
        indices = self.split_among_workers(indices)
        batch = PartialChainBatch(self.tokenizer.pad_token_id)
        for i in indices:
            partial_row = self.partials.iloc[i]
            qa_with_facts: EvalChainTuple = self.get_qa_with_facts(q_idx=partial_row.q_idx,
                                                                   fact_idxs=partial_row.partial_expl.tolist(),
                                                                   partial_idx=i)

            if len(qa_with_facts.fact_idxs) > 0 and self.config.predict_stop_expl:
                # append partial expl without candidate fact, to allow stopping
                batch.append(**self.prepare_without_candidate(qa_with_facts))

                if self.is_batch_full(len(batch.question_idxs), batch.max_length * len(batch.question_idxs)):
                    yield batch.yield_batch()  # gold_ids=qa_with_facts.gold
                    batch = PartialChainBatch(self.pad_token_id)

            for vf_index in qa_with_facts.visible_facts:
                batch.append(**self.prepare_with_candidate(qa_with_facts, vf_index))

                if self.is_batch_full(len(batch.question_idxs), batch.max_length * len(batch.question_idxs)):
                    yield batch.yield_batch()  # gold_ids=qa_with_facts.gold
                    batch = PartialChainBatch(self.pad_token_id)

            if (True or self.single_question_per_batch) and len(batch.question_idxs) > 0:
                yield batch.yield_batch()  # gold_ids=qa_with_facts.gold
                batch = PartialChainBatch(self.pad_token_id)
        if len(batch.question_idxs) > 0:
            yield batch.yield_batch()


class EvalChainDataset(EvalChainMixin, EvalDatasetMixin):
    pass


class GreedyBeamEvalChainDataset(EvalChainMixin, GreedyBeamEvalDatasetMixin):
    pass


class SampleBeamEvalChainDataset(EvalChainMixin, SampleBeamEvalDatasetMixin):
    pass
