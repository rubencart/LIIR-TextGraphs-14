
import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical
from transformers import top_k_top_p_filtering

from datasets.eval.eval_dataset import EvalDatasetMixin


class BeamEvalDatasetMixin(EvalDatasetMixin):

    def reset_partials(self):
        columns = ['id', 'gold_facts'] if self.supply_gold else ['id']
        partials = self.qa_feats[columns].copy()
        if self.supply_gold:
            partials.gold_facts = partials.gold_facts.apply(np.array)

        partials['q_idx'] = partials.index.copy()
        partials['beam_score'] = 0.0
        partials['beam_idx'] = 0
        partials['beam_score_relative'] = 0.0
        partials['candidates'] = [np.array([], dtype=int)] * len(partials)
        partials['candidate_scores'] = [np.array([], dtype=int)] * len(partials)

        partials['partial_expl'] = [np.array([], dtype=int)] * len(partials)
        partials['scores_batch'] = [np.array([], dtype=float)] * len(partials)
        partials['scores'] = [np.array([], dtype=float)] * len(partials)
        partials['all_scores'] = [np.array([], dtype=float)] * len(partials)
        partials['finished'] = False
        self.partials = partials

    def process_scores(self):
        cols_to_edit = ['scores_batch', 'scores', 'candidates', 'beam_score', 'all_scores']
        self.partials[cols_to_edit] = self.partials.apply(
            lambda x: self.get_candidates(x) if not x.finished else x[cols_to_edit],
            axis=1, result_type='expand'
        )
        self.partials = self.partials.explode('candidates')

        cols_to_edit = ['partial_expl', 'finished', 'candidates', 'beam_score']
        self.partials[cols_to_edit] = self.partials.apply(
            lambda x: self.process_candidates(x) if not x.finished else x[cols_to_edit],
            axis=1, result_type='expand'
        )

        # df = self.partials.groupby(by='q_idx', sort=False) \
        #       .apply(lambda x: x[x['beam_score'] == x['beam_score'].max()])
        if self.config.beam_score_relative_size and self.config.beam_score_acc == 'sum':
            self.partials.beam_score_relative = self.partials.beam_score / self.partials.partial_expl.apply(len)
        else:
            self.partials.beam_score_relative = self.partials.beam_score

        self.partials = self.select_beams()

    def get_candidates(self, row):
        beam_score = row.beam_score
        candidates = self.select_facts(row)
        scores_batch = np.array([], dtype=float)
        scores, all_scores = self.stack_scores(row)
        return scores_batch, scores, candidates, beam_score, all_scores

    def select_facts(self, row):
        raise NotImplementedError

    def stack_scores(self, row):
        all_scores = self.stacked_array(row.all_scores, row.scores_batch)
        if self.config.average_scores_over_partials:
            return self.stacked_array(row.scores, row.scores_batch), all_scores
        else:
            return row.scores_batch, all_scores
        # return self.stacked_array(row.scores, row.scores_batch)

    def process_candidates(self, row):
        assert not row.finished

        selected_fact, cand_score = row.candidates
        partial = np.concatenate((row.partial_expl, [int(selected_fact)]))
        finished = int(selected_fact) == self.stop_explanation_id
        candidates = None
        if self.config.beam_score_acc == 'sum':
            beam_score = row.beam_score + cand_score  # todo how to accumulate beam scores?
        else:
            assert self.config.beam_score_acc == 'last'
            beam_score = cand_score
        return partial, finished, candidates, beam_score

    def select_beams(self):
        df = self.partials.groupby(by='q_idx', sort=False) \
            .apply(lambda grp: grp.nlargest(self.config.beam_size, columns='beam_score_relative')) \
            .reset_index(drop=True)

        if self.config.average_scores_over_beams_intermed:
            df_stacked_scores = self.partials.groupby(by='q_idx', sort=False)['scores'] \
                .apply(lambda x: np.vstack(x))
            df_stacked_scores = df_stacked_scores \
                .iloc[np.arange(len(df_stacked_scores.index)).repeat(self.config.beam_size)] \
                .reset_index(drop=True)
            df.loc[:, 'scores'] = df_stacked_scores
        return df

    def rank_all(self):
        df = self.partials.sort_values('beam_score_relative', ascending=False) \
            .drop_duplicates(['q_idx']) \
            .sort_values('q_idx', ascending=True) \
            .reset_index(drop=True)

        # rank facts from other beams between partial expl and considered but not chosen
        if self.config.facts_from_unused_beams_second:
            df_extended_beams = self.partials.groupby(by='q_idx', sort=False)['partial_expl'] \
                                .apply(lambda x: np.concatenate(x.tolist())) \
                                .reset_index(drop=True) \
                                .apply(pd.unique)
            df.loc[:, 'partial_expl'] = df_extended_beams

        if self.config.average_scores_over_beams:
            df_weighted = self.partials.copy()
            if self.config.average_scores_weighted:
                if self.config.eval_use_logprobs or self.config.beam_fact_selection == 'sample':   # weight by probs
                    # to estimate expected value: just take average of samples, don't multiply with probs
                    pass
                else:       # weight by scores (logits)
                    df_weighted['weights'] = df_weighted.beam_score_relative + min(df_weighted.beam_score_relative)
                    df_weighted['sq_weights'] = df_weighted.weights ** 2
                    df_weighted.weights = df_weighted.weights.div(
                        df_weighted.groupby(by='q_idx', sort=False)['sq_weights']
                                .transform('sum')
                                .apply(lambda x: np.sqrt(x))
                    )
                df_weighted.scores = df_weighted.scores * df_weighted.weights
            df_stacked_scores = df_weighted.groupby(by='q_idx', sort=False)['scores'] \
                .apply(lambda x: np.vstack(x)) \
                .reset_index(drop=True)
            df.loc[:, 'scores'] = df_stacked_scores

        self.partials = df
        return super().rank_all()


class GreedyBeamEvalDatasetMixin(BeamEvalDatasetMixin):

    def select_facts(self, row):
        sc_copy = np.copy(row.scores_batch)
        sc_copy[self.stop_explanation_id] -= self.config.stop_delta
        if len(row.partial_expl) < self.config.min_expl_length:
            sc_copy[self.stop_explanation_id] = -np.inf

        # argsort will sort nan as greatest value
        argsorted = np.flip(np.argsort(sc_copy)[:len(sc_copy[~np.isnan(sc_copy)])])
        sc_sorted = np.flip(np.sort(sc_copy)[:len(sc_copy[~np.isnan(sc_copy)])])

        if len(argsorted) < self.config.beam_size:
            selected = np.array(self.stop_explanation_id).repeat(self.config.beam_size)
            scores = np.array(0.0).repeat(self.config.beam_size)
            selected[:len(argsorted)] = argsorted
            scores[:len(sc_sorted)] = sc_sorted
            return tuple(zip(selected, scores))

        return tuple(zip(argsorted[:self.config.beam_size], sc_sorted[:self.config.beam_size]))


class SampleBeamEvalDatasetMixin(BeamEvalDatasetMixin):

    def reset_partials(self):
        self.logits = torch.empty((len(self.qa_feats.index), self.config.beam_size, self.nb_facts))
        super().reset_partials()

    def store_scores_batch(self, partial_idx, logits):
        # fact idxs are already stored
        q_idx = partial_idx // self.config.beam_size
        beam_idx = partial_idx % self.config.beam_size
        self.partials.at[partial_idx, 'beam_idx'] = beam_idx
        self.logits[q_idx, beam_idx] = logits

    def select_facts(self, row):
        logits = self.logits[row.q_idx, row.beam_idx].clone().detach()
        assert logits.shape == (self.nb_facts,), logits.shape

        logits[self.stop_explanation_id] -= self.config.stop_delta
        if len(row.partial_expl) < self.config.min_expl_length:
            logits[self.stop_explanation_id] = -float('Inf')

        logits[torch.isnan(logits)] = -float('Inf')
        logits = logits.unsqueeze(0)
        logits = top_k_top_p_filtering(logits, top_p=self.config.beam_decode_top_p)
        logits = logits.squeeze(0)
        distrib = Categorical(logits=logits)

        idxs = distrib.sample((self.config.beam_size,))     # beam_size
        scores = logits[idxs]
        scores_logprobs = distrib.log_prob(idxs).reshape(idxs.shape)

        result = tuple(zip(idxs.tolist(), scores_logprobs.tolist()))
        return result

    def stack_scores(self, row):
        # row.scores_batch is empty because logits stored in self.logits
        scores_batch = self.logits[row.q_idx, row.beam_idx].cpu().numpy()
        all_scores = self.stacked_array(row.all_scores, scores_batch)
        if self.config.average_scores_over_partials:
            return self.stacked_array(row.scores, scores_batch), all_scores
        else:
            return scores_batch, all_scores
