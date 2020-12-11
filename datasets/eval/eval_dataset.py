import logging
import os
from abc import ABC

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_distances

logger = logging.getLogger()


class EvalDatasetMixin(ABC):

    def store_scores_batch(self, q_idx, scores):
        # fact idxs are already stored
        if self.config.eval_use_logprobs:
            scores = self.logits_2_logprobs(scores)
        self.partials.at[q_idx, 'scores_batch'] = scores.cpu().numpy()

    @staticmethod
    def logits_2_logprobs(logits):
        probs = logits.clone()
        probs[~torch.isnan(logits)] = F.log_softmax(logits[~torch.isnan(logits)], dim=0)
        # probs[logits == np.nan] = 0.0
        return probs

    @staticmethod
    def stacked_array(old_array, array):
        return np.vstack((old_array, array)) if len(old_array) > 0 else array[None, :]

    def process_scores(self):
        # process logits in batch -> add best facts to partial expl
        #                           keep logits to rank considered but unselected facts
        cols_to_edit = ['partial_expl', 'finished', 'scores_batch', 'scores', 'all_scores']
        self.partials[cols_to_edit] = self.partials.apply(
            lambda x: self.handle_row(x) if not x.finished else x[cols_to_edit],
            axis=1, result_type='expand'
        )

    def handle_row(self, row):
        assert not row.finished
        selected_fact = self.select_fact(row)
        partial = np.concatenate((row.partial_expl, [selected_fact]))
        finished = selected_fact == self.stop_explanation_id
        scores_batch = np.array([], dtype=float)
        if self.config.average_scores_over_partials:
            scores = self.stacked_array(row.scores, row.scores_batch)
        else:
            scores = row.scores_batch
        all_scores = self.stacked_array(row.all_scores, row.scores_batch)
        return partial, finished, scores_batch, scores, all_scores

    def select_fact(self, row):
        if len(row.scores_batch[~np.isnan(row.scores_batch)]) == 0:
            return self.stop_explanation_id

        # argsort will sort nan as greatest value
        argsorted = np.argsort(row.scores_batch)[:len(row.scores_batch[~np.isnan(row.scores_batch)])]
        selected_fact_idx, score = argsorted[-1], row.scores_batch[argsorted[-1]]

        if selected_fact_idx == self.stop_explanation_id:
            if (score < row.scores_batch[argsorted[-2]] + self.config.stop_delta
                    or len(row.partial_expl) < self.config.min_expl_length):
                selected_fact_idx = argsorted[-2]
                logger.info('Stop selected but undone. Score: %s - 2nd score: %s - delta: %s - '
                            'length: %s - min length: %s'
                            % (score, row.scores_batch[argsorted[-2]], self.config.stop_delta,
                               len(row.partial_expl), self.config.min_expl_length))
        return selected_fact_idx

    def is_finished(self):
        is_finished = max(self.partials.partial_expl.apply(len)) >= self.config.max_expl_length
        if self.config.predict_stop_expl:
            is_finished = is_finished or all(self.partials.partial_expl.apply(
                lambda part: len(part) > 0 and part[-1] == self.stop_explanation_id
            ))
        return is_finished

    def rank_all(self):
        # squash scores -> but make sure facts that were considered only in some iterations dont get 0 for
        #                   for skipped iterations  (done by np.nanmean())
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.partials.scores = self.partials.scores.apply(
                lambda scores: np.nanmean(scores, axis=0) if len(scores.shape) > 1 else scores
            )
        # nan > any real nb
        self.partials['fact_idxs_sorted'] = self.partials.apply(
            lambda row: self.prepare_fact_idxs(row),
            axis=1
        )
        result = {
            self.qa_idx_2_uid[q_idx]: [
                self.fact_idx_2_uid[f_idx] for f_idx in f_idxs if f_idx != self.stop_explanation_id
            ]
            for q_idx, f_idxs in enumerate(self.partials.fact_idxs_sorted)
        }
        self.reset_partials()
        return result

    def prepare_fact_idxs(self, row):
        scores = row.scores.copy()
        if self.config.rank_rest == 'random':
            scores[np.isnan(scores)] = np.random.rand(np.count_nonzero(np.isnan(scores))) \
                                       - 1.0 + scores[~np.isnan(scores)].min()
        elif self.config.rank_rest == 'seq':
            scores[np.isnan(scores)] = np.arange(-1, -np.count_nonzero(np.isnan(scores))-1, -1) \
                                       + scores[~np.isnan(scores)].min()
        elif self.config.rank_rest == 'tf_idf':
            # distances so positive and lower is better, after - : max is best is 0
            tf_idf_dists = np.concatenate((- self.tf_idf_distances(row), [np.nan]))[np.isnan(scores)]
            scores[np.isnan(scores)] = tf_idf_dists + scores[~np.isnan(scores)].min()

        if not self.config.use_partial_expl_in_ranking:
            # still only use non-nan because rank_rest might be none
            sorted_all = np.flip(np.argsort(scores)[:len(scores[~np.isnan(scores)])])
        elif not self.config.rank_scored_but_unused_2nd:
            sorted_all = np.array(row.partial_expl)
        else:
            sorted_all = np.concatenate((
                row.partial_expl,
                np.setdiff1d(np.flip(np.argsort(scores)[:len(scores[~np.isnan(scores)])]),
                             # assume_unique=True because numpy sorts values otherwise
                             row.partial_expl, assume_unique=True)
                if len(row.scores) > 0 else []
            ))
        return sorted_all

    def tf_idf_distances(self, row):
        partial_expl = row.partial_expl[np.where(row.partial_expl != self.stop_explanation_id)]
        stemmed_f = ' '.join(self.fact_feats.stemmed.iloc[partial_expl].apply(lambda x: ' '.join(x)))
        stemmed_q = ' '.join(self.qa_feats.stemmed.iloc[row.q_idx])
        transformed_expl = self.tfidf.vectorizer.transform([stemmed_q + stemmed_f])
        cos_distances = cosine_distances(transformed_expl, self.transformed_facts)
        return cos_distances[0]

    def reset_partials(self):
        if self.supply_gold:
            partials = self.qa_feats[['gold_facts']].copy().apply(np.array)
        else:
            partials = pd.DataFrame(index=self.qa_feats.index)

        partials['q_idx'] = partials.index.copy()
        partials['partial_expl'] = [np.array([], dtype=int)] * len(partials)
        partials['scores_batch'] = [np.array([], dtype=float)] * len(partials)
        partials['scores'] = [np.array([], dtype=float)] * len(partials)
        partials['all_scores'] = [np.array([], dtype=float)] * len(partials)
        partials['finished'] = False
        self.partials = partials

    def save_predictions_dataframe(self, output_dir):
        filename = os.path.join(output_dir, 'predictions_df.bin')
        logger.info('Saving predictions dataframe to %s' % filename)
        with open(filename, 'wb') as f:
            torch.save(self.partials, f)
