import argparse
import json
import logging
import os
import time

from datasets.factory import DatasetFactory
from rankers.chain_ranker import set_up_experiment, ChainRanker
from tools import utils

logger = logging.getLogger()


class HyperparamSearch:
    def __init__(self, config, tokenizer, num_eval):
        self.num_evaluations = num_eval
        self.config = config
        self.tokenizer = tokenizer

    def __call__(self, param, *args, **kwargs):
        results = {}
        sampler = self.sampler(param)
        try:
            for i in range(self.num_evaluations):
                try:
                    logger.info('#### Hyperparam search iteration %s ####' % i)
                    logger.info('Output dir: %s' % self.config.output_dir)
                    hyperparams = next(sampler)

                    self.config.set_nearest_k_visible(hyperparams['nearest_k_visible'])
                    self.config.set_max_expl_length(hyperparams['max_expl_length'])
                    self.config.set_min_expl_length(hyperparams.get('min_expl_length', 1))
                    self.config.set_beam_size(hyperparams.get('beam_size', 1))
                    self.config.set_use_partial_expl_in_ranking(hyperparams.get('use_partial_expl_in_ranking', True))
                    self.config.set_beam_score_relative_size(hyperparams.get('beam_score_relative_size', False))
                    self.config.set_average_scores_over_beams(hyperparams.get('average_scores_over_beams', True))
                    self.config.set_average_scores_weighted(hyperparams.get('average_scores_weighted', False))
                    self.config.set_eval_use_logprobs(hyperparams.get('eval_use_logprobs', False))
                    self.config.set_beam_score_acc(hyperparams.get('beam_score_acc', 'last'))
                    self.config.set_beam_fact_selection(hyperparams.get('beam_fact_selection', 'greedy'))
                    self.config.set_beam_decode_top_p(hyperparams.get('beam_decode_top_p', 0.3))
                    self.config.set_average_scores_over_partials(hyperparams.get('average_scores_over_partials', False))
                    self.config.set_rank_rest(hyperparams.get('rank_rest', 'seq'))
                    self.config.set_unused_beams_second(hyperparams.get('facts_from_unused_beams_second', True))
                    self.config.set_rank_scored_but_unused_2nd(hyperparams.get('rank_scored_but_unused_2nd', True))

                    with open(os.path.join(self.config.output_dir, "training_args_{}.json".format(i)), 'w') as f:
                        json.dump(self.config.to_dict(), f, indent=4)

                    tds = DatasetFactory.load_and_cache_dataset(self.config, self.tokenizer, self.config.test_qa_path,
                                                                infer=True)
                    ranker = ChainRanker(self.config, self.tokenizer)

                    logger.info('Evaluating hyperparam candidates: %s' % hyperparams)
                    it_result, _ = ranker.evaluate(tds, epoch_step='hp', step=i)

                    logger.info('Results it %s: %s' % (i, it_result))
                    results[i] = {'hyperparams': hyperparams, 'results': it_result}
                except KeyboardInterrupt:
                    time.sleep(5)
                    continue
                except StopIteration:
                    break
        except KeyboardInterrupt:
            pass
        # self.save_results(results)
        return results

    @staticmethod
    def sampler(param):
        # nearest_k = 290
        if param == 'k':
            params = [
                {
                    "nearest_k_visible": 90,
                    "max_expl_length": 8,
                    "min_expl_length": 1,
                    "rank_rest": "tf_idf",
                    "rank_scored_but_unused_2nd": True,
                },
                {
                    "nearest_k_visible": 130,
                    "max_expl_length": 8,
                    "min_expl_length": 1,
                    "rank_rest": "tf_idf",
                    "rank_scored_but_unused_2nd": True,
                },
                {
                    "nearest_k_visible": 180,
                    "max_expl_length": 8,
                    "min_expl_length": 1,
                    "rank_rest": "tf_idf",
                    "rank_scored_but_unused_2nd": True,
                },
                {
                    "nearest_k_visible": 210,
                    "max_expl_length": 8,
                    "min_expl_length": 1,
                    "rank_rest": "tf_idf",
                    "rank_scored_but_unused_2nd": True,
                },
                {
                    "nearest_k_visible": 290,
                    "max_expl_length": 8,
                    "min_expl_length": 1,
                    "rank_rest": "tf_idf",
                    "rank_scored_but_unused_2nd": True,
                },
            ]
        elif param == 'L':
            params = [
                {
                    "nearest_k_visible": 180,
                    "max_expl_length": 1,
                    "min_expl_length": 1,
                    "rank_rest": "tf_idf",
                    "rank_scored_but_unused_2nd": True,
                },
                {
                    "nearest_k_visible": 180,
                    "max_expl_length": 2,
                    "min_expl_length": 1,
                    "rank_rest": "tf_idf",
                    "rank_scored_but_unused_2nd": True,
                },
                {
                    "nearest_k_visible": 180,
                    "max_expl_length": 3,
                    "min_expl_length": 1,
                    "rank_rest": "tf_idf",
                    "rank_scored_but_unused_2nd": True,
                },
                {
                    "nearest_k_visible": 180,
                    "max_expl_length": 4,
                    "min_expl_length": 1,
                    "rank_rest": "tf_idf",
                    "rank_scored_but_unused_2nd": True,
                },
                {
                    "nearest_k_visible": 180,
                    "max_expl_length": 5,
                    "min_expl_length": 1,
                    "rank_rest": "tf_idf",
                    "rank_scored_but_unused_2nd": True,
                },
                {
                    "nearest_k_visible": 180,
                    "max_expl_length": 6,
                    "min_expl_length": 1,
                    "rank_rest": "tf_idf",
                    "rank_scored_but_unused_2nd": True,
                },
                {
                    "nearest_k_visible": 180,
                    "max_expl_length": 7,
                    "min_expl_length": 1,
                    "rank_rest": "tf_idf",
                    "rank_scored_but_unused_2nd": True,
                },
                {
                    "nearest_k_visible": 180,
                    "max_expl_length": 8,
                    "min_expl_length": 1,
                    "rank_rest": "tf_idf",
                    "rank_scored_but_unused_2nd": True,
                },
                {
                    "nearest_k_visible": 180,
                    "max_expl_length": 9,
                    "min_expl_length": 1,
                    "rank_rest": "tf_idf",
                    "rank_scored_but_unused_2nd": True,
                },
                {
                    "nearest_k_visible": 180,
                    "max_expl_length": 10,
                    "min_expl_length": 1,
                    "rank_rest": "tf_idf",
                    "rank_scored_but_unused_2nd": True,
                },
            ]
        else:
            assert param == 'rr', param
            params = [
                {
                    "nearest_k_visible": 180,
                    "max_expl_length": 8,
                    "min_expl_length": 1,
                    "rank_rest": "random",
                    "rank_scored_but_unused_2nd": True,
                },
                {
                    "nearest_k_visible": 180,
                    "max_expl_length": 8,
                    "min_expl_length": 1,
                    "rank_rest": "none",
                    "rank_scored_but_unused_2nd": True,
                },
                {
                    "nearest_k_visible": 180,
                    "max_expl_length": 8,
                    "min_expl_length": 1,
                    "rank_rest": "none",
                    "rank_scored_but_unused_2nd": False,
                },
                {
                    "nearest_k_visible": 290,
                    "max_expl_length": 9,
                    "min_expl_length": 3,
                    "rank_rest": "tf_idf",
                    "rank_scored_but_unused_2nd": True,
                },
            ]

        for param in params:
            yield param


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None, )
    parser.add_argument('--num_eval', type=int, default=20,)
    parser.add_argument('--param', type=str, default='L', choices=('L', 'k', 'rr'))
    args = parser.parse_args()

    _, config, tokenizer = set_up_experiment(args.config_path)
    utils.initialize_logging(config=config)

    search = HyperparamSearch(config, tokenizer, args.num_eval)
    res = search(args.param)
