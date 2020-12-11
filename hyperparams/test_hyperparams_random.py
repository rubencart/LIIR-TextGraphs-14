import argparse
import json
import logging
import os
import random
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

    def __call__(self, *args, **kwargs):
        results = {}
        sampler = self.sampler()
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

                    val_dataset = DatasetFactory.load_and_cache_dataset(self.config, self.tokenizer, self.config.val_qa_path,
                                                                        valid=True)
                    ranker = ChainRanker(self.config, self.tokenizer)

                    logger.info('Evaluating hyperparam candidates: %s' % hyperparams)
                    it_result, _ = ranker.evaluate(val_dataset, epoch_step='debug', step=i)

                    logger.info('Results it %s: %s' % (i, it_result))
                    results[i] = {'hyperparams': hyperparams, 'results': it_result}
                except KeyboardInterrupt:
                    time.sleep(5)
                    continue
                except StopIteration:
                    break
        except KeyboardInterrupt:
            pass
        self.save_results(results)
        return results

    @staticmethod
    def _sample():
        max_expl_length = random.randrange(3, 11)
        min_expl_length = random.randrange(1, min(max_expl_length, 4))
        yield {
            'nearest_k_visible': 130,
            'beam_size': random.choice((1, 3, 5, 10)),
            'max_expl_length': max_expl_length,
            'min_expl_length': min_expl_length,
            'use_partial_expl_in_ranking': bool(random.randrange(0, 2)),
        }

    @staticmethod
    def sampler():
        # nearest_k = 290
        params = [

        ]

        for param in params:
            yield param

    def save_results(self, results):
        config = self.config
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)

        json_filename = 'hyperparams.json'
        preds_eval_json = os.path.join(config.output_dir, json_filename)
        logger.info('Saving results to %s' % preds_eval_json)
        with open(preds_eval_json, 'w') as f:
            json.dump(results, f, indent=4)

        logger.info(
            'Search results: %s'
            % json.dumps(sorted([res for i, res in results.items()], key=lambda x: x['results']['map']), indent=4)
        )
        logger.info('Best results: %s' % json.dumps(self.find_best_hyperparams(results), indent=4))

    @staticmethod
    def find_best_hyperparams(results):
        it_results, _ = max(
            [(it_results, it_results['results']['map'])
             for i, it_results in results.items()],
            key=lambda x: x[1]
        )
        return it_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None, )
    parser.add_argument('--num_eval', type=int, default=20,)
    args = parser.parse_args()

    _, config, tokenizer = set_up_experiment(args.config_path)
    utils.initialize_logging(config=config)

    search = HyperparamSearch(config, tokenizer, args.num_eval)
    res = search()
