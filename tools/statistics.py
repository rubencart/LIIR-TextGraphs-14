import json
import logging
import time

logger = logging.getLogger(__name__)


class Statistics(object):
    """
    Wrapper class for a bunch of running statistics
    """

    def __init__(self, config):
        self.step_start_time = time.time()
        self.total_time_elapsed = 0
        self.train_time_elapsed = 0
        self.eval_time_elapsed = 0

        self.n_gpu = config.n_gpu
        self.train_batch_size = config.train_batch_size
        self.train_batch_tokens = config.train_tokens_per_batch
        self.eval_batch_size = config.eval_batch_size
        self.eval_batch_tokens = config.eval_tokens_per_batch

        self.stat_points = []

    def load_stats(self, file):
        with open(file, 'r') as f:
            infos = json.load(f)
        self.stat_points = infos.get('stat_points', [])

    def to_dict(self):
        return {
            'stat_points': self.stat_points,
        }

    def start_step(self):
        self.step_start_time = time.time()

    def toggle_to_train(self):
        end = time.time()
        time_elapsed = end - self.step_start_time
        self.total_time_elapsed += time_elapsed
        self.eval_time_elapsed += time_elapsed
        self.step_start_time = end

    def toggle_to_eval(self):
        end = time.time()
        time_elapsed = end - self.step_start_time
        self.total_time_elapsed += time_elapsed
        self.train_time_elapsed += time_elapsed
        self.step_start_time = end

    def record(self, step, lr, loss, eval_results=None, other=None) -> None:
        stats_dict = {
            'step': step,
            'loss': loss,
            'lr': lr,
            'map': eval_results['map'] if (eval_results is not None and 'map' in eval_results) else None,
            "total_time_elapsed": self.total_time_elapsed,
            "train_time_elapsed": self.train_time_elapsed,
            "eval_time_elapsed": self.eval_time_elapsed,
            "n_gpu": self.n_gpu,
            "train_batch_size": self.train_batch_size,
            "train_batch_tokens": self.train_batch_tokens,
            "eval_batch_size": self.eval_batch_size,
            "eval_batch_tokens": self.eval_batch_tokens,
        }
        if other:
            stats_dict.update({'other': other})
        self.stat_points.append(stats_dict)

    def log_stats(self):
        if len(self.stat_points) < 1:
            return
        stats = self.stat_points[-1]
        logger.info('---------------- STATS ----------------')
        logger.info('Step: %s', stats['step'])
        logger.info('Avg loss: %s', stats['loss'])
        logger.info('LR: %s', stats['lr'])
        if 'map' in stats:
            logger.info('MAP: %s', stats['map'])
        if 'other' in stats:
            logger.info('Other: %s', stats['other'])
        # logger.info('Avg eval loss: %s', stats[''])
        logger.info('Time elapsed total: %s', stats['total_time_elapsed'])
        logger.info('Time elapsed training: %s', stats['train_time_elapsed'])
        # logger.info('ALT MAP: %s', self.map_2_stats[-1][1])

    def write_stats(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
