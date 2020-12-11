import copy
import logging
import os

import torch
from torch.nn import Module

logger = logging.getLogger(__name__)


class Podium(object):
    def __init__(self, output_dir, margin=0.0):
        self.champion = None
        self.best_score = 0.0
        self.margin = margin
        self.best_iteration = 0
        self.output_dir = output_dir

    def process_candidate(self, score: float, model: Module, iteration: int):
        if score > self.best_score + self.margin:
            logger.info('New best validation score: %s' % score)
            self.handle_new_champion(score, model, iteration)
        else:
            logger.info('Best validation score not beaten, still: %s, from iteration %s'
                        % (self.best_score, self.best_iteration))

    def get_best_model(self) -> Module:
        return self.champion

    def handle_new_champion(self, score: float, model: Module, iteration: int):
        self.best_score = float(score)
        self.champion = copy.deepcopy(model)
        torch.save(self.champion,
                   os.path.join(self.output_dir, 'best_model_{:.2f}.pt'.format(self.best_score)))
        self.best_iteration = int(iteration)
