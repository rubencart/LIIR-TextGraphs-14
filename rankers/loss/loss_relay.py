
from torch.nn import CrossEntropyLoss

from datasets.utils import NOISE_CONTRASTIVE_LOSSES
from rankers.loss.loss_functions import BCEWithLogitsLossWrapper, RankNetLoss, BinaryNCELoss, RankingNCELoss, MarginRankingLossWrapper
from tools.config import ChainConfig


class LossRelay:
    def __init__(self, config: ChainConfig):
        self.huggingface_loss = False
        self.loss_index, self.logits_index = None, 0
        self.num_labels = 2
        if config.loss == 'ranking-nce':
            self.num_labels = config.num_labels
            self.loss_fct = RankingNCELoss()
        elif config.loss == 'binary-nce':
            self.num_labels = config.num_labels
            self.loss_fct = BinaryNCELoss()
        elif config.loss == 'xent':
            self.num_labels = 1
            self.loss_fct = BCEWithLogitsLossWrapper()
        elif config.loss == 'ranknet':
            self.loss_fct = RankNetLoss()
        elif config.loss == 'margin-pairs':
            self.loss_fct = MarginRankingLossWrapper(margin=config.margin)
        else:
            self.huggingface_loss = True
            if config.loss == 'xent-2':
                self.loss_fct = CrossEntropyLoss()
            else:  # mse
                assert config.loss == 'mse', config.loss
                self.num_labels = 1
                self.loss_fct = id
            self.loss_index, self.logits_index = 0, 1

        self.config = config

    def __call__(self, outputs, **kwargs):
        if self.config.loss in NOISE_CONTRASTIVE_LOSSES:
            loss = self.loss_fct(outputs[self.logits_index], kwargs['labels'], kwargs['noise_logprobs'])
        elif not self.huggingface_loss:
            loss = self.loss_fct(outputs[self.logits_index], kwargs['labels'].float())
        else:
            loss = outputs[self.loss_index]  # model outputs are always tuple in transformers (see doc)
        if len(loss.shape) > 0 and loss.shape[0] > 1:
            loss = loss.mean()
        return loss
