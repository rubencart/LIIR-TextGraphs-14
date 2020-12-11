"""
Code:
RankNet & LambdaRank: https://github.com/haowei01/pytorch-examples
Idem + more: https://github.com/allegro/allRank
"""
import logging
import math
from itertools import product

import torch
from torch import nn
from torch.nn import MarginRankingLoss, BCEWithLogitsLoss, CrossEntropyLoss

logger = logging.getLogger(__name__)


class BCEWithLogitsLossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = BCEWithLogitsLoss(reduction='mean')

    def forward(self, logits, labels):
        return self.loss(logits.squeeze(-1), labels)


class MarginRankingLossWrapper(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.loss = MarginRankingLoss(margin=margin)
        self.margin = margin

    def forward(self, logits, labels):
        """
        logits: bs x 2,  scores (higher = pos)
        labels: bs,      0 (neg) or 1 (pos)
        """
        preds = logits.clone()

        pos_idxs = (labels > 0.5).nonzero().squeeze(-1).tolist()
        neg_idxs = (labels < 0.5).nonzero().squeeze(-1).tolist()
        if not len(pos_idxs) > 0 or not len(neg_idxs) > 0:
            return torch.tensor(0.0)

        preds = torch.sub(preds[:, 1], preds[:, 0]).unsqueeze(-1)  # subtract neg from pos

        pairs_idxs = list(product(pos_idxs, neg_idxs))  # cartesian product
        pred_pairs = preds[pairs_idxs, :]  # shape len(pairs_idxs), 2, 1
        # pred_diffs = pred_pairs[:, 0] - pred_pairs[:, 1]  # shape len(pairs_idxs), 1

        return self.loss(pred_pairs[:, 0], pred_pairs[:, 1], torch.ones_like(pred_pairs[:, 0]))


class RankNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = None
        self.loss = BCEWithLogitsLoss(weight=self.weight, reduction='mean')

    def forward(self, logits, labels):
        """
        logits: bs x 2,  scores (higher = pos)
        labels: bs,      0 (neg) or 1 (pos)
        """
        preds = logits.clone()

        pos_idxs = (labels > 0.5).nonzero().squeeze(-1).tolist()
        neg_idxs = (labels < 0.5).nonzero().squeeze(-1).tolist()
        if not len(pos_idxs) > 0 or not len(neg_idxs) > 0:
            return torch.tensor(0.0)

        preds = torch.sub(preds[:, 1], preds[:, 0]).unsqueeze(-1)  # subtract neg from pos

        pairs_idxs = list(product(pos_idxs, neg_idxs))  # cartesian product
        pred_pairs = preds[pairs_idxs, :]                # shape len(pairs_idxs), 2, 1
        pred_diffs = pred_pairs[:, 0] - pred_pairs[:, 1]  # shape len(pairs_idxs), 1

        return self.loss(pred_diffs, torch.ones_like(pred_diffs))


class BinaryNCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = None
        self.bce_with_logits_loss = BCEWithLogitsLoss(weight=self.weight, reduction='mean')

    def forward(self, logits_model, labels, logprobs_noise=None, logits_noise=None):
        """
        Based on https://github.com/Stonesjtu/Pytorch-NCE/blob/master/nce/nce_loss.py .

        logits_model: logits (before log regr sigmoid) of all samples (pos and neg) computed by model
                      --> probs that samples are from data distribution
                shape: bs x 1
        labels: 0 (neg, from p_noise) or 1 (pos, from p_data)
                shape: bs
        logprobs_noise: logprobs of all samples (the same samples, both pos and neg) from noise distribution
                      --> probs that samples are from noise distribution
                      E.g. logprobs computed by model in previous iteration of SCE
                           or logprobs of
                shape: bs x 1
        """
        if logprobs_noise is None:
            assert logits_noise is not None
            logprobs_noise = logits_noise
            # logits_noise = torch.log(torch.exp(logprobs_noise) / (1 - torch.exp(logprobs_noise)))

        if logits_model.shape[1] == 2:
            logits_model = torch.sub(logits_model[:, 1], logits_model[:, 0]).unsqueeze(-1)  # subtract neg from pos

        if len(logprobs_noise.shape) < 2:
            logprobs_noise = logprobs_noise.unsqueeze(-1)

        pos_idxs = (labels > 0.5).nonzero().squeeze(-1).tolist()
        neg_idxs = (labels < 0.5).nonzero().squeeze(-1).tolist()
        if not len(pos_idxs) > 0 or not len(neg_idxs) > 0:
            logger.error('No positives or no negatives')
            return torch.tensor(0.0)

        noise_ratio = math.ceil(len(neg_idxs) / len(pos_idxs))
        logits = logits_model - logprobs_noise - math.log(noise_ratio)
        # todo - gamma? see Ma & Collins , or word embeddings neg sampling papers using NCE

        return self.bce_with_logits_loss(logits.squeeze(-1), labels.float())


class RankingNCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = None
        self.xent_loss = CrossEntropyLoss(weight=self.weight, reduction='mean')

    def forward(self, logits_model, labels, logprobs_noise=None, logits_noise=None):
        """
        Based on https://github.com/Stonesjtu/Pytorch-NCE/blob/master/nce/nce_loss.py .

        logits_model: logits (before log regr sigmoid) of all samples (pos and neg) computed by model
                      --> probs that samples are from data distribution
                shape: bs x 1
        labels: 0 (neg, from p_noise) or 1 (pos, from p_data)
                shape: bs
        logprobs_noise: logprobs of all samples (the same samples, both pos and neg) from noise distribution
                      --> probs that samples are from noise distribution
                      E.g. logprobs computed by model in previous iteration of SCE
                           or logprobs of
                shape: bs x 1
        """
        # see https://arxiv.org/pdf/1809.01812.pdf , ranking version
        if logprobs_noise is None:
            assert logits_noise is not None
            logprobs_noise = logits_noise
            # logits_noise = torch.log(torch.exp(logprobs_noise) / (1 - torch.exp(logprobs_noise)))

        if logits_model.shape[1] == 2:
            logits_model = torch.sub(logits_model[:, 1], logits_model[:, 0]).unsqueeze(-1)  # subtract neg from pos

        if len(logprobs_noise.shape) < 2:
            logprobs_noise = logprobs_noise.unsqueeze(-1)

        pos_idxs = (labels > 0.5).nonzero().squeeze(-1)  # .tolist()
        neg_idxs = (labels < 0.5).nonzero().squeeze(-1)  # .tolist()
        if not len(pos_idxs) > 0 or not len(neg_idxs) > 0:
            logger.error('No positives or no negatives')
            return torch.tensor(0.0)
        assert len(pos_idxs) == 1

        # noise_ratio = math.ceil(len(neg_idxs) / len(pos_idxs))
        logits = logits_model - logprobs_noise  # - math.log(noise_ratio)

        # print(logits.unsqueeze(0).squeeze(-1), torch.where(labels > 0)[0])
        return self.xent_loss(logits.unsqueeze(0).squeeze(-1), torch.where(labels > 0)[0])
