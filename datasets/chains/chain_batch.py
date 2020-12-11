import logging
import random
from abc import ABC
from typing import Tuple, List

import torch
from torch import Tensor

from rankers.utils import pad_2d_tensors

logger = logging.getLogger()


class ChainTuple:
    def __init__(self, original: str, explanation: str, gold: Tuple[int], question_id: str, question_idx: int,
                 fact_idxs: Tuple[int], visible_facts: Tuple[int], noise_logprob: float = None):
        self.original: str = original
        self.explanation: str = explanation
        self.gold: Tuple[int] = gold
        self.question_id: str = question_id
        self.question_idx: int = question_idx
        self.fact_idxs: Tuple[int] = fact_idxs
        self.visible_facts: Tuple[int] = visible_facts
        self.noise_logprob: float = noise_logprob

    def __str__(self):
        return self.original + self.explanation


class EvalChainTuple(ChainTuple):
    def __init__(self, partial_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.partial_idx = partial_idx


class FisherTuple:
    def __init__(self, original: str, explanation: str, label: bool, question_id: str,
                 question_idx: int, fact_idxs: Tuple[int], weight: float, noise_logprob: float = None):
        self.original: str = original
        self.explanation = explanation
        self.label = label
        self.weight = weight
        self.question_id: str = question_id
        self.question_idx: int = question_idx
        self.fact_idxs: Tuple[int] = fact_idxs
        self.noise_logprob: float = noise_logprob

    def __str__(self):
        return self.original + '' + self.explanation


class FisherEvalTuple(FisherTuple):
    def __init__(self, partial_idx: int, visible_facts: Tuple[int], gold: Tuple[int], **kwargs):
        super().__init__(**kwargs)
        self.partial_idx: int = partial_idx
        self.visible_facts: Tuple[int] = visible_facts
        self.gold: Tuple[int] = gold


class Batch:
    def __init__(self, tokenized: Tensor, token_type_ids: Tensor, attention_mask: Tensor, labels: Tensor,
                 question_idxs: Tuple[int], cand_fact_idxs: Tuple[int], partial_idxs: Tuple[int] = None,
                 noise_logprobs: Tensor = None,):
        self.tokenized: Tensor = tokenized
        self.token_type_ids: Tensor = token_type_ids
        self.attention_mask: Tensor = attention_mask
        self.labels: Tensor = labels
        self.question_idxs: Tuple[int] = question_idxs
        self.partial_idxs: Tuple[int] = partial_idxs
        self.cand_fact_idxs: Tuple[int] = cand_fact_idxs
        self.noise_logprobs: Tensor = noise_logprobs

    def to_device(self, device):
        self.tokenized = self.tokenized.to(device)
        if self.token_type_ids is not None:
            self.token_type_ids = self.token_type_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.labels = self.labels.to(device)
        if self.noise_logprobs is not None:
            self.noise_logprobs = self.noise_logprobs.to(device)

    def __str__(self):
        return '\n'.join(
            ['-----BATCH-----']
            + ['%s: %s' % (str(k), str(v)) for k, v in vars(self).items()]
            + ['-------------']
        )

    def to_dict(self, without_labels=False):
        dct = {
            'input_ids': self.tokenized,
            'attention_mask': self.attention_mask,
        }
        if self.token_type_ids is not None:
            dct.update({'token_type_ids': self.token_type_ids})
        if not without_labels:
            dct.update({'labels': self.labels})
        return dct


class ChainBatch(Batch):
    pass


class PartialBatch(ABC):
    def __init__(self, pad_token_id):
        self.tokenized: List[Tensor] = []
        self.token_type_ids: List[Tensor] = []
        self.attention_mask: List[Tensor] = []
        self.labels: List[int] = []
        self.question_idxs: List[int] = []
        self.partial_idxs: List[int] = []
        self.cand_fact_idxs: List[int] = []
        self.pad_token_id: int = pad_token_id
        # todo fix if we ever use XLNet
        self.pad_att_mask_id = 0    # XLNet: "0 for tokens that are NOT MASKED", rest 1
        self.pad_tok_type_id = 0    # XLNet: "0 for tokens that are NOT MASKED", rest 1
        self.max_length = 0

        self.noise_logprobs: List[float] = []

    def append(self, tokenized: List[int], token_type_ids: List[int], attention_mask: List[int],
               label: int,
               question_idx: int,
               partial_idx: int = -1,
               cand_fact_idx: int = -1,
               noise_logprob: float = None,
               ):
        self.tokenized.append(torch.tensor(tokenized, dtype=torch.long))
        if token_type_ids:
            self.token_type_ids.append(torch.tensor(token_type_ids))
        self.attention_mask.append(torch.tensor(attention_mask, dtype=torch.long))
        self.labels.append(label)
        self.question_idxs.append(question_idx)
        if partial_idx > -1:
            self.partial_idxs.append(partial_idx)
        self.max_length = max(self.max_length, len(tokenized))
        if cand_fact_idx > -1:
            self.cand_fact_idxs.append(cand_fact_idx)
        if noise_logprob is not None:
            self.noise_logprobs.append(noise_logprob)

    def shuffle(self):
        permutation = random.sample(range(len(self.question_idxs)), len(self.question_idxs))
        self.tokenized = [self.tokenized[i] for i in permutation]
        if self.token_type_ids:
            self.token_type_ids = [self.token_type_ids[i] for i in permutation]
        self.attention_mask = [self.attention_mask[i] for i in permutation]
        self.labels = [self.labels[i] for i in permutation]
        self.question_idxs = [self.question_idxs[i] for i in permutation]
        if self.partial_idxs:
            self.partial_idxs = [self.partial_idxs[i] for i in permutation]
        self.cand_fact_idxs = [self.cand_fact_idxs[i] for i in permutation]
        self.noise_logprobs = [self.noise_logprobs[i] for i in permutation]
        return permutation


class PartialChainBatch(PartialBatch):
    def yield_batch(self, last=False) -> ChainBatch:
        # todo batch_encode_plus?
        return ChainBatch(
            tokenized=pad_2d_tensors(self.tokenized, self.pad_token_id),
            token_type_ids=(
                pad_2d_tensors(self.token_type_ids, self.pad_tok_type_id)
                if self.token_type_ids else None
            ),
            attention_mask=pad_2d_tensors(self.attention_mask, self.pad_att_mask_id),
            labels=torch.tensor(self.labels, dtype=torch.long),
            question_idxs=tuple(self.question_idxs),
            partial_idxs=tuple(self.partial_idxs) if self.partial_idxs else None,
            cand_fact_idxs=tuple(self.cand_fact_idxs),
            noise_logprobs=torch.tensor(self.noise_logprobs, dtype=torch.float) if self.noise_logprobs else None,
        )
