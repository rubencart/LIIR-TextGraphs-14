import logging
import math
import os
import random
from typing import Tuple, List, NamedTuple, Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import PreTrainedTokenizer

from datasets.utils import CONTRASTIVE_LOSSES
from datasets.worldtree_dataset import WorldTreeDataset
from tools.config import SingleFactConfig
from rankers.utils import pad_2d_tensors

logger = logging.getLogger(__name__)


class ClfQATuple(NamedTuple):
    tokenized: List[int]
    token_type_ids: List[int]
    attention_mask: List[int]
    gold: bool
    question_id: str
    fact_id: str
    question_idx: int
    noise_logprob: float


class ClfQABatch:
    def __init__(self, tokenized, token_type_ids, attention_mask, labels, question_ids, fact_ids, noise_logprobs=None):
        self.tokenized: Tensor = tokenized
        self.token_type_ids: Tensor = token_type_ids
        self.attention_mask: Tensor = attention_mask
        self.labels: Tensor = labels
        self.question_ids: Tuple[str] = question_ids
        self.fact_ids: Tuple[str] = fact_ids
        self.noise_logprobs = noise_logprobs

    def to_device(self, device):
        self.tokenized = self.tokenized.to(device)
        if self.token_type_ids is not None:
            self.token_type_ids = self.token_type_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.labels = self.labels.to(device)
        if self.noise_logprobs is not None:
            self.noise_logprobs = self.noise_logprobs.to(device)

    def __str__(self):
        return str(vars(self))

    def type_label(self, labels, ltype):
        # return labels.long()
        if ltype == 'float':
            return labels.float()
        else:
            return labels.long()

    def to_dict(self, without_labels=False, labels_as='float'):
        dct = {
            'input_ids': self.tokenized,
            'attention_mask': self.attention_mask,
        }
        if self.token_type_ids is not None:
            dct.update({'token_type_ids': self.token_type_ids})
        if not without_labels:
            dct.update({'labels': self.type_label(self.labels, ltype=labels_as)})
        return dct


class PartialSingleBatch:
    def __init__(self, pad_token_id):
        self.tokenized: List[Tensor] = []
        self.token_type_ids: List[Tensor] = []
        self.attention_mask: List[Tensor] = []
        self.labels: List[int] = []
        self.question_ids: List[str] = []
        self.cand_fact_ids: List[str] = []
        self.noise_logprobs: List[float] = []

        self.pad_token_id: int = pad_token_id
        # todo fix if we ever use XLNet
        self.pad_att_mask_id = 0    # XLNet: "0 for tokens that are NOT MASKED", rest 1
        self.pad_tok_type_id = 0    # XLNet: "0 for tokens that are NOT MASKED", rest 1

        self.max_length = 0

    def append(self, qa: ClfQATuple):
        self.tokenized.append(torch.tensor(qa.tokenized, dtype=torch.long))
        if qa.token_type_ids:
            self.token_type_ids.append(torch.tensor(qa.token_type_ids, dtype=torch.long))
        self.attention_mask.append(torch.tensor(qa.attention_mask, dtype=torch.long))
        self.labels.append(qa.gold)
        self.question_ids.append(qa.question_id)
        self.max_length = max(self.max_length, len(qa.tokenized))
        self.cand_fact_ids.append(qa.fact_id)
        if qa.noise_logprob is not None:
            self.noise_logprobs.append(qa.noise_logprob)

    def yield_batch(self, last=False) -> ClfQABatch:
        return ClfQABatch(
            tokenized=pad_2d_tensors(self.tokenized, self.pad_token_id),
            token_type_ids=(pad_2d_tensors(self.token_type_ids, self.pad_tok_type_id)
                            if self.token_type_ids else None),
            attention_mask=pad_2d_tensors(self.attention_mask, self.pad_att_mask_id),
            labels=torch.tensor(self.labels, dtype=torch.long),
            question_ids=tuple(self.question_ids),
            fact_ids=tuple(self.cand_fact_ids),
            noise_logprobs=torch.tensor(self.noise_logprobs, dtype=torch.float) if self.noise_logprobs else None,
        )

    def from_lists(self, tokenized, tt_ids, att_mask_ids, labels, q_ids, f_ids):
        return ClfQABatch(
            tokenized=pad_2d_tensors([torch.tensor(t, dtype=torch.long) for t in tokenized], self.pad_token_id),
            token_type_ids=(
                pad_2d_tensors([torch.tensor(t, dtype=torch.long) for t in tt_ids], self.pad_tok_type_id)
                if self.token_type_ids else None
            ),
            attention_mask=pad_2d_tensors([torch.tensor(t, dtype=torch.long) for t in att_mask_ids],
                                          self.pad_att_mask_id),
            labels=torch.tensor(labels, dtype=torch.long),
            question_ids=tuple(q_ids),
            fact_ids=tuple(f_ids),
        )


class SingleFactDatasetMixin(WorldTreeDataset):
    """
        Returns a QA and a fact, either a gold fact or a negative fact
    """
    def __init__(self,
                 args: SingleFactConfig,
                 path_to_qas: str,
                 tokenizer: PreTrainedTokenizer,
                 inference: bool = False,
                 validation: bool = False,
                 filter_empty: bool = True):
        super().__init__(args, path_to_qas, tokenizer, inference, validation,
                         filter_empty, tokenize=True)
        assert 0 <= args.train_neg_sample_rate < 1
        self.negative_sample_rate: float = args.train_neg_sample_rate

        # todo check with model max length from huggingface
        self.max_qa_length = max([len(seq) for seq in self.qa_feats.tokenized])
        self.max_fact_length = max([len(seq) for seq in self.fact_feats.tokenized])
        self.max_seq_length = args.max_seq_length or (self.max_fact_length + self.max_qa_length)
        logger.info('Dataset: using %s as max sequence length' % self.max_seq_length)

        self.pad_token_id = self.tokenizer.pad_token_id
        self.one_positive_in_every_N = round(1 / (1 - self.negative_sample_rate))
        self.noise_logprob = math.log(1 / self.nb_facts)

        if self.training:
            pair_file = args.train_pair_file
        elif self.validation:
            pair_file = args.dev_pair_file
        else:
            pair_file = args.test_pair_file

        if pair_file and os.path.exists(pair_file):
            # these files only contain idx -> idx mappings, no encoded text, so not model dependent
            logger.info('Loading qa-fact combinations from {}'.format(pair_file))
            self.all_pairs_df = torch.load(pair_file)
        else:
            logger.info('Exploding all qa-fact combinations...')
            self.all_pairs_df = self.get_all_pairs(self.qa_feats, self.fact_feats)
            logger.info('Saving pos and neg qa-fact combinations to {}'.format(pair_file))
            torch.save(self.all_pairs_df, pair_file)

    def prepare_item(self, qa_idx, fact_idx, is_positive=False, qa_row=None) -> ClfQATuple:
        if qa_row is None:
            qa_row = self.qa_feats.iloc[qa_idx]
        qa_id, qa_txt = qa_row.id, qa_row.original

        fact = self.fact_feats.iloc[fact_idx]
        fact_id, fact_txt = fact.id, fact.original

        encoded: Dict[str, List[int]] = self.tokenizer.encode_plus(qa_txt, fact_txt, add_special_tokens=True,
                                                                   max_length=self.args.max_seq_length,
                                                                   truncation_strategy='longest_first')

        return ClfQATuple(tokenized=encoded['input_ids'],
                          token_type_ids=encoded['token_type_ids'] if self.use_segment_ids else None,
                          attention_mask=encoded['attention_mask'],
                          gold=is_positive,
                          question_id=str(qa_id),
                          fact_id=str(fact_id),
                          question_idx=qa_idx,
                          noise_logprob=self.noise_logprob)

    def __getitem__(self, index: int) -> ClfQATuple:
        row = self.all_pairs_df.iloc[index]
        is_positive = row.gold if not self.inference else False
        qa_idx, fact_idx = row.qa_idx, row.fact_idx
        try:
            return self.prepare_item(qa_idx, fact_idx, is_positive)
        except IndexError:
            logger.error('single positional index is out-of-bounds in {} dataset: {}'
                         .format(self.mode, qa_idx))

    def __len__(self) -> int:
        return len(self.all_pairs_df.index)

    def get_all_pairs(self, qa_feats: pd.DataFrame, fact_feats: pd.DataFrame) -> pd.DataFrame:
        temp = qa_feats.copy()
        temp['fact_idx'] = [np.arange(0, len(fact_feats.index)) for _ in range(0, len(temp.index))]
        temp = temp.explode('fact_idx')
        if not self.inference:
            temp['gold'] = temp.apply(func=lambda row: row.fact_idx in row.gold_facts, axis=1)
            temp = temp.drop(['tokenized', 'original', 'gold_facts'], axis=1)
        else:
            temp = temp.drop(['tokenized', 'original'], axis=1)
        temp['qa_idx'] = pd.Series(range(len(temp.index)))
        temp = temp.reset_index(drop=True)
        return temp.copy()

    def create_dataloader(self, **kwargs):
        pad_token_id = self.pad_token_id

        def collate(data: List[ClfQATuple]) -> ClfQABatch:
            """
            data: list of ClfQATuple
            """
            seq, token_type_ids, attention_masks, labels, question_ids, fact_ids, q_idxs, logprobs = zip(*data)
            return PartialSingleBatch(pad_token_id).from_lists(
                # todo other possibility = pad to max length, then cut off with
                #   batch.tokenized[:, batch.tokenized.ne(pad_token_id).sum(dim=0) != 0]
                tokenized=seq,
                tt_ids=token_type_ids,
                att_mask_ids=attention_masks,
                labels=labels,
                q_ids=question_ids, f_ids=fact_ids,
            )

        # todo Pin memory? If dataset does not fit in memory and is loaded on CPU,
        # in our case, fits in memory: https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/2
        dl = DataLoader(self, collate_fn=collate, pin_memory=True, **kwargs)
        return dl


class SingleFactDataset(SingleFactDatasetMixin, Dataset):
    # This is here so all functionality is in an abstract class SingleFactDatasetMixin, which can be subclassed
    # without automatically inheriting the PyTorch Dataset class as well, so a subclass can still implement
    # an IterableDataset instead, like ContrastiveSingleFactDataset
    pass


class ContrastiveSingleFactDataset(SingleFactDatasetMixin, IterableDataset):

    def __init__(self, config: SingleFactConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.positives = self.all_pairs_df[self.all_pairs_df.gold]
        self.negatives = self.all_pairs_df[~self.all_pairs_df.gold]
        self.qa_idx_2_neg_idxs = {
            qa_idx: self.negatives[self.negatives.qa_idx == qa_idx].index.tolist()
            for qa_idx in self.qa_feats.index.tolist()
        }

    def __len__(self):
        # nb of batches = nb of positives
        return len(self.positives)

    def __iter__(self):
        pos_indices = random.sample(self.positives.index.tolist(), len(self.positives.index))
        # neg_indices = self.negatives.index.tolist()
        # only split positives over workers
        split_pos_indices = self.split_among_workers(pos_indices)

        batch = PartialSingleBatch(self.pad_token_id)
        for p_idx in split_pos_indices:
            p_qa: ClfQATuple = self[p_idx]
            batch.append(p_qa)

            while batch.max_length * len(batch.question_ids) < self.args.train_tokens_per_batch:
                n_qa: ClfQATuple = self[random.choice(self.qa_idx_2_neg_idxs[p_qa.question_idx])]
                batch.append(n_qa)

            yield batch.yield_batch()
            batch = PartialSingleBatch(self.pad_token_id)

    def create_dataloader(self, **kwargs):
        # todo Pin memory? If dataset does not fit in memory and is loaded on CPU,
        # in our case, fits in memory: https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/2
        dl = DataLoader(self, batch_size=None, pin_memory=True, **kwargs)
        return dl


def load_and_cache_dataset(args: SingleFactConfig, tokenizer, qa_path, valid=False, infer=False):
    # todo only load cached if all DB settings are identical
    mode = 'dev' if valid else ('test' if infer else 'train')
    cached_dataset_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            args.task,
            mode,
            args.model_type,
            'single-fact',
            args.max_seq_length,
            args.answer_choices,
            args.mark_correct_in_qa,
            args.no_lower_case,
            args.downsample_negatives,
            args.upsample_positives,
        ),
    )
    if args.v2:
        cached_dataset_file += '_v2'
    if mode == 'train' and args.loss in CONTRASTIVE_LOSSES:
        cached_dataset_file += '_contrastive'
    cached_dataset_file += '_debug' if args.debug else ''
    logger.info('Looking for DS file %s' % cached_dataset_file)

    if os.path.exists(cached_dataset_file) and not args.overwrite_cache:
        logger.info('Loading features from cached file %s', cached_dataset_file)
        dataset = torch.load(cached_dataset_file)
    else:
        logger.info('Creating features from dataset file at %s', args.data_dir)
        if mode == 'train':
            assert args.loss in CONTRASTIVE_LOSSES
            dataset_class = ContrastiveSingleFactDataset
            dataset = dataset_class(args, path_to_qas=qa_path,
                                    tokenizer=tokenizer, validation=valid, inference=infer)
        else:
            dataset = SingleFactDataset(args, path_to_qas=qa_path,
                                        tokenizer=tokenizer, validation=valid, inference=infer)
        logger.info('Saving features into cached file %s', cached_dataset_file)
        torch.save(dataset, cached_dataset_file)

    idx = random.randint(0, len(dataset))
    logger.info('Example QA+fact: %s - Gold: %s'
                % (tokenizer.decode(dataset[idx].tokenized), dataset[idx].gold))
    dataset.args = args
    return dataset
