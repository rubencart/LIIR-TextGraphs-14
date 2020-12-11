import logging
import math
import os
import re
from abc import ABC
from collections import OrderedDict
from typing import Tuple, List, NamedTuple

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from transformers import PreTrainedTokenizer

from tg2020task.baseline_tfidf import read_explanations

from datasets.utils import read_explanations_wo_fill, worker_init_fn
from tools.config import Config
from rankers.utils import pad_2d_tensors

logger = logging.getLogger()


class QATuple(NamedTuple):
    question_id: str
    question_tok: Tensor
    answer_key: str
    gold_fact_ids: List[str] = []
    gold_facts_tok: List[Tensor] = []


class QABatch(NamedTuple):
    question_ids: Tuple[str]
    questions_tok: Tensor
    answer_keys: Tuple[str]
    gold_fact_ids: Tuple[List[str]] = []
    gold_facts_tok: Tuple[List[Tensor]] = []


class WorldTreeDataset(ABC):

    def __init__(self,
                 args: Config,
                 path_to_qas: str,
                 tokenizer: PreTrainedTokenizer = None,
                 inference: bool = False,
                 validation: bool = False,
                 filter_empty: bool = True,
                 tokenize: bool = False,
                 ):
        """
        Args should at least contain:
            - args.answer_choices
            - args.mark_correct_in_qa
        """
        super().__init__()
        self.args: Config = args
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.path_to_facts: str = args.fact_path
        self.inference: bool = inference
        self.supply_gold: bool = not self.inference
        self.validation = validation
        self.training = not self.inference and not self.validation
        self.mode = 'train' if self.training else ('dev' if self.validation else 'test')
        self.path_to_qas: str = path_to_qas
        self.filter_empty: bool = filter_empty
        self.tokenize: bool = tokenize
        if self.tokenize:
            self.pad_token_id = self.tokenizer.pad_token_id
        self.batch_size = args.train_batch_size if self.mode == 'train' else args.eval_batch_size

        facts = []
        for path, _, files in os.walk(self.path_to_facts):
            for file in files:
                facts += read_explanations(os.path.join(path, file))

        self.df_qas = pd.read_csv(self.path_to_qas, sep='\t', dtype=str)
        self.df_qas.columns = map(str.lower, self.df_qas.columns)

        q_and_a_columns = ['questionid', 'explanation']
        self.df_qas[q_and_a_columns] = self.df_qas[q_and_a_columns].apply(lambda x: x.str.lower())
        if self.filter_empty:
            if self.supply_gold:
                self.df_qas = self.df_qas[self.df_qas['explanation'].notnull()]
            index = self.df_qas['question'].notnull() & self.df_qas['question'] != ''
            self.df_qas = self.df_qas[index]
            logger.info('Removing %s questions because they did not have question text' % len(self.df_qas[~index]))

        self.df_facts = pd.DataFrame(facts, columns=('uid', 'text'))  # .apply(lambda x: x.str.lower())
        self.df_facts.uid = self.df_facts.uid.apply(lambda x: x.lower())
        self.df_facts.columns = map(str.lower, self.df_facts.columns)
        if self.supply_gold:
            self.gold = self.construct_gold_dict()

        self.nb_qas = len(self.df_qas.index)
        self.nb_facts = len(self.df_facts.index)
        self.all_fact_idxs = set(range(len(self.df_facts.index)))
        self.fact_uid_2_idx = {row.uid.lower(): i for i, row in self.df_facts.iterrows()}
        self.fact_idx_2_uid = [row.uid.lower() for _, row in self.df_facts.iterrows()]
        self.qa_uid_2_idx = {row.questionid.lower(): i for i, row in self.df_qas.iterrows()}
        self.qa_idx_2_uid = [row.questionid.lower() for _, row in self.df_qas.iterrows()]

        self.process_answer_choices(args.answer_choices, args.mark_correct_in_qa, args.mark_answer_in_qa)
        self.use_segment_ids = args.use_segment_ids

        if args.task == '19':
            self.non_existent_facts = ('f3bd-0290-ae95-8e0a', '5124-b03e-ff20-b4d6')
        else:
            self.non_existent_facts = ()

        def encode(x, **kwargs):
            y = self.tokenizer.encode(x, add_special_tokens=False, **kwargs)
            y = np.array(y, dtype=np.int)
            return y

        def append_punct(sentence):
            return sentence.strip().rstrip('.') + '. '

        self.qa_feats = pd.DataFrame(columns=('id', 'tokenized', 'original'))
        self.qa_feats.id = self.df_qas.questionid.copy()
        self.qa_feats.original = self.df_qas.question.copy()
        self.qa_feats.original = self.qa_feats.original.apply(lambda x: append_punct(x))
        if self.tokenize:
            self.qa_feats.tokenized = self.qa_feats.original.apply(lambda x: encode(x))

        if self.supply_gold:
            # training or validation: gold facts available
            self.qa_feats['gold_facts'] = self.df_qas.explanation \
                .apply(lambda x: x.split()) \
                .apply(lambda x: [f.split('|')[0] for f in x]) \
                .apply(lambda x: [self.fact_uid_2_idx[uid] for uid in x
                                  if uid not in self.non_existent_facts]) \
                .apply(lambda x: np.array(x))
        self.qa_feats = self.qa_feats.reset_index(drop=True)

        self.fact_feats = pd.DataFrame(columns=('id', 'tokenized', 'original'))
        self.fact_feats.id = self.df_facts.uid.copy()
        self.fact_feats.original = self.df_facts.text.copy()
        self.fact_feats.original = self.fact_feats.original.apply(lambda x: append_punct(x))
        if self.tokenize:
            self.fact_feats.tokenized = self.fact_feats.original.apply(lambda x: encode(x))
        self.fact_feats = self.fact_feats.reset_index(drop=True)

        if self.args.compute_overlap or self.validation:
            facts = []
            for path, _, files in os.walk(self.path_to_facts):
                for file in files:
                    facts += read_explanations_wo_fill(os.path.join(path, file))

            df_facts = pd.DataFrame(facts, columns=('uid', 'text'))  # .apply(lambda x: x.str.lower())
            df_facts.uid = df_facts.uid.apply(lambda x: x.lower())
            df_facts.columns = map(str.lower, df_facts.columns)

            # we assume order is same as before
            self.fact_feats['original_wo_fill'] = df_facts.text.apply(lambda x: append_punct(x))

    def process_answer_choices(self, answer_choices: str, mark_correct_in_qa: bool, mark_answer_in_qa: bool):
        if answer_choices == 'all':
            if mark_correct_in_qa:
                # insert "(correct)" before each correct answer in all questions
                self.df_qas['question'] = self.df_qas.apply(
                    lambda qa: re.sub(r'(\({key}\))([\s\S]*)'.format(key=qa['answerkey'].upper()),
                                      r'\1 (correct)\2', qa['question']),
                    axis=1
                )
        else:
            self.df_qas['question'] = self.df_qas.apply(
                lambda qa: self.remove_wrong_answer_choices(qa, answer_choices, mark_answer_in_qa),
                axis=1
            )

    def get_all_qa_ids(self):
        return list(self.qa_feats.id)

    def get_all_qa_idxs(self):
        return list(self.qa_feats.index)

    @staticmethod
    def remove_wrong_answer_choices(row, choices, mark_answer=False):
        answer_key_map = {str(i+1): 'ABCDEFGHIJKL'[i] for i in range(10)}
        question = re.sub(r'\(1\)|\(2\)|\(3\)|\(4\)|\(5\)|\(6\)|\(7\)|\(8\)',
                          lambda match: (
                              {'(%s)' % key: '(%s)' % val for key, val in answer_key_map.items()}[match.group(0)]
                          ),
                          row["question"])
        correct_choice = row["answerkey"] if row["answerkey"] in 'ABCDEFGHIJK' else answer_key_map[row["answerkey"]]
        option_start_loc = question.rfind("(A)")
        split0, split1 = question[:option_start_loc], question[option_start_loc:]

        split0 = (split0.rstrip() + ' (answer)') if mark_answer else split0.rstrip()

        if choices == "none":
            return split0

        if correct_choice == "A" and "(B)" in split1:
            answer = (split1[3:split1.rfind("(B)")])
        elif correct_choice == "A":
            answer = (split1[3:])
        elif correct_choice == "B" and "(C)" in split1:
            answer = (split1[split1.rfind("(B)") + 3:split1.rfind("(C)")])
        elif correct_choice == "B":
            answer = (split1[split1.rfind("(B)") + 3:])
        elif correct_choice == "C" and "(D)" in split1:
            answer = (split1[split1.rfind("(C)") + 3:split1.rfind("(D)")])
        elif correct_choice == "C":
            answer = (split1[split1.rfind("(C)") + 3:])
        elif correct_choice == "D" and "(E)" in split1:
            answer = (split1[split1.rfind("D)") + 3:split1.rfind("(E)")])
        elif correct_choice == "D":
            answer = (split1[split1.rfind("D)") + 3:])
        elif correct_choice == "E" and "(F)" in split1:
            answer = (split1[split1.rfind("(E)") + 3:split1.rfind("(F)")])
        elif correct_choice == "E":
            answer = (split1[split1.rfind("(E)") + 3:])
        else:
            raise ValueError("Unhandled option type: {}".format(correct_choice))

        split0 += ' ' + answer.lstrip()
        return split0

    def get_facts(self) -> DataFrame:
        return self.df_facts

    def construct_gold_dict(self):
        """
        From tg2019task code
        """
        gold = OrderedDict()
        if self.filter_empty:
            df = self.df_qas[['questionid', 'explanation']].dropna()
        else:
            df = self.df_qas[['questionid', 'explanation']]
        for _, row in df.iterrows():
            # if row.notnull()['explanation']:
            #     explanations = [
            #         uid.lower()
            #         for e in row['explanation'].split()
            #         for uid, _ in (e.split('|', 1),)
            #     ]
            #     explanations = OrderedDict((uid.lower(), Explanation(uid.lower(), role))
            #                                for e in row['explanation'].split()
            #                                for uid, role in (e.split('|', 1),))
            # else:
            #     explanations = OrderedDict()
            # question = Question(row['questionid'].lower(), explanations)
            gold[row['questionid'].lower()] = [
                uid.lower() for e in row['explanation'].split()
                for uid, _ in (e.split('|', 1),)
            ]
        return gold

    def create_dataloader(self, batch_size: int, shuffle: bool = True, **kwargs):
        raise NotImplementedError

        pad_token = self.tokenizer.pad_token_id

        def collate(data: List[QATuple]) -> QABatch:
            """
            data: list of QATuples (str, tensor, str, list, list)
            """
            question_ids, questions, answer_keys, gold_fact_ids, gold_facts_tok = zip(*data)
            return QABatch(
                question_ids, pad_2d_tensors(questions, pad_token), answer_keys, gold_fact_ids, gold_facts_tok
            )

        # Pin memory? If dataset does not fit in memory and is loaded on CPU,
        # in our case, fits in memory: https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/2
        dl = DataLoader(self, batch_size=batch_size, pin_memory=True,
                        collate_fn=collate, shuffle=shuffle, **kwargs, worker_init_fn=worker_init_fn)
        return dl

    def split_indices_for_workers(self, indices, worker_info):
        num_workers = worker_info.num_workers
        per_worker = int(math.ceil(len(indices) / float(num_workers)))
        worker_id = worker_info.id
        start = per_worker * worker_id
        end = min(per_worker * (worker_id + 1), len(indices))
        # logger.info('worker %s doing indices %s to %s of len %s' % (worker_id, start, end, len(indices)))
        return end, start

    def split_among_workers(self, indices):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return indices
        else:
            end, start = self.split_indices_for_workers(indices, worker_info)
            return indices[start:end]

    def is_batch_full(self, samples_in_batch, tokens_in_batch):
        if self.batch_size:
            return samples_in_batch >= self.batch_size
        else:
            assert bool(self.tokens_per_batch), self.tokens_per_batch
            return tokens_in_batch >= self.tokens_per_batch
