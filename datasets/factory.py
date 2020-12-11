import os
import random

import torch

from datasets.chains.chain_dataset import ChainDataset
from datasets.chains.eval_chain_dataset import GreedyBeamEvalChainDataset, SampleBeamEvalChainDataset, EvalChainDataset
from datasets.chains.train_chain_dataset import PointwiseTrainChainDataset, ContrastiveTrainChainDataset
from datasets.eval.beam_eval_dataset import BeamEvalDatasetMixin
from datasets.eval.eval_dataset import EvalDatasetMixin
from datasets.utils import POINTWISE_LOSSES, logger, CONTRASTIVE_LOSSES
from tools.config import ChainConfig


class DatasetFactory:
    @staticmethod
    def load_and_cache_dataset(args: ChainConfig, tokenizer, qa_path, valid=False, infer=False):
        mode = 'dev' if valid else ('test' if infer else 'train')
        train = not (valid or infer)
        cached_dataset_file = os.path.join(
            args.data_dir,
            'cached_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.task,
                mode,
                args.model_type,
                args.algo,
                args.answer_choices,
                args.mark_correct_in_qa,
                args.no_lower_case,
                args.powerset_complete_combs,
                args.mark_answer_in_qa,
                args.distance_mode,
            )
        )
        cached_dataset_file += (
            '_{}'.format('pointwise' if args.loss in POINTWISE_LOSSES else 'batchwise')
            if train and args.algo == 'chains' else ''
        )
        if args.beam_size > 1 and not train:
            cached_dataset_file += ('_beam_sample' if args.beam_fact_selection == 'sample' else '_beam_greedy')
        cached_dataset_file += ('_debug' if args.debug else '')
        cached_dataset_file += ('_v2' if args.v2 else '')
        logger.info('Looking for DS file %s' % cached_dataset_file)

        if os.path.exists(cached_dataset_file) and not args.overwrite_cache:
            logger.info('Loading features from cached file %s', cached_dataset_file)
            dataset: ChainDataset = torch.load(cached_dataset_file)

            dataset.set_settings(args, train=train, valid=valid, infer=infer)

            # logger.info('Saving features into cached file %s', cached_dataset_file)
            # torch.save(dataset, cached_dataset_file)
        else:
            logger.info('Creating features from dataset file at %s', args.data_dir)
            dataset = DatasetFactory.create_dataset(args, tokenizer, qa_path, valid, infer)

            logger.info('Saving features into cached file %s', cached_dataset_file)
            torch.save(dataset, cached_dataset_file)

        logger.info(type(dataset))
        try:
            idx = random.randint(0, 10)
            if not (isinstance(dataset, EvalDatasetMixin) or isinstance(dataset, BeamEvalDatasetMixin)):
                sample = dataset[idx]
            else:
                sample = dataset.get_qa_with_facts(idx, random.sample(range(100), random.randrange(1, 4)))

            logger.info('Example QA+facts: %s ' % str(sample))
            dl = dataset.create_dataloader()
            batch = next(iter(dl))
            logger.info('Example batch:\n %s' % str(batch))
            logger.info('Tokenized shape:\n %s' % str(batch.tokenized.shape))
            logger.info('Decode tokenized:\n %s' % tokenizer.decode(batch.tokenized[1]))
        except (TypeError, IndexError, AttributeError) as e:
            logger.error('Got error while taking dataset sample: %s' % e)
            pass
        return dataset

    @staticmethod
    def create_dataset(args, tokenizer, qa_path, valid, infer):
        mode = 'dev' if valid else ('test' if infer else 'train')
        assert args.algo == 'chains'
        if mode == 'train':
            if args.loss in POINTWISE_LOSSES:
                dataset = PointwiseTrainChainDataset(args, path_to_qas=qa_path, tokenizer=tokenizer)
            else:
                assert args.loss in CONTRASTIVE_LOSSES
                dataset = ContrastiveTrainChainDataset(args, path_to_qas=qa_path, tokenizer=tokenizer)
        else:
            if args.beam_size > 1:
                if args.beam_fact_selection == 'greedy':
                    dataset = GreedyBeamEvalChainDataset(args, path_to_qas=qa_path,
                                                         tokenizer=tokenizer, validation=valid, inference=infer)
                else:
                    assert args.beam_fact_selection == 'sample'
                    dataset = SampleBeamEvalChainDataset(args, path_to_qas=qa_path,
                                                         tokenizer=tokenizer, validation=valid, inference=infer)
            else:
                dataset = EvalChainDataset(args, path_to_qas=qa_path,
                                           tokenizer=tokenizer, validation=valid, inference=infer)
        return dataset
