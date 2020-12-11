import json
import logging
import os

from tqdm import tqdm
from transformers import PreTrainedTokenizer

from rankers import utils as rankutils
from datasets.chains.chain_batch import Batch
from datasets.factory import DatasetFactory
from datasets.utils import NOISE_CONTRASTIVE_LOSSES
from rankers.bert_ranker import BertRanker, MODEL_CLASSES
from tools import utils
from tools.config import ChainConfig

logger = logging.getLogger()


class ChainRanker(BertRanker):
    def __init__(self, args: ChainConfig, tokenizer: PreTrainedTokenizer):
        super().__init__(args, tokenizer)
        self.anneal_cond_on_negs_scheduler = self.init_cond_neg_curriculum(args)

    def train_epoch(self, train_dataloader, train_dataset, val_dataset,):
        ctrs = self.counters
        logger.info("  Starting epoch nb %s" % ctrs.epoch)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        self.curriculum_step(self.counters.epoch)

        for step, batch in enumerate(epoch_iterator):
            self.counters.step = step
            try:
                assert isinstance(batch, Batch), type(batch)

                # Skip past any already trained steps if resuming training
                if ctrs.steps_trained_in_current_epoch > 0:
                    ctrs.steps_trained_in_current_epoch -= 1
                    continue

                self.model.train()
                batch.to_device(self.args.device)
                inputs = batch.to_dict(without_labels=not self.loss_handler.huggingface_loss)
                outputs = self.model(**inputs)

                kwargs = {'outputs': outputs, 'labels': batch.labels}
                if self.args.loss in NOISE_CONTRASTIVE_LOSSES:
                    kwargs['noise_logprobs'] = batch.noise_logprobs
                loss = self.loss_handler(**kwargs)
                self.losses.running_loss += loss.item()
                self.optim_step(loss)

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.log_stats(val_dataset)

            except Exception as e:
                # output_dir = os.path.join(self.args.output_dir, "checkpoint_F-{}_{}".format(ctrs.epoch, step))
                # rankers.utils.save_checkpoint(self.args, self.model, self.tokenizer, self.gear, output_dir=output_dir)
                raise e

            if ctrs.global_step > self.args.max_steps > 0:
                epoch_iterator.close()
                break

    @staticmethod
    def init_cond_neg_curriculum(args):
        condition_neg_schedule = None
        if args.anneal_cond_on_negs_schedule == 'linear':
            condition_neg_schedule = rankutils.linear_incr_schedule(args.num_train_epochs)
        elif args.anneal_cond_on_negs_schedule == 'steps':
            condition_neg_schedule = rankutils.stepwise_incr_schedule(args.num_train_epochs, args.anneal_cn_steps)
        return condition_neg_schedule

    def curriculum_step(self, epoch_step):
        if self.args.anneal_cond_on_negs_schedule is not None:
            prev = self.args.condition_on_negs_rate
            self.args.condition_on_negs_rate = (
                self.args.max_condition_on_negs_rate * self.anneal_cond_on_negs_scheduler(epoch_step)
                + self.args.init_condition_on_negs_rate
            )
            if self.args.condition_on_negs_rate != prev:
                logger.info("  Using condition on negs rate: %s" % self.args.condition_on_negs_rate)


def set_up_experiment(config_path):
    config = ChainConfig(path=config_path)

    utils.set_seed(config)

    new_output_dir = utils.get_output_dir(config)
    # new_output_dir = args.output_dir
    config.set_output_dir(new_output_dir)

    if (
            os.path.exists(config.output_dir)
            and os.listdir(config.output_dir)
            and config.do_train
            and not config.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome."
                .format(config.output_dir)
        )

    config.model_type = config.model_type.lower()
    _, _, tokenizer_class = MODEL_CLASSES[config.model_type]
    tokenizer_kwconfig = {
        'cache_dir': config.cache_dir if config.cache_dir else None,
        'do_lower_case': not config.no_lower_case
    }
    tokenizer: PreTrainedTokenizer = tokenizer_class.from_pretrained(
        config.tokenizer_name if config.tokenizer_name else config.model_name_or_path,
        **tokenizer_kwconfig
    )

    ranker = ChainRanker(config, tokenizer)
    config.set_pad_token(tokenizer.pad_token_id)

    return ranker, config, tokenizer


if __name__ == '__main__':
    _args = utils.parse_args()
    ranker, config, tokenizer = set_up_experiment(_args.config_path)
    utils.initialize_logging(config=config)

    logger.info("Training/evaluation parameters %s", json.dumps(config.to_dict(), indent=4))
    if config.do_train:
        train_dataset = DatasetFactory.load_and_cache_dataset(config, tokenizer, config.train_qa_path)
        if config.evaluate_during_training:
            val_dataset = DatasetFactory.load_and_cache_dataset(config, tokenizer, config.val_qa_path,
                                                                valid=True)
        else:
            val_dataset = None
        ranker.train(train_dataset, val_dataset)
        # try:
        # except Exception as e:
        #     model_to_save = ranker.podium.get_best_model() or ranker.model
        #     output_dir = os.path.join(args.output_dir, "checkpoint_F")
        #     save_checkpoint(config, model_to_save, tokenizer)
        #     raise e
        # model_to_save = ranker.podium.get_best_model() or ranker.model
        # output_dir = os.path.join(config.output_dir, "checkpoint_best")
        # rankutils.save_checkpoint(config, model_to_save, tokenizer, output_dir=output_dir)

    if config.do_eval:
        val_dataset = DatasetFactory.load_and_cache_dataset(config, tokenizer, config.val_qa_path,
                                                            valid=True)
        results, _ = ranker.evaluate(val_dataset)
        logger.info('Eval results: %s' % results)

    if config.do_test:
        test_dataset = DatasetFactory.load_and_cache_dataset(config, tokenizer, config.test_qa_path,
                                                             infer=True)
        ranker.evaluate(test_dataset)
