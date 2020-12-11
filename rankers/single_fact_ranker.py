import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import SequentialSampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer

import rankers.utils
from datasets.single_fact.single_fact_dataset import load_and_cache_dataset, SingleFactDataset
from datasets.utils import CONTRASTIVE_LOSSES
from rankers.bert_ranker import BertRanker, MODEL_CLASSES
from tools import utils
from tools.config import SingleFactConfig
from rankers.utils import compute_metrics_predictions

logger = logging.getLogger(__name__)


class SingleFactRanker(BertRanker):

    def train_epoch(self, train_dataloader, train_dataset: SingleFactDataset, val_dataset):
        ctrs = self.counters
        logger.info("  Starting epoch nb %s" % ctrs.epoch)

        kwargs = {'num_workers': self.args.num_train_workers}
        assert self.args.loss in CONTRASTIVE_LOSSES
        train_dataloader = train_dataset.create_dataloader(**kwargs)

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        ctrs, args = self.counters, self.args

        for step, batch in enumerate(epoch_iterator):
            ctrs.step = step

            # Skip past any already trained steps if resuming training
            if ctrs.steps_trained_in_current_epoch > 0:
                ctrs.steps_trained_in_current_epoch -= 1
                continue

            self.model.train()
            batch.to_device(args.device)
            inputs = batch.to_dict(without_labels=not self.loss_handler.huggingface_loss, labels_as=args.labels_as)

            outputs = self.model(**inputs)

            loss = self.loss_handler(outputs=outputs, labels=batch.labels)
            self.losses.running_loss += loss.item()
            self.optim_step(loss)

            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.log_stats(val_dataset)

            if self.counters.global_step > args.max_steps > 0:
                epoch_iterator.close()
                break

    def evaluate_per_length(self, eval_dataset, model, output_dir=None):
        args = self.args
        model = self.model

        if not args.no_cuda and args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # PT 2
        # has to be sequential!
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = eval_dataset.create_dataloader(sampler=eval_sampler,
                                                         batch_size=args.eval_batch_size,
                                                         num_workers=args.num_eval_workers)

        # eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=False):
            model.eval()
            batch.to_device(args.device)

            with torch.no_grad():
                inputs = batch.to_dict(without_labels=True)
                outputs = model(**inputs)

                logits = outputs[0]
                if logits.shape[1] == 2:
                    logits = torch.sub(logits[:, 1], logits[:, 0])  # don't unsqueeze ... .unsqueeze(-1)

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()  # shape batch_size x num_classes (2)
                if not args.do_test and hasattr(eval_dataset, 'gold'):
                    out_label_ids = batch.labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)  # stack along batch dimension
                if not args.do_test and hasattr(eval_dataset, 'gold'):
                    out_label_ids = np.append(out_label_ids, batch.labels.detach().cpu().numpy(), axis=0)

        output_dir = output_dir if output_dir is not None else args.output_dir
        step = self.counters.global_step if self.counters else 0
        np.save(os.path.join(output_dir, "eval_preds{}.npy".format('_{}'.format(step))),
                preds)

        mean_ap, predictions = compute_metrics_predictions(eval_dataset, preds, out_label_ids,
                                                           compute_map=not args.do_test and hasattr(eval_dataset,
                                                                                                    'gold'))
        return {
            'predictions': predictions,
            'metrics': {
                # 'eval_loss': eval_loss / (nb_eval_steps * args.eval_batch_size),
                'map': mean_ap,
            }
        }


def set_up_experiment(config_path):
    config = SingleFactConfig(path=config_path)

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

    ranker = SingleFactRanker(config, tokenizer)
    config.set_pad_token(tokenizer.pad_token_id)

    return ranker, config, tokenizer


if __name__ == '__main__':

    _args = utils.parse_args()
    ranker, config, tokenizer = set_up_experiment(_args.config_path)
    utils.initialize_logging(config=config)

    logger.info("Training/evaluation parameters %s", json.dumps(config.to_dict(), indent=4))
    if config.do_train:
        train_dataset = load_and_cache_dataset(config, tokenizer, config.train_qa_path)
        if config.evaluate_during_training:
            val_dataset = load_and_cache_dataset(config, tokenizer, config.val_qa_path,
                                                 valid=True)
        else:
            val_dataset = None
        ranker.train(train_dataset, val_dataset)
        model_to_save = ranker.podium.get_best_model() or ranker.model
        output_dir = os.path.join(config.output_dir, "checkpoint_best")
        rankers.utils.save_checkpoint(config, model_to_save, tokenizer, output_dir=output_dir)

    elif config.do_eval:
        val_dataset = load_and_cache_dataset(config, tokenizer, config.val_qa_path,
                                             valid=True)
        results, _ = ranker.evaluate(val_dataset)
        logger.info('Eval results: %s' % results)

    elif config.do_test:
        test_dataset = load_and_cache_dataset(config, tokenizer, config.test_qa_path,
                                              infer=True)
        ranker.evaluate(test_dataset)
