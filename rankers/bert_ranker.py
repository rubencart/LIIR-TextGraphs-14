
import json
import logging
import os
import time
from collections import namedtuple
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import trange, tqdm
from transformers import (
    DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig, AdamW,
    get_linear_schedule_with_warmup, BertConfig, RobertaConfig, AlbertConfig,
    BertForSequenceClassification, BertTokenizer, RobertaForSequenceClassification, RobertaTokenizer, AlbertTokenizer,
    AlbertForSequenceClassification, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer,
    AutoModel)

import rankers.utils as rutils
from datasets.chains.chain_dataset import ChainDataset
from datasets.chains.eval_chain_dataset import EvalChainDataset
from datasets.eval.eval_dataset import EvalDatasetMixin
from rankers.fact_ranker import FactRanker
from rankers.loss.loss_relay import LossRelay
from tg2020task.evaluate import mean_average_precision_score

from tools import utils
from tools.config import ChainConfig
from tools.podium import Podium
from tools.statistics import Statistics

logger = logging.getLogger(__name__)


ALL_MODELS = sum(
        (tuple(conf.pretrained_config_archive_map.keys())
         for conf in (BertConfig,
                      RobertaConfig,
                      DistilBertConfig,
                      AlbertConfig)
         ),
        (),
    )


MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "auto": (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer),
}

Gear = namedtuple('Gear', field_names=('optimizer', 'scheduler', 'amp'))


class Counters:
    def __init__(self, global_step, epoch, step, epochs_trained, steps_trained_in_current_epoch):
        self.global_step = global_step
        self.epoch = epoch
        self.step = step
        self.epochs_trained = epochs_trained
        self.steps_trained_in_current_epoch = steps_trained_in_current_epoch


class Losses:
    def __init__(self, running_loss, logging_loss):
        self.running_loss = running_loss
        self.logging_loss = logging_loss


class BertRanker(FactRanker):

    def __init__(self, args: ChainConfig, tokenizer: PreTrainedTokenizer):
        """
        Simple bert based ranker. Classification layer with one class on top of DistilBert.
            Input is concatenation of QA and facts. Trained to assign high probability to QA + positive facts,
            low probability to  QA + sampled negative facts.
        """
        config_class, model_class, _ = MODEL_CLASSES[args.model_type]

        self.train_per_question = False
        self.loss_handler = LossRelay(args)
        # self.num_labels = sel

        self.model_config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=self.loss_handler.num_labels,
            cache_dir=args.cache_dir if args.cache_dir else None,
            # max_position_embeddings=args.max_position_embeddings
        )
        if args.from_scratch:
            self.model = model_class(config=self.model_config)
        else:
            self.model = model_class.from_pretrained(
                args.model_name_or_path,
                config=self.model_config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
        self.model.to(args.device)
        self.args = args
        self.podium = Podium(output_dir=args.output_dir)
        self.stats = Statistics(args)
        self.tokenizer = tokenizer
        self.gear = None
        self.counters = self.initialize_training_counters()
        self.losses = None

    def train(self, train_dataset: ChainDataset, val_dataset: EvalDatasetMixin):
        args = self.args
        logger.info('Creating train dataloader from dataset type: %s' % type(train_dataset))
        train_dataloader = train_dataset.create_dataloader(num_workers=args.num_train_workers)

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # put into train mode (default is eval mode)
        self.model.train()
        # Prepare optimizer and schedule (linear warmup and decay)
        self.model, self.gear = self.initialize_training_gear(args, self.model)
        model = self.model

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * args.gradient_accumulation_steps
        )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Number of trainable parameters = %d", sum(p.numel() for p in model.parameters()
                                                                 if p.requires_grad))
        logger.info("  Device, nb = %s, %s", str(args.device), args.n_gpu)
        logger.info("  LR = %s", self.gear.scheduler.get_last_lr()[0])

        self.counters = self.initialize_training_counters(train_dataloader)

        self.losses = Losses(running_loss=0.0, logging_loss=0.0)
        model.zero_grad()
        train_iterator = trange(
            self.counters.epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=False,
        )
        # train_iterator = range(epochs_trained, int(args.num_train_epochs))
        utils.set_seed(args)  # Added here for reproducibility
        self.stats.start_step()

        for epoch_step in train_iterator:
            self.counters.epoch, self.counters.step = epoch_step, 0

            self.train_epoch(train_dataloader, train_dataset, val_dataset)
            self.counters.steps_trained_in_current_epoch = 0    # todo only skip steps in first epoch?

            if args.lr_decay_per_epoch and args.lr_decay:
                self.gear.scheduler.step()
                logger.info("  LR = %s", self.gear.scheduler.get_last_lr()[0])

            if 0 < args.max_steps < self.counters.global_step:
                train_iterator.close()
                break

        self.log_stats(val_dataset, force=True)

    def train_epoch(self, train_dataloader, train_dataset, val_dataset):
        raise NotImplementedError

    def optim_step(self, loss, backward_arg=None):
        optimizer, scheduler, amp = self.gear
        bw_arg = torch.tensor(1.0, dtype=torch.double, device=loss.device) if backward_arg is None else backward_arg
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(bw_arg)
        else:
            loss.backward(bw_arg)

        if (self.counters.step + 1) % self.args.gradient_accumulation_steps == 0:
            if self.args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

            optimizer.step()
            if not self.args.lr_decay_per_epoch and self.args.lr_decay:
                scheduler.step()  # Update learning rate schedule
            self.model.zero_grad()  # todo not when single loss spans multiple batches!!! (not the case now)
            self.counters.global_step += 1

    def initialize_training_counters(self, train_dataloader=None):
        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if train_dataloader is not None and os.path.exists(self.args.model_name_or_path) and self.args.init_counters_from_checkpoint:
            # set global_step to gobal_step of last saved checkpoint from model path
            if '/checkpoint' and '-' in self.args.model_name_or_path:
                # global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
                both_steps = self.args.model_name_or_path.split("-")[-1].split("/")[0]
                if '_' in both_steps:
                    epochs_trained, steps_trained_in_current_epoch = int(both_steps.split('_')[0]), \
                                                                     int(both_steps.split('_')[1])
                    global_step = epochs_trained * len(train_dataloader) + steps_trained_in_current_epoch
                else:
                    # todo this is not correct anymore because we don't know exact training DS length :'(
                    global_step = int(self.args.model_name_or_path.split("-")[-1].split("/")[0])
                    epochs_trained = global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                    steps_trained_in_current_epoch = global_step % (
                            len(train_dataloader) // self.args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        return Counters(global_step, 0, 0, epochs_trained, steps_trained_in_current_epoch)

    def initialize_training_gear(self, args, model):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
                          weight_decay=args.weight_decay, betas=(args.adam_beta_1, args.adam_beta_2))
        num_training_steps = (args.num_train_epochs if args.lr_decay_per_epoch else
                              args.approx_num_steps * args.num_train_epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
        )
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if not args.no_cuda and args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Check if saved optimizer or scheduler states exist
        if args.init_gear_from_checkpoint:
            if (os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
                and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))):
                # Load in optimizer and scheduler states
                optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"),
                                                     map_location=args.device))
                scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"),
                                                     map_location=args.device))

            if args.fp16 and os.path.isfile(os.path.join(args.model_name_or_path, "amp.pt")):
                amp.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "amp.pt"),
                                               map_location=args.device))

        if os.path.isfile(os.path.join(args.model_name_or_path, "stats.json")):
            self.stats.load_stats(os.path.join(args.model_name_or_path, "stats.json"))

        return model, Gear(optimizer, scheduler, (amp if args.fp16 else None))

    def log_stats(self, val_dataset, other_stats=None, force=False):
        ctrs = self.counters
        output_dir = os.path.join(self.args.output_dir, "checkpoint-{}_{}".format(ctrs.epoch, ctrs.step + 1))
        evaluated, eval_results = False, {}

        if self.args.save_steps > 0 and (force or ctrs.global_step % self.args.save_steps == 0):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            rutils.save_checkpoint(self.args, self.model, self.tokenizer, self.gear, output_dir=output_dir)

        if self.args.eval_steps > 0 and (force or ctrs.global_step % self.args.eval_steps == 0):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            self.stats.toggle_to_eval()
            eval_results, _ = self.evaluate(val_dataset, epoch_step=ctrs.epoch, step=ctrs.step + 1) \
                if self.args.evaluate_during_training else ({}, None)
            evaluated = True
            self.stats.toggle_to_train()
            time.sleep(10)

        if self.args.logging_steps > 0 and (force or ctrs.global_step % self.args.logging_steps == 0):
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)
            logs = {
                "step": ctrs.global_step,
                "lr": self.gear.scheduler.get_last_lr()[0],
                "loss": (self.losses.running_loss - self.losses.logging_loss) / self.args.logging_steps,
            }
            if evaluated:
                logs.update({"eval_results": eval_results})
            if other_stats:
                logs.update({"other": other_stats})

            self.losses.logging_loss = self.losses.running_loss

            logging.info(logs)
            self.stats.record(**logs)
            self.stats.log_stats()
            if evaluated:
                self.stats.write_stats(os.path.join(self.args.output_dir, 'stats.json'))
                self.stats.write_stats(os.path.join(output_dir, 'stats.json'))

    def evaluate(self, vds: EvalChainDataset, prefix='', epoch_step=0, step=0) -> Tuple[Dict, Dict]:
        args = self.args
        model = self.model
        mean_ap = None

        if not args.no_cuda and args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        logger.info("***** Running evaluation {} *****".format(''))
        logger.info("  Num examples = %d", len(vds.get_all_qa_ids()))
        logger.info("  Batch size = %d", args.eval_batch_size)

        output_dir = os.path.join(args.output_dir, "checkpoint-{}_{}".format(epoch_step, step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        predictions: Dict = self.evaluate_per_length(vds, model, output_dir)
        if 'metrics' in predictions:
            mean_ap = predictions['metrics']['map']
            predictions = predictions['predictions']

        preds_filename = "eval_predictions-{}_{}.txt".format(epoch_step, step) if step > 0 else "eval_predictions.txt"
        preds_eval_file = os.path.join(output_dir, prefix + preds_filename)
        with open(preds_eval_file, "w") as f:
            for qa_id, preds in predictions.items():
                for fact_id in preds:
                    f.write("%s\t%s\n" % (qa_id, fact_id))
        logger.info('Saved predictions to file %s' % preds_eval_file)

        json_filename = 'eval_predictions-{}_{}.json'.format(epoch_step, step) if step > 0 else 'eval_predictions.json'
        preds_eval_json = os.path.join(output_dir, json_filename)
        with open(preds_eval_json, 'w') as f:
            json.dump(predictions, f)

        if not args.do_test and hasattr(vds, 'gold'):
            # if mean_ap is None:
            result = rutils.analyze(args, mean_ap, predictions, vds)
            # else:
            #     result = {'map': mean_ap}
            result_filename = 'eval_results-{}_{}.json'.format(epoch_step, step) if step > 0 else 'eval_results.json'
            res_eval_file = os.path.join(output_dir, prefix + result_filename)
            with open(res_eval_file, "w") as f:
                json.dump(result, f, indent=4)

            logger.info('Eval results: %s' % result)

            return result, predictions
        return {}, predictions

    def evaluate_per_length(self, eval_dataset, model, output_dir=None):
        args = self.args
        nb_eval_steps = 0

        unfinished_expls = True
        eval_dataset.reset_partials()
        eval_dataloader = eval_dataset.create_dataloader(num_workers=args.num_eval_workers)

        logger.info('Using stop id: %s, predict stop: %s, delta: %s, beam size: %s'
                    % (eval_dataset.stop_explanation_id, eval_dataset.config.predict_stop_expl,
                       eval_dataset.config.stop_delta, eval_dataset.config.beam_size))

        eval_iterator = trange(args.max_expl_length, desc="Length", disable=False)
        for _ in eval_iterator:
            if not unfinished_expls:
                break

            eval_iterator = tqdm(eval_dataloader, desc="Evaluating...", disable=False)
            prev_question = -1
            all_scores = torch.empty(eval_dataset.nb_facts).fill_(np.nan).to(args.device)

            for step, batch in enumerate(eval_iterator):
                # assume single question per batch for now?
                # assert isinstance(batch, ChainBatch), type(batch)
                model.eval()

                q_idx = batch.partial_idxs[0]
                prev_question = prev_question if prev_question > -1 else q_idx
                if prev_question != batch.partial_idxs[0]:
                    # all_scores = all_scores.cpu().numpy()
                    eval_dataset.store_scores_batch(prev_question, all_scores)
                    all_scores = torch.empty(eval_dataset.nb_facts).fill_(np.nan).to(args.device)
                    prev_question = q_idx

                if batch.tokenized.shape[1] > self.model_config.max_position_embeddings - 20:
                    logger.warning('Batch was too long, had to skip.')
                    continue

                batch.to_device(args.device)
                fact_idxs = torch.LongTensor(batch.cand_fact_idxs).to(args.device)
                with torch.no_grad():
                    # inputs = batch.to_dict(without_labels=not self.loss_handler.huggingface_loss)
                    inputs = batch.to_dict(without_labels=True)
                    outputs = model(**inputs)
                    logits = outputs[0]  # was `self.loss_handler.logits_index` but should be 0 bc without_labels=True
                    # logits = outputs[self.loss_handler.logits_index]
                    # loss = self.loss_handler(outputs, batch)
                    if logits.shape[1] == 2:
                        logits = torch.sub(logits[:, 1], logits[:, 0]).unsqueeze(-1)

                all_scores = all_scores.scatter(dim=0, index=fact_idxs, src=logits.squeeze(-1))
                nb_eval_steps += 1

            # all_scores = all_scores.cpu().numpy()
            eval_dataset.store_scores_batch(prev_question, all_scores)

            eval_dataset.process_scores()
            unfinished_expls = not eval_dataset.is_finished()
            # max_current_length = max(eval_dataset.partials.partial_expl.apply(len))
        # if output_dir is not None:
        #     eval_dataset.save_predictions_dataframe(output_dir)
        predictions = eval_dataset.rank_all()
        return predictions
