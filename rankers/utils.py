import copy
import json
import os
import warnings
from typing import Callable, List, Optional, Dict
import numpy as np
import pandas as pd
import torch

from tg2020task.evaluate import average_precision_score, mean_average_precision_score
from tools.utils import logger


def linear_incr_schedule(num_steps: int) -> Callable:

    def lr_lambda(current_step):
        return float(current_step) / float(num_steps)

    return lr_lambda


def stepwise_incr_schedule(tot_num_steps: int, num_incr_steps: int, incl_zero=True, incl_max=True) -> Callable:
    per_step = 1.0 / num_incr_steps
    max_term = 1 if incl_max else 0
    steps_per_incr = round(float(tot_num_steps) / (num_incr_steps + max_term))
    zero_term = 0 if incl_zero else 1

    def lr_lambda(current_step):
        return min(1.0, ((current_step // steps_per_incr) + zero_term) * per_step)

    return lr_lambda


def pad_2d_tensors(tensors: List[torch.Tensor], pad_token: int) -> torch.Tensor:
    padded_tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=pad_token)
    return padded_tensors


def compute_AP(sorted_labels):
    num_hits = 0.0
    ap = 0.0

    if np.sum(sorted_labels) == 0:
        return 0.0

    for i, ll in enumerate(sorted_labels):
        if ll == 0:
            continue
        num_hits += 1
        ap += num_hits / (i + 1.)

    return ap / np.sum(sorted_labels)


def compute_metrics_predictions(examples_ds, logits, labels=None, compute_map=False):
    examples = list(examples_ds)  # if isinstance(examples_ds, Dataset) else examples_ds

    idx_start = 0
    prev_query = examples[0].question_id

    ap_list = []
    predictions = {}

    for i, example in enumerate(examples):
        if example.question_id == prev_query:  # continue as long as same question
            continue

        relevant_logits = logits[idx_start:i]
        relevant_examples = examples[idx_start:i]
        sorted_examples, ap = metrics_and_predictions_for_q(compute_map, i, idx_start, labels,
                                                            relevant_examples, relevant_logits)
        predictions.update(sorted_examples)
        ap_list.append(ap)

        prev_query = example.question_id
        idx_start = i

    relevant_logits = logits[idx_start:]
    relevant_examples = examples[idx_start:]
    sorted_examples, ap = metrics_and_predictions_for_q(compute_map, len(logits), idx_start, labels,
                                                        relevant_examples, relevant_logits)
    predictions.update(sorted_examples)
    ap_list.append(ap)

    return np.mean(ap_list), predictions


def metrics_and_predictions_for_q(compute_map, i, idx_start, labels, relevant_examples, relevant_logits):

    if compute_map:
        relevant_labels = labels[idx_start:i]
        sorted_preds, sorted_labels, sorted_examples = zip(*sorted(zip(relevant_logits, relevant_labels,
                                                                       relevant_examples), key=lambda e: e[0],
                                                                   reverse=True))
        ap = compute_AP(sorted_labels)
    else:
        sorted_preds, sorted_examples = zip(
            *sorted(zip(relevant_logits, relevant_examples), key=lambda e: e[0], reverse=True)
        )
        ap = -1

    return {sorted_examples[0].question_id: [se.fact_id for se in sorted_examples]}, ap


def compute_metrics_predictions_2(examples_ds, logits, labels=None, compute_map=False):
    examples = list(examples_ds)  # if isinstance(examples_ds, Dataset) else examples_ds

    idx_start = 0
    prev_query = examples[0].text_a

    ap_list = []
    predictions = {}

    for i, example in enumerate(examples):
        if example.text_a == prev_query:  # continue as long as same question
            continue

        relevant_logits = logits[idx_start:i]
        relevant_examples = examples[idx_start:i]
        sorted_examples, ap = metrics_and_predictions_for_q_2(compute_map, i, idx_start, labels,
                                                            relevant_examples, relevant_logits)
        predictions.update(sorted_examples)
        ap_list.append(ap)

        prev_query = example.text_a
        idx_start = i

    relevant_logits = logits[idx_start:]
    relevant_examples = examples[idx_start:]
    sorted_examples, ap = metrics_and_predictions_for_q_2(compute_map, len(logits), idx_start, labels,
                                                        relevant_examples, relevant_logits)
    predictions.update(sorted_examples)
    ap_list.append(ap)

    return np.mean(ap_list), predictions


def metrics_and_predictions_for_q_2(compute_map, i, idx_start, labels, relevant_examples, relevant_logits):

    if compute_map:
        relevant_labels = labels[idx_start:i]
        sorted_preds, sorted_labels, sorted_examples = zip(*sorted(zip(relevant_logits, relevant_labels,
                                                                       relevant_examples), key=lambda e: e[0],
                                                                   reverse=True))
        ap = compute_AP(sorted_labels)
    else:
        sorted_preds, sorted_examples = zip(
            *sorted(zip(relevant_logits, relevant_examples), key=lambda e: e[0], reverse=True)
        )
        ap = -1

    return {sorted_examples[0].guid.split('###')[0].lower():
                [se.guid.split('###')[1].lower() for se in sorted_examples]}, ap


def save_checkpoint(args,
                    model,
                    tokenizer,
                    gear=None,
                    output_dir=None):
    if not output_dir:
        output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("Saving model checkpoint to %s", output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    with open(os.path.join(args.output_dir, "training_args.json"), 'w') as f:
        json.dump(args.to_dict(), f, indent=4)

    # Load a trained model and vocabulary that you have fine-tuned
    if gear:
        optimizer, scheduler, amp = gear
        if optimizer:
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        if scheduler:
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        if amp:
            torch.save(amp.state_dict(), os.path.join(output_dir, "amp.pt"))
    # logger.info("Saving optimizer and scheduler states to %s", output_dir)


def average_precision_per_qid(golds, preds, df_qas):
    aps = {}
    sum_ap = 0.

    if not golds or not preds:
        return 0.

    for id, gold in golds.items():
        if id in preds:
            pred = preds[id]

            score = average_precision_score(gold, pred)
            aps[id] = score
            sum_ap += score
            df_qas.loc[df_qas.id == id, 'avg_precision'] = score

    return aps, df_qas, sum_ap / len(golds)


def k_hops_map_breakdown(golds, preds, k_hops):
    def is_neg_or_x_hops_away(q_id, f_id, x):
        # k_hops contains only golds
        return (
            # len(k_hops.loc[(k_hops.q_id == q_id) & (k_hops.f_id == f_id)]) == 0 or
            k_hops.loc[(k_hops.q_id == q_id) & (k_hops.f_id == f_id), 'hops'].iloc[0] == x
        )

    def filter_dict(dct, i_hops, golds):
        result = {}
        for q_id, facts in dct.items():
            filt_facts = copy.deepcopy(facts)
            for gf_id in golds.get(q_id, []):
                if not is_neg_or_x_hops_away(q_id, gf_id, i_hops):
                    filt_facts.remove(gf_id)
            # filt_facts = [f_id for f_id in facts if is_neg_or_x_hops_away(q_id, f_id, i_hops)]
            if len(filt_facts) > 0:
                result[q_id] = filt_facts
        return result

    result = {}
    for i_hops in list(range(1, max(k_hops.hops))) + [-1]:
        filtered_golds, filtered_preds = filter_dict(golds, i_hops, golds), filter_dict(preds, i_hops, golds)
        mean_ap = mean_average_precision_score(filtered_golds, filtered_preds)
        result[i_hops] = mean_ap
    return result


def role_map_breakdown(golds, preds, roles):
    def is_neg_or_role(q_id, f_id, role):
        return (
            # len(roles.loc[(roles.q_id == q_id) & (roles.f_id == f_id)]) == 0 or
            roles.loc[(roles.q_id == q_id) & (roles.f_id == f_id), 'role'].iloc[0] == role
        )

    def filter_dict(dct, role, golds):
        result = {}
        for q_id, facts in dct.items():
            filt_facts = copy.deepcopy(facts)
            for gf_id in golds.get(q_id, []):
                if not is_neg_or_role(q_id, gf_id, role):
                    filt_facts.remove(gf_id)
            # filt_facts = [f_id for f_id in facts if is_neg_or_role(q_id, f_id, role)]
            if len(filt_facts) > 0:
                result[q_id] = filt_facts
        return result

    result = {}
    for role in ['central', 'lexglue', 'grounding']:
        filtered_golds, filtered_preds = filter_dict(golds, role, golds), filter_dict(preds, role, golds)
        mean_ap = mean_average_precision_score(filtered_golds, filtered_preds)
        result[role] = mean_ap
    return result


def map_breakdown(golds, preds, k_hops, lex_hops, roles):
    # hops away: for k in range(): only keep facts that are k hops away in golds and preds, then compute map
    # same for roles
    k_hops_results = k_hops_map_breakdown(golds, preds, k_hops)
    lex_hops_results = k_hops_map_breakdown(golds, preds, lex_hops)
    role_breakdown = role_map_breakdown(golds, preds, roles)
    return {'k_hops': k_hops_results,
            'lex_hops': lex_hops_results,
            'roles': role_breakdown}


def analyze(args, mean_ap, predictions, vds):
    prec_per_qid, qa_feats, map_og = average_precision_per_qid(golds=vds.gold, preds=predictions,
                                                               df_qas=vds.qa_feats)
    try:
        if args.algo != 'single-fact':
            analysis_results = map_breakdown(golds=vds.gold, preds=predictions,
                                             k_hops=vds.qf_k_hops, lex_hops=vds.qf_lex_hops,
                                             roles=vds.qf_roles)
        else:
            analysis_results = {}
    except Exception as e:
        logger.error(e)
        analysis_results = {}

    # eval_dataset.qa_feats = df_qas
    result = {
        'map': np.mean(list(prec_per_qid.values())),
        'map_sanity': np.sum(list(prec_per_qid.values())) / len(vds.gold),
        'map_sanity_2': map_og,
        'map_tg_eval': mean_average_precision_score(golds=vds.gold, preds=predictions),
        'map_single_fact': mean_ap,
        # 'eval_loss': eval_loss / (nb_eval_steps * args.eval_batch_size),
        'map_challenge': np.mean(
            qa_feats[vds.df_qas.reset_index(drop=True)[args.lvl_col_name] == 'Challenge'].avg_precision
        ),
        'map_easy': np.mean(
            qa_feats[vds.df_qas.reset_index(drop=True)[args.lvl_col_name] == 'Easy'].avg_precision
        ),
        'map_length': list(
            pd.concat(
                (vds.qa_feats.gold_facts.apply(len), vds.qa_feats.avg_precision),
                axis=1
            ) \
                .groupby('gold_facts')['avg_precision'] \
                .apply(np.mean).to_dict().items()
        ),
        **analysis_results,
    }
    return result
