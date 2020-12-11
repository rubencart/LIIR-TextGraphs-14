import copy
import json
import os
from typing import Dict

import torch


class Config(object):

    def __init__(self, path: str = None):
        assert (path is None or (isinstance(path, str)
                                 and path[-4:] == 'json'
                                 and os.path.exists(path)))

        if path:
            with open(path, 'r') as config_file:
                self.config_dict = json.load(config_file)
        self.config_json_path = path

        self.debug = self.config_dict['debug'] if 'debug' in self.config_dict else False

        self.seed = self.config_dict['seed']
        self.do_eval = self.config_dict['do_eval']
        self.do_test = self.config_dict['do_test']
        self.do_train = self.config_dict.get('do_train', True)
        # self.do_train = not self.do_eval and not self.do_test
        # assert not self.do_test or not self.do_eval

        self.fact_path = self.config_dict['fact_path']
        self.qa_dir = self.config_dict['qa_dir']
        self.train_qa_path = os.path.join(self.qa_dir, self.config_dict['train_qa_file'])
        self.val_qa_path = os.path.join(self.qa_dir, self.config_dict['val_qa_file'])
        self.test_qa_path = os.path.join(self.qa_dir, self.config_dict['test_qa_file'])
        self.task = self.config_dict['task']
        self.v2 = self.config_dict.get('2020v2', False)
        if 'lvl_col_name' in self.config_dict:
            self.lvl_col_name = self.config_dict['lvl_col_name']
        else:
            self.lvl_col_name = 'focusnotes' if self.task == '20' else 'fold'
        self.algo = self.config_dict['algo']

        self.data_dir = self.config_dict['data_dir']
        self.output_dir = self.config_dict['output_dir']
        self.cache_dir = self.config_dict['cache_dir']

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.no_lower_case = self.config_dict['no_lower_case']
        assert not self.no_lower_case
        self.model_type = self.config_dict['model_type']

        self.no_cuda = self.config_dict.get('no_cuda', False)
        device = torch.device('cuda' if torch.cuda.is_available() and not self.no_cuda else 'cpu')
        self.n_gpu = torch.cuda.device_count()
        self.device = device

        self.per_gpu_train_batch_size = self.config_dict.get('per_gpu_train_batch_size', 0)
        self.per_gpu_eval_batch_size = self.config_dict.get('per_gpu_eval_batch_size', 0)
        self.eval_batch_size = self.per_gpu_eval_batch_size * max(1, self.n_gpu)
        self.train_batch_size = self.per_gpu_train_batch_size * max(1, self.n_gpu)

        self.per_gpu_train_tokens_per_batch = self.config_dict.get('per_gpu_train_tokens_per_batch', 0)
        self.per_gpu_eval_tokens_per_batch = self.config_dict.get('per_gpu_eval_tokens_per_batch', 0)
        self.train_tokens_per_batch = self.per_gpu_train_tokens_per_batch * max(1, self.n_gpu)
        self.eval_tokens_per_batch = self.per_gpu_eval_tokens_per_batch * max(1, self.n_gpu)

        self.loss = self.config_dict.get('loss', None)
        self.margin = self.config_dict.get('margin', 1.0)

        self.answer_choices = self.config_dict['answer_choices']
        self.mark_correct_in_qa = self.config_dict['mark_correct_in_qa']
        self.mark_answer_in_qa = self.config_dict['mark_answer_in_qa']

        self.overwrite_cache = self.config_dict['overwrite_cache']
        self.pad_token_id = ValueError
        self.use_segment_ids = self.model_type in ["bert", "xlnet", "albert"]

        self.labels_as = 'float' if self.loss == 'mse' else 'long'
        self.model_name_or_path = self.config_dict.get('model_name_or_path', '')
        self.from_scratch = self.config_dict.get('from_scratch', False)
        self.config_name = self.config_dict.get('config_name', '')

        self.tokenizer_name = self.config_dict.get('tokenizer_name', '')

        self.logging_steps = self.config_dict.get('logging_steps', 0)
        self.save_steps = self.config_dict.get('save_steps', 0)
        self.eval_steps = self.config_dict.get('eval_steps', self.logging_steps)
        self.evaluate_during_training = self.config_dict.get('evaluate_during_training', False)
        # self.eval_all_checkpoints = self.config_dict['eval_all_checkpoints']

        self.fp16 = self.config_dict.get('fp16', False)
        self.fp16_opt_level = self.config_dict.get('fp16_opt_level', "O1")

        self.overwrite_output_dir = self.config_dict['overwrite_output_dir']

        self.gradient_accumulation_steps = self.config_dict.get('gradient_accumulation_steps', 1)
        self.learning_rate = self.config_dict.get('learning_rate', -1)
        self.weight_decay = self.config_dict.get('weight_decay', -1)
        self.adam_epsilon = self.config_dict.get('adam_epsilon', -1)
        self.max_grad_norm = self.config_dict.get('max_grad_norm', -1)

        self.max_steps = self.config_dict.get('max_steps', -1)
        self.approx_num_steps = self.config_dict.get('approx_num_steps', None)
        self.warmup_steps = self.config_dict.get('warmup_steps', 0)
        self.num_train_epochs = self.config_dict.get('num_train_epochs', -1)
        self.num_train_workers = self.config_dict.get('num_train_workers', -1)
        self.num_eval_workers = self.config_dict.get('num_eval_workers', -1)

        self.lr_decay_per_epoch = self.config_dict.get('lr_decay_per_epoch', True)
        self.lr_decay = self.config_dict.get('lr_decay', True)

        self.eval_top_k = self.config_dict.get('eval_top_k', 9727)
        self.max_position_embeddings = self.config_dict.get('max_position_embeddings', None)

        self.adam_beta_1 = self.config_dict.get('adam_beta_1', 0.9)
        self.adam_beta_2 = self.config_dict.get('adam_beta_2', 0.999)

        self.init_counters_from_checkpoint = self.config_dict.get('init_counters_from_checkpoint', False)
        self.init_gear_from_checkpoint = self.config_dict.get('init_gear_from_checkpoint', False)

        self.compute_overlap = self.config_dict.get('compute_overlap', False)

        # only for tf idf ranker
        self.stem_before_tf_idf = self.config_dict.get('stem_before_tf_idf', False)

    def set_pad_token(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def set_output_dir(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def to_dict(self) -> Dict:
        dct = copy.deepcopy(vars(self))
        # dct.pop('default_config_dict')
        # dct.pop('config_dict')
        dct.pop('device')
        # dct = self.config_dict
        # dct['n_gpu']
        return dct


class ChainConfig(Config):
    def __init__(self, path: str = None):
        super().__init__(path)

        self.mark_expl = self.config_dict['mark_expl']

        self.train_chains_file = self.config_dict['train_chains_file']
        self.powerset_complete_combs = self.config_dict['powerset_complete_combs']
        self.nearest_k_visible = self.config_dict['nearest_k_visible']
        self.single_question_per_batch = self.config_dict['single_question_per_batch']
        self.predict_stop_expl = self.config_dict['predict_stop_expl']
        # self.stop_explanation_id = self.config_dict['stop_explanation_id']
        self.use_partial_expl_in_ranking = self.config_dict['use_partial_expl_in_ranking']
        self.average_scores_over_partials = self.config_dict.get('average_scores_over_partials', False)

        self.distance_mode = self.config_dict['distance_mode']
        self.distance_func = self.config_dict['distance_func']
        self.embedder_name_or_path = self.config_dict['embedder_name_or_path']
        self.embedder_aggregate_func = self.config_dict['embedder_aggregate_func']
        self.overwrite_embeddings = self.config_dict['overwrite_embeddings']
        self.distance_wo_fill = self.config_dict.get('distance_wo_fill', False)

        self.use_all_negatives = self.config_dict['use_all_negatives']
        self.condition_on_negs_rate = self.config_dict.get('condition_on_negs_rate', 0.0)
        self.init_condition_on_negs_rate = self.config_dict.get('init_condition_on_negs_rate', 0.0)
        self.max_condition_on_negs_rate = self.config_dict.get('max_condition_on_negs_rate', 0.0)
        self.sample_condition_on_negs_rate = self.config_dict.get('sample_condition_on_negs_rate', False)
        self.anneal_cond_on_negs_schedule = self.config_dict.get('anneal_cond_on_negs_schedule', None)
        self.anneal_cn_steps = self.config_dict.get('anneal_cn_steps', 0)

        self.stop_delta = self.config_dict['stop_delta']
        self.max_expl_length = self.config_dict['max_expl_length']
        self.min_expl_length = self.config_dict['min_expl_length']
        # self.train_max_expl_length = self.config_dict['train_max_expl_length']
        self.categorical_expl_length = self.config_dict.get('categorical_expl_length', False)

        self.rank_rest = self.config_dict['rank_rest']
        self.rank_scored_but_unused_2nd = self.config_dict.get('rank_scored_but_unused_2nd', True)
        self.rank_rest_tf_idf_q_only = self.config_dict.get('rank_rest_tf_idf_q_only', False)
        self.rank_rest_tf_idf_f_in_q = self.config_dict.get('rank_rest_tf_idf_f_in_q', None)

        self.lambdaloss_scheme = self.config_dict['lambdaloss_scheme'] if self.loss == 'lambdaloss' else None
        self.beam_size = self.config_dict['beam_size'] if 'beam_size' in self.config_dict else 1
        self.eval_use_logprobs = self.config_dict.get('eval_use_logprobs', False)
        self.average_scores_over_beams = self.config_dict.get('average_scores_over_beams', True)
        self.average_scores_over_beams_intermed = self.config_dict.get('average_scores_over_beams_intermed', False)
        self.beam_score_relative_size = self.config_dict.get('beam_score_relative_size', True)
        self.average_scores_weighted = self.config_dict.get('average_scores_weighted', False)
        self.beam_score_acc = self.config_dict.get('beam_score_acc', 'sum')
        self.beam_decode_top_p = self.config_dict.get('beam_decode_top_p', 0.95)
        self.beam_fact_selection = self.config_dict.get('beam_fact_selection', 'greedy')
        self.facts_from_unused_beams_second = self.config_dict.get('facts_from_unused_beams_second', True)

        self.fisher_rho = self.config_dict['fisher_rho'] if 'fisher_rho' in self.config_dict else 1e-6
        self.repeat_dataset_in_epoch = self.config_dict['repeat_dataset_in_epoch'] \
            if 'repeat_dataset_in_epoch' in self.config_dict else 1
        self.train_visible_rate = self.config_dict.get('train_visible_rate', 0.0)
        self.max_train_visible_rate = self.config_dict.get('max_train_visible_rate', 0.0)
        self.init_train_visible_rate = self.config_dict.get('init_train_visible_rate', 0.0)
        self.train_pos_in_negs_rate = self.config_dict.get('train_pos_in_negs_rate', 0.0)
        self.max_train_pos_in_negs_rate = self.config_dict.get('max_train_pos_in_negs_rate', 0.0)
        self.init_train_pos_in_negs_rate = self.config_dict.get('init_train_pos_in_negs_rate', 0.0)
        self.sample_pos_in_negs_rate = self.config_dict.get('sample_pos_in_negs_rate', False)
        self.anneal_vr_schedule = self.config_dict.get('anneal_vr_schedule', None)
        self.anneal_pinr_schedule = self.config_dict.get('anneal_pinr_schedule', None)
        self.anneal_vr_steps = self.config_dict.get('anneal_vr_steps', 0)
        self.fisher_weights = self.config_dict.get('fisher_weights', False)

        self.num_labels = self.config_dict.get('num_labels', 1)

        self.expl_length_per_batch = self.config_dict.get('expl_length_per_batch', False)
        self.fisher_use_all_golds = self.config_dict['fisher_use_all_golds'] \
            if 'fisher_use_all_golds' in self.config_dict else False

        self.encoding = self.config_dict.get('encoding', 'single_candidate')
        assert self.encoding in ('single_candidate', 'explanation_set'), self.encoding

    def set_stop_delta(self, stop_delta):
        self.stop_delta = stop_delta

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size

    def set_beam_score_relative_size(self, beam_score_relative_size):
        self.beam_score_relative_size = beam_score_relative_size

    def set_average_scores_weighted(self, average_scores_weighted):
        self.average_scores_weighted = average_scores_weighted

    def set_average_scores_over_beams(self, average_scores_over_beams):
        self.average_scores_over_beams = average_scores_over_beams

    def set_max_expl_length(self, max_expl_length):
        self.max_expl_length = max_expl_length

    def set_min_expl_length(self, min_expl_length):
        self.min_expl_length = min_expl_length

    def set_nearest_k_visible(self, nearest_k_visible):
        self.nearest_k_visible = nearest_k_visible

    def set_predict_stop_expl(self, predict_stop_expl):
        self.predict_stop_expl = predict_stop_expl

    def set_use_partial_expl_in_ranking(self, use_partial_expl_in_ranking):
        self.use_partial_expl_in_ranking = use_partial_expl_in_ranking

    def set_eval_use_logprobs(self, val):
        self.eval_use_logprobs = val

    def set_beam_score_acc(self, val):
        self.beam_score_acc = val

    def set_beam_fact_selection(self, val):
        self.beam_fact_selection = val

    def set_beam_decode_top_p(self, val):
        self.beam_decode_top_p = val

    def set_average_scores_over_partials(self, val):
        self.average_scores_over_partials = val

    def set_rank_rest(self, val):
        self.rank_rest = val

    def set_rank_scored_but_unused_2nd(self, val):
        self.rank_scored_but_unused_2nd = val

    def set_unused_beams_second(self, val):
        self.facts_from_unused_beams_second = val


class SingleFactConfig(Config):
    def __init__(self, path: str = None):
        super().__init__(path)
        self.max_seq_length = self.config_dict.get('max_seq_length', 72)

        self.train_pair_file = self.config_dict['train_pair_file']
        self.dev_pair_file = self.config_dict['dev_pair_file']
        self.test_pair_file = self.config_dict['test_pair_file']
