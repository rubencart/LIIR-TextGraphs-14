import logging
import os

import pandas as pd
import torch
from torch import Tensor, FloatTensor
from tqdm import tqdm
from transformers import PreTrainedTokenizer, BertConfig, RobertaConfig, DistilBertConfig, AlbertConfig, BertModel, \
    RobertaModel, DistilBertModel, AlbertTokenizer, AutoTokenizer, AutoModel, AutoConfig, AlbertModel, \
    RobertaTokenizer, DistilBertTokenizer, BertTokenizer

from representations.representation import Representation
from tools.config import ChainConfig

logger = logging.getLogger()


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
    "bert": (BertConfig, BertModel, BertTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertModel, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
    "auto": (AutoConfig, AutoModel, AutoTokenizer),
}


class SentenceEmbedder(Representation):
    def __init__(self, config: ChainConfig):
        self.config = config

        config_class, model_class, tokenizer_class = MODEL_CLASSES['auto']

        tokenizer_kwconfig = {
            'cache_dir': config.cache_dir if config.cache_dir else None,
            'do_lower_case': not config.no_lower_case
        }
        self.tokenizer: PreTrainedTokenizer = tokenizer_class.from_pretrained(config.embedder_name_or_path,
                                                                              **tokenizer_kwconfig)
        self.model_config = config_class.from_pretrained(config.embedder_name_or_path,
                                                         cache_dir=config.cache_dir if config.cache_dir else None,)
        self.model = model_class.from_pretrained(config.embedder_name_or_path,
                                                 config=self.model_config,
                                                 cache_dir=config.cache_dir if config.cache_dir else None,)
        self.model.to(config.device)

    def __call__(self, raw: pd.Series, mode: str = 'train', sents: str = 'facts', **kwargs) -> Tensor:
        cached_embeddings_fn = self.cached_embeddings_filename(mode, sents, self.config.debug)
        if os.path.exists(cached_embeddings_fn) and not self.config.overwrite_embeddings:
            logger.info('Loading embeddings from cached file %s', cached_embeddings_fn)
            return torch.load(cached_embeddings_fn)

        else:
            embeddings = self.compute_embeddings(raw)
            logger.info('Saving embeddings to file %s', cached_embeddings_fn)
            torch.save(embeddings, cached_embeddings_fn)
            return embeddings

    def compute_embeddings(self, raw):
        logger.info('Computing embeddings...')
        model = self.model
        tokenizer = self.tokenizer
        if not self.config.no_cuda and self.config.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        result = torch.zeros((len(raw), self.model_config.hidden_size))
        model.eval()
        with torch.no_grad():
            for i, sentence in enumerate(tqdm(raw, desc='Computing sentence embeddings...')):
                encoded = tokenizer.encode_plus(sentence, add_special_tokens=True)
                inputs = {
                    'input_ids': torch.LongTensor(encoded['input_ids']).to(self.config.device).unsqueeze(0),
                    'token_type_ids': (torch.LongTensor(encoded['token_type_ids']).to(self.config.device).unsqueeze(0)
                                       if self.config.use_segment_ids else None),
                    'attention_mask': torch.LongTensor(encoded['attention_mask']).to(self.config.device).unsqueeze(0),
                }
                outputs = model(**inputs)
                sequence_output, pooled_output = outputs[0], outputs[1]
                result[i, :] = self.sequence_to_embedding(sequence_output, pooled_output)
        return result

    def sequence_to_embedding(self, sequence_output: FloatTensor, pooled_output: FloatTensor) -> Tensor:
        """
            sequence_output: bs (=1) x L x hidden_dim
        """
        if self.config.embedder_aggregate_func == 'max_pool':
            values, _ = sequence_output.max(dim=1)
            return values
        elif self.config.embedder_aggregate_func == 'avg_pool':
            return sequence_output.mean(dim=1)
        else:
            assert self.config.embedder_aggregate_func == 'pooled'
            return pooled_output

    def cached_embeddings_filename(self, mode, sents, debug=False):
        mode_sents = 'facts' if sents == 'facts' else '%s_%s' % (mode, sents)
        cached_embeddings_file = os.path.join(
            self.config.data_dir,
            'cached_embeddings_{}_{}_{}_{}_{}'.format(
                self.config.task,
                mode_sents,
                self.model.config.model_type,
                self.config.embedder_name_or_path.replace('/', '-'),
                self.config.embedder_aggregate_func,
            )
        )
        cached_embeddings_file += '_debug' if (debug and sents != 'facts') else ''
        return cached_embeddings_file
