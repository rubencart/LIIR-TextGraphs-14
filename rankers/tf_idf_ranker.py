import json
import logging
import os
import time
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

from rankers.fact_ranker import FactRanker
from tg2020task.evaluate import mean_average_precision_score

from representations.tf_idf import TfIdf
from tools import utils
from tools.config import Config
from datasets.worldtree_dataset import WorldTreeDataset
from tools.utils import parse_args

logger = logging.getLogger(__name__)


class TfIdfRanker(FactRanker):
    def __init__(self, config: Config):
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(lowercase=not config.no_lower_case)
        # self.nearest_k = args.eval_top_k
        self.train_ds = None
        self.lower_case = not config.no_lower_case
        self.config = config
        self.tfidf = TfIdf(config)

    def train(self, dataset: WorldTreeDataset):
        self.tfidf.fit_vectorizer(pd.concat((dataset.df_facts['text'],
                                             dataset.df_qas['question'])), stem_first=self.config.stem_before_tf_idf)
        self.train_ds = dataset

    def evaluate(self, dataset: WorldTreeDataset) -> float:
        predictions = self.predict_facts_for_qa_dataset(dataset)
        mean_avg_precision = mean_average_precision_score(golds=dataset.gold, preds=predictions)
        return mean_avg_precision

    def predict(self, dataset: WorldTreeDataset):
        start = time.time()
        logger.info('Predicting...')
        predictions = self.predict_facts_for_qa_dataset(dataset)
        logger.info('Predicted. Took %s sec' % (time.time() - start))

        preds_filename = "eval_predictions_tfidf.txt"
        preds_eval_file = os.path.join(self.config.output_dir, preds_filename)
        with open(preds_eval_file, "w") as f:
            for qa_id, preds in predictions.items():
                for fact_id in preds:
                    f.write("%s\t%s\n" % (qa_id, fact_id))
        logger.info('Saved predictions to file %s' % preds_eval_file)

        json_filename = 'eval_predictions_tfidf.json'
        preds_eval_json = os.path.join(self.config.output_dir, json_filename)
        with open(preds_eval_json, 'w') as f:
            json.dump(predictions, f)

    def predict_facts_for_qa_dataset(self, dataset: WorldTreeDataset) -> Dict[str, List[str]]:
        X_q = self.tfidf(dataset.df_qas['question'], stem_first=self.config.stem_before_tf_idf,
                         fit_vectorizer=False)
        X_f = self.tfidf(self.train_ds.df_facts['text'], stem_first=self.config.stem_before_tf_idf,
                         fit_vectorizer=False)
        X_dist = cosine_distances(X_q, X_f)
        predictions = {}

        for i_question, distances in tqdm(enumerate(X_dist)):
            predictions.update({
                dataset.qa_idx_2_uid[i_question]:
                    [dataset.fact_idx_2_uid[idx] for idx in np.argsort(distances)]  # [:self.nearest_k]]
            })
        return predictions


if __name__ == '__main__':
    args = parse_args()
    config = Config(path=args.config_path)
    new_output_dir = utils.get_output_dir(config)
    config.set_output_dir(new_output_dir)
    utils.initialize_logging(config=config)

    ds = WorldTreeDataset(config, path_to_qas=config.train_qa_path)
    vds = WorldTreeDataset(config, path_to_qas=config.val_qa_path, validation=True)
    ranker = TfIdfRanker(config)
    ranker.train(ds)
    print(ranker.evaluate(vds))
