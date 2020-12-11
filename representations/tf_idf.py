import logging

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from datasets.nlp import NLP
from representations.representation import Representation

logger = logging.getLogger()


class TfIdf(Representation):
    def __init__(self, args):
        self.args = args
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(lowercase=not self.args.no_lower_case, )
        self.lower_case = not self.args.no_lower_case
        self.nlp = NLP()

    def fit_vectorizer(self, text, stem_first=True):
        if stem_first:
            stemmed = self.stem(text)
            self.vectorizer.fit(stemmed.apply(lambda x: ' '.join(x)))
        else:
            self.vectorizer.fit(text)

    def __call__(self, raw: pd.Series, fit_vectorizer: bool = True, stem_first=True) -> csr_matrix:
        if stem_first:
            logger.info('Stemming...')
            data = self.stem(raw).apply(lambda x: ' '.join(x))
        else:
            data = raw

        if fit_vectorizer:
            logger.info('Fitting TF-IDF...')
            self.vectorizer.fit(data)

        result = self.vectorizer.transform(data)
        return result

    def stem(self, raw: pd.Series):
        return raw.apply(
            lambda x: [
                self.nlp.stem(word) for word in self.nlp.tokenize(x) if self.nlp.stem(word) is not None
            ]
        )

    def get_vectorizer(self):
        return self.vectorizer
