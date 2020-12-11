import re
from typing import List

import nltk
from nltk.corpus import stopwords


class NLP:
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('punkt')
        self.stemmer = nltk.PorterStemmer()
        self.stopwords = set(stopwords.words('english'))  # + get_stop_words('en'))

    def interesting(self, word: str) -> bool:
        return (
                bool(re.search('[a-zA-Z]', self.stemmer.stem(word)))
                # mean nb of overlapping facts including stopwords is 3062
                # excluding nltk stopwords 594
                and word not in self.stopwords
        )

    def stem(self, word: str) -> str:
        if self.interesting(word) and self.interesting(self.stemmer.stem(word)):
            return self.stemmer.stem(word)

    def tokenize(self, seq: str) -> List[str]:
        return nltk.word_tokenize(seq.lower())
