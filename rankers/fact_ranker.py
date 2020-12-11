from abc import ABC


class FactRanker(ABC):

    def train(self, *args):
        raise NotImplementedError

    def evaluate(self, *args):
        raise NotImplementedError
