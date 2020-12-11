from abc import ABC
from typing import Any

import pandas as pd


class Representation(ABC):
    def __call__(self, raw: pd.Series, **kwargs) -> Any:
        raise NotImplementedError
