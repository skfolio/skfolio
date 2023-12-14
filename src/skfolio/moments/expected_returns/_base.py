"""Base Expected returns estimators."""

# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import sklearn.base as skb


class BaseMu(skb.BaseEstimator, ABC):
    """Base class for all expected returns estimators in skfolio.

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    mu_: np.ndarray

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y=None):
        pass
