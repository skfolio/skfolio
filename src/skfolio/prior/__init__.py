"""Prior module."""

from skfolio.prior._base import BasePrior, ReturnDistribution
from skfolio.prior._black_litterman import BlackLitterman
from skfolio.prior._empirical import EmpiricalPrior
from skfolio.prior._entropy_pooling import EntropyPooling
from skfolio.prior._opinion_pooling import OpinionPooling
from skfolio.prior._synthetic_data import SyntheticData
from skfolio.prior._time_series_factor_model import (
    BaseLoadingMatrix,
    LoadingMatrixRegression,
    TimeSeriesFactorModel,
)

__all__ = [
    "BaseLoadingMatrix",
    "BasePrior",
    "BlackLitterman",
    "EmpiricalPrior",
    "EntropyPooling",
    "LoadingMatrixRegression",
    "OpinionPooling",
    "ReturnDistribution",
    "SyntheticData",
    "TimeSeriesFactorModel",
]

_DEPRECATED_NAMES = {
    "FactorModel": "TimeSeriesFactorModel",
}


def __getattr__(name):
    if name in _DEPRECATED_NAMES:
        import warnings

        new_name = _DEPRECATED_NAMES[name]
        warnings.warn(
            f"`{name}` has been renamed to `{new_name}` and will be removed "
            "in version 1.0.0.",
            FutureWarning,
            stacklevel=2,
        )
        return globals()[new_name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
