"""Model selection module"""

# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause

from skfolio.model_selection._combinatorial import (
    BaseCombinatorialCV,
    CombinatorialPurgedCV,
    optimal_folds_number,
)
from skfolio.model_selection._validation import cross_val_predict
from skfolio.model_selection._walk_forward import WalkForward

__all__ = [
    "cross_val_predict",
    "WalkForward",
    "BaseCombinatorialCV",
    "CombinatorialPurgedCV",
    "optimal_folds_number",
]
