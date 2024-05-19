"""Combinatorial module"""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# Implementation derived from:
# scikit-portfolio, Copyright (c) 2022, Carlo Nicolini, Licensed under MIT Licence.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

import itertools
import math
import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
import sklearn.model_selection as skm
import sklearn.utils as sku

import skfolio.typing as skt


class BaseCombinatorialCV(ABC):
    """Base class for all combinatorial cross-validators.

    Implementations must define `split` or `get_path_ids`.
    """

    @abstractmethod
    def split(self, X: npt.ArrayLike, y=None) -> tuple[np.ndarray, list[np.ndarray]]:
        pass

    @abstractmethod
    def get_path_ids(self) -> np.ndarray:
        """Return the path id of each test sets in each split"""
        pass

    __repr__ = skm.BaseCrossValidator.__repr__


# TODO: review params and function naming
class CombinatorialPurgedCV(BaseCombinatorialCV):
    """Combinatorial Purged Cross-Validation.

    Provides train/test indices to split time series data samples based on
    Combinatorial Purged Cross-Validation [1]_.

    Compared to `KFold`, which splits the data into `k` folds with `1` fold for the test
    set and `k - 1` folds for the training set, `CombinatorialPurgedCV` uses `k - p`
    folds for the training set with `p > 1` being the number of test folds.

    `KFold` can recombine one single testing path while `CombinatorialPurgedCV` can
    recombine multiple testing paths from the combinations of the train/test sets.

    To avoid data leakage, purging and embargoing can be performed.

    Purging consist of removing from the training set all observations whose labels
    overlapped in time with those labels included in the testing set.

    Embargoing consist of removing from the training set all observations that
    immediately follow an observation in the testing set, since financial features
    often incorporate series that exhibit serial correlation (like ARMA processes).

    Parameters
    ----------
    n_folds : int, default=10
        Number of folds. Must be at least 3.

    n_test_folds : int, default=8
        Number of test folds. Must be at least 2.
        For only one test fold, use `sklearn.model_validation.KFold`.

    purged_size : int, default=0
        Number of observations to exclude from the start of each train set that are
        after a test set **and** the number of observations to exclude from the end of
        each training set that are before a test set.

    embargo_size : int, default=0
        Number of observations to exclude from the start of each training set that are
        after a test set.

    Attributes
    ----------
    index_train_test_ : ndarray of shape (n_observations, n_splits)

    Examples
    --------
    >>> import numpy as np
    >>> from skfolio.model_selection import CombinatorialPurgedCV
    >>> X = np.random.randn(12, 2)
    >>> cv = CombinatorialPurgedCV(n_folds=3, n_test_folds=2)
    >>> for i, (train_index, tests) in enumerate(cv.split(X)):
    ...     print(f"Split {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     for j, test_index in enumerate(tests):
    ...         print(f"  Test {j}:  index={test_index}")
    Split 0:
      Train: index=[ 8  9 10 11]
      Test 0:  index=[0 1 2 3]
      Test 1:  index=[4 5 6 7]
    Split 1:
      Train: index=[4 5 6 7]
      Test 0:  index=[0 1 2 3]
      Test 1:  index=[ 8  9 10 11]
    Split 2:
      Train: index=[0 1 2 3]
      Test 0:  index=[4 5 6 7]
      Test 1:  index=[ 8  9 10 11]
    >>> cv = CombinatorialPurgedCV(n_folds=3, n_test_folds=2, purged_size=1)
    >>> for i, (train_index, tests) in enumerate(cv.split(X)):
    ...     print(f"Split {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     for j, test_index in enumerate(tests):
    ...         print(f"  Test {j}:  index={test_index}")
    Split 0:
      Train: index=[ 9 10 11]
      Test 0:  index=[0 1 2 3]
      Test 1:  index=[4 5 6 7]
    Split 1:
      Train: index=[5 6]
      Test 0:  index=[0 1 2 3]
      Test 1:  index=[ 8  9 10 11]
    Split 2:
      Train: index=[0 1 2]
      Test 0:  index=[4 5 6 7]
      Test 1:  index=[ 8  9 10 11]
    >>> cv = CombinatorialPurgedCV(n_folds=3, n_test_folds=2, embargo_size=1)
    >>> for i, (train_index, tests) in enumerate(cv.split(X)):
    ...     print(f"Split {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     for j, test_index in enumerate(tests):
    ...         print(f"  Test {j}:  index={test_index}")
    Split 0:
      Train: index=[ 9 10 11]
      Test 0:  index=[0 1 2 3]
      Test 1:  index=[4 5 6 7]
    Split 1:
      Train: index=[5 6 7]
      Test 0:  index=[0 1 2 3]
      Test 1:  index=[ 8  9 10 11]
    Split 2:
      Train: index=[0 1 2 3]
      Test 0:  index=[4 5 6 7]
      Test 1:  index=[ 8  9 10 11]

    References
    ----------
    .. [1]  "Advances in Financial Machine Learning",
        Marcos López de Prado (2018)
    """

    index_train_test_: np.ndarray

    def __init__(
        self,
        n_folds: int = 10,
        n_test_folds: int = 8,
        purged_size: int = 0,
        embargo_size: int = 0,
    ):
        if not isinstance(n_folds, numbers.Integral):
            raise ValueError(
                "The number of folds must be of Integral type. "
                f"{n_folds} of type {type(n_folds)} was passed."
            )
        n_folds = int(n_folds)

        if n_folds <= 2:
            raise ValueError(f"`n_folds` must be at least 3`, got `n_folds={n_folds}`.")

        if n_test_folds <= 1:
            raise ValueError(
                f"`n_test_folds` must at least 2, got `n_test_folds={n_test_folds}`."
            )

        if n_test_folds >= n_folds:
            raise ValueError(
                "Combinatorial purged cross-validation requires `n_folds` "
                "to be greater than `n_test_folds`."
            )

        if purged_size < 0:
            raise ValueError("`purged_size` cannot be negative")

        if embargo_size < 0:
            raise ValueError("`embargo_size` cannot be negative")

        self.n_folds = n_folds
        self.n_test_folds = n_test_folds
        self.purged_size = purged_size
        self.embargo_size = embargo_size

    @property
    def n_splits(self) -> int:
        """Number of splits"""
        return _n_splits(n_folds=self.n_folds, n_test_folds=self.n_test_folds)

    @property
    def n_test_paths(self) -> int:
        """Number of test paths that can be reconstructed from the train/test
        combinations"""
        return _n_test_paths(n_folds=self.n_folds, n_test_folds=self.n_test_folds)

    @property
    def test_set_index(self) -> np.ndarray:
        """Location of each test set"""
        return np.array(
            list(itertools.combinations(np.arange(self.n_folds), self.n_test_folds))
        ).reshape(-1, self.n_test_folds)

    @property
    def binary_train_test_sets(self) -> np.ndarray:
        """Identify training and test folds for each combinations by assigning `0` to
        training folds and `1` to test folds"""
        folds_train_test = np.zeros((self.n_folds, self.n_splits))
        folds_train_test[
            self.test_set_index, np.arange(self.n_splits)[:, np.newaxis]
        ] = 1
        return folds_train_test

    @property
    def recombined_paths(self) -> np.ndarray:
        """Recombine each test path by returning the test set location in each split."""
        return np.argwhere(self.binary_train_test_sets == 1)[:, 1].reshape(
            self.n_folds, -1
        )

    def get_path_ids(self) -> np.ndarray:
        """Return the path id of each test sets in each split"""
        recombine_paths = self.recombined_paths
        path_ids = np.zeros((self.n_splits, self.n_test_folds), dtype=int)
        for i in range(self.n_splits):
            for j in range(self.n_test_folds):
                path_ids[i, j] = np.argwhere(recombine_paths == i)[j][1]
        return path_ids

    def split(
        self, X: npt.ArrayLike, y=None, groups=None
    ) -> Iterator[tuple[np.ndarray, list[np.ndarray]]]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), optional
            The (multi-)target variable

        groups : array-like of shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        test_set_index = self.test_set_index
        recombine_paths = self.recombined_paths

        X, y = sku.indexable(X, y)
        n_samples = X.shape[0]
        min_fold_size = n_samples // self.n_folds
        if self.purged_size + self.embargo_size >= min_fold_size - 1:
            raise ValueError(
                "The sum of `purged_size` and `embargo_size` must be smaller than the"
                f" size of a train fold which is {min_fold_size}"
            )

        fold_index_num = np.arange(n_samples) // (n_samples // self.n_folds)
        fold_index_num[fold_index_num == self.n_folds] = self.n_folds - 1

        index_train_test = np.zeros((n_samples, self.n_splits))
        for i in range(self.n_splits):
            index_train_test[
                np.argwhere([fold_index_num == j for j in test_set_index[i]])[:, 1], i
            ] = 1

        diff = np.diff(index_train_test, axis=0)

        # Purge before
        before_index = np.argwhere(diff == 1)
        for k in range(self.purged_size):
            index_train_test[
                np.maximum(0, before_index[:, 0] - k), before_index[:, 1]
            ] = -1

        # Purge after and Embargo
        after_index = np.argwhere(diff == -1)
        for k in range(self.purged_size + self.embargo_size):
            index_train_test[
                np.minimum(n_samples - 1, after_index[:, 0] + k + 1), after_index[:, 1]
            ] = -1
        self.index_train_test_ = index_train_test

        fold_index = {
            fold_id: np.argwhere(fold_index_num == fold_id).reshape(-1)
            for fold_id in range(self.n_folds)
        }
        for i in range(self.n_splits):
            train_index = np.argwhere(index_train_test[:, i] == 0).reshape(-1)
            test_index_list = [
                fold_index[fold_id] for fold_id, _ in np.argwhere(recombine_paths == i)
            ]
            yield train_index, test_index_list

    def summary(self, X) -> pd.Series:
        n_observations = X.shape[0]
        avg_train_size = _avg_train_size(
            n_observations=n_observations,
            n_folds=self.n_folds,
            n_test_folds=self.n_test_folds,
        )
        return pd.Series(
            {
                "Number of Observations": n_observations,
                "Total Number of Folds": self.n_folds,
                "Number of Test Folds": self.n_test_folds,
                "Purge Size": self.purged_size,
                "Embargo Size": self.embargo_size,
                "Average Training Size": int(avg_train_size),
                "Number of Test Paths": self.n_test_paths,
                "Number of Training Combinations": self.n_splits,
            }
        )

    def plot_train_test_folds(self) -> skt.Figure:
        """Plot the train/test fold locations"""
        values = self.binary_train_test_sets
        fill_color = np.where(values == 0, "blue", "red")
        fill_color = fill_color.astype(object)
        fill_color = np.insert(
            fill_color, 0, np.array(["darkblue" for _ in range(self.n_splits)]), axis=0
        )
        values = np.insert(values, 0, np.arange(self.n_splits), axis=0)
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=["Train Combinations"]
                        + [f"Fold {i}" for i in range(self.n_folds)],
                        fill_color="darkblue",
                        font=dict(color="white"),
                        align="left",
                    ),
                    cells=dict(
                        values=values,
                        font=dict(color="white"),
                        fill_color=fill_color,
                        line_color="grey",
                        align="left",
                    ),
                )
            ]
        )
        fig.update_layout(title="Split Train (0) /Test (1) Folds per Combination")
        return fig

    def plot_train_test_index(self, X) -> skt.Figure:
        """Plot the training and test indices for each combinations by assigning `0` to
        training, `1` to test and `-1` to both purge and embargo indices."""
        next(self.split(X))
        n_samples = X.shape[0]
        cond = [
            self.index_train_test_ == -1,
            self.index_train_test_ == 0,
            self.index_train_test_ == 1,
        ]
        values = self.index_train_test_.T
        values = np.insert(values, 0, np.arange(n_samples), axis=0)
        fill_color = np.select(cond, ["green", "blue", "red"]).T
        fill_color = fill_color.astype(object)
        fill_color = np.insert(
            fill_color, 0, np.array(["darkblue" for _ in range(n_samples)]), axis=0
        )
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=["observations"]
                        + [f"Split {i}" for i in range(self.n_splits)],
                        fill_color="darkblue",
                        font=dict(color="white"),
                        align="left",
                    ),
                    cells=dict(
                        values=values,
                        font=dict(color="white"),
                        fill_color=fill_color,
                        line_color="grey",
                        align="left",
                    ),
                )
            ]
        )
        fig.update_layout(
            title="Train (0), Test (1) and Purge/Embargo (-1) observations per splits"
        )

        return fig


def _n_splits(n_folds: int, n_test_folds: int) -> int:
    """Number of splits.

    Parameters
    ----------
    n_folds : int
        Number of folds.

    n_test_folds : int
        Number of test folds.

    Returns
    -------
    n_splits : int
        Number of splits
    """
    return int(math.comb(n_folds, n_test_folds))


def _n_test_paths(n_folds: int, n_test_folds: int) -> int:
    """Number of test paths that can be reconstructed from the train/test
    combinations

    Parameters
    ----------
    n_folds : int
        Number of folds.

    n_test_folds : int
        Number of test folds.

    Returns
    -------
    n_splits : int
        Number of test paths.
    """
    return (
        _n_splits(n_folds=n_folds, n_test_folds=n_test_folds) * n_test_folds // n_folds
    )


def _avg_train_size(n_observations: int, n_folds: int, n_test_folds: int) -> float:
    """Average number of observations contained in each training set.

    Parameters
    ----------
    n_observations : int
        Number of observations.

    n_folds : int
        Number of folds.

    n_test_folds : int
        Number of test folds.

    Returns
    -------
    avg_train_size : float
        Average number of observations contained in each training set.
    """
    return n_observations / n_folds * (n_folds - n_test_folds)


def optimal_folds_number(
    n_observations: int,
    target_train_size: int,
    target_n_test_paths: int,
    weight_train_size: float = 1,
    weight_n_test_paths: float = 1,
) -> tuple[int, int]:
    r"""Find the optimal number of folds (total folds and test folds) for a target
    training size and a target number of test paths.

    We find `x = n_folds` and `y = n_test_folds` that minimizes the below
    cost function of the relative distance from the two targets:

    .. math::
           cost(x,y) = w_{f} \times \lvert\frac{f(x,y)-f_{target}}{f_{target}}\rvert + w_{g} \times \lvert\frac{g(x,y)-g_{target}}{g_{target}}\rvert

    with :math:`w_{f}` and :math:`w_{g}` the weights assigned to the distance
    from each target and :math:`f(x,y)` and :math:`g(x,y)` the average training size
    and the number of test paths as a function of the number of total folds and test
    folds.

    This is a combinatorial problem with :math:`\frac{T\times(T-3)}{2}` combinations,
    with :math:`T` the number of observations.

    We reduce the search space by using the combinatorial symetry
    :math:`{n \choose k}={n \choose n-k}` and skipping cost computation above 1e5.

    Parameters
    ----------
    n_observations : int
        Number of observations.

    target_train_size : int
        The target number of observation in the training set.

    target_n_test_paths : int
        The target number of test paths (that can be reconstructed from the train/test
        combinations).

    weight_train_size : float, default=1
        The weight assigned to the distance from the target train size.
        The default value is 1.

    weight_n_test_paths : float, default=1
        The weight assigned to the distance from the target number of test paths.
        The default value is 1.

    Returns
    -------
    n_folds : int
        Optimal number of total folds.

    n_test_folds : int
        Optimal number of test folds.
    """

    def _cost(
        x: int,
        y: int,
    ) -> float:
        n_test_paths = _n_test_paths(n_folds=x, n_test_folds=y)
        avg_train_size = _avg_train_size(
            n_observations=n_observations, n_folds=x, n_test_folds=y
        )
        return (
            weight_n_test_paths
            * abs(n_test_paths - target_n_test_paths)
            / target_n_test_paths
            + weight_train_size
            * abs(avg_train_size - target_train_size)
            / target_train_size
        )

    costs = []
    res = []
    for n_folds in range(3, n_observations + 1):
        i = None
        for n_test_folds in range(2, n_folds):
            if i is None or n_folds - n_test_folds <= i:
                cost = _cost(x=n_folds, y=n_test_folds)
                costs.append(cost)
                res.append((n_folds, n_test_folds))
                if i is None and cost > 1e5:
                    i = n_test_folds

    j = np.argmin(costs)
    return res[j]
