"""Walk Forward cross-validator."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-portfolio, Copyright (c) 2022, Carlo Nicolini, Licensed under MIT Licence.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

import datetime as dt
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.model_selection as sks
import sklearn.utils as sku


class WalkForward(sks.BaseCrossValidator):
    """Walk Forward Cross-Validator.

    Provides train/test indices to split time series data samples using a walk-forward
    logic.

    In each split, test indices must be higher than the previous ones; therefore,
    shuffling in cross-validator is inappropriate.

    Compared to `sklearn.model_selection.TimeSeriesSplit`, you control the train/test
    folds by specifying the number of training and test samples instead of the number
    of splits, making it more suitable for portfolio cross-validation.

    If your data is a DataFrame indexed with a DatetimeIndex, you can split the data
    using specific datetime frequencies and offsets.

    Parameters
    ----------
    test_size : int
        Length of each test set.
        If `freq` is `None` (default), it represents the number of observations.
        Otherwise, it represents the number of periods defined by `freq`.

    train_size : int | pandas.offsets.DateOffset | datetime.timedelta
        Length of each training set.
        If `freq` is `None` (default), it represents the number of observations.
        Otherwise, for integers, it represents the number of periods defined by `freq`;
        for pandas DateOffset or datetime timedelta it represents the date offset
        applied to the start of each period.

    freq : str | pandas.offsets.DateOffset, optional
        If provided, it must be a frequency string or a pandas DateOffset, and the
        returns `X` must be a DataFrame with an index of type `DatetimeIndex`.
        For a list of pandas frequencies and offsets, see `here <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases>`_.
        The defaul (`None`) means `test_size` and `train_size` represent the number of
        observations.

        Below are some common examples:

            * Rebalancing    : Montly on the first day
            * Test Duration  : 1 month
            * Train Duration : 6 months

            >>> cv = WalkForward(test_size=1, train_size=6, freq="MS")

            * Rebalancing    : Quarterly on the first day
            * Test Duration  : 1 quarter
            * Train Duration : 2 months

            >>> cv = WalkForward(test_size=1, train_size=pd.DateOffset(months=2), freq="QS")

            * Rebalancing    : Montly on the third Friday
            * Test Duration  : 1 month
            * Train Duration : 6 weeks

            >>> cv = WalkForward(test_size=1, train_size=pd.offsets.Week(6), freq= "WOM-3FRI")

            * Rebalancing    : Semi-annually on the last day
            * Test Duration  : 6 months
            * Train Duration : 1 year

            >>> cv = WalkForward(test_size=1, train_size=2, freq=pd.offsets.SemiMonthEnd())

            * Rebalancing    : Every 2 months on the second day
            * Test Duration  : 2 months
            * Train Duration : 6 months

            >>> cv = WalkForward(test_size=2, train_size=6, freq="MS", freq_offset=dt.timedelta(days=2))

    freq_offset : pandas DateOffset | datetime timedelta, optional
        Only used if `freq` is provided. Offsets the `freq` by a pandas DateOffset or a
        datetime timedelta offset.

    previous : bool, default=False
        Only used if `freq` is provided. If set to `True`, and if the period start
        or period end is not in the `DatetimeIndex`, the previous observation is used;
        otherwise, the next observation is used (default).

    expend_train : bool, default=False
        If set to `True`, each subsequent training set after the first one will
        use all past observations.
        The default is `False`.

    reduce_test : bool, default=False
        If set to `True`, the last train/test split will be returned even if the
        test set is partial (i.e., it contains fewer observations than `test_size`),
        otherwise, it will be ignored.
        The default is `False`.

    purged_size : int, default=0
        The number of observations to exclude from the end of each training set before
        the test set.
        The default value is `0`.

    Examples
    --------
    >>> import numpy as np
    >>> from skfolio.model_selection import WalkForward
    >>> X = np.random.randn(6, 2)
    >>> cv = WalkForward(test_size=1, train_size=2)
    >>> for i, (train_index, test_index) in enumerate(cv.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[0 1]
      Test:  index=[2]
    Fold 1:
      Train: index=[1 2]
      Test:  index=[3]
    Fold 2:
      Train: index=[2 3]
      Test:  index=[4]
    Fold 3:
      Train: index=[3 4]
      Test:  index=[5]
    >>> cv = WalkForward(test_size=1, train_size=2, purged_size=1)
    >>> for i, (train_index, test_index) in enumerate(cv.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[0 1]
      Test:  index=[3]
    Fold 1:
      Train: index=[1 2]
      Test:  index=[4]
    Fold 2:
      Train: index=[2 3]
      Test:  index=[5]
    >>> cv = WalkForward(test_size=2, train_size=3)
    >>> for i, (train_index, test_index) in enumerate(cv.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[0 1 2]
      Test:  index=[3 4]
    >>> cv = WalkForward(test_size=2, train_size=3, reduce_test=True)
    >>> for i, (train_index, test_index) in enumerate(cv.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[0 1 2]
      Test:  index=[3 4]
    Fold 1:
      Train: index=[2 3 4]
      Test:  index=[5]
    >>> cv = WalkForward(test_size=2, train_size=3, expend_train=True, reduce_test=True)
    >>> for i, (train_index, test_index) in enumerate(cv.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[0 1 2]
      Test:  index=[3 4]
    Fold 1:
      Train: index=[0 1 2 3 4]
      Test:  index=[5]
    """

    def __init__(
        self,
        test_size: int,
        train_size: int | pd.offsets.BaseOffset | dt.timedelta,
        freq: str | pd.offsets.BaseOffset | None = None,
        freq_offset: pd.offsets.BaseOffset | dt.timedelta | None = None,
        previous: bool = False,
        expend_train: bool = False,
        reduce_test: bool = False,
        purged_size: int = 0,
    ):
        self.test_size = test_size
        self.train_size = train_size
        self.freq = freq
        self.freq_offset = freq_offset
        self.previous = previous
        self.expend_train = expend_train
        self.reduce_test = reduce_test
        self.purged_size = purged_size

    def split(
        self, X: npt.ArrayLike, y=None, groups=None
    ) -> Iterator[np.ndarray, np.ndarray]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like of shape (n_observations, n_targets)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_observations,)
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y = sku.indexable(X, y)
        n_samples = X.shape[0]

        if not isinstance(self.test_size, int):
            raise ValueError("test_size` must be an integer")

        if self.freq is None:
            if not isinstance(self.train_size, int):
                raise ValueError("When `freq` is None, `train_size` must be an integer")
            return _split_without_period(
                n_samples=n_samples,
                train_size=self.train_size,
                test_size=self.test_size,
                purged_size=self.purged_size,
                expend_train=self.expend_train,
                reduce_test=self.reduce_test,
            )

        if not hasattr(X, "index") or not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError(
                "X must be a DataFrame with an index of type DatetimeIndex"
            )
        if isinstance(self.train_size, int):
            return _split_from_period_without_train_offset(
                n_samples=n_samples,
                train_size=self.train_size,
                test_size=self.test_size,
                freq=self.freq,
                freq_offset=self.freq_offset,
                previous=self.previous,
                purged_size=self.purged_size,
                expend_train=self.expend_train,
                reduce_test=self.reduce_test,
                ts_index=X.index,
            )
        return _split_from_period_with_train_offset(
            n_samples=n_samples,
            train_size=self.train_size,
            test_size=self.test_size,
            freq=self.freq,
            freq_offset=self.freq_offset,
            previous=self.previous,
            purged_size=self.purged_size,
            expend_train=self.expend_train,
            reduce_test=self.reduce_test,
            ts_index=X.index,
        )

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
         X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like of shape (n_observations, n_targets)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_observations,)
            Always ignored, exists for compatibility.

        Returns
        -------
        n_folds : int
            Returns the number of splitting iterations in the cross-validator.
        """
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        X, y = sku.indexable(X, y)
        n_samples = X.shape[0]
        n = n_samples - self.train_size - self.purged_size

        if self.reduce_test and n % self.test_size != 0:
            return n // self.test_size + 1
        return n // self.test_size


def _split_without_period(
    n_samples: int,
    train_size: int,
    test_size: int,
    purged_size: int,
    expend_train: bool,
    reduce_test: bool,
) -> Iterator[np.ndarray, np.ndarray]:
    if train_size + purged_size >= n_samples:
        raise ValueError(
            "The sum of `train_size` with `purged_size` "
            f"({train_size + purged_size}) cannot be greater than the"
            f" number of samples ({n_samples})."
        )

    indices = np.arange(n_samples)

    test_start = train_size + purged_size
    while True:
        if test_start >= n_samples:
            return
        test_end = test_start + test_size
        train_end = test_start - purged_size
        if expend_train:
            train_start = 0
        else:
            train_start = train_end - train_size

        if test_end > n_samples:
            if not reduce_test:
                return
            test_indices = indices[test_start:]
        else:
            test_indices = indices[test_start:test_end]
        train_indices = indices[train_start:train_end]
        yield train_indices, test_indices

        test_start = test_end


def _split_from_period_without_train_offset(
    n_samples: int,
    train_size: int,
    test_size: int,
    freq: str,
    freq_offset: pd.offsets.BaseOffset | dt.timedelta | None,
    previous: bool,
    purged_size: int,
    expend_train: bool,
    reduce_test: bool,
    ts_index,
) -> Iterator[np.ndarray, np.ndarray]:
    start = ts_index[0]
    end = ts_index[-1]
    if freq_offset is not None:
        start = min(start, start - freq_offset)

    date_range = pd.date_range(start=start, end=end, freq=freq)
    if freq_offset is not None:
        date_range += freq_offset

    idx = ts_index.get_indexer(date_range, method="ffill" if previous else "bfill")
    n = len(idx)
    i = 0
    while True:
        if i + train_size >= n:
            return

        if i + train_size + test_size >= n:
            if not reduce_test:
                return
            test_indices = np.arange(idx[i + train_size], n_samples)

        else:
            test_indices = np.arange(
                idx[i + train_size], idx[i + train_size + test_size]
            )
        if expend_train:
            train_start = 0
        else:
            train_start = idx[i]
        train_indices = np.arange(train_start, idx[i + train_size] - purged_size)
        yield train_indices, test_indices

        i += test_size


def _split_from_period_with_train_offset(
    n_samples: int,
    train_size: pd.offsets.BaseOffset | dt.timedelta,
    test_size: int,
    freq: str,
    freq_offset: pd.offsets.BaseOffset | dt.timedelta | None,
    previous: bool,
    purged_size: int,
    expend_train: bool,
    reduce_test: bool,
    ts_index,
) -> Iterator[np.ndarray, np.ndarray]:
    start = ts_index[0]
    end = ts_index[-1]
    if freq_offset is not None:
        start = min(start, start - freq_offset)

    date_range = pd.date_range(start=start, end=end, freq=freq)
    if freq_offset is not None:
        date_range += freq_offset

    idx = ts_index.get_indexer(date_range, method="ffill" if previous else "bfill")
    train_idx = ts_index.get_indexer(date_range - train_size, method="ffill")

    n = len(idx)

    if np.all(train_idx == -1):
        return

    i = np.argmax(train_idx > -1)
    while True:
        if i >= n:
            return

        if i + test_size >= n:
            if not reduce_test:
                return
            test_indices = np.arange(idx[i], n_samples)
        else:
            test_indices = np.arange(idx[i], idx[i + test_size] - purged_size)

        if expend_train:
            train_start = 0
        else:
            train_start = train_idx[i]
        train_indices = np.arange(train_start, idx[i])
        yield train_indices, test_indices

        i += test_size
