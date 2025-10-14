"""Tools module."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

import warnings
from collections.abc import Callable, Iterator
from enum import Enum
from functools import wraps
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp
import sklearn as sk
import sklearn.base as skb

__all__ = [
    "AutoEnum",
    "args_names",
    "bisection",
    "cache_method",
    "cached_property_slots",
    "check_estimator",
    "deduplicate_names",
    "default_asset_names",
    "fit_and_predict",
    "fit_single_estimator",
    "format_measure",
    "get_feature_names",
    "input_to_array",
    "optimal_rounding_decimals",
    "safe_indexing",
    "safe_split",
    "validate_input_list",
]

GenericAlias = type(list[int])


class AutoEnum(str, Enum):
    """Base Enum class used in `skfolio`."""

    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: Any
    ) -> str:
        """Overriding `auto()`."""
        return name.lower()

    @classmethod
    def has(cls, value: str) -> bool:
        """Check if a value is in the Enum.

        Parameters
        ----------
        value : str
            Input value.

        Returns
        -------
        x : bool
            True if the value is in the Enum, False otherwise.
        """
        return value in cls._value2member_map_

    def __repr__(self) -> str:
        """Representation of the Enum."""
        return self.name


# noinspection PyPep8Naming
class cached_property_slots:
    """Cached property decorator for slots."""

    def __init__(self, func):
        self.func = func
        self.public_name = None
        self.private_name = None
        self.__doc__ = func.__doc__

    def __set_name__(self, owner, name):
        """Set Name."""
        self.public_name = name
        self.private_name = f"_{name}"

    def __get__(self, instance, owner=None):
        """Getter."""
        if instance is None:
            return self
        if self.private_name is None:
            raise TypeError(
                "Cannot use cached_property instance without calling __set_name__"
                " on it."
            )
        try:
            value = getattr(instance, self.private_name)
        except AttributeError:
            value = self.func(instance)
            setattr(instance, self.private_name, value)
        return value

    def __set__(self, instance, owner=None):
        """Setter."""
        raise AttributeError(
            f"'{type(instance).__name__}' object attribute '{self.public_name}' is"
            " read-only"
        )

    __class_getitem__ = classmethod(GenericAlias)


def _make_key(args, kwds) -> int:
    """Make a cache key from optionally typed positional and keyword arguments."""
    key = args
    if kwds:
        for item in kwds.items():
            key += item
    return hash(key)


def _make_indexable(iterable):
    """Ensure iterable supports indexing or convert to an indexable variant.

    Convert sparse matrices to csr and other non-indexable iterable to arrays.
    Let `None` and indexable objects (e.g. pandas dataframes) pass unchanged.

    Parameters
    ----------
    iterable : {list, dataframe, ndarray, sparse matrix} or None
        Object to be converted to an indexable iterable.
    """
    if sp.issparse(iterable):
        return iterable.tocsr()
    elif hasattr(iterable, "__getitem__") or hasattr(iterable, "iloc"):
        return iterable
    elif iterable is None:
        return iterable
    return np.array(iterable)


def _check_method_params(
    X: npt.ArrayLike, params: dict, indices: np.ndarray = None, axis: int = 0
):
    """Check and validate the parameters passed to a specific
    method like `fit`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data array.

    params : dict
        Dictionary containing the parameters passed to the method.

    indices : ndarray of shape (n_samples,), default=None
        Indices to be selected if the parameter has the same size as `X`.

    axis : int, default=0
        The axis along which `X` will be sub-sampled. `axis=0` will select
        rows while `axis=1` will select columns.

    Returns
    -------
    method_params_validated : dict
        Validated parameters. We ensure that the values support indexing.
    """
    # noinspection PyUnresolvedReferences
    n_observations = X.shape[0]
    method_params_validated = {}
    for param_key, param_value in params.items():
        if param_value.shape[0] != n_observations:
            raise ValueError(
                f"param_key has wrong number of observations, "
                f"received={param_value.shape[0]}, "
                f"expected={n_observations}"
            )
        method_params_validated[param_key] = _make_indexable(param_value)
        method_params_validated[param_key] = safe_indexing(
            X=method_params_validated[param_key], indices=indices, axis=axis
        )
    return method_params_validated


def safe_indexing(
    X: npt.ArrayLike | pd.DataFrame, indices: npt.ArrayLike | None, axis: int = 0
):
    """Return rows, items or columns of X using indices.

    Parameters
    ----------
    X : array-like
        Data from which to sample rows.

    indices : array-like, optional
        Indices of rows or columns.
        The default (`None`) is to select the entire data.

    axis : int, default=0
        The axis along which `X` will be sub-sampled. `axis=0` will select
        rows while `axis=1` will select columns.

    Returns
    -------
    subset :
        Subset of X on axis 0.
    """
    if indices is None:
        return X
    if hasattr(X, "iloc"):
        return X.take(indices, axis=axis)
    if axis == 0:
        return X[indices]
    return X[:, indices]


def safe_split(
    X: npt.ArrayLike,
    y: npt.ArrayLike | None = None,
    indices: np.ndarray | None = None,
    axis: int = 0,
):
    """Create subset of dataset.

    Slice X, y according to indices for cross-validation.

    Parameters
    ----------
    X : array-like
        Data to be indexed.

    y : array-like
        Data to be indexed.

    indices : ndarray of int, optional
        Rows or columns to select from X and y.
        The default (`None`) is to select the entire data.

    axis : int, default=0
        The axis along which `X` will be sub-sampled. `axis=0` will select
        rows while `axis=1` will select columns.

    Returns
    -------
    X_subset : array-like
        Indexed data.

    y_subset : array-like
        Indexed targets.
    """
    X_subset = safe_indexing(X, indices=indices, axis=axis)
    if y is not None:
        y_subset = safe_indexing(y, indices=indices, axis=axis)
    else:
        y_subset = None
    return X_subset, y_subset


def cache_method(cache_name: str) -> Callable:
    """Decorator that caches class methods results into a class dictionary.

    Parameters
    ----------
    cache_name : str
        Name of the dictionary class attribute.

    Returns
    -------
    func : Callable
        Decorating function that caches class methods.
    """

    # To avoid memory leakage and proper garbage collection, self should not be part of
    # the cache key.
    # This is a known issue when we use functools.lru_cache on class methods.
    def decorating_function(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            func_name = method.__name__
            key = _make_key(args, kwargs)
            try:
                cache = getattr(self, cache_name)
            except AttributeError:
                raise AttributeError(
                    "You first need to create a dictionary class attribute named "
                    f"'{cache_name}'"
                ) from None
            if not isinstance(cache, dict):
                raise AttributeError(
                    f"'The cache named '{cache_name}' must be a "
                    f"dictionary, got {type(cache)}"
                )
            if func_name not in cache:
                cache[func_name] = {}
            c = cache[func_name]
            if key not in c:
                c[key] = method(self, *args, **kwargs)
            return c[key]

        return wrapper

    return decorating_function


def args_names(func: object) -> list[str]:
    """Returns the argument names of a function.

    Parameters
    ----------
    func : object
        Function.

    Returns
    -------
    args : list[str]
        The list of function arguments.
    """
    return [
        v for v in func.__code__.co_varnames[: func.__code__.co_argcount] if v != "self"
    ]


def check_estimator(
    estimator: skb.BaseEstimator | None,
    default: skb.BaseEstimator | None,
    check_type: Any,
):
    """Check the estimator type and returns its cloned version it provided, otherwise
     return the default estimator.

    Parameters
    ----------
    estimator : BaseEstimator, optional
        Estimator.

    default : BaseEstimator, optional
        Default estimator to return when `estimator` is `None`.

    check_type : Any
        Expected type of the estimator to check against.

    Returns
    -------
    estimator : Estimator
        The checked estimator or the default.
    """
    if estimator is None:
        return default
    if not isinstance(estimator, check_type):
        raise TypeError(f"Expected type {check_type}, got {type(estimator)}")
    return sk.clone(estimator)


def input_to_array(
    items: dict | npt.ArrayLike,
    n_assets: int,
    fill_value: Any,
    dim: int,
    assets_names: np.ndarray | None,
    name: str,
) -> np.ndarray:
    """Convert a collection of items (array-like or dictionary) into
    a numpy array and verify its shape.

    Parameters
    ----------
    items : np.ndarray | dict | list
        Items to verify and convert to array.

    n_assets : int
        Expected number of assets.
        Used to verify the shape of the converted array.

    fill_value : Any
        When `items` is a dictionary, elements that are not in `asset_names` are filled
        with `fill_value` in the converted array.

    dim : int
        Dimension of the final array.
        Possible values are `1` or `2`.

    assets_names : ndarray, optional
        Asset names used when `items` is a dictionary.

    name : str
        Name of the items used for error messages.

    Returns
    -------
    values : ndarray of shape (n_assets) for dim=1 or (n_groups, n_assets) for dim=2
        Converted array.
    """
    if dim not in [1, 2]:
        raise ValueError(f"dim must be 1 or 2, got {dim}")
    if isinstance(items, dict):
        if assets_names is None:
            raise ValueError(
                f"If `{name}` is provided as a dictionary, you must input `X` as a"
                " DataFrame with assets names in columns"
            )
        if dim == 1:
            arr = np.array([items.get(asset, fill_value) for asset in assets_names])
        else:
            # add assets and convert dict to ordered array
            arr = {}
            for asset in assets_names:
                elem = items.get(asset)
                if elem is None:
                    elem = [asset]
                elif np.isscalar(elem):
                    elem = [asset, elem]
                else:
                    elem = [asset, *elem]
                arr[asset] = elem
            arr = (
                pd.DataFrame.from_dict(arr, orient="index")
                .loc[assets_names]
                .to_numpy()
                .T
            )
    else:
        arr = np.asarray(items)

    if arr.ndim != dim:
        raise ValueError(f"`{name}` must be a {dim}D array, got a {arr.ndim}D array")

    if not isinstance(fill_value, str) and np.isnan(arr).any():
        raise ValueError(f"`{name}` contains NaN")

    if arr.shape[-1] != n_assets:
        if dim == 1:
            s = "(n_assets,)"
        else:
            s = "(n_groups, n_assets)"
        raise ValueError(
            f"`{name}` must be a of shape {s} with n_assets={n_assets}, "
            f"got {arr.shape[0]}"
        )
    return arr


def validate_input_list(
    items: list[int | str],
    n_assets: int,
    assets_names: np.ndarray[str] | None,
    name: str,
    raise_if_string_missing: bool = True,
) -> list[int]:
    """Convert a list of items (asset indices or asset names) into a list of
    validated asset indices.

    Parameters
    ----------
    items : list[int | str]
       List of asset indices or asset names.

    n_assets : int
       Expected number of assets.
       Used for verification.

    assets_names : ndarray, optional
       Asset names used when `items` contain strings.

    name : str
       Name of the items used for error messages.

    raise_if_string_missing : bool, default=True
        If set to True, raises an error if an item string is missing from assets_names;
        otherwise, issue a User Warning.

    Returns
    -------
    values : list[int]
       Converted and validated list.
    """
    if len(set(items)) != len(items):
        raise ValueError(f"Duplicates found in {items}")

    asset_indices = set(range(n_assets))
    res = []
    for asset in items:
        if isinstance(asset, str):
            if assets_names is None:
                raise ValueError(
                    f"If `{name}` is provided as a list of string, you must input `X` "
                    f"as a DataFrame with assets names in columns."
                )
            mask = assets_names == asset
            if np.any(mask):
                res.append(int(np.where(mask)[0][0]))
            else:
                if raise_if_string_missing:
                    raise ValueError(f"{asset} not found in {assets_names}")
                else:
                    warnings.warn(f"{asset} not found in {assets_names}", stacklevel=2)
        else:
            if asset not in asset_indices:
                raise ValueError(f"`central_assets` {asset} is not in {asset_indices}.")
            res.append(int(asset))
    return res


def format_measure(x: float, percent: bool = False) -> str:
    """Format a measure number into a user-friendly string.

    Parameters
    ----------
    x : float
        Number to format.

    percent : bool, default=False
        If this is set to True, the number is formatted in percentage.

    Returns
    -------
    formatted : str
        Formatted string.
    """
    if np.isnan(x):
        return str(x)
    if percent:
        xn = x * 100
        f = "%"
    else:
        xn = x
        f = "f"
    if xn == 0:
        n = 0
    else:
        n = optimal_rounding_decimals(xn)
    return "{value:{fmt}}".format(value=x, fmt=f".{n}{f}")


def optimal_rounding_decimals(x: float) -> int:
    """Return the optimal rounding decimal number for a user-friendly formatting.

    Parameters
    ----------
    x : float
        Number to round.

    Returns
    -------
    n : int
        Rounding decimal number.
    """
    if np.isclose(x, 0.0):
        return 2
    return min(6, max(int(-np.log10(abs(x))) + 2, 2))


def bisection(x: list[np.ndarray]) -> Iterator[list[np.ndarray, np.ndarray]]:
    """Generator to bisect a list of array.

    Parameters
    ----------
    x : list[ndarray]
        A list of array.

    Yields
    ------
    arr :  Iterator[list[ndarray, ndarray]]
        Bisected array.
    """
    for e in x:
        n = len(e)
        if n > 1:
            mid = n // 2
            yield [e[0:mid], e[mid:n]]


def fit_single_estimator(
    estimator: Any,
    X: npt.ArrayLike,
    y: npt.ArrayLike | None,
    fit_params: dict,
    indices: np.ndarray | None = None,
    axis: int = 0,
):
    """Function used to fit an estimator within a job.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.

    X : array-like of shape (n_observations, n_assets)
        The data to fit.

    y : array-like of shape (n_observations, n_targets), optional
        The target array if provided.

    fit_params : dict
        Parameters that will be passed to `estimator.fit`.

    indices : ndarray of int, optional
        Rows or columns to select from X and y.
        The default (`None`) is to select the entire data.

    axis : int, default=0
        The axis along which `X` will be sub-sampled. `axis=0` will select
        rows while `axis=1` will select columns.

    Returns
    -------
    fitted_estimator : estimator
        The fitted estimator.
    """
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_method_params(X, params=fit_params, indices=indices, axis=axis)

    X, y = safe_split(X, y, indices=indices, axis=axis)
    estimator.fit(X, y, **fit_params)
    return estimator


def fit_and_predict(
    estimator: Any,
    X: npt.ArrayLike,
    y: npt.ArrayLike | None,
    train: np.ndarray,
    test: np.ndarray | list[np.ndarray],
    fit_params: dict,
    method: str,
    column_indices: np.ndarray | None = None,
) -> npt.ArrayLike | list[npt.ArrayLike]:
    """Fit the estimator and predict values for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.

    X : array-like of shape (n_observations, n_assets)
        The data to fit.

    y : array-like of shape (n_observations, n_factors) or None
        The factor array if provided

    train : ndarray of int of shape (n_train_observations,)
        Indices of training samples.

    test : ndarray of int of shape (n_test_samples,) or list of ndarray
        Indices of test samples or list of indices.

    fit_params : dict
        Parameters that will be passed to `estimator.fit`.

    method : str
        Invokes the passed method name of the passed estimator.

    column_indices : ndarray, optional
        Indices of columns to select.
        The default (`None`) is to select all columns.

    Returns
    -------
    predictions : array-like or list of array-like
        If `test` is an array, it returns the array-like result of calling
        'estimator.method' on `test`.
        Otherwise, if `test` is a list of arrays, it returns the list of array-like
        results of calling 'estimator.method' on each test set in `test`.
    """
    fit_params = fit_params if fit_params is not None else {}
    if column_indices is not None:
        fit_params = _check_method_params(
            X, params=fit_params, indices=column_indices, axis=1
        )
    fit_params = _check_method_params(X, params=fit_params, indices=train, axis=0)

    X, y = safe_split(X, y, indices=column_indices, axis=1)
    X_train, y_train = safe_split(X, y, indices=train, axis=0)
    if y_train is None:
        estimator.fit(X_train, **fit_params)
    else:
        estimator.fit(X_train, y_train, **fit_params)
    func = getattr(estimator, method)

    if isinstance(test, list):
        predictions = []
        for t in test:
            X_test, _ = safe_split(X, indices=t, axis=0)
            predictions.append(func(X_test))
    else:
        X_test, _ = safe_split(X, indices=test, axis=0)
        predictions = func(X_test)

    return predictions


def default_asset_names(n_assets: int) -> np.ndarray:
    """Default asset names are `["x0", "x1", ..., "x(n_assets - 1)"]`.

    Parameters
    ----------
    n_assets : int
        Number of assets.

    Returns
    -------
    asset_names : ndarray of str
        Default assets names.
    """
    return np.asarray([f"x{i}" for i in range(n_assets)], dtype=object)


def deduplicate_names(names: npt.ArrayLike) -> list[str]:
    """Rename duplicated names by appending "_{duplicate_nb}" at the end.

    This function is inspired by the pandas function `_maybe_dedup_names`.

    Parameters
    ----------
    names : array-like of shape (n_names,)
        List of names.

    Returns
    -------
    names : list[str]
        Deduplicate names.
    """
    names = list(names)
    counts = {}
    for i, col in enumerate(names):
        cur_count = counts.get(col, 0)
        if cur_count > 0:
            names[i] = f"{col}_{cur_count}"
        counts[col] = cur_count + 1
    return names


def get_feature_names(X):
    """Get feature names from X.

    Support for other array containers should place its implementation here.

    Parameters
    ----------
    X : {ndarray, dataframe} of shape (n_samples, n_features)
        Array container to extract feature names.

        - pandas dataframe : The columns will be considered to be feature
          names. If the dataframe contains non-string feature names, `None` is
          returned.
        - All other array containers will return `None`.

    Returns
    -------
    names: ndarray or None
        Feature names of `X`. Unrecognized array containers will return `None`.
    """
    feature_names = None

    # extract feature names for support array containers
    if isinstance(X, pd.DataFrame):
        # Make sure we can inspect columns names from pandas, even with
        # versions too old to expose a working implementation of
        # __dataframe__.column_names() and avoid introducing any
        # additional copy.
        # TODO: remove the pandas-specific branch once the minimum supported
        # version of pandas has a working implementation of
        # __dataframe__.column_names() that is guaranteed to not introduce any
        # additional copy of the data without having to impose allow_copy=False
        # that could fail with other libraries. Note: in the longer term, we
        # could decide to instead rely on the __dataframe_namespace__ API once
        # adopted by our minimally supported pandas version.
        feature_names = np.asarray(X.columns, dtype=object)
    elif hasattr(X, "__dataframe__"):
        df_protocol = X.__dataframe__()
        feature_names = np.asarray(list(df_protocol.column_names()), dtype=object)

    if feature_names is None or len(feature_names) == 0:
        return

    types = sorted(t.__qualname__ for t in set(type(v) for v in feature_names))

    # mixed type of string and non-string is not supported
    if len(types) > 1 and "str" in types:
        raise TypeError(
            "Feature names are only supported if all input features have string names, "
            f"but your input has {types} as feature name / column name types. "
            "If you want feature names to be stored and validated, you must convert "
            "them all to strings, by using X.columns = X.columns.astype(str) for "
            "example. Otherwise you can remove feature / column names from your input "
            "data, or convert them all to a non-string data type."
        )

    # Only feature names of all strings are supported
    if len(types) == 1 and types[0] == "str":
        return feature_names
