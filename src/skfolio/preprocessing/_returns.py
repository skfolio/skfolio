"""Preprocessing module to transform X to returns."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

import numpy as np
import pandas as pd

import skfolio.typing as skt


def prices_to_returns(
    X: pd.DataFrame,
    y: pd.DataFrame | None = None,
    return_type: skt.ReturnType = "linear",
    log_returns: bool = False,
    nan_threshold: float = 1,
    join: Literal["left", "right", "inner", "outer", "cross"] = "outer",
    drop_inceptions_nan: bool = True,
    fill_nan: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    r"""Transform a DataFrame of prices to linear or logarithmic returns.

    Linear returns (also called simple returns) are defined as:
        .. math:: \frac{S_{t}}{S_{t-1}} - 1

    Logarithmic returns (also called continuously compounded return) are defined as:
        .. math:: ln\Biggl(\frac{S_{t}}{S_{t-1}}\Biggr)

    With :math:`S_{t}` the asset price at time :math:`t`.

    .. warning::

        The linear returns aggregate across securities, meaning that the linear return
        of the portfolio is the weighted average of the linear returns of the
        securities. For this reason, **portfolio optimization should be performed
        using linear returns** [1]_.

        On the other hand, the logarithmic returns aggregate across time, meaning that
        the total logarithmic return over K time periods is the sum of all K
        single-period logarithmic returns.

    .. seealso::

        :ref:`data preparation <data_preparation>`

    Parameters
    ----------
    X : DataFrame
        The DataFrame of assets prices.

    y : DataFrame, optional
        The DataFrame of target or factors prices.
        If provided, it is joined with the DataFrame of prices to ensure identical
        observations.

    return_type: {'linear', 'log', 'arithmetic'}, default='linear'
        The type of returns to compute.
        - 'linear': Linear returns (simple returns): $\frac{X_t}{X_{t-1}} - 1$.
        - 'log': Logarithmic returns (continuously compounded returns): $\ln\Biggl(\frac{X_t}{X_{t-1}}\Biggr)$.
        - 'arithmetic': the difference between successive prices: $X_t - X_{t-1}$.
        
    log_returns : bool, default=True
        Deprecated since version X.X.X. If this is set to True, logarithmic returns are used instead of simple returns.

    join : str, default="outer"
        The join method between `X` and `y` when `y` is provided.

    nan_threshold : float, default=1.0
        Drop observations (rows) that have a percentage of missing assets prices above
        this threshold. The default (`1.0`) is to keep all the observations.

    drop_inceptions_nan : bool, default=True
        If set to True, observations at the beginning are dropped if any of
        the asset values are missing, otherwise we keep the NaNs. This is useful when
        you work with a large universe of assets with different inception dates coupled
        with a pre-selection Transformer.

    fill_nan : bool, default=True
        If set to True, missing prices (NaNs) are forward filled using the previous
        price. Otherwise, NaNs are kept.

    Returns
    -------
    X : DataFrame
        The DataFrame of price returns of the input `X`.

    y : DataFrame, optional
        The DataFrame of price returns of the input `y` when provided.

    References
    ----------
    .. [1]  "Linear vs. Compounded Returns - Common Pitfalls in Portfolio Management".
        GARP Risk Professional.
        Attilio Meucci (2010).
    """
    if log_returns:
        return_type = "log"
        raise DeprecationWarning(
            "`log_returns` is deprecated since version X.X.X. "
            "Please use `return_type='log'` instead."
        )

    if not isinstance(X, pd.DataFrame):
        raise TypeError("`X` must be a DataFrame")

    if y is None:
        df = X.copy()
    else:
        if not isinstance(y, pd.DataFrame):
            raise TypeError("`y` must be a DataFrame")
        df = X.join(y, how=join)

    _, n_assets = X.shape

    # Remove observations with missing X above threshold
    if nan_threshold is not None:
        nan_threshold = float(nan_threshold)
        if not 0 < nan_threshold <= 1:
            raise ValueError("`nan_threshold` must be between 0 and 1")
        count_nan = df.isna().sum(axis=1)
        to_drop = count_nan[count_nan > n_assets * nan_threshold].index
        if len(to_drop) > 0:
            df.drop(to_drop, axis=0, inplace=True)

    # Forward fill missing values
    if fill_nan:
        df.ffill(inplace=True)
    # Drop rows according to drop_inceptions_nan
    # noinspection PyTypeChecker
    df.dropna(how="any" if drop_inceptions_nan else "all", inplace=True)
    # Drop column if all its values are missing
    df.dropna(axis=1, how="all", inplace=True)

    # returns
    
    match return_type:
        case "linear":
            all_returns = df.pct_change(fill_method=None).iloc[1:]
        case "log":
            all_returns = np.log1p(df).diff().iloc[1:]
        case "arithmetic":
            all_returns = df.diff().iloc[1:]
        case _:
            raise ValueError(
                "`return_type` must be one of {'linear', 'log', 'arithmetic'}"
            )

    if y is None:
        return all_returns

    returns = all_returns[[x for x in X.columns if x in df.columns]]
    factor_returns = all_returns[[x for x in y.columns if x in df.columns]]
    return returns, factor_returns
