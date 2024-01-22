"""Preprocessing module to transform X to returns."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd


def prices_to_returns(
    X: pd.DataFrame,
    y: pd.DataFrame | None = None,
    log_returns: bool = False,
    nan_threshold: float = 1,
    join: str = "outer",
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    r"""Transforms a DataFrame of prices to linear or logarithmic returns.

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

    log_returns : bool, default=True
        If this is set to True, logarithmic returns are used instead of simple returns.

    join : str, default='outer
        The join method between `X` and `y` when `y` is provided.

    nan_threshold : float, default=1.0
        Drop observations (rows) that have a percentage of missing assets prices above
        this threshold. The default (`1.0`) is to keep all the observations.

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
    if not isinstance(X, pd.DataFrame):
        raise TypeError("`X` must be a DataFrame")

    if y is None:
        df = X.copy()
    else:
        if not isinstance(y, pd.DataFrame):
            raise TypeError("`y` must be a DataFrame")
        df = X.join(y, how="outer")

    n_observations, n_assets = X.shape

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
    df.ffill(inplace=True)
    # Drop rows if any of its values is missing
    df.dropna(axis=0, how="any", inplace=True)
    # Drop column if all its values are missing
    df.dropna(axis=1, how="all", inplace=True)

    # returns
    all_returns = df.pct_change().dropna()
    if log_returns:
        all_returns = np.log1p(all_returns)

    if y is None:
        return all_returns
    returns = all_returns[[x for x in X.columns if x in df.columns]]
    factor_returns = all_returns[[x for x in y.columns if x in df.columns]]
    return returns, factor_returns
