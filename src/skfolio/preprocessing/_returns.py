"""Preprocessing module to transform X to returns."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause

from typing import Literal

import numpy as np
import pandas as pd


def prices_to_returns(
    X: pd.DataFrame,
    y: pd.DataFrame | None = None,
    log_returns: bool = False,
    nan_threshold: float = 1,
    join: Literal["left", "right", "inner", "outer", "cross"] = "outer",
    drop_inceptions_nan: bool = True,
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

    join : str, default="outer"
        The join method between `X` and `y` when `y` is provided.

    nan_threshold : float, default=1.0
        Drop observations (rows) that have a percentage of missing assets prices above
        this threshold. The default (`1.0`) is to keep all the observations.

    drop_inceptions_nan : bool, default=True
        If this is set to True, observations at the beginning are dropped if any of
        the asset values are missing, otherwise we keep the NaNs. This is useful when
        you work with a large universe of assets with different inception dates coupled
        with a pre-selection Transformer.

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
        df = X.join(y, how=join)

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
    # Drop rows according to drop_inceptions_nan
    # noinspection PyTypeChecker
    df.dropna(how="any" if drop_inceptions_nan else "all", inplace=True)
    # Drop column if all its values are missing
    df.dropna(axis=1, how="all", inplace=True)

    # returns
    all_returns = df.pct_change().iloc[1:]
    if log_returns:
        all_returns = np.log1p(all_returns)

    if y is None:
        return all_returns

    returns = all_returns[[x for x in X.columns if x in df.columns]]
    factor_returns = all_returns[[x for x in y.columns if x in df.columns]]
    return returns, factor_returns


def yields_to_returns(
    X: pd.DataFrame,
    X_maturity: pd.DataFrame,
    y: pd.DataFrame | None = None,
    y_maturity: pd.DataFrame | None = None,
    log_returns: bool = False,
    nan_threshold: float = 1,
    join: Literal["left", "right", "inner", "outer", "cross"] = "outer",
    drop_inceptions_nan: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    r"""Calculates via yields and maturity of bonds the linear or logarithmic returns.

    Return of bond is defined as:
        .. math:: R_t = \frac{\text{ending price} + \text{ending accrued interest } + \text{coupon payments between } t_0 \text{ and } t_1}{\text{beginning price} + \text{beginning accrued interest}} - 1

    However, it is often the case that one does not have coupon data, but yields and maturities are often available. In this case, we can use the following approximation:
        .. math:: R_t = \text{yield income} - \text{duration} \cdot \Delta y + \frac{1}{2} \cdot \text{convexity} \cdot (\Delta y)^2

    Where:
        - :math:`\text{yield income}_t = (1 + \text{yield}_{t-1}) ^ \text{n} - 1`
        - :math:`\text{duration}_t = \frac{1}{\text{yield}_t} \left( 1 - \frac{1}{(1 + 0.5 \cdot \text{yield}) ^ {2 \cdot \text{M}}} \right)`
        - :math:`\text{convexity}_t = \frac{2}{\text{yield}_t^2} \left( 1 - \frac{1}{(1 + 0.5 \cdot \text{yield}_t) ^ {2 \cdot \text{M}}} \right) - \frac{2 \cdot \text{M}}{\text{yield}_t \cdot (1 + 0.5 \cdot \text{yield}_t) ^ {2 \cdot \text{M} + 1}}`

    With :math:`M_{t}` Maturity at time :math:`t` and :math:`\text{n}` being the frequency of the data.

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
        The DataFrame of yields in decimal of bonds.

    X_maturity : DataFrame
        The DataFrame of bond maturities.

    y : DataFrame, optional
        The DataFrame of target or factors yields in decimal.
        If provided, it is joined with the DataFrame of yields to ensure identical
        observations.

    Y_maturity : DataFrame, optional
        The DataFrame of target or factors maturities.
        If provided, it is joined with the DataFrame of X_maturities to ensure identical
        observations.


    log_returns : bool, default=True
        If this is set to True, logarithmic returns are used instead of simple returns.

    join : str, default="outer"
        The join method between `X` and `y` when `y` is provided.

    nan_threshold : float, default=1.0
        Drop observations (rows) that have a percentage of missing assets prices above
        this threshold. The default (`1.0`) is to keep all the observations.

    drop_inceptions_nan : bool, default=True
        If this is set to True, observations at the beginning are dropped if any of
        the asset values are missing, otherwise we keep the NaNs. This is useful when
        you work with a large universe of assets with different inception dates coupled
        with a pre-selection Transformer.

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
    .. [2] "Treasury Bond Return Data Starting in 1962".
        Multidisciplinary Digital Publishing Institute.
        Laurens Swinkels (2019).
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("`X` must be a DataFrame")

    if not isinstance(X_maturity, pd.DataFrame):
        raise TypeError("`maturity` must be a DataFrame")

    if y is None:
        df = X.copy()
        df_maturity = X_maturity.copy()
    else:
        if not isinstance(y, pd.DataFrame):
            raise TypeError("`y` must be a DataFrame")
        if not isinstance(y_maturity, pd.DataFrame):
            raise TypeError("`y` must be a DataFrame")
        df = X.join(y, how=join)
        df_maturity = X_maturity.join(y_maturity, how=join)

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

    # Drop column if all its values are missing
    df.dropna(axis=1, how="all", inplace=True)

    # Infer data frequency
    frequency = pd.infer_freq(X.index)
    frequency = 1 / _frequency_to_numeric(frequency)

    # Yield income
    yield_income = (1 + df.shift()) ** frequency - 1

    # Duration
    duration = (1 / df) * (1 - 1 / (1 + 0.5 * df) ** (2 * df_maturity))

    # Convexity
    convexity = (2 / df**2) * (1 - 1 / (1 + 0.5 * df) ** (2 * df_maturity)) - (
        2 * df_maturity
    ) / (df * (1 + 0.5 * df) ** (2 * df_maturity + 1))

    all_returns = (
        yield_income
        - duration * (df - df.shift())
        + 0.5 * convexity * (df - df.shift()) ** 2
    )

    if log_returns:
        all_returns = np.log1p(all_returns)

    if y is None:
        return all_returns

    returns = all_returns[[x for x in X.columns if x in df.columns]]
    factor_returns = all_returns[[x for x in y.columns if x in df.columns]]
    return returns, factor_returns


def _frequency_to_numeric(freq: str) -> int:
    r"""Converts a frequency string to its numeric equivalent.

    The function takes a frequency string as input and returns its numeric equivalent based on a predefined dictionary of frequencies.

    Parameters
    ----------
    freq : str
        The frequency string to be converted.

    Returns
    -------
    int
        The numeric equivalent of the frequency.

    Examples
    --------
    >>> frequency_to_numeric('MS')
    12
    >>> frequency_to_numeric('W')
    52

    """
    freq_dict = {
        "B": 365,
        "C": 365,
        "D": 365,
        "W": 52,
        "MS": 12,
        "ME": 12,
        "SME": 12,
        "BME": 12,
        "QS": 4,
        "QE": 4,
        "YS": 1,
        "YE": 1,
    }

    return freq_dict.get(freq, "Unknown frequency")
