"""Portfolio module.
`Portfolio` is returned by the `predict` method of Optimization estimators.
"""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
import sklearn.utils.validation as skv

import skfolio.typing as skt
from skfolio._constants import (
    _MANAGEMENT_FEES,
    _PREVIOUS_WEIGHTS,
    _TRANSACTION_COSTS,
)
from skfolio.attribution import Attribution
from skfolio.measures import RiskMeasure, effective_number_assets
from skfolio.portfolio._base import _ZERO_THRESHOLD, BasePortfolio
from skfolio.typing import AnyArray, ArrayLike, FloatArray, IntArray, StrArray
from skfolio.utils.tools import (
    _get_liquidation_turnover_and_cost,
    args_names,
    cached_property_slots,
    default_asset_names,
    input_to_array,
)

if TYPE_CHECKING:
    from skfolio.prior import FactorModel


class Portfolio(BasePortfolio):
    r"""
    Portfolio class.

    `Portfolio` is returned by the `predict` method of Optimization estimators.

    By default, each observation is evaluated at the target `weights`. Portfolio
    returns are the dot product of those weights and the asset returns, minus
    transaction costs and management fees. This constant-weight convention
    (`weight_drift=False`) is consistent with the optimizer's linear portfolio return
    definition and evaluates allocation skill independently of subsequent changes in
    weights caused by relative asset returns.

    With `weight_drift=True`, the portfolio starts at the target weights and holds the
    resulting positions throughout the observation window of `X`. Position values
    change with asset returns, so portfolio weights evolve with the relative
    performance of the assets. Combined with `compounded=True`, this produces a
    compounded wealth path for evaluating realized capital growth and other
    path-dependent quantities.

    `weight_drift` changes the observation-level portfolio return series, while
    `compounded` changes how that series is accumulated. See
    :ref:`backtesting_and_evaluation`.

    Parameters
    ----------
    X : array-like of shape (n_observations, n_assets)
        Price returns of the assets.
        If `X` is a DataFrame or another array containers that implements 'columns'
        and 'index', the columns will be considered as assets names and the
        indices will be considered as observations.
        Otherwise, we use `["x0", "x1", ..., "x(n_assets - 1)"]` as asset names
        and `[0, 1, ..., n_observations]` as observations.
        `NaN` values are treated as zero returns for the portfolio return
        computation (e.g. non-investable assets, delisted assets or trading
        holidays), while the original `X` is preserved.

    weights : array-like of shape (n_assets,) | dict[str, float]
        Portfolio weights.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset weight) and `X` must be a DataFrame with assets names
        in columns.

    transaction_costs : float | dict[str, float] | array-like of shape (n_assets, ), optional
        Linear transaction costs of the assets. The Portfolio total transaction cost
        is:

        .. math:: total\_cost = \sum_{i=1}^{N} c_{i} \times |w_{i} - w\_prev_{i}|

        with :math:`c_{i}` the transaction cost of asset i, :math:`w_{i}` its weight
        and :math:`w\_prev_{i}` its previous weight (defined in `previous_weights`).
        The float :math:`total\_cost` is used in the portfolio returns:

        .. math:: ptf\_returns = R \cdot w - total\_cost

        with :math:`R` the matrix of assets returns and :math:`w` the vector of
        assets weights.

        If a float is provided, it is applied to each asset.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset weight) and `X` must be a DataFrame with assets names
        in columns.
        The default (`None`) means no transaction costs.

        .. warning::

            To be consistent with the optimization problems, the periodicity of the
            transaction costs must match the periodicity of the
            returns `X`. For example, if `X` is composed of **daily** returns,
            the `transaction_costs` need to be expressed in **daily** transaction costs.

    management_fees : float | dict[str, float] | array-like of shape (n_assets, ), optional
        Linear management fees of the assets. The Portfolio total management cost
        is:

        .. math:: total\_fee = \sum_{i=1}^{N} f_{i} \times w_{i}

        with :math:`f_{i}` the management fee of asset i and :math:`w_{i}` its weight.
        The float :math:`total\_fee` is used in the portfolio returns:

        .. math:: ptf\_returns = R \cdot w - total\_fee

        with :math:`R` the matrix of assets returns and :math:`w` the vector of
        assets weights.

        If a float is provided, it is applied to each asset.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset weight) and `X` must be a DataFrame with assets names
        in columns.
        The default (`None`) means no management fees.

        .. warning::

            To be consistent with the optimization problems, the periodicity of the
            management fees must match the periodicity of the
            returns `X`. For example, if `X` is composed of **daily** returns,
            the `management_fees` need to be expressed in **daily** fees.

    previous_weights : float | dict[str, float] | array-like of shape (n_assets, ), optional
        Previous portfolio weights.
        Previous weights are used to compute turnover and transaction costs.
        For named positions in assets absent from `X`, these calculations assume
        full liquidation. To specify their transaction costs, `transaction_costs`
        must be a single rate applied to all assets or a dictionary keyed by asset name.
        If a float is provided, it is applied to each asset.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset previous weight) and `X` must be a DataFrame with assets names
        in columns.
        The default (`None`) means no previous weights.

    name : str, optional
        Name of the portfolio.
        The default (`None`) is to use the object id.

    tag : str, optional
        Tag given to the portfolio.
        Tags are used to manipulate groups of Portfolios from a `Population`.

    fitness_measures : list[measures], optional
        List of fitness measures.
        Fitness measures are used to compute the portfolio fitness which is used to
        compute domination.
        The default (`None`) is to use the list [PerfMeasure.MEAN, RiskMeasure.VARIANCE]

    annualization_factor : float, default=252.0
        Factor used to annualize the below measures using the square-root rule:

            * Annualized Mean = Mean * factor
            * Annualized Variance = Variance * factor
            * Annualized Semi-Variance = Semi-Variance * factor
            * Annualized Standard-Deviation = Standard-Deviation * sqrt(factor)
            * Annualized Semi-Deviation = Semi-Deviation * sqrt(factor)
            * Annualized Sharpe Ratio = Sharpe Ratio * sqrt(factor)
            * Annualized Sortino Ratio = Sortino Ratio * sqrt(factor)

    risk_free_rate : float, default=0.0
        Risk-free rate. The default value is `0.0`.

    compounded : bool, default=False
        If `True`, cumulative returns are compounded.
        The default is `False`.

    weight_drift : bool, default=False
        If `True`, the portfolio starts at the target `weights` and the
        weights used for subsequent observations evolve with asset returns following
        the self-financing identity
        :math:`u_{t+1} = u_t \circ (1 + r_t) / (1 + u_t \cdot r_t)`.
        Drift accumulates over the entire window of `X`, and the implicit cash position
        :math:`1 - \sum_i w_i` earns zero. The same transaction-cost and management-fee
        formulas are used with either setting. With the default (`False`), every
        observation is evaluated at the target `weights`. This attribute is read-only.
        See :ref:`backtesting_and_evaluation`.

    sample_weight : ndarray of shape (n_observations,), optional
        Sample weights for each observation. If None, equal weights are assumed.

    min_acceptable_return : float, optional
        The minimum acceptable return used to distinguish "downside" and "upside"
        returns for the computation of lower partial moments:

            * First Lower Partial Moment
            * Semi-Variance
            * Semi-Deviation

        The default (`None`) is to use the mean.

    value_at_risk_beta : float, default=0.95
        The confidence level of the Portfolio VaR (Value At Risk) which represents
        the return on the worst (1-beta)% observations.
        The default value is `0.95`.

    entropic_risk_measure_theta : float, default=1.0
        The risk aversion level of the Portfolio Entropic Risk Measure.
        The default value is `1.0`.

    entropic_risk_measure_beta : float, default=0.95
        The confidence level of the Portfolio Entropic Risk Measure.
        The default value is `0.95`.

    cvar_beta : float, default=0.95
        The confidence level of the Portfolio CVaR (Conditional Value at Risk) which
        represents the expected VaR on the worst (1-beta)% observations.
        The default value is `0.95`.

    evar_beta : float, default=0.95
        The confidence level of the Portfolio EVaR (Entropic Value at Risk).
        The default value is `0.95`.

    drawdown_at_risk_beta : float, default=0.95
        The confidence level of the Portfolio Drawdown at Risk (DaR) which represents
        the drawdown on the worst (1-beta)% observations.
        The default value is `0.95`.

    cdar_beta : float, default=0.95
        The confidence level of the Portfolio CDaR (Conditional Drawdown at Risk) which
        represents the expected drawdown on the worst (1-beta)% observations.
        The default value is `0.95`.

    edar_beta : float, default=0.95
        The confidence level of the Portfolio EDaR (Entropic Drawdown at Risk).
        The default value is `0.95`.

    fallback_chain : list[tuple[str, str]] | None, optional
        Sequence describing the optimization fallback attempts. Each element is
        a pair `(estimator_repr, outcome)` where:

        * `estimator_repr` is the string representation of the primary
          estimator or a fallback (e.g. `"EqualWeighted()"`,
          `"previous_weights"`).
        * `outcome` is `"success"` if that step produced a valid solution,
          otherwise the stringified error message.

        For successful fits without any fallback, this is `None`. When
        fallbacks are provided and the primary fails, the chain starts with
        `(primary_repr, primary_error)` and is followed by one entry per
        fallback that was attempted, ending with the first `"success"` or the
        last error if all fail. This is set by the optimization estimator and
        propagated to the resulting portfolio.

    Attributes
    ----------
    n_observations : float
        Number of observations.

    mean : float
        Mean of the portfolio returns.

    annualized_mean : float
        Mean annualized by :math:`mean \times annualization\_factor`

    mean_absolute_deviation : float
        Mean Absolute Deviation. The deviation is the difference between the
        return and a minimum acceptable return (`min_acceptable_return`).

    first_lower_partial_moment : float
        First Lower Partial Moment. The First Lower Partial Moment is the mean of the
        returns below a minimum acceptable return (`min_acceptable_return`).

    variance : float
        Variance (Second Moment)

    annualized_variance : float
        Variance annualized by :math:`variance \times annualization\_factor`

    semi_variance : float
        Semi-variance (Second Lower Partial Moment).
        The semi-variance is the variance of the returns below a minimum acceptable
        return (`min_acceptable_return`).

    annualized_semi_variance : float
        Semi-variance annualized by
        :math:`semi\_variance \times annualization\_factor`

    standard_deviation : float
        Standard Deviation (Square Root of the Second Moment).

    annualized_standard_deviation : float
        Standard Deviation annualized by
        :math:`standard\_deviation \times \sqrt{annualization\_factor}`

    semi_deviation : float
        Semi-deviation (Square Root of the Second Lower Partial Moment).
        The Semi Standard Deviation is the Standard Deviation of the returns below a
        minimum acceptable return (`min_acceptable_return`).

    annualized_semi_deviation : float
        Semi-deviation annualized by
        :math:`semi\_deviation \times \sqrt{annualization\_factor}`

    skew : float
        Skew. The Skew is a measure of the lopsidedness of the distribution.
        A symmetric distribution have a Skew of zero.
        Higher Skew corresponds to longer right tail.

    kurtosis : float
        Kurtosis. It is a measure of the heaviness of the tail of the distribution.
        Higher Kurtosis corresponds to greater extremity of deviations (fat tails).

    fourth_central_moment : float
       Fourth Central Moment.

    fourth_lower_partial_moment : float
        Fourth Lower Partial Moment. It is a measure of the heaviness of the downside
        tail of the returns below a minimum acceptable return (`min_acceptable_return`).
        Higher Fourth Lower Partial Moment corresponds to greater extremity of downside
        deviations (downside fat tail).

    worst_realization : float
        Worst Realization which is the worst return.

    value_at_risk : float
        Historical VaR (Value at Risk).
        The VaR is the maximum loss at a given confidence level (`value_at_risk_beta`).

    cvar : float
        Historical CVaR (Conditional Value at Risk). The CVaR (or Tail VaR) represents
        the mean shortfall at a specified confidence level (`cvar_beta`).

    entropic_risk_measure : float
        Historical Entropic Risk Measure. It is a risk measure which depends on the
        risk aversion defined by the investor (`entropic_risk_measure_theta`) through
        the exponential utility function at a given confidence level
        (`entropic_risk_measure_beta`).

    evar : float
         Historical EVaR (Entropic Value at Risk). It is a coherent risk measure which
         is an upper bound for the VaR and the CVaR, obtained from the Chernoff
         inequality at a given confidence level (`evar_beta`). The EVaR can be
         represented by using the concept of relative entropy.

    drawdown_at_risk : float
        Historical Drawdown at Risk. It is the maximum drawdown at a given
        confidence level (`drawdown_at_risk_beta`).

    cdar : float
        Historical CDaR (Conditional Drawdown at Risk) at a given confidence level
        (`cdar_beta`).

    max_drawdown : float
        Maximum Drawdown.

    average_drawdown : float
        Average Drawdown.

    edar : float
        EDaR (Entropic Drawdown at Risk). It is a coherent risk measure which is an
        upper bound for the Drawdown at Risk and the CDaR, obtained from the Chernoff
        inequality at a given confidence level (`edar_beta`). The EDaR can be
        represented by using the concept of relative entropy.

    ulcer_index : float
        Ulcer Index

    gini_mean_difference : float
        Gini Mean Difference (GMD). It is the expected absolute difference between two
        realizations. The GMD is a superior measure of variability  for non-normal
        distribution than the variance. It can be used to form necessary conditions
        for second-degree stochastic dominance, while the variance cannot.

    mean_absolute_deviation_ratio : float
        Mean Absolute Deviation ratio.
        It is the excess mean (mean - risk_free_rate) divided by the MaD.

    first_lower_partial_moment_ratio : float
        First Lower Partial Moment ratio.
        It is the excess mean (mean - risk_free_rate) divided by the First Lower
        Partial Moment.

    sharpe_ratio : float
        Sharpe ratio.
        It is the excess mean (mean - risk_free_rate) divided by the standard-deviation.

    annualized_sharpe_ratio : float
        Sharpe ratio annualized by
        :math:`sharpe\_ratio \times \sqrt{annualization\_factor}`.

    sortino_ratio : float
        Sortino ratio.
        It is the excess mean (mean - risk_free_rate) divided by the semi
        standard-deviation.

    annualized_sortino_ratio : float
        Sortino ratio annualized by
        :math:`sortino\_ratio \times \sqrt{annualization\_factor}`.

    value_at_risk_ratio : float
        VaR ratio.
        It is the excess mean (mean - risk_free_rate) divided by the Value at Risk
        (VaR).

    cvar_ratio : float
        CVaR ratio.
        It is the excess mean (mean - risk_free_rate) divided by the Conditional Value
        at Risk (CVaR).

    entropic_risk_measure_ratio : float
        Entropic risk measure ratio.
        It is the excess mean (mean - risk_free_rate) divided by the Entropic risk
        measure.

    evar_ratio : float
        EVaR ratio.
        It is the excess mean (mean - risk_free_rate) divided by the EVaR (Entropic
        Value at Risk).

    worst_realization_ratio : float
        Worst Realization ratio.
        It is the excess mean (mean - risk_free_rate) divided by the Worst Realization
        (worst return).

    drawdown_at_risk_ratio : float
        Drawdown at Risk ratio.
        It is the excess mean (mean - risk_free_rate) divided by the drawdown at
        risk.

    cdar_ratio : float
        CDaR ratio.
        It is the excess mean (mean - risk_free_rate) divided by the CDaR (conditional
        drawdown at risk).

    calmar_ratio : float
        Calmar ratio.
        It is the excess mean (mean - risk_free_rate) divided by the Maximum Drawdown.

    average_drawdown_ratio : float
        Average Drawdown ratio.
        It is the excess mean (mean - risk_free_rate) divided by the Average Drawdown.

    edar_ratio : float
        EDaR ratio.
        It is the excess mean (mean - risk_free_rate) divided by the EDaR (Entropic
        Drawdown at Risk).

    ulcer_index_ratio : float
        Ulcer Index ratio.
        It is the excess mean (mean - risk_free_rate) divided by the Ulcer Index.

    gini_mean_difference_ratio : float
        Gini Mean Difference ratio.
        It is the excess mean (mean - risk_free_rate) divided by the Gini Mean
        Difference.

    ending_weights : ndarray of shape (n_assets,)
        Asset weights immediately after the final observation. With
        `weight_drift=False`, they equal the target `weights`. With
        `weight_drift=True`, they reflect the effect of asset returns through the final
        observation. They are calculated before transaction costs and management fees.
        In a sequential evaluation, the `ending_weights` of a successful `Portfolio`
        are used as `previous_weights` for the next optimization. A `FailedPortfolio`
        contains only NaN ending weights.

    turnover : float
        Total absolute weight change, assuming full liquidation of positions in
        assets absent from `X`.
    """

    _read_only_attrs: ClassVar[set] = BasePortfolio._read_only_attrs.copy()
    _read_only_attrs.update(
        {
            "X",
            "assets",
            "weights",
            _PREVIOUS_WEIGHTS,
            _TRANSACTION_COSTS,
            _MANAGEMENT_FEES,
            "n_assets",
            "total_cost",
            "total_fee",
            "weight_drift",
            "ending_weights",
        }
    )

    __slots__ = {
        # read-only
        "X",
        "weights",
        _PREVIOUS_WEIGHTS,
        _TRANSACTION_COSTS,
        _MANAGEMENT_FEES,
        "assets",
        "n_assets",
        "total_cost",
        "total_fee",
        "weight_drift",
        "ending_weights",
        # custom getter (read-only and cached)
        "_nonzero_assets",
        "_nonzero_assets_index",
        # private state
        "_original_named_inputs",
        "_liquidation_turnover",
        # private cache
        "_weights_path",
        # read-write
        "fallback_chain",
    }

    def __init__(
        self,
        X: ArrayLike,
        weights: skt.MultiInput | None,
        previous_weights: skt.MultiInput | None = None,
        transaction_costs: skt.MultiInput | None = None,
        management_fees: skt.MultiInput | None = None,
        risk_free_rate: float = 0,
        name: str | None = None,
        tag: str | None = None,
        annualization_factor: float | None = None,
        fitness_measures: list[skt.Measure] | None = None,
        compounded: bool = False,
        weight_drift: bool = False,
        sample_weight: FloatArray | None = None,
        min_acceptable_return: float | None = None,
        value_at_risk_beta: float = 0.95,
        entropic_risk_measure_theta: float = 1,
        entropic_risk_measure_beta: float = 0.95,
        cvar_beta: float = 0.95,
        evar_beta: float = 0.95,
        drawdown_at_risk_beta: float = 0.95,
        cdar_beta: float = 0.95,
        edar_beta: float = 0.95,
        fallback_chain: list[tuple[str, str]] | None = None,
        **kwargs,
    ):
        weights_provided = weights is not None
        rets = _to_numpy_returns(X) if weights_provided else None
        # extract assets names from X
        assets = None
        observations = None
        if hasattr(X, "columns"):
            assets = np.asarray(X.columns, dtype=object)
            observations = np.asarray(X.index)

        shape = rets.shape if weights_provided else np.shape(X)
        if len(shape) != 2:
            raise ValueError("`X` must be a 2D array-like")

        n_observations, n_assets = shape

        # Preserve excluded assets and their cost rates when reconstructing a portfolio.
        original_named_inputs = {}
        if isinstance(previous_weights, dict):
            original_named_inputs[_PREVIOUS_WEIGHTS] = previous_weights.copy()
        if isinstance(transaction_costs, dict):
            original_named_inputs[_TRANSACTION_COSTS] = transaction_costs.copy()

        liquidation_turnover = 0.0
        liquidation_cost = 0.0
        if not weights_provided:
            weights = np.full(n_assets, np.nan)
        else:
            liquidation_turnover, liquidation_cost = _get_liquidation_turnover_and_cost(
                previous_weights=previous_weights,
                transaction_costs=transaction_costs,
                assets_names=assets,
            )
            weights = input_to_array(
                items=weights,
                n_assets=n_assets,
                fill_value=0,
                dim=1,
                assets_names=assets,
                name="weights",
            )

        if previous_weights is None:
            previous_weights = np.zeros(n_assets)
        else:
            previous_weights = input_to_array(
                items=previous_weights,
                n_assets=n_assets,
                fill_value=0,
                dim=1,
                assets_names=assets,
                name=_PREVIOUS_WEIGHTS,
            )

        if transaction_costs is None:
            transaction_costs = 0
        elif not np.isscalar(transaction_costs):
            transaction_costs = input_to_array(
                items=transaction_costs,
                n_assets=n_assets,
                fill_value=0,
                dim=1,
                assets_names=assets,
                name=_TRANSACTION_COSTS,
            )

        if management_fees is None:
            management_fees = 0
        elif not np.isscalar(management_fees):
            management_fees = input_to_array(
                items=management_fees,
                n_assets=n_assets,
                fill_value=0,
                dim=1,
                assets_names=assets,
                name=_MANAGEMENT_FEES,
            )

        # Default observations and assets if X is not a DataFrame
        if observations is None:
            observations = np.arange(n_observations)

        if assets is None or len(assets) == 0:
            assets = default_asset_names(n_assets=n_assets)

        # Computing portfolio returns
        if np.isscalar(transaction_costs) and transaction_costs == 0:
            total_cost = 0
        else:
            total_cost = (transaction_costs * abs(previous_weights - weights)).sum()
        total_cost += liquidation_cost

        if np.isscalar(management_fees) and management_fees == 0:
            total_fee = 0
        else:
            total_fee = (management_fees * weights).sum()

        ending_weights = weights.copy()
        if weights_provided:
            rets_clean = _nan_to_zero(rets)
            if weight_drift and n_observations > 0:
                position_values, wealth = _position_values_and_wealth(
                    returns=rets_clean,
                    weights=weights,
                    observations=observations,
                )
                previous_wealth = np.concatenate(([1.0], wealth[:-1]))
                returns = wealth / previous_wealth - 1 - total_cost - total_fee
                ending_weights = position_values[-1] / wealth[-1]
            else:
                returns = weights @ rets_clean.T - total_cost - total_fee
        else:
            returns = np.full(n_observations, np.nan)

        super().__init__(
            returns=returns,
            observations=observations,
            name=name,
            tag=tag,
            fitness_measures=fitness_measures,
            compounded=compounded,
            sample_weight=sample_weight,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor,
            min_acceptable_return=min_acceptable_return,
            value_at_risk_beta=value_at_risk_beta,
            cvar_beta=cvar_beta,
            entropic_risk_measure_theta=entropic_risk_measure_theta,
            entropic_risk_measure_beta=entropic_risk_measure_beta,
            evar_beta=evar_beta,
            drawdown_at_risk_beta=drawdown_at_risk_beta,
            cdar_beta=cdar_beta,
            edar_beta=edar_beta,
            **kwargs,
        )
        self._loaded = False
        self._original_named_inputs = original_named_inputs
        self._liquidation_turnover = liquidation_turnover
        # We save the original array-like object and not the numpy copy for improved
        # memory
        self.X = X
        self.assets = assets
        self.n_assets = n_assets
        self.weights = weights
        self.transaction_costs = transaction_costs
        self.management_fees = management_fees
        self.previous_weights = previous_weights
        self.total_cost = total_cost
        self.total_fee = total_fee
        self.weight_drift = weight_drift
        self.ending_weights = ending_weights
        # Keep attribute name aligned with Optimization API (fallback_chain_)
        self.fallback_chain = fallback_chain
        self._loaded = True
        self._weights_path = None

    @property
    def _is_failed_portfolio(self) -> bool:
        return self.__class__.__name__ == "FailedPortfolio"

    def _get_init_params(self) -> dict:
        params = super()._get_init_params()
        params.update(self._original_named_inputs)
        return params

    def _check_compatible_parameters(self, other: Portfolio) -> None:
        """Check that portfolios differ only in weights, name or tag."""
        assets = set(self.assets)
        for name in args_names(self.__init__):
            if name in ("weights", "name", "tag"):
                continue
            if not np.array_equal(getattr(self, name), getattr(other, name)):
                raise ValueError(
                    f"Cannot combine two Portfolios with different `{name}`"
                )
        # Aligned arrays do not include positions and rates outside X.
        for name in (_PREVIOUS_WEIGHTS, _TRANSACTION_COSTS):
            named = self._original_named_inputs.get(name, {})
            other_named = other._original_named_inputs.get(name, {})
            excluded_assets = (named.keys() | other_named.keys()) - assets
            if any(
                named.get(asset, 0) != other_named.get(asset, 0)
                for asset in excluded_assets
            ):
                raise ValueError(
                    f"Cannot combine two Portfolios with different `{name}`"
                )

    def __neg__(self):
        if self._is_failed_portfolio:
            return self.copy()
        args = self._get_init_params()
        args["weights"] = -self.weights
        return self.__class__(**args)

    def __abs__(self):
        if self._is_failed_portfolio:
            return self.copy()
        args = self._get_init_params()
        args["weights"] = np.abs(self.weights)
        return self.__class__(**args)

    def __round__(self, n: int):
        if self._is_failed_portfolio:
            return self.copy()
        args = self._get_init_params()
        args["weights"] = np.round(self.weights, n)
        return self.__class__(**args)

    def __add__(self, other):
        if not isinstance(other, Portfolio):
            raise TypeError(
                f"Cannot add a Portfolio with an object of type {type(other)}"
            )
        if self._is_failed_portfolio:
            return self.copy()
        if other._is_failed_portfolio:
            return other.copy()
        self._check_compatible_parameters(other=other)
        args = self._get_init_params()
        args["weights"] = self.weights + other.weights
        return self.__class__(**args)

    def __sub__(self, other):
        if not isinstance(other, Portfolio):
            raise TypeError(
                f"Cannot add a Portfolio with an object of type {type(other)}"
            )
        if self._is_failed_portfolio:
            return self.copy()
        if other._is_failed_portfolio:
            return other.copy()
        self._check_compatible_parameters(other=other)
        args = self._get_init_params()
        args["weights"] = self.weights - other.weights
        return self.__class__(**args)

    def __mul__(self, other: numbers.Number):
        if not isinstance(other, numbers.Number):
            raise TypeError(
                "Portfolio can only be multiplied by a number, but received a"
                f" {type(other)}"
            )
        if self._is_failed_portfolio:
            return self.copy()
        args = self._get_init_params()
        args["weights"] = other * self.weights
        return self.__class__(**args)

    __rmul__ = __mul__

    def __floordiv__(self, other: numbers.Number):
        if not isinstance(other, numbers.Number):
            raise TypeError(
                "Portfolio can only be floor divided by a number, but received a"
                f" {type(other)}"
            )
        if self._is_failed_portfolio:
            return self.copy()
        args = self._get_init_params()
        args["weights"] = np.floor_divide(self.weights, other)
        return self.__class__(**args)

    def __truediv__(self, other: numbers.Number):
        if not isinstance(other, numbers.Number):
            raise TypeError(
                "Portfolio can only be divided by a number, but received a"
                f" {type(other)}"
            )
        if self._is_failed_portfolio:
            return self.copy()
        args = self._get_init_params()
        args["weights"] = self.weights / other
        return self.__class__(**args)

    # Custom attribute getter (read-only and cached)
    @cached_property_slots
    def nonzero_assets(self) -> StrArray:
        """Invested asset :math:`abs(weights) > 0.001%`."""
        return self.assets[self.nonzero_assets_index]

    @cached_property_slots
    def nonzero_assets_index(self) -> IntArray:
        """Indices of invested asset :math:`abs(weights) > 0.001%`."""
        return np.flatnonzero(
            np.isnan(self.weights) | (abs(self.weights) > _ZERO_THRESHOLD)
        )

    @property
    def composition(self) -> pd.DataFrame:
        """DataFrame of portfolio composition (weights). Rows with zero weights are
        filtered out. Use `weights_dict` to access all weights, including zeros.
        """
        weights = self.weights[self.nonzero_assets_index]
        df = pd.DataFrame({"asset": self.nonzero_assets, "weight": weights})
        df.sort_values(by="weight", ascending=False, inplace=True)
        df.rename(columns={"weight": self.name}, inplace=True)
        df.set_index("asset", inplace=True)
        return df

    @property
    def weights_dict(self) -> dict[str, float]:
        """Dict mapping asset name to weight; includes zeros."""
        return {
            asset: float(weight)
            for asset, weight in zip(self.assets, self.weights, strict=True)
        }

    @property
    def previous_weights_dict(self) -> dict[str, float]:
        """Dict mapping asset name to previous weight; includes zeros."""
        return {
            asset: float(weight)
            for asset, weight in zip(self.assets, self.previous_weights, strict=True)
        }

    @property
    def ending_weights_dict(self) -> dict[str, float]:
        """Dict mapping asset name to ending weight; includes zeros."""
        return {
            asset: float(weight)
            for asset, weight in zip(self.assets, self.ending_weights, strict=True)
        }

    @property
    def turnover(self) -> float:
        """Total absolute weight traded at the start of the period.

        In a sequential evaluation, `previous_weights` come from the last successful
        Portfolio. With `weight_drift=False`, target turnover compares successive
        target allocations. With `weight_drift=True`, executed turnover compares the
        previous period's ending weights with the new target allocation. When
        `previous_weights` is None, it defaults to zero. Turnover includes the full
        absolute weight of positions in assets absent from `X`.
        """
        if self._is_failed_portfolio:
            return np.nan
        return (
            float(np.abs(self.weights - self.previous_weights).sum())
            + self._liquidation_turnover
        )

    def _get_weights_path(self) -> FloatArray:
        """Return the portfolio's weight path across the observation window.

        Returns
        -------
        weights_path : ndarray of shape (n_observations, n_assets)
            Row `t` contains the asset weights at the start of observation `t`. The
            first row contains the target `weights`, and each subsequent row reflects
            asset returns from the preceding observations. `ending_weights` contains
            the weights immediately after the final observation. The matrix is built
            on first use and cached in `_weights_path`.
        """
        if self._weights_path is not None:
            return self._weights_path

        if self._is_failed_portfolio:
            path = np.full((self.n_observations, self.n_assets), np.nan)
        elif self.n_observations == 0:
            path = np.empty((0, self.n_assets))
        else:
            position_values, wealth = _position_values_and_wealth(
                returns=_nan_to_zero(_to_numpy_returns(self.X)),
                weights=self.weights,
                observations=self.observations,
            )
            previous_values = np.vstack((self.weights, position_values[:-1]))
            previous_wealth = np.concatenate(([1.0], wealth[:-1]))
            path = previous_values / previous_wealth[:, None]
        self._weights_path = path
        return path

    @property
    def weights_per_observation(self) -> pd.DataFrame:
        """DataFrame of asset weights at the start of each observation.

        With `weight_drift=False`, every row contains the target `weights`. With
        `weight_drift=True`, each row incorporates the effect of preceding asset
        returns. `ending_weights` contains the weights immediately after the final
        observation.
        """
        idx = self.nonzero_assets_index
        assets = self.assets[idx]
        if self.weight_drift:
            weights = self._get_weights_path()[:, idx]
        else:
            weights = np.ones((len(self.observations), len(assets))) * self.weights[idx]
        df = pd.DataFrame(weights, index=self.observations, columns=assets)
        return df

    @property
    def diversification(self) -> float:
        """Weighted average of volatility divided by the portfolio volatility."""
        if self._is_failed_portfolio:
            return np.nan
        rets = _to_numpy_returns(self.X)
        return self.weights @ np.std(rets, axis=0) / self.standard_deviation

    @property
    def sric(self) -> float:
        """Sharpe Ratio Information Criterion (SRIC).

        It is an unbiased estimator of the Sharpe Ratio adjusting for both sources of
        bias which are noise fit and estimation error [1]_.

        References
        ----------
        .. [1]  "Noise Fit, Estimation Error and a Sharpe Information Criterion",
            Dirk Paulsen (2019)
        """
        return self.sharpe_ratio - self.n_assets / (
            self.n_observations * self.sharpe_ratio
        )

    @property
    def effective_number_assets(self) -> float:
        r"""Computes the effective number of assets, defined as the inverse of the
        Herfindahl index.

        .. math:: N_{eff} = \frac{1}{\Vert w \Vert_{2}^{2}}

        It quantifies portfolio concentration, with a higher value indicating a more
        diversified portfolio.

        Returns
        -------
        value : float
            Effective number of assets.

        References
        ----------
        .. [1] "Banking and Financial Institutions Law in a Nutshell".
            Lovett, William Anthony (1988)
        """
        return effective_number_assets(weights=self.weights)

    # Public methods
    def expected_returns_from_assets(
        self, assets_expected_returns: FloatArray
    ) -> float:
        """Compute the portfolio expected return from expected asset returns,
        weights, management costs and transaction fees.

        Parameters
        ----------
        assets_expected_returns : ndarray of shape (n_assets,)
            The vector of expected asset returns.

        Returns
        -------
        value : float
            The portfolio expected return.
        """
        return (
            self.weights @ assets_expected_returns.T - self.total_cost - self.total_fee
        )

    def variance_from_assets(self, assets_covariance: FloatArray) -> float:
        """Compute the Portfolio variance expectation from the assets covariance and
        weights.

        Parameters
        ----------
        assets_covariance : ndarray of shape (n_assets,n_assets)
            The matrix of assets covariance expectation.

        Returns
        -------
        value : float
            The Portfolio variance from the assets covariance.
        """
        return float(self.weights @ assets_covariance @ self.weights.T)

    def contribution(
        self, measure: skt.Measure, spacing: float | None = None, to_df: bool = False
    ) -> FloatArray | pd.DataFrame:
        r"""Compute the contribution of each asset to a given measure.

        With `weight_drift=True`, the contributions are finite-difference sensitivities
        to the target weights. Because drifted returns are nonlinear in the target
        weights, the contributions are not guaranteed to sum exactly to the measure.

        Parameters
        ----------
        measure : Measure
            The measure used for the contribution computation.

        spacing : float, optional
            Spacing "h" of the finite difference:
            :math:`contribution(wi)= \frac{measure(wi-h) - measure(wi+h)}{2h}`

        to_df : bool, default=False
            If set to True, a DataFrame with asset names in index is returned,
            otherwise a numpy array is returned. When a DataFrame is returned, the
            values are sorted in descending order and assets with zero weights are
            removed.

        Returns
        -------
        values : numpy array of shape (n_assets,) or DataFrame
            The measure contribution of each asset.
        """
        if self._is_failed_portfolio:
            assets = self.assets
            contribution = np.full(len(assets), np.nan)
        else:
            if spacing is None:
                if measure in [
                    RiskMeasure.MAX_DRAWDOWN,
                    RiskMeasure.AVERAGE_DRAWDOWN,
                    RiskMeasure.CDAR,
                    RiskMeasure.EDAR,
                ]:
                    spacing = 1e-1
                else:
                    spacing = 1e-5
            args = self._get_init_params()
            args.pop("weights")

            contribution, assets = _compute_contribution(
                args=args,
                weights=self.weights,
                assets=self.assets,
                measure=measure,
                h=spacing,
                drop_zero_weights=to_df,
            )

        if not to_df:
            return np.array(contribution)
        df = pd.DataFrame(contribution, index=assets, columns=[self.name])
        df.sort_values(by=self.name, ascending=False, inplace=True)
        return df

    def summary(self, formatted: bool = True) -> pd.Series:
        """Portfolio summary of all its measures.

        Parameters
        ----------
        formatted : bool, default=True
            If this is set to True, the measures are formatted into rounded string
            with units.

        Returns
        -------
        summary : series
            Portfolio summary.
        """
        df = super().summary(formatted=formatted)
        assets_number = self.n_assets
        effective_nb_assets = self.effective_number_assets
        if formatted:
            assets_number = str(assets_number)
            effective_nb_assets = str(effective_nb_assets)
        df["Effective Number of Assets"] = effective_nb_assets
        df["Assets Number"] = assets_number
        return df

    def get_weight(self, asset: str) -> float:
        """Get the weight of a given asset.

        Parameters
        ----------
        asset : str
            Name of the asset.

        Returns
        -------
        weight : float
            Weight of the asset.
        """
        try:
            return self.weights[np.where(self.assets == asset)[0][0]]
        except IndexError:
            raise IndexError("{asset} is not a valid asset name.") from None

    def predicted_attribution(
        self,
        factor_model: FactorModel,
        compute_asset_breakdowns: bool = True,
    ) -> Attribution:
        r"""Ex-ante (predicted) factor risk and performance attribution.

        Decomposes the portfolio's predicted risk and expected return into contributions
        from individual factors and an idiosyncratic component using the factor model's
        latest forecast estimates (`loading_matrix`, `factor_covariance`,
        `idio_covariance`, `factor_mu`, `idio_mu`).

        The annualization scaling uses `self.annualization_factor`.

        Predicted attribution uses only these latest forecast estimates, so no
        observation alignment is required. The `factor_model` may therefore cover a
        different observation window than the portfolio.

        The portfolio may hold a subset of the assets covered by the factor model and
        weights are zero-filled for missing assets.

        See :func:`~skfolio.attribution.predicted_factor_attribution`
        for the full mathematical description.

        Parameters
        ----------
        factor_model : FactorModel
            Factor model whose latest forecast estimates are used. Every asset in
            `self.assets` must appear in `factor_model.asset_names`.

        compute_asset_breakdowns : bool, default=True
            If `True`, compute per-asset systematic/idiosyncratic decomposition. Set to
            `False` for faster computation when only portfolio-level results are needed.

        Returns
        -------
        attribution : Attribution
            Component-level, factor-level, and optionally asset-level attribution
            results.

        Raises
        ------
        ValueError
            If the portfolio is a failed portfolio or if it holds assets not covered by
            the factor model.
        """
        if self._is_failed_portfolio:
            raise ValueError("Cannot compute factor attribution on a failed portfolio.")
        aligned_weights = _align_weights(
            self.weights, self.assets, factor_model.asset_names
        )
        return factor_model.predicted_attribution(
            weights=aligned_weights,
            annualization_factor=self.annualization_factor,
            compute_asset_breakdowns=compute_asset_breakdowns,
        )

    def realized_attribution(
        self,
        factor_model: FactorModel,
        compute_asset_breakdowns: bool = True,
        compute_uncertainty: bool = True,
    ) -> Attribution:
        r"""Realized (ex-post) factor risk and performance attribution.

        Decomposes the portfolio's realized risk and return into contributions from
        individual factors and an idiosyncratic component using actual historical factor
        returns, exposures, and residuals.

        The annualization scaling uses `self.annualization_factor`.

        Realized attribution uses the target weights when `weight_drift=False` and the
        weights held during each observation when `weight_drift=True`.

        Realized attribution is computed on the overlapping observation window between
        the portfolio and the factor model. Portfolio observations outside the factor
        model window, commonly caused by factor-model warmup or exposure lag, are
        excluded. Missing portfolio observations inside the overlapping window raise
        `ValueError`. Time-varying exposures follow the as-of indexing convention
        described in
        :func:`~skfolio.attribution.realized_factor_attribution`:
        when `exposure_lag > 0`, exposures known at observation
        :math:`t-\ell` are aligned with returns at observation :math:`t`.

        The portfolio may hold a subset of the assets covered by the factor model and
        weights are zero-filled for missing assets.

        See :func:`~skfolio.attribution.realized_factor_attribution`
        for the full mathematical description.

        Parameters
        ----------
        factor_model : FactorModel
            Factor model containing time-varying fields (`factor_returns`, `exposures`,
            `idio_returns`) that overlap with the portfolio's observation period. Every
            asset in `self.assets` must appear in `factor_model.asset_names`.

        compute_asset_breakdowns : bool, default=True
            If `True`, compute per-asset systematic/idiosyncratic attribution. Set to
            `False` for faster computation when only portfolio-level results are needed.

        compute_uncertainty : bool, default=True
            If `True`, compute attribution uncertainty (standard errors on the factor
            and idiosyncratic mean-return split).

        Returns
        -------
        attribution : Attribution
            Component-level, factor-level, and optionally asset-level attribution
            results.

        Raises
        ------
        ValueError
            If the portfolio is a failed portfolio, if it holds assets not covered by
            the factor model, if no portfolio observations overlap with the factor model
            or if portfolio observations are missing inside the overlapping window.
        """
        aligned_weights, portfolio_returns, aligned_factor_model = (
            _prepare_realized_attribution_inputs(self, factor_model)
        )
        return aligned_factor_model.realized_attribution(
            weights=aligned_weights,
            portfolio_returns=portfolio_returns,
            annualization_factor=self.annualization_factor,
            compute_asset_breakdowns=compute_asset_breakdowns,
            compute_uncertainty=compute_uncertainty,
        )

    def rolling_realized_attribution(
        self,
        factor_model: FactorModel,
        window_size: int = 60,
        step: int = 21,
        compute_asset_breakdowns: bool = True,
        compute_asset_factor_contribs: bool = False,
        compute_uncertainty: bool = True,
    ) -> Attribution:
        r"""Rolling realized (ex-post) factor risk and performance attribution.

        Computes :func:`~skfolio.attribution.realized_factor_attribution`
        over rolling windows, returning an :class:`~skfolio.attribution.Attribution`
        where all numeric fields carry an additional leading dimension for the number
        of windows.

        Rolling realized attribution is computed on the overlapping observation window
        between the portfolio and the factor model. Portfolio observations outside the
        factor model window, commonly caused by factor-model warmup or exposure lag, are
        excluded. Missing portfolio observations inside the overlapping window raise
        `ValueError`. Time-varying exposures follow the as-of indexing convention
        described in
        :func:`~skfolio.attribution.rolling_realized_factor_attribution`.

        Each rolling window uses the target weights when `weight_drift=False` and the
        weights held during its observations when `weight_drift=True`.

        The portfolio may hold a subset of the assets covered by the factor model and
        weights are zero-filled for missing assets.

        See :func:`~skfolio.attribution.rolling_realized_factor_attribution`
        for the full mathematical description.

        Parameters
        ----------
        factor_model : FactorModel
            Factor model containing time-varying fields that overlap with the
            portfolio's observation period.

        window_size : int, default=60
            Number of effective return periods in each rolling window.

        step : int, default=21
            Number of observations to advance between consecutive windows. The default
            of 21 produces approximately monthly output for daily data.

        compute_asset_breakdowns : bool, default=True
            If `True`, compute per-asset attribution for each window.

        compute_asset_factor_contribs : bool, default=False
            If `True`, compute asset-factor matrix for each window.

        compute_uncertainty : bool, default=True
            If `True`, compute per-window attribution uncertainty.

        Returns
        -------
        attribution : Attribution
            Rolling attribution results.

        Raises
        ------
        ValueError
            If the portfolio is a failed portfolio, if it holds assets not covered by
            the factor model, if no portfolio observations overlap with the factor model
            or if `window_size` exceeds the number of overlapping observations.
        """
        aligned_weights, portfolio_returns, aligned_factor_model = (
            _prepare_realized_attribution_inputs(self, factor_model)
        )
        return aligned_factor_model.rolling_realized_attribution(
            weights=aligned_weights,
            portfolio_returns=portfolio_returns,
            annualization_factor=self.annualization_factor,
            window_size=window_size,
            step=step,
            compute_asset_breakdowns=compute_asset_breakdowns,
            compute_asset_factor_contribs=compute_asset_factor_contribs,
            compute_uncertainty=compute_uncertainty,
        )


def _to_numpy_returns(X: ArrayLike) -> FloatArray:
    """Convert real numeric returns to float64, preserving missing values."""
    if isinstance(X, pd.DataFrame) and all(dtype.kind in "biuf" for dtype in X.dtypes):
        # Convert real pandas dtypes directly, avoiding nullable object arrays.
        # Leave other dtypes to check_array so complex values are not cast away.
        X = X.to_numpy(dtype=float, na_value=np.nan)
    return skv.check_array(
        X,
        dtype=float,
        ensure_all_finite="allow-nan",
        ensure_min_samples=0,
        ensure_min_features=0,
    )


def _nan_to_zero(returns: FloatArray) -> FloatArray:
    """Replace NaN asset returns by zero, returning the input when it has no NaN."""
    mask = np.isnan(returns)
    if mask.any():
        returns = returns.copy()
        returns[mask] = 0.0
    return returns


def _position_values_and_wealth(
    returns: FloatArray,
    weights: FloatArray,
    observations: AnyArray,
) -> tuple[FloatArray, FloatArray]:
    """Position values and wealth of the drifted weights, starting from unit wealth.

    Each position grows with its own asset return and is not rebalanced within the
    window. Both outputs are gross of transaction costs and management fees. They serve
    to derive the drifted weights `position_values / wealth` and the single-period
    returns `wealth[t] / wealth[t - 1] - 1`, which are both scale-free.

    Parameters
    ----------
    returns : ndarray of shape (n_observations, n_assets)
        Asset returns without NaN.

    weights : ndarray of shape (n_assets,)
        Weights held on the first observation. The implicit cash position
        `1 - weights.sum()` earns zero.

    observations : ndarray of shape (n_observations,)
        Observation labels, used in the error message.

    Returns
    -------
    position_values : ndarray of shape (n_observations, n_assets)
        Value of each position at the end of each observation.

    wealth : ndarray of shape (n_observations,)
        Portfolio wealth at the end of each observation, positions plus cash.

    Raises
    ------
    ValueError
        If the wealth is non-positive at some observation, in which case the drifted
        weights are undefined from that observation on.
    """
    position_values = np.cumprod(1 + returns, axis=0) * weights
    cash = 1 - weights.sum()
    wealth = position_values.sum(axis=1) + cash
    non_positive = np.flatnonzero(wealth <= 0)
    if non_positive.size:
        raise ValueError(
            "The portfolio wealth is non-positive at observation "
            f"{observations[non_positive[0]]!r}, so the drifted weights are undefined "
            "from that observation on."
        )
    return position_values, wealth


def _align_weights(
    weights: FloatArray,
    portfolio_assets: StrArray,
    model_assets: StrArray,
) -> FloatArray:
    """Map portfolio weights into the factor model's asset ordering.

    Parameters
    ----------
    weights : ndarray of shape (..., n_portfolio_assets)
        Portfolio weights. The last axis corresponds to `portfolio_assets`.

    portfolio_assets : ndarray of shape (n_portfolio_assets,)
        Asset names of the portfolio.

    model_assets : ndarray of shape (n_model_assets,)
        Asset names of the factor model (the target ordering).

    Returns
    -------
    aligned : ndarray of shape (..., n_model_assets)
        Weights aligned to `model_assets`. Assets present in the portfolio keep their
        weight; assets only in the model receive zero.

    Raises
    ------
    ValueError
        If any asset in `portfolio_assets` is not found in `model_assets`.
    """
    if np.array_equal(portfolio_assets, model_assets):
        return weights

    model_indices = pd.Index(model_assets).get_indexer(portfolio_assets)
    missing_mask = model_indices < 0
    if np.any(missing_mask):
        missing = portfolio_assets[missing_mask]
        raise ValueError(
            f"Portfolio contains {len(missing)} asset(s) not in the factor "
            f"model: {missing[:5].tolist()}{'...' if len(missing) > 5 else ''}."
        )
    aligned_weights = np.zeros(
        (*weights.shape[:-1], len(model_assets)), dtype=weights.dtype
    )
    aligned_weights[..., model_indices] = weights
    return aligned_weights


def _prepare_realized_attribution_inputs(
    portfolio: Portfolio,
    factor_model: FactorModel,
) -> tuple[FloatArray, FloatArray, FactorModel]:
    """Prepare aligned weights and factor model data for realized attribution.

    Parameters
    ----------
    portfolio : Portfolio
        The portfolio to prepare.

    factor_model : FactorModel
        Factor model to restrict.

    Returns
    -------
    aligned_weights : ndarray
        Weights aligned to the factor model's asset ordering: the target weights of
        shape `(n_model_assets,)` when `weight_drift=False`, or the weights held during
        each observation of shape `(n_observations, n_model_assets)` when
        `weight_drift=True`.

    portfolio_returns : ndarray of shape (n_observations,)
        Portfolio returns restricted to the overlapping factor model window.

    aligned_factor_model : FactorModel
        Factor model restricted to the overlapping portfolio observation window.
    """
    if portfolio._is_failed_portfolio:
        raise ValueError("Cannot compute factor attribution on a failed portfolio.")
    portfolio_indices, aligned_factor_model = _select_realized_observation_window(
        observations=portfolio.observations,
        factor_model=factor_model,
    )
    weights = (
        portfolio._get_weights_path()[portfolio_indices]
        if portfolio.weight_drift
        else portfolio.weights
    )
    aligned_weights = _align_weights(
        weights, portfolio.assets, factor_model.asset_names
    )
    return aligned_weights, portfolio.returns[portfolio_indices], aligned_factor_model


def _select_realized_observation_window(
    observations: AnyArray,
    factor_model: FactorModel,
) -> tuple[IntArray, FactorModel]:
    """Select the overlapping realized attribution observation window.

    Leading or trailing portfolio observations outside the factor model are excluded.
    Missing portfolio observations inside the overlapping window are treated as a data
    alignment error.
    """
    factor_model_observations = factor_model.observations
    portfolio_observation_mask = np.isin(observations, factor_model_observations)
    portfolio_indices = np.flatnonzero(portfolio_observation_mask).astype(
        np.intp, copy=False
    )

    if len(portfolio_indices) == 0:
        raise ValueError(
            "Portfolio observations not found in FactorModel. "
            f"First five: {observations[:5].tolist()}"
        )

    first_index = portfolio_indices[0]
    last_index = portfolio_indices[-1]
    internal_missing = observations[first_index : last_index + 1][
        ~portfolio_observation_mask[first_index : last_index + 1]
    ]
    if len(internal_missing) > 0:
        raise ValueError(
            f"{len(internal_missing)} portfolio observation(s) inside the "
            "overlapping factor model window were not found in FactorModel. "
            f"First five: {internal_missing[:5].tolist()}"
        )

    selected_observations = observations[portfolio_indices]
    model_observation_mask = np.isin(factor_model_observations, selected_observations)
    factor_model_indices = np.flatnonzero(model_observation_mask).astype(
        np.intp, copy=False
    )
    if not np.array_equal(
        factor_model_observations[factor_model_indices], selected_observations
    ):
        raise ValueError(
            "Portfolio observations inside the overlapping factor model window must be "
            "a duplicate-free subset of FactorModel observations in the same relative "
            "order."
        )

    aligned_factor_model = factor_model.select_observations(factor_model_indices)
    return portfolio_indices, aligned_factor_model


def _get_risk(
    args: dict, weights: FloatArray, measure: skt.Measure, i: int, h: float
) -> float:
    """Get the Portfolio risk measure when the weight of asset `i` is increased by `h`."""
    assert "weights" not in args
    weights = weights.copy()
    weights[i] += h
    return getattr(Portfolio(weights=weights, **args), measure.value)


def _compute_contribution(
    args: dict,
    weights: FloatArray,
    assets: StrArray,
    measure: skt.Measure,
    h: float,
    drop_zero_weights: bool,
) -> tuple[list[float], list[str]]:
    """Compute the contribution of each asset to a given measure using finite
    difference.
    """
    contributions = []
    _assets = []
    for i, (weight, asset) in enumerate(zip(weights, assets, strict=True)):
        if weight == 0:
            if not drop_zero_weights:
                _assets.append(asset)
                contributions.append(0)
        else:
            _assets.append(asset)
            contributions.append(
                (
                    _get_risk(args, weights, measure, i, h)
                    - _get_risk(args, weights, measure, i, -h)
                )
                / (2 * h)
                * weight
            )
    return contributions, _assets
