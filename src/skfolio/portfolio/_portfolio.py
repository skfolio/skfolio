"""Portfolio module.
`Portfolio` is returned by the `predict` method of Optimization estimators.
It needs to be homogenous to the convex optimization problems meaning that `Portfolio`
is the dot product of the assets weights with the assets returns.
"""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause

import numbers
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px

import skfolio.typing as skt
from skfolio.measures import RiskMeasure, effective_number_assets
from skfolio.portfolio._base import _ZERO_THRESHOLD, BasePortfolio
from skfolio.utils.tools import (
    args_names,
    cached_property_slots,
    default_asset_names,
    input_to_array,
)

pd.options.plotting.backend = "plotly"


class Portfolio(BasePortfolio):
    r"""
    Portfolio class.

    `Portfolio` is returned by the `predict` method of Optimization estimators.
    It is homogenous to the convex optimization problems meaning that `Portfolio` is
    the dot product of the assets weights with the assets returns.

    Parameters
    ----------
    X : array-like of shape (n_observations, n_assets)
        Price returns of the assets.
        If `X` is a DataFrame or another array containers that implements 'columns'
        and 'index', the columns will be considered as assets names and the
        indices will be considered as observations.
        Otherwise, we use `["x0", "x1", ..., "x(n_assets - 1)"]` as asset names
        and `[0, 1, ..., n_observations]` as observations.

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

        with :math:`R` the matrix af assets returns and :math:`w` the vector of
        assets weights.

        If a float is provided, it is applied to each asset.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset weight) and `X` must be a DataFrame with assets names
        in columns.
        The default (`None`) means no transaction costs.

        .. warning::

            To be homogenous to the optimization problems, the periodicity of the
            transaction costs needs to be homogenous to the periodicity of the
            returns `X`. For example, if `X` is composed of **daily** returns,
            the `transaction_costs` need to be expressed in **daily** transaction costs.

    management_fees : float | dict[str, float] | array-like of shape (n_assets, ), optional
        Linear management fees of the assets. The Portfolio total management cost
        is:

        .. math:: total\_fee = \sum_{i=1}^{N} f_{i} \times w_{i}

        with :math:`f_{i}` the management fee of asset i and :math:`w_{i}` its weight.
        The float :math:`total\_fee` is used in the portfolio returns:

        .. math:: ptf\_returns = R \cdot w - total\_fee

        with :math:`R` the matrix af assets returns and :math:`w` the vector of
        assets weights.

        If a float is provided, it is applied to each asset.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset weight) and `X` must be a DataFrame with assets names
        in columns.
        The default (`None`) means no management fees.

        .. warning::

            To be homogenous to the optimization problems, the periodicity of the
            management fees needs to be homogenous to the periodicity of the
            returns `X`. For example, if `X` is composed of **daily** returns,
            the `management_fees` need to be expressed in **daily** fees.

    previous_weights : float | dict[str, float] | array-like of shape (n_assets, ), optional
        Previous portfolio weights.
        Previous weights are used to compute the total portfolio cost.
        If `transaction_costs` is 0, `previous_weights` will have no impact.
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

    annualized_factor : float, default=252.0
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
        If this is set to True, cumulative returns are compounded.
        The default is `False`.

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
    """

    _read_only_attrs: ClassVar[set] = BasePortfolio._read_only_attrs.copy()
    _read_only_attrs.update(
        {
            "X",
            "assets",
            "weights",
            "previous_weights",
            "transaction_costs",
            "management_fees",
            "n_assets",
            "total_cost",
            "total_fee",
        }
    )

    __slots__ = {
        # read-only
        "X",
        "weights",
        "previous_weights",
        "transaction_costs",
        "management_fees",
        "assets",
        "n_assets",
        "total_cost",
        "total_fee",
        # custom getter (read-only and cached)
        "_nonzero_assets",
        "_nonzero_assets_index",
    }

    def __init__(
        self,
        X: npt.ArrayLike,
        weights: skt.MultiInput,
        previous_weights: skt.MultiInput = None,
        transaction_costs: skt.MultiInput = None,
        management_fees: skt.MultiInput = None,
        risk_free_rate: float = 0,
        name: str | None = None,
        tag: str | None = None,
        annualized_factor: float = 252,
        fitness_measures: list[skt.Measure] | None = None,
        compounded: bool = False,
        min_acceptable_return: float | None = None,
        value_at_risk_beta: float = 0.95,
        entropic_risk_measure_theta: float = 1,
        entropic_risk_measure_beta: float = 0.95,
        cvar_beta: float = 0.95,
        evar_beta: float = 0.95,
        drawdown_at_risk_beta: float = 0.95,
        cdar_beta: float = 0.95,
        edar_beta: float = 0.95,
    ):
        # extract assets names from X
        assets = None
        observations = None
        if hasattr(X, "columns"):
            assets = np.asarray(X.columns, dtype=object)
            observations = np.asarray(X.index)

        # We don't perform extensive checks (like in check_X) for faster instantiation.
        rets = np.asarray(X)
        if rets.ndim != 2:
            raise ValueError("`X` must be a 2D array-like")

        n_observations, n_assets = rets.shape

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
                name="previous_weights",
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
                name="transaction_costs",
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
                name="management_fees",
            )

        # Default observations and assets if X is not a DataFrame
        if observations is None or len(observations) == 0:
            observations = np.arange(n_observations)

        if assets is None or len(assets) == 0:
            assets = default_asset_names(n_assets=n_assets)

        # Computing portfolio returns
        if np.isscalar(transaction_costs) and transaction_costs == 0:
            total_cost = 0
        else:
            total_cost = (transaction_costs * abs(previous_weights - weights)).sum()

        if np.isscalar(management_fees) and management_fees == 0:
            total_fee = 0
        else:
            total_fee = (management_fees * weights).sum()

        returns = weights @ rets.T - total_cost - total_fee

        if np.any(np.isnan(returns)):
            raise ValueError("NaN found in `returns`")

        super().__init__(
            returns=returns,
            observations=observations,
            name=name,
            tag=tag,
            fitness_measures=fitness_measures,
            compounded=compounded,
            risk_free_rate=risk_free_rate,
            annualized_factor=annualized_factor,
            min_acceptable_return=min_acceptable_return,
            value_at_risk_beta=value_at_risk_beta,
            cvar_beta=cvar_beta,
            entropic_risk_measure_theta=entropic_risk_measure_theta,
            entropic_risk_measure_beta=entropic_risk_measure_beta,
            evar_beta=evar_beta,
            drawdown_at_risk_beta=drawdown_at_risk_beta,
            cdar_beta=cdar_beta,
            edar_beta=edar_beta,
        )
        self._loaded = False
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
        self._loaded = True

    def __neg__(self):
        args = {arg: getattr(self, arg) for arg in args_names(self.__init__)}
        args["weights"] = -self.weights
        return self.__class__(**args)

    def __abs__(self):
        args = {arg: getattr(self, arg) for arg in args_names(self.__init__)}
        args["weights"] = np.abs(self.weights)
        return self.__class__(**args)

    def __round__(self, n: int):
        args = {arg: getattr(self, arg) for arg in args_names(self.__init__)}
        args["weights"] = np.round(self.weights, n)
        return self.__class__(**args)

    def __add__(self, other):
        if not isinstance(other, Portfolio):
            raise TypeError(
                f"Cannot add a Portfolio with an object of type {type(other)}"
            )
        args = args_names(self.__init__)
        for arg in args:
            if arg not in ["weights", "name", "tag"] and not np.array_equal(
                getattr(self, arg), getattr(other, arg)
            ):
                raise ValueError(f"Cannot add two Portfolios with different `{arg}`")
        args = {arg: getattr(self, arg) for arg in args}
        args["weights"] = self.weights + other.weights
        return self.__class__(**args)

    def __sub__(self, other):
        if not isinstance(other, Portfolio):
            raise TypeError(
                f"Cannot add a Portfolio with an object of type {type(other)}"
            )
        args = args_names(self.__init__)
        for arg in args:
            if arg not in ["weights", "name", "tag"] and not np.array_equal(
                getattr(self, arg), getattr(other, arg)
            ):
                raise ValueError(
                    f"Cannot subtract two Portfolios with different `{arg}`"
                )
        args = {arg: getattr(self, arg) for arg in args}
        args["weights"] = self.weights - other.weights
        return self.__class__(**args)

    def __mul__(self, other: numbers.Number):
        if not isinstance(other, numbers.Number):
            raise TypeError(
                "Portfolio can only be multiplied by a number, but received a"
                f" {type(other)}"
            )
        args = {arg: getattr(self, arg) for arg in args_names(self.__init__)}
        args["weights"] = other * self.weights
        return self.__class__(**args)

    __rmul__ = __mul__

    def __floordiv__(self, other: numbers.Number):
        if not isinstance(other, numbers.Number):
            raise TypeError(
                "Portfolio can only be floor divided by a number, but received a"
                f" {type(other)}"
            )
        args = {arg: getattr(self, arg) for arg in args_names(self.__init__)}
        args["weights"] = np.floor_divide(self.weights, other)
        return self.__class__(**args)

    def __truediv__(self, other: numbers.Number):
        if not isinstance(other, numbers.Number):
            raise TypeError(
                "Portfolio can only be divided by a number, but received a"
                f" {type(other)}"
            )
        args = {arg: getattr(self, arg) for arg in args_names(self.__init__)}
        args["weights"] = self.weights / other
        return self.__class__(**args)

    # Custom attribute getter (read-only and cached)
    @cached_property_slots
    def nonzero_assets(self) -> np.ndarray:
        """Invested asset :math:`abs(weights) > 0.001%`"""
        return self.assets[self.nonzero_assets_index]

    @cached_property_slots
    def nonzero_assets_index(self) -> np.ndarray:
        """Indices of invested asset :math:`abs(weights) > 0.001%`"""
        return np.flatnonzero(abs(self.weights) > _ZERO_THRESHOLD)

    @property
    def composition(self) -> pd.DataFrame:
        """DataFrame of the Portfolio composition."""
        weights = self.weights[self.nonzero_assets_index]
        df = pd.DataFrame({"asset": self.nonzero_assets, "weight": weights})
        df.sort_values(by="weight", ascending=False, inplace=True)
        df.rename(columns={"weight": self.name}, inplace=True)
        df.set_index("asset", inplace=True)
        return df

    @property
    def diversification(self):
        """Weighted average of volatility divided by the portfolio volatility."""
        return (
            self.weights @ np.std(np.asarray(self.X), axis=0) / self.standard_deviation
        )

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
        Herfindahl index [1]_:

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
        self, assets_expected_returns: np.ndarray
    ) -> float:
        """Compute the Portfolio expected returns from the assets expected returns,
        weights, management costs and transaction fees.

        Parameters
        ----------
        assets_expected_returns : ndarray of shape (n_assets,)
            The vector of assets expected returns.

        Returns
        -------
        value : float
            The Portfolio expected returns.
        """
        return (
            self.weights @ assets_expected_returns.T - self.total_cost - self.total_fee
        )

    def variance_from_assets(self, assets_covariance: np.ndarray) -> float:
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
        self, measure: skt.Measure, spacing: float | None = None
    ) -> np.ndarray:
        r"""Compute the contribution of each asset to a given measure.

        Parameters
        ----------
        measure : Measure
            The measure used for the contribution computation.

        spacing : float, optional
            Spacing "h" of the finite difference:
            :math:`contribution(wi)= \frac{measure(wi-h) - measure(wi+h)}{2h}`

        Returns
        -------
        values : ndrray of shape (n_assets,)
            The measure contribution of each asset.
        """
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
        args = {arg: getattr(self, arg) for arg in args_names(self.__init__)}

        def get_risk(i: int, h: float) -> float:
            a = args.copy()
            w = a["weights"].copy()
            w[i] += h
            a["weights"] = w
            return getattr(Portfolio(**a), measure.value)

        cont = [
            (get_risk(i, h=spacing) - get_risk(i, h=-spacing))
            / (2 * spacing)
            * self.weights[i]
            for i in range(len(self.weights))
        ]
        return np.array(cont)

    def plot_contribution(self, measure: skt.Measure, spacing: float | None = None):
        r"""Plot the contribution of each asset to a given measure.

        Parameters
        ----------
        measure : Measure
            The measure used for the contribution computation.

        spacing : float, optional
            Spacing "h" of the finite difference:
            :math:`contribution(wi)= \frac{measure(wi-h) - measure(wi+h)}{2h}`

        Returns
        -------
        plot : Figure
            The plotly Figure of assets contribution to the measure.
        """
        cont = self.contribution(measure=measure, spacing=spacing)
        df = pd.DataFrame(cont, index=self.assets, columns=["contribution"])
        fig = px.bar(df, x=df.index, y=df.columns)
        fig.update_layout(
            title=f"{measure} contribution",
            xaxis_title="Asset",
            yaxis_title=f"{measure} contribution",
        )
        return fig

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
