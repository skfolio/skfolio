"""Multi Period Portfolio module.
`MultiPeriodPortfolio` is returned by the `predict` method of Optimization estimators.
`MultiPeriodPortfolio` is a list of `Portfolio`.
"""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

import numbers
from collections.abc import Iterator

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import skfolio.typing as skt
from skfolio.portfolio._base import BasePortfolio
from skfolio.portfolio._failed_portfolio import FailedPortfolio
from skfolio.portfolio._portfolio import Portfolio
from skfolio.utils.tools import deduplicate_names


class MultiPeriodPortfolio(BasePortfolio):
    r"""Multi-Period Portfolio class.

    A Multi-Period Portfolio is composed of a list of :class:`Portfolio`.

    Parameters
    ----------
    portfolios : list[Portfolio], optional
       A list of :class:`Portfolio`. The default (`None`) is to initialize with an
       empty list.

    name : str, optional
        Name of the multi-period portfolio.
        The default (`None`) is to use the object id.

    tag : str, optional
        Tag given to the multi-period portfolio.
        Tags are used to manipulate groups of portfolios from a `Population`.

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
        The confidence level of the portfolio VaR (Value At Risk) which represents
        the return on the worst (1-beta)% observations.
        The default value is `0.95`.

    entropic_risk_measure_theta : float, default=1.0
        The risk aversion level of the portfolio Entropic Risk Measure.
        The default value is `1.0`.

    entropic_risk_measure_beta : float, default=0.95
        The confidence level of the portfolio Entropic Risk Measure.
        The default value is `0.95`.

    cvar_beta : float, default=0.95
        The confidence level of the portfolio CVaR (Conditional Value at Risk) which
        represents the expected VaR on the worst (1-beta)% observations.
        The default value is `0.95`.

    evar_beta : float, default=0.95
        The confidence level of the portfolio EVaR (Entropic Value at Risk).
        The default value is `0.95`.

    drawdown_at_risk_beta : float, default=0.95
        The confidence level of the portfolio Drawdown at Risk (DaR) which represents
        the drawdown on the worst (1-beta)% observations.
        The default value is `0.95`.

    cdar_beta : float, default=0.95
        The confidence level of the portfolio CDaR (Conditional Drawdown at Risk) which
        represents the expected drawdown on the worst (1-beta)% observations.
        The default value is `0.95`.

    edar_beta : float, default=0.95
        The confidence level of the portfolio EDaR (Entropic Drawdown at Risk).
        The default value is `0.95`.

    check_observations_order : bool, default=False
        If this is set to True, and if the list of portfolios is not chronologically
        sorted, an error is raised. The chronological order is determined by comparing
        the first and last observations of each portfolio.
        The default is `False`.

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

    __slots__ = {
        # read-only
        "_portfolios",
        "check_observations_order",
    }

    def __init__(
        self,
        portfolios: list[Portfolio] | None = None,
        name: str | None = None,
        tag: str | None = None,
        risk_free_rate: float = 0,
        annualized_factor: float = 252.0,
        fitness_measures: list[skt.Measure] | None = None,
        compounded: bool = False,
        sample_weight: np.ndarray | None = None,
        min_acceptable_return: float | None = None,
        value_at_risk_beta: float = 0.95,
        entropic_risk_measure_theta: float = 1,
        entropic_risk_measure_beta: float = 0.95,
        cvar_beta: float = 0.95,
        evar_beta: float = 0.95,
        drawdown_at_risk_beta: float = 0.95,
        cdar_beta: float = 0.95,
        edar_beta: float = 0.95,
        check_observations_order: bool = False,
    ):
        super().__init__(
            returns=np.array([]),
            observations=np.array([]),
            name=name,
            tag=tag,
            risk_free_rate=risk_free_rate,
            annualized_factor=annualized_factor,
            fitness_measures=fitness_measures,
            compounded=compounded,
            sample_weight=sample_weight,
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
        self.check_observations_order = check_observations_order
        self._set_portfolios(portfolios=portfolios)

    def __len__(self) -> int:
        return len(self.portfolios)

    def __getitem__(self, key: int | slice) -> Portfolio | list[Portfolio]:
        return self._portfolios[key]

    def __setitem__(self, key: int, value: Portfolio) -> None:
        if not isinstance(value, Portfolio):
            raise TypeError(f"Cannot set a value with type {type(value)}")
        new_portfolios = self._portfolios.copy()
        new_portfolios[key] = value
        self._set_portfolios(portfolios=new_portfolios)
        self.clear()

    def __delitem__(self, key: int) -> None:
        new_portfolios = self._portfolios.copy()
        del new_portfolios[key]
        self._set_portfolios(portfolios=new_portfolios)
        self.clear()

    def __iter__(self) -> Iterator[Portfolio]:
        return iter(self._portfolios)

    def __contains__(self, value: Portfolio) -> bool:
        if not isinstance(value, Portfolio):
            return False
        return value in self._portfolios

    def __neg__(self):
        return self.__class__(
            portfolios=[-p for p in self],
            tag=self.tag,
            fitness_measures=self.fitness_measures,
        )

    def __abs__(self):
        return self.__class__(
            portfolios=[abs(p) for p in self],
            tag=self.tag,
            fitness_measures=self.fitness_measures,
        )

    def __round__(self, n: int):
        return self.__class__(
            portfolios=[p.__round__(n) for p in self],
            tag=self.tag,
            fitness_measures=self.fitness_measures,
        )

    def __floor__(self):
        return self.__class__(
            portfolios=[np.floor(p) for p in self],
            tag=self.tag,
            fitness_measures=self.fitness_measures,
        )

    def __trunc__(self):
        return self.__class__(
            portfolios=[np.trunc(p) for p in self],
            tag=self.tag,
            fitness_measures=self.fitness_measures,
        )

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(
                "Cannot add a MultiPeriodPortfolio with an object of type"
                f" {type(other)}"
            )
        if len(self) != len(other):
            raise TypeError("Cannot add two MultiPeriodPortfolio of different sizes")
        return self.__class__(
            portfolios=[p1 + p2 for p1, p2 in zip(self, other, strict=True)],
            tag=self.tag,
            fitness_measures=self.fitness_measures,
        )

    def __sub__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(
                "Cannot subtract a MultiPeriodPortfolio with an object of type"
                f" {type(other)}"
            )
        if len(self) != len(other):
            raise TypeError(
                "Cannot subtract two MultiPeriodPortfolio of different sizes"
            )
        return self.__class__(
            portfolios=[p1 - p2 for p1, p2 in zip(self, other, strict=True)],
            tag=self.tag,
            fitness_measures=self.fitness_measures,
        )

    def __mul__(self, other: numbers.Number | list[numbers.Number] | np.ndarray):
        if np.isscalar(other):
            portfolios = [p * other for p in self]
        else:
            portfolios = [p * a for p, a in zip(self, other, strict=True)]
        return self.__class__(
            portfolios=portfolios, tag=self.tag, fitness_measures=self.fitness_measures
        )

    __rmul__ = __mul__

    def __floordiv__(self, other: numbers.Number | list[numbers.Number] | np.ndarray):
        if np.isscalar(other):
            portfolios = [p // other for p in self]
        else:
            portfolios = [p // a for p, a in zip(self, other, strict=True)]
        return self.__class__(
            portfolios=portfolios, tag=self.tag, fitness_measures=self.fitness_measures
        )

    def __truediv__(self, other: numbers.Number | list[numbers.Number] | np.ndarray):
        if np.isscalar(other):
            portfolios = [p / other for p in self]
        else:
            portfolios = [p / a for p, a in zip(self, other, strict=True)]
        return self.__class__(
            portfolios=portfolios, tag=self.tag, fitness_measures=self.fitness_measures
        )

    # Private method
    def _set_portfolios(self, portfolios: list[Portfolio] | None = None) -> None:
        """Set the returns, observations and portfolios list.

        Parameters
        ----------
        portfolios : list[Portfolio], optional
            The list of Portfolios. The default (`None`) is to use an empty list.
        """
        returns = []
        observations = []
        if portfolios is None:
            portfolios = []
        if len(portfolios) != 0:
            for item in portfolios:
                if not isinstance(item, BasePortfolio | Portfolio):
                    raise TypeError(
                        "`portfolios` items must be of type `Portfolio`, got"
                        f" {type(item).__name__}"
                    )
                returns.append(item.returns)
                observations.append(item.observations)
            returns = np.concatenate(returns)
            observations = np.concatenate(observations)
            if self.check_observations_order:
                iteration = iter(portfolios)
                prev_p = next(iteration)
                while (p := next(iteration, None)) is not None:
                    if p.observations[0] <= prev_p.observations[-1]:
                        raise ValueError(
                            "Portfolios observations should not overlap:"
                            f" {p} overlapping {prev_p}"
                        )
                    prev_p = p
        self._loaded = False
        self._portfolios = portfolios
        self.returns = np.asarray(returns)
        self.observations = np.asarray(observations)
        self._loaded = True

    # Custom attribute setter and getter
    @property
    def portfolios(self) -> list[Portfolio]:
        """List of portfolios composing the mutli-period portfolio."""
        return self._portfolios

    @portfolios.setter
    def portfolios(self, value: list[Portfolio] | None = None):
        """Set the list of Portfolios and clear the attributes cache linked to the
        list of portfolios.
        """
        self._set_portfolios(portfolios=value)
        self.clear()

    # Classic property
    @property
    def failed_portfolios(self) -> list[FailedPortfolio]:
        """Return the list of `FailedPortfolio` in the multi-period portfolio."""
        return [x for x in self if isinstance(x, FailedPortfolio)]

    @property
    def fallback_portfolios(self) -> list[Portfolio]:
        """
        Return the list of portfolios in the multi-period portfolio that used a
        fallback (i.e., have a non-None `fallback_chain`). This includes
        `FailedPortfolio` instances when fallbacks were attempted.
        """
        return [x for x in self if getattr(x, "fallback_chain", None) is not None]

    @property
    def n_failed_portfolios(self) -> int:
        """Number of `FailedPortfolio` in the multi-period portfolio."""
        return len(self.failed_portfolios)

    @property
    def n_fallback_portfolios(self) -> int:
        """Number of portfolios in the multi-period portfolio with a fallback."""
        return len(self.fallback_portfolios)

    @property
    def assets(self) -> list:
        """List of assets names in each Portfolio."""
        return [p.assets for p in self]

    @property
    def composition(self) -> pd.DataFrame:
        """DataFrame of the Portfolio composition."""
        df = pd.concat([p.composition for p in self], axis=1)
        df.columns = deduplicate_names(df.columns)
        # Leave columns of only NaNs untouched
        mask = ~df.isna().all(axis=0)
        df.loc[:, mask] = df.loc[:, mask].fillna(0)
        return df

    @property
    def weights_dict(self) -> dict[str, dict[str, float]]:
        """Dictionary mapping each Portfolio name to its asset weight allocation."""
        names = deduplicate_names([ptf.name for ptf in self.portfolios])
        return {
            name: ptf.weights_dict
            for name, ptf in zip(names, self.portfolios, strict=True)
        }

    @property
    def previous_weights_dict(self) -> dict[str, dict[str, float]]:
        """Dictionary mapping Portfolio name to its previous asset weight allocation."""
        names = deduplicate_names([ptf.name for ptf in self.portfolios])
        return {
            name: ptf.previous_weights_dict
            for name, ptf in zip(names, self.portfolios, strict=True)
        }

    @property
    def weights_per_observation(self) -> pd.DataFrame:
        """DataFrame of the Portfolio weights per observation."""
        return (
            pd.concat([p.weights_per_observation for p in self], axis=0)
            .fillna(0)
            .sort_index()
        )

    def contribution(
        self, measure: skt.Measure, spacing: float | None = None, to_df: bool = True
    ) -> np.ndarray | pd.DataFrame:
        r"""Compute the contribution of each asset to a given measure for each
        portfolio.

        Parameters
        ----------
        measure : Measure
            The measure used for the contribution computation.

        spacing : float, optional
            Spacing "h" of the finite difference:
            :math:`contribution(wi)= \frac{measure(wi-h) - measure(wi+h)}{2h}`

        to_df : bool, default=False
            If this is set to True, a DataFrame with asset names in index and portfolio
            names in columns is returned, otherwise a list of numpy array is returned.
            When a DataFrame is returned, the assets with zero weights are removed.

        Returns
        -------
        values : list of numpy array of shape (n_assets,) for each portfolio or a DataFrame
            The measure contribution of each asset for each portfolio.
        """
        contributions = [
            ptf.contribution(measure=measure, spacing=spacing, to_df=to_df)
            for ptf in self
        ]
        if not to_df:
            return contributions
        df = pd.concat(contributions, axis=1)
        df.columns = deduplicate_names(df.columns)
        # Leave columns of only NaNs untouched
        mask = ~df.isna().all(axis=0)
        df.loc[:, mask] = df.loc[:, mask].fillna(0)
        return df

    def summary(self, formatted: bool = True) -> pd.Series:
        """Portfolio summary of all its measures.

        Parameters
        ----------
        formatted : bool, default=True
            If this is set to True, the measures are formatted into rounded string with
            units.

        Returns
        -------
        summary : series
            Portfolio summary of all its measures.
        """
        df = super().summary(formatted=formatted)
        avg_assets_per_portfolio = np.mean([p.n_assets for p in self])
        n_portfolios = len(self)
        n_failed_portfolios = self.n_failed_portfolios
        n_fallback_portfolios = self.n_fallback_portfolios

        if formatted:
            avg_assets_per_portfolio = f"{avg_assets_per_portfolio:0.1f}"
            n_portfolios = str(int(n_portfolios))
            n_failed_portfolios = str(n_failed_portfolios)
            n_fallback_portfolios = str(n_fallback_portfolios)

        df["Avg nb of Assets per Portfolio"] = avg_assets_per_portfolio
        df["Number of Portfolios"] = n_portfolios
        df["Number of Failed Portfolios"] = n_failed_portfolios
        df["Number of Fallback Portfolios"] = n_fallback_portfolios

        return df

    # Public methods
    def append(self, portfolio: Portfolio) -> None:
        """Append a Portfolio to the Portfolio list.

        Parameters
        ----------
        portfolio : Portfolio
            The Portfolio to append.
        """
        if self.check_observations_order and len(self) != 0:
            start_date = portfolio.observations[0]
            prev_last_date = self[-1].observations[-1]
            if start_date < prev_last_date:
                raise ValueError(
                    f"Portfolios observations should not overlap: {prev_last_date} ->"
                    f" {start_date} "
                )
        self._loaded = False
        self._portfolios.append(portfolio)
        if len(self.observations) == 0:
            # We don't concatenate an empty array as we cannot know the dtype before.
            self.observations = portfolio.observations
            self.returns = portfolio.returns
        else:
            self.observations = np.concatenate(
                [self.observations, portfolio.observations], axis=0
            )
            self.returns = np.concatenate([self.returns, portfolio.returns], axis=0)
        self._loaded = True
        self.clear()

    def plot_weights_per_observation(self):
        """Plot portfolio weights per observation as a stacked-area chart.

        This shows the composition of the portfolio over time, with each asset's weight
        stacked to illustrate how allocations shift.

        Returns
        -------
        plot : Figure
            Returns the plot Figure object.
        """
        df = self.weights_per_observation

        fig = go.Figure()

        for asset in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[asset],
                    mode="lines",
                    name=asset,
                    stackgroup="one",  # stack all series
                    line=dict(width=0.5),
                    hovertemplate=(
                        "%{x|%Y-%m-%d}<br>"  # date
                        f"{asset}: "  # asset name
                        "%{y:.2%}"  # two-decimals percent
                        "<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            title="Weight allocation over time",
            xaxis_title="Date",
            yaxis_title="Weight (%)",
            legend_title_text="Assets",
        )

        fig.update_yaxes(
            tickformat=".0%",
            zeroline=True,
            zerolinecolor="gray",
        )

        return fig
