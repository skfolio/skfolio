"""Base Portfolio module"""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause

# The Portfolio class contains more than 40 measures than can be computationally
# expensive. The use of __slots__ instead of __dict__ is based on the following
# consideration:
#   * Fast Portfolio instantiation.
#   * Compute a measure only when needed.
#   * Reuse the measures functions in measures.py module independently of the
#     Portfolio class.
#   * Have the measures as Class attributes and not as Class generic
#     methods for better usability.
#   * Caching of the 40 measures.
#   * DRY by not re-writing @cached_property decorated methods for all the 40 measures.
#
# We define 7 types of attributes:
#     * Public (read and right)
#     * Private (read and right for private usage)
#     * Read-only (handled in __setattr__)
#     * Global abd local measures arguments: when they change, we clear the cache of
#       all the measures (handled in __setattr__)
#     * Attributes with custom getter and setter (using @property + private name
#       in __slots__)
#     * Attributes with custom getter without setter (read-only) that caches the result
#       (using custom decorator @cached_property_slots + private name in __slots__)
#     * Measures that are cached (handled in __getattribute__)
#
#  In order to generate the measures attributes we call the measure functions and their
#  arguments dynamically from the measures.py module. The function arguments are
#  retrieved from the class attributes following the below rules:
#     * Global measures function arguments (defined in GLOBAL_ARGS) need to be defined
#       in the class attributes with identical name.
#     * Local measures function arguments (defined in LOCAL_ARGS) need to be defined in
#       the class attributes with the argument name preceded by the measure name and
#       separated by '_'.

import warnings
from abc import abstractmethod
from typing import ClassVar

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import skfolio.typing as skt
from skfolio import measures as mt
from skfolio.measures import (
    ExtraRiskMeasure,
    PerfMeasure,
    RatioMeasure,
    RiskMeasure,
)
from skfolio.utils.sorting import dominate
from skfolio.utils.tools import (
    args_names,
    cached_property_slots,
    format_measure,
)

# TODO: remove and use plotly express
pd.options.plotting.backend = "plotly"


_ZERO_THRESHOLD = 1e-5
_MEASURES = {
    e for enu in [PerfMeasure, RiskMeasure, ExtraRiskMeasure, RatioMeasure] for e in enu
}
_MEASURES_VALUES = {e.value: e for e in _MEASURES}


class BasePortfolio:
    r"""Base Portfolio class for all portfolios in skfolio.

    Parameters
    ----------
    returns : array-like of shape (n_observations,)
        Vector of portfolio returns.

    observations : array-like of shape (n_observations,)
        Vector of portfolio observations.

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

    _read_only_attrs: ClassVar[set] = {
        "returns",
        "observations",
    }

    # Arguments globally used in measures computation
    _measure_global_args: ClassVar[set] = {
        "returns",
        "cumulative_returns",
        "drawdowns",
        "min_acceptable_return",
        "compounded",
        "risk_free_rate",
    }

    # Arguments locally used in measures computation
    _measure_local_args: ClassVar[set] = {
        "value_at_risk_beta",
        "cvar_beta",
        "entropic_risk_measure_theta",
        "entropic_risk_measure_beta",
        "evar_beta",
        "drawdown_at_risk_beta",
        "cdar_beta",
        "edar_beta",
    }

    __slots__ = {
        # public
        "tag",
        "name",
        # public read-only
        "returns",
        "observations",
        # private
        "_loaded",
        # custom getter and setter
        "_fitness_measures",
        "_annualized_factor",
        # custom getter (read-only and cached)
        "_fitness",
        "_cumulative_returns",
        "_drawdowns",
        # global args
        "min_acceptable_return",
        "compounded",
        "risk_free_rate",
        # local args
        "value_at_risk_beta",
        "cvar_beta",
        "entropic_risk_measure_theta",
        "entropic_risk_measure_beta",
        "evar_beta",
        "drawdown_at_risk_beta",
        "cdar_beta",
        "edar_beta",
        # measures
        # perf
        "mean",
        # annualized
        "annualized_mean",
        # risk measure
        "mean_absolute_deviation",
        "first_lower_partial_moment",
        "variance",
        "standard_deviation",
        "semi_variance",
        "semi_deviation",
        "fourth_central_moment",
        "fourth_lower_partial_moment",
        "value_at_risk",
        "cvar",
        "entropic_risk_measure",
        "evar",
        "worst_realization",
        "drawdown_at_risk",
        "cdar",
        "max_drawdown",
        "average_drawdown",
        "edar",
        "ulcer_index",
        "gini_mean_difference",
        "skew",
        "kurtosis",
        # annualized
        "annualized_variance",
        "annualized_semi_variance",
        "annualized_standard_deviation",
        "annualized_semi_deviation",
        # ratio
        "mean_absolute_deviation_ratio",
        "first_lower_partial_moment_ratio",
        "sharpe_ratio",
        "sortino_ratio",
        "value_at_risk_ratio",
        "cvar_ratio",
        "entropic_risk_measure_ratio",
        "evar_ratio",
        "worst_realization_ratio",
        "drawdown_at_risk_ratio",
        "cdar_ratio",
        "calmar_ratio",
        "average_drawdown_ratio",
        "edar_ratio",
        "ulcer_index_ratio",
        "gini_mean_difference_ratio",
        # annualized
        "annualized_sharpe_ratio",
        "annualized_sortino_ratio",
    }

    def __init__(
        self,
        returns: np.ndarray | list,
        observations: np.ndarray | list,
        name: str | None = None,
        tag: str | None = None,
        annualized_factor: float = 252.0,
        fitness_measures: list[skt.Measure] | None = None,
        risk_free_rate: float = 0.0,
        compounded: bool = False,
        min_acceptable_return: float | None = None,
        value_at_risk_beta: float = 0.95,
        entropic_risk_measure_theta: float = 1.0,
        entropic_risk_measure_beta: float = 0.95,
        cvar_beta: float = 0.95,
        evar_beta: float = 0.95,
        drawdown_at_risk_beta: float = 0.95,
        cdar_beta: float = 0.95,
        edar_beta: float = 0.95,
    ):
        self._loaded = False
        self._annualized_factor = annualized_factor
        self.returns = np.asarray(returns)
        self.observations = np.asarray(observations)
        self.risk_free_rate = risk_free_rate
        self.tag = tag
        self.compounded = compounded
        self.min_acceptable_return = min_acceptable_return
        self.value_at_risk_beta = value_at_risk_beta
        self.entropic_risk_measure_theta = entropic_risk_measure_theta
        self.entropic_risk_measure_beta = entropic_risk_measure_beta
        self.cvar_beta = cvar_beta
        self.evar_beta = evar_beta
        self.drawdown_at_risk_beta = drawdown_at_risk_beta
        self.cdar_beta = cdar_beta
        self.edar_beta = edar_beta

        self.name = str(id(self)) if name is None else name
        if fitness_measures is None:
            self._fitness_measures = [PerfMeasure.MEAN, RiskMeasure.VARIANCE]
        else:
            self._fitness_measures = fitness_measures
        self._loaded = True

    def __reduce__(self):
        # For fast serialization and deserialization
        # We don't want to serialize generic slots but only init arguments
        return self.__class__, tuple(
            [getattr(self, arg) for arg in args_names(self.__init__)]
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"

    def __eq__(self, other) -> bool:
        return isinstance(other, BasePortfolio) and np.array_equal(
            self.fitness, other.fitness
        )

    def __gt__(self, other) -> bool:
        if not isinstance(other, BasePortfolio):
            raise TypeError(
                "`>` not supported between instances of `Portfolio` and"
                f" `{type(other)}`"
            )
        return self.dominates(other)

    def __ge__(self, other) -> bool:
        if not isinstance(other, BasePortfolio):
            raise TypeError(
                "`>=` not supported between instances of `Portfolio` and"
                f" `{type(other)}`"
            )
        return self.__eq__(other) or self.__gt__(other)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result._loaded = False
        for attr in self._slots():
            if attr not in _MEASURES_VALUES and attr != "_loaded":
                try:
                    setattr(result, attr, getattr(self, attr))
                except AttributeError:
                    pass
        result._loaded = True
        return result

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError as e:
            # The Measures are the only attributes in __slots__ that are not yet
            # assigned.
            # We assign their values dynamically the first time they are called.
            if name not in _MEASURES_VALUES:
                raise AttributeError(e) from None
            measure = _MEASURES_VALUES[name]
            value = self.get_measure(measure=measure)
            setattr(self, name, value)
            return value

    def __setattr__(self, name, value):
        if name != "_loaded" and self._loaded:
            if name in self._read_only_attrs:
                raise AttributeError(
                    f"can't set attribute '{name}' because it is read-only"
                )
            if name in self._measure_global_args or name in self._measure_local_args:
                # When an attribute in GLOBAL_ARGS or LOCAL_ARGS is set, we reset all
                # the measures
                self.clear()
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        # We only want to raise an error when the attribute doesn't exist and we don't
        # want to raise an error when it's a valid attribute that has not been assigned
        # a value.
        try:
            object.__delattr__(self, name)
        except AttributeError:
            if name not in self._slots():
                raise AttributeError(
                    f"`{type(self).__name__}` object has no attribute '{name}'"
                ) from None

    def __array__(self) -> np.ndarray:
        return self.returns

    # Private methods
    def _slots(self) -> set[str]:
        slots = set()
        for s in self.__class__.__mro__:
            slots.update(getattr(s, "__slots__", set()))
        return slots

    @property
    @abstractmethod
    def composition(self) -> pd.DataFrame:
        """DataFrame of the Portfolio composition"""
        pass

    # Custom attribute setter and getter
    @property
    def fitness_measures(self) -> list[skt.Measure]:
        """Portfolio fitness measures."""
        return self._fitness_measures

    @fitness_measures.setter
    def fitness_measures(self, value: list[skt.Measure]) -> None:
        if not isinstance(value, list) or len(value) == 0:
            raise TypeError("`fitness_measures` must be a non-empty list of Measure")
        for val in value:
            if not isinstance(
                val, PerfMeasure | RiskMeasure | ExtraRiskMeasure | RatioMeasure
            ):
                raise TypeError("`fitness_measures` must be a list of Measure")
        self._fitness_measures = value
        delattr(self, "_fitness")

    @property
    def annualized_factor(self) -> float:
        """Portfolio annualized factor."""
        return self._annualized_factor

    @annualized_factor.setter
    def annualized_factor(self, value: float) -> None:
        self._annualized_factor = value
        self.clear()

    # Custom attribute getter (read-only and cached)
    @cached_property_slots
    def fitness(self) -> np.ndarray:
        """The Portfolio fitness."""
        res = []
        for measure in self.fitness_measures:
            if isinstance(measure, PerfMeasure | RatioMeasure):
                sign = 1
            else:
                sign = -1
            res.append(sign * getattr(self, str(measure.value)))
        return np.array(res)

    @cached_property_slots
    def cumulative_returns(self) -> np.ndarray:
        """Portfolio cumulative returns array."""
        return mt.get_cumulative_returns(
            returns=self.returns, compounded=self.compounded
        )

    @cached_property_slots
    def drawdowns(self) -> np.ndarray:
        """Portfolio drawdowns array."""
        return mt.get_drawdowns(returns=self.returns, compounded=self.compounded)

    # Classic property
    @property
    def n_observations(self) -> int:
        """Number of observations"""
        return len(self.observations)

    @property
    def returns_df(self) -> pd.Series:
        """Portfolio returns DataFrame."""
        return pd.Series(index=self.observations, data=self.returns, name="returns")

    @property
    def cumulative_returns_df(self) -> pd.Series:
        """Portfolio cumulative returns Series."""
        return pd.Series(
            index=self.observations,
            data=self.cumulative_returns,
            name="cumulative_returns",
        )

    @property
    def measures_df(self) -> pd.DataFrame:
        """DataFrame of all measures."""
        idx = [e.value for enu in [PerfMeasure, RiskMeasure, RatioMeasure] for e in enu]
        res = [getattr(self, attr) for attr in idx]
        return pd.DataFrame(res, index=idx, columns=["measures"])

    # Public methods
    def copy(self):
        """Copy the Portfolio attributes without its measures values."""
        return self.__copy__()

    def clear(self) -> None:
        """Clear all measures, fitness, cumulative returns and drawdowns in slots"""
        attrs = ["_fitness", "_cumulative_returns", "_drawdowns"]
        for attr in attrs + list(_MEASURES_VALUES):
            delattr(self, attr)

    def get_measure(self, measure: skt.Measure) -> float:
        """Returns the value of a given measure.

        Parameters
        ----------
        measure : PerfMeasure | RiskMeasure | ExtraRiskMeasure | RatioMeasure
            The input measure.

        Returns
        -------
        value : float
            The measure value.
        """
        if isinstance(measure, PerfMeasure | RiskMeasure | ExtraRiskMeasure):
            # We call the measure functions and their arguments dynamically.
            # The measure functions are called from the "measures" module.
            # The function arguments are retrieved from the class attributes following
            # the below rules:
            # Global measures function arguments (defined in GLOBAL_ARGS) need to be
            # defined in the class attributes with identical name.
            # Local measures function arguments need to be defined in the class
            # attributes with the argument name preceded by the measure name and
            # separated by "_".
            if measure.is_annualized:
                func = getattr(mt, str(measure.non_annualized_measure.value))
            else:
                func = getattr(mt, str(measure.value))

            args = {
                arg: (
                    getattr(self, arg)
                    if arg in self._measure_global_args
                    else getattr(self, f"{measure.value}_{arg}")
                )
                for arg in args_names(func)
            }
            try:
                value = func(**args)
                if measure in [
                    PerfMeasure.ANNUALIZED_MEAN,
                    RiskMeasure.ANNUALIZED_VARIANCE,
                    RiskMeasure.ANNUALIZED_SEMI_VARIANCE,
                ]:
                    value *= self.annualized_factor
                elif measure in [
                    RiskMeasure.ANNUALIZED_STANDARD_DEVIATION,
                    RiskMeasure.ANNUALIZED_SEMI_DEVIATION,
                ]:
                    value *= np.sqrt(self.annualized_factor)
            except Exception as e:
                warnings.warn(
                    f"Unable to calculate the portfolio '{measure.value}' with"
                    f" error: {e}",
                    stacklevel=2,
                )
                value = np.nan
        elif isinstance(measure, RatioMeasure):
            # ratio
            if measure.is_annualized:
                mean = self.annualized_mean
            else:
                mean = self.mean
            risk = getattr(self, str(measure.linked_risk_measure.value))
            value = (mean - self.risk_free_rate) / risk
        else:
            raise ValueError(f"{measure} is not a Measure.")
        return value

    def dominates(
        self, other: "BasePortfolio", idx: slice | np.ndarray | None = None
    ) -> bool:
        """Portfolio domination.

        Returns true if each objective of the current portfolio fitness is not
        strictly worse than the corresponding objective of the other portfolio fitness
        and at least one objective is strictly better.

        Parameters
        ----------
        other : BasePortfolio
            The other portfolio.

        idx : slice | array, optional
            Indexes or slice indicating on which objectives the domination is performed.
            The default (`None`) is to use all objectives.

        Returns
        -------
        value : bool
            Returns True if the Portfolio dominates the other one.
        """
        if idx is None:
            idx = slice(None)
        return dominate(self.fitness[idx], other.fitness[idx])

    def rolling_measure(
        self, measure: skt.Measure = RatioMeasure.SHARPE_RATIO, window: int = 30
    ) -> pd.Series:
        """Compute the measure over a rolling window.

        Parameters
        ----------
        measure : ct.Measure, default=RatioMeasure.SHARPE_RATIO
            The measure. The default measure is the Sharpe Ratio.

        window : int, default=30
            The window size. The default value is `30`.

        Returns
        -------
        series : pandas Series
            The rolling measure Series.
        """
        if measure.is_annualized:
            non_annualized_measure = measure.non_annualized_measure
        else:
            non_annualized_measure = measure

        if measure.is_perf:
            perf_measure = non_annualized_measure
            risk_measure = None
        elif measure.is_ratio:
            perf_measure = PerfMeasure.MEAN
            risk_measure = non_annualized_measure.linked_risk_measure
        else:
            perf_measure = None
            risk_measure = non_annualized_measure

        if risk_measure is not None:
            risk_func = getattr(mt, str(risk_measure.value))
            risk_func_args = {
                arg: (
                    getattr(self, arg)
                    if arg in self._measure_global_args
                    else getattr(self, f"{risk_measure.value}_{arg}")
                )
                for arg in args_names(risk_func)
            }

            if "drawdowns" in risk_func_args:
                del risk_func_args["drawdowns"]

                def meta_risk_func(returns):
                    drawdowns = mt.get_drawdowns(returns, compounded=self.compounded)
                    return risk_func(drawdowns=drawdowns, **risk_func_args)

            else:
                del risk_func_args["returns"]

                def meta_risk_func(returns):
                    return risk_func(returns=returns, **risk_func_args)

            if perf_measure is not None:
                perf_func = getattr(mt, str(perf_measure.value))

                def func(returns):
                    return (perf_func(returns) - self.risk_free_rate) / meta_risk_func(
                        returns
                    )

            else:
                func = meta_risk_func
        else:
            perf_func = getattr(mt, str(perf_measure.value))

            def func(returns):
                return perf_func(returns)

        rolling = (
            pd.Series(self.returns, index=self.observations)
            .rolling(window=window)
            .apply(func)
        )
        if measure.is_annualized:
            if measure in [
                PerfMeasure.ANNUALIZED_MEAN,
                RiskMeasure.ANNUALIZED_VARIANCE,
                RiskMeasure.ANNUALIZED_SEMI_VARIANCE,
            ]:
                rolling *= self.annualized_factor
            elif measure in [
                RiskMeasure.ANNUALIZED_STANDARD_DEVIATION,
                RiskMeasure.ANNUALIZED_SEMI_DEVIATION,
                RatioMeasure.ANNUALIZED_SHARPE_RATIO,
                RatioMeasure.ANNUALIZED_SORTINO_RATIO,
            ]:
                rolling *= np.sqrt(self.annualized_factor)
        return rolling

    def summary(self, formatted: bool = True) -> pd.Series:
        """Portfolio summary of all its measures.

        Parameters
        ----------
        formatted : bool, default=True
            If this is set to True, the measures are formatted into rounded string
            with units.

        Returns
        -------
        summary : pandas Series
            The Portfolio summary.
        """
        measures = (
            e
            for enu in [PerfMeasure, RiskMeasure, ExtraRiskMeasure, RatioMeasure]
            for e in enu
        )
        summary = {}
        for e in measures:
            e: skt.Measure
            try:
                if e.is_ratio:
                    base_measure = e.linked_risk_measure
                else:
                    base_measure = e
                beta = getattr(self, f"{base_measure.value}_beta")
                key = f"{e!s} at {beta:.0%}"
            except AttributeError:
                key = str(e)
            if isinstance(e, RatioMeasure) or e in [
                ExtraRiskMeasure.ENTROPIC_RISK_MEASURE,
                RiskMeasure.ULCER_INDEX,
                ExtraRiskMeasure.SKEW,
                ExtraRiskMeasure.KURTOSIS,
            ]:
                percent = False
            else:
                percent = True
            if formatted:
                value = format_measure(getattr(self, str(e.value)), percent=percent)
            else:
                value = getattr(self, str(e.value))
            summary[key] = value
        return pd.Series(summary)

    def plot_cumulative_returns(
        self, log_scale: bool = False, idx: slice | np.ndarray | None = None
    ) -> go.Figure:
        """Plot the Portfolio cumulative returns.
        Non-compounded cumulative returns start at 0.
        Compounded cumulative returns are rescaled to start at 1000.

        Parameters
        ----------
        log_scale : bool, default=False
            If this is set to True, the cumulative returns are displayed with a
            logarithm scale on the y-axis and rebased at 1000. The cumulative returns
            must be compounded otherwise an exception is raised.

        idx : slice | array, optional
            Indexes or slice of the observations to plot.
            The default (`None`) is to plot all observations.

        Returns
        -------
        plot : Figure
            Returns the plot Figure object.
        """
        if idx is None:
            idx = slice(None)
        df = self.cumulative_returns_df.iloc[idx]
        title = "Cumulative Returns"
        if self.compounded:
            yaxis_title = f"{title} (rebased at 1000)"
            if log_scale:
                title = f"{title} (compounded & log scaled)"
            else:
                title = f"{title} (compounded)"
        else:
            if log_scale:
                raise ValueError(
                    "Plotting with logarithm scaling must be done on cumulative "
                    "returns that are compounded as opposed to non-compounded."
                    "You can change to compounded with `compounded=True`"
                )
            yaxis_title = title
            title = f"{title} (non-compounded)"

        fig = df.plot()
        fig.update_layout(
            title=title,
            xaxis_title="Observations",
            yaxis_title=yaxis_title,
            showlegend=False,
        )
        if self.compounded:
            fig.update_yaxes(tickformat=".0f")
        else:
            fig.update_yaxes(tickformat=".2%")
        if log_scale:
            fig.update_yaxes(type="log")
        return fig

    def plot_returns(self, idx: slice | np.ndarray | None = None) -> go.Figure:
        """Plot the Portfolio returns

        Parameters
        ----------
        idx : slice | array, optional
            Indexes or slice of the observations to plot.
            The default (`None`) is to plot all observations.

        Returns
        -------
        plot : Figure
            Returns the plot Figure object
        """
        if idx is None:
            idx = slice(None)
        fig = self.returns_df.iloc[idx].plot()
        fig.update_layout(
            title="Returns",
            xaxis_title="Observations",
            yaxis_title="Returns",
            showlegend=False,
        )
        return fig

    def plot_rolling_measure(
        self,
        measure: skt.Measure = RatioMeasure.SHARPE_RATIO,
        window: int = 30,
    ) -> go.Figure:
        """Plot the measure over a rolling window.

        Parameters
        ----------
        measure : ct.Measure, default = RatioMeasure.SHARPE_RATIO
           The measure.

        window : int, default=30
           The window size.

        Returns
        -------
        plot : Figure
            Returns the plot Figure object
        """
        rolling = self.rolling_measure(measure=measure, window=window)
        rolling.name = f"{measure} {window} observations"
        fig = rolling.plot()
        fig.add_hline(
            y=getattr(self, measure.value),
            line_width=1,
            line_dash="dash",
            line_color="blue",
        )
        max_val = rolling.max()
        min_val = rolling.min()
        if max_val > 0:
            fig.add_hrect(
                y0=0, y1=max_val * 1.3, line_width=0, fillcolor="green", opacity=0.1
            )
        if min_val < 0:
            fig.add_hrect(
                y0=min_val * 1.3, y1=0, line_width=0, fillcolor="red", opacity=0.1
            )

        fig.update_layout(
            title=f"rolling {measure} - {window} observations window",
            xaxis_title="Observations",
            yaxis_title=str(measure),
            showlegend=False,
        )
        return fig

    def plot_composition(self) -> go.Figure:
        """Plot the Portfolio composition.

        Returns
        -------
        plot : Figure
            Returns the plot Figure object.
        """
        df = self.composition.T
        fig = px.bar(df, x=df.index, y=df.columns)
        fig.update_layout(
            title="Portfolio Composition",
            xaxis_title="Portfolio",
            yaxis_title="Weight",
            legend_title_text="Assets",
        )
        return fig
