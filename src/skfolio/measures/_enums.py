"""Module that includes all Measures enums used across `skfolio`."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause

from abc import abstractmethod
from enum import auto

from skfolio.utils.tools import AutoEnum


class BaseMeasure(AutoEnum):
    """Base Enum of measures"""

    def __repr__(self) -> str:
        """Enum representation for improved reading"""
        words = [
            (
                word.capitalize()
                if len(word) > 3
                else word.upper()
                if len(word) != 2
                else word.lower()
            )
            for word in self.value.split("_")
        ]
        if len(words[0]) in [3, 4] and words[0][-2:].lower() == "ar":
            s = list(words[0].upper())
            s[-2] = str(s[-2].lower())
            words[0] = "".join(s)
        string = " ".join(words).replace("Semi ", "Semi-")
        return string

    def __str__(self) -> str:
        return self.__repr__()

    @property
    @abstractmethod
    def is_perf(self):
        pass

    @property
    @abstractmethod
    def is_risk(self):
        pass

    @property
    @abstractmethod
    def is_ratio(self):
        pass

    @property
    def is_annualized(self) -> bool:
        return self.name[:10] == "ANNUALIZED"

    @property
    def annualized_measure(self):
        if self.is_annualized:
            raise ValueError(f"{self.name} is already an annualized measure")
        try:
            return getattr(self.__class__, f"ANNUALIZED_{self.name}")
        except AttributeError:
            raise AttributeError(
                f"{self.name} doesn't have a annualized version"
            ) from None

    @property
    def non_annualized_measure(self):
        if not self.is_annualized:
            raise ValueError(f"{self.name} is already a non-annualized measure")
        return getattr(self.__class__, self.name[11:])


class PerfMeasure(BaseMeasure):
    """Enumeration of performance measures

    Attributes
    ----------
    MEAN : str
        Mean

    ANNUALIZED_MEAN : str
       Annualized Mean
    """

    MEAN = auto()

    # Annualized measures
    ANNUALIZED_MEAN = auto()

    @property
    def is_perf(self) -> bool:
        return True

    @property
    def is_risk(self) -> bool:
        return False

    @property
    def is_ratio(self) -> bool:
        return False


class RiskMeasure(BaseMeasure):
    """Enumeration of risk measures

    Attributes
    ----------
    VARIANCE : str
        Variance.

    ANNUALIZED_VARIANCE : str
        Annualized Variance.

    SEMI_VARIANCE : str
        Semi-variance (Second Lower Partial Moment or Downside Variance).

    ANNUALIZED_SEMI_VARIANCE : str
        Annualized Semi-variance.

    STANDARD_DEVIATION : str
        Standard-deviation

    ANNUALIZED_STANDARD_DEVIATION : str
        Annualized Standard-deviation.

    SEMI_DEVIATION : str
        Semi-deviation.

    ANNUALIZED_SEMI_DEVIATION : str
        Annualized Semi-deviation.

    MEAN_ABSOLUTE_DEVIATION : str
        Mean Absolute Deviation.

    CVAR : str
        Conditional Value at Risk or Expected Shortfall.

    EVAR : str
        Entropic Value at Risk.

    WORST_REALIZATION : str
        Worst Realization (Worst Return).

    CDAR : str
        Conditional Drawdown at Risk.

    MAX_DRAWDOWN : str
        Maximum Drawdown.

    AVERAGE_DRAWDOWN : str
        Average Drawdown.

    EDAR : str
        Entropic Drawdown at Risk.

    FIRST_LOWER_PARTIAL_MOMENT : str
        First Lower Partial Moment.

    ULCER_INDEX : str
        Ulcer Index.

    GINI_MEAN_DIFFERENCE : str
        Gini Mean Difference.
    """

    VARIANCE = auto()
    ANNUALIZED_VARIANCE = auto()
    SEMI_VARIANCE = auto()
    ANNUALIZED_SEMI_VARIANCE = auto()
    STANDARD_DEVIATION = auto()
    ANNUALIZED_STANDARD_DEVIATION = auto()
    SEMI_DEVIATION = auto()
    ANNUALIZED_SEMI_DEVIATION = auto()
    MEAN_ABSOLUTE_DEVIATION = auto()
    CVAR = auto()
    EVAR = auto()
    WORST_REALIZATION = auto()
    CDAR = auto()
    MAX_DRAWDOWN = auto()
    AVERAGE_DRAWDOWN = auto()
    EDAR = auto()
    FIRST_LOWER_PARTIAL_MOMENT = auto()
    ULCER_INDEX = auto()
    GINI_MEAN_DIFFERENCE = auto()

    @property
    def is_perf(self) -> bool:
        return False

    @property
    def is_risk(self) -> bool:
        return True

    @property
    def is_ratio(self) -> bool:
        return False


class ExtraRiskMeasure(BaseMeasure):
    """Enumeration of other risk measures not used in convex optimization

    Attributes
    ----------
    VALUE_AT_RISK : str
        Value at Risk (VaR).

    DRAWDOWN_AT_RISK : str
        Drawdown at Risk.

    ENTROPIC_RISK_MEASURE : str
        Entropic Risk Measure.

    FOURTH_CENTRAL_MOMENT : str
        Fourth Central Moment.

    FOURTH_LOWER_PARTIAL_MOMENT : str
         Fourth Lower Central Moment.

    SKEW : str
        Skew.

    KURTOSIS : str
        Kurtosis.
    """

    VALUE_AT_RISK = auto()
    DRAWDOWN_AT_RISK = auto()
    ENTROPIC_RISK_MEASURE = auto()
    FOURTH_CENTRAL_MOMENT = auto()
    FOURTH_LOWER_PARTIAL_MOMENT = auto()
    SKEW = auto()
    KURTOSIS = auto()

    @property
    def is_perf(self) -> bool:
        return False

    @property
    def is_risk(self) -> bool:
        return True

    @property
    def is_ratio(self) -> bool:
        return False


class RatioMeasure(BaseMeasure):
    """Enumeration of ratio measures

    Attributes
    ----------
    SHARPE_RATIO : str
        Ratio of the excess Mean divided by the Standard-deviation.

    ANNUALIZED_SHARPE_RATIO : str
        Annualized Sharpe ratio.

    SORTINO_RATIO : str
         Ratio of the excess Mean divided by the Semi standard-deviation.

    ANNUALIZED_SORTINO_RATIO : str
        Annualized Sortino ratio.

    MEAN_ABSOLUTE_DEVIATION_RATIO : str
        Ratio of the excess Mean divided by the Mean Absolute Deviation.

    FIRST_LOWER_PARTIAL_MOMENT_RATIO : str
        Ratio of the excess Mean divided by the First Lower Partial Moment.

    VALUE_AT_RISK_RATIO : str
         Ratio of the excess Mean divided by the Value at Risk.

    CVAR_RATIO : str
         Ratio of the excess Mean divided by the Conditional Value at Risk.

    ENTROPIC_RISK_MEASURE_RATIO : str
         Ratio of the excess Mean divided by the Entropic Risk Measure.

    EVAR_RATIO : str
         Ratio of the excess Mean divided by the Entropic Value at Risk.

    WORST_REALIZATION_RATIO : str
         Ratio of the excess Mean divided by the Worst Realization.

    DRAWDOWN_AT_RISK_RATIO : str
         Ratio of the excess Mean divided by the Drawdown at Risk.

    CDAR_RATIO : str
         Ratio of the excess Mean divided by the Conditional Drawdown at Risk.

    CALMAR_RATIO : str
         Ratio of the excess Mean divided by the Maximum Drawdown.

    AVERAGE_DRAWDOWN_RATIO : str
         Ratio of the excess Mean divided by the Average Drawdown.

    EDAR_RATIO : str
         Ratio of the excess Mean divided by the Entropic Drawdown at Risk.

    ULCER_INDEX_RATIO : str
         Ratio of the excess Mean divided by the Ulcer Index.

    GINI_MEAN_DIFFERENCE_RATIO : str
         Ratio of the excess Mean divided by the Gini Mean Difference.
    """

    SHARPE_RATIO = auto()
    ANNUALIZED_SHARPE_RATIO = auto()
    SORTINO_RATIO = auto()
    ANNUALIZED_SORTINO_RATIO = auto()
    MEAN_ABSOLUTE_DEVIATION_RATIO = auto()
    FIRST_LOWER_PARTIAL_MOMENT_RATIO = auto()
    VALUE_AT_RISK_RATIO = auto()
    CVAR_RATIO = auto()
    ENTROPIC_RISK_MEASURE_RATIO = auto()
    EVAR_RATIO = auto()
    WORST_REALIZATION_RATIO = auto()
    DRAWDOWN_AT_RISK_RATIO = auto()
    CDAR_RATIO = auto()
    CALMAR_RATIO = auto()
    AVERAGE_DRAWDOWN_RATIO = auto()
    EDAR_RATIO = auto()
    ULCER_INDEX_RATIO = auto()
    GINI_MEAN_DIFFERENCE_RATIO = auto()

    @property
    def is_perf(self) -> bool:
        return False

    @property
    def is_risk(self) -> bool:
        return False

    @property
    def is_ratio(self) -> bool:
        return True

    @property
    def linked_risk_measure(self) -> RiskMeasure | ExtraRiskMeasure:
        match self:
            case RatioMeasure.SHARPE_RATIO:
                return RiskMeasure.STANDARD_DEVIATION
            case RatioMeasure.ANNUALIZED_SHARPE_RATIO:
                return RiskMeasure.ANNUALIZED_STANDARD_DEVIATION
            case RatioMeasure.SORTINO_RATIO:
                return RiskMeasure.SEMI_DEVIATION
            case RatioMeasure.ANNUALIZED_SORTINO_RATIO:
                return RiskMeasure.ANNUALIZED_SEMI_DEVIATION
            case RatioMeasure.CALMAR_RATIO:
                return RiskMeasure.MAX_DRAWDOWN
            case _:
                risk_measure = str(self.value).replace("_ratio", "")
                try:
                    return RiskMeasure(risk_measure)
                except ValueError:
                    return ExtraRiskMeasure(risk_measure)
