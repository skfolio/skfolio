import datetime as dt
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv
from pandas.tseries.frequencies import to_offset
from sklearn.compose import make_column_selector

import skfolio.measures as sm
import skfolio.typing as skt
from skfolio.prior import BasePrior, EmpiricalPrior, ReturnDistribution
from skfolio.utils.tools import check_estimator


class MarketContext:
    """The market context class contains a date and a dictionary of pricing parameters.
    Ideally, these parameters would include all data needed to price instruments, such that
    an instrument priced with a given MarketContext will always return the same price.

    MarketContexts will, typically, be constructed from DataFrames passed to SKLearn
    transformers. So why create a special MarketContext class when we could just use
    a pandas series? The reason being that MarketContexts may include more than just
    prices or simple market quotes. For example, a MarketContext may include discount curves
    or other complex data structures. That said, for ease of use we provide for a `from_series`
    method to convert between pandas Series and MarketContexts.

    No instrument can be priced without reference to a pricing date so we require that all
    MarketContexts include a `date` attribute.
    """

    data: dict[str, Any]
    date: dt.date

    def __init__(self, date: dt.date | None = None, **kwargs: Any):
        if (date is not None) and (not isinstance(date, dt.date)):
            raise ValueError(
                f"MarketContext requires a valid date of type datetime.date, not {type(date)}"
            )

        self.date = date
        self.data = dict(**kwargs)

    @classmethod
    def from_series(cls, series: pd.Series) -> "MarketContext":
        date = series.name
        return cls(date=date, **series)

    def update_date(self, date: dt.date) -> "MarketContext":
        self.date = date
        return self

    def update_from_series(self, series: pd.Series, overwrite=True) -> "MarketContext":
        if (self.date is None) or overwrite:
            if isinstance(series.name, dt.date):
                self.date = series.name
            elif isinstance(series.name, pd.Timestamp):
                self.date = series.name.date()

        if overwrite:
            self.data.update(**series)
        else:
            self.data = dict(series) | self.data

        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return self.data

    def __len__(self) -> int:
        return self.data.__len__()

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __setitem__(self, key, value) -> None:
        self.data.__setitem__(key, value)

    def __delitem__(self, key) -> None:
        self.data.__delitem__(key)

    def __iter__(self):
        return self.data.__iter__()

    def __getattr__(self, attr):
        return getattr(self.data, attr)


class Instrument(ABC):
    @abstractmethod
    def price(self, market_context: MarketContext) -> float:
        pass


class InstrumentAdapter(Instrument):
    def __init__(self, instrument):
        self.instrument = instrument

    def __getattr__(self, attr):
        """All non-adapted calls are passed to the original instrument object."""
        return getattr(self.instrument, attr)


class PortfolioInstruments(dict):
    """
    A collection of instruments in a portfolio, identified by string ids.
    Note that this class does not properly define a "portfolio" because it does not
    include the weighting of each instrument in the portfolio. In contrast, weighting
    are assigned in the SKfolio `Portfolio` class. This class is simply designed to
    help translate `MarketContext`s into price series for the instruments in the `Portfolio`.

    For this reason the method `price` returns a pd.Series of prices for each instrument
    in the portfolio, rather than a single price for the entire portfolio.
    """

    def __init__(self, **kwargs: dict[str, Instrument]):
        if not all(isinstance(instr, Instrument) for instr in kwargs.values()):
            raise ValueError("All portfolio items must be of type Instrument")

        if not all(isinstance(instr_id, str) for instr_id in kwargs.keys()):
            raise ValueError("All portfolio instrument ids must be of type str")

        super().__init__(**kwargs)

    def price(self, market_context: MarketContext) -> pd.Series:
        return pd.Series(
            {instr_id: instr.price(market_context) for instr_id, instr in self.items()},
            name=market_context.date,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return self


def price_df(
    X: pd.DataFrame,
    portfolio_instruments: PortfolioInstruments,
    reference_market_context: MarketContext | None = None,
    market_data_parser: Callable[[pd.Series], dict] | None = None,
    use_date_index: bool = True,
) -> pd.DataFrame:
    
    reference_market_context = (
        reference_market_context if reference_market_context is not None 
        else MarketContext()
    )

    rows = []
    for idx, market_data in X.iterrows():
        if market_data_parser:
            market_data = market_data_parser(market_data)
        reference_market_context.update(**market_data)

        if use_date_index and isinstance(idx, dt.date):
            reference_market_context.update_date(idx)

        rows.append(portfolio_instruments.price(reference_market_context))

    return pd.concat(rows, axis=1).T


class ReturnsProcessor:
    def __init__(
        self,
        periods: int = 1,
        freq: str | None = None,
        return_types: skt.ReturnType
        | dict[str, skt.ReturnType]
        | dict[make_column_selector, skt.ReturnType] = "linear",
    ):
        self.periods = periods
        self.freq = freq
        self.return_types = return_types

    def linear_returns(self, X: pd.DataFrame) -> pd.DataFrame:
        if np.any(X <= 0):
            raise ValueError("Prices must be positive to compute returns.")

        return X.pct_change(freq=self.freq, periods=self.periods).iloc[self.periods :]

    def log_returns(self, X: pd.DataFrame) -> pd.DataFrame:
        return np.log1p(self.linear_returns(X)).iloc[self.periods :]

    def arithmetic_returns(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.diff(periods=self.periods).iloc[self.periods :]

    def prices_to_returns(
        self, X: pd.DataFrame, return_type: skt.ReturnType
    ) -> pd.DataFrame:
        match return_type:
            case "linear":
                return self.linear_returns(X)
            case "log":
                return self.log_returns(X)
            case "arithmetic":
                return self.arithmetic_returns(X)
            case _:
                raise ValueError(f"Unknown return_type: {return_type}")

    def returns_to_prices(
        self,
        returns: pd.DataFrame,
        reference_prices: pd.Series,
        return_type: skt.ReturnType,
    ) -> pd.DataFrame:
        match return_type:
            case "linear":
                prices = reference_prices.values * (1 + returns)
            case "log":
                prices = reference_prices.values * np.exp(returns)
            case "arithmetic":
                prices = reference_prices.values + returns
            case _:
                raise ValueError(f"Unknown return_type: {return_type}")
        return pd.DataFrame(prices, index=returns.index, columns=returns.columns)

    def df_to_returns(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(self.return_types, str):
            return self.prices_to_returns(X, self.return_types)
        elif isinstance(self.return_types, dict):
            returns_list = []
            for key, ret_type in self.return_types.items():
                if callable(key):
                    cols = key(X)
                elif isinstance(key, str):
                    cols = [key]
                else:
                    raise ValueError(
                        f"Invalid key type in return_type dict: {type(key)} for key {key}"
                    )

                X_subset = X[cols]
                returns_subset = self.prices_to_returns(X_subset, ret_type)
                returns_list.append(returns_subset)

            return pd.concat(returns_list, axis=1)
        else:
            raise ValueError(f"Invalid type for return_type: {type(self.return_types)}")

    def returns_to_df(
        self, returns: pd.DataFrame, reference_prices: pd.Series
    ) -> pd.DataFrame:
        if isinstance(self.return_types, str):
            return self.returns_to_prices(returns, reference_prices, self.return_types)
        elif isinstance(self.return_types, dict):
            prices_list = []
            for key, return_type in self.return_types.items():
                if callable(key):
                    cols = key(returns)
                elif isinstance(key, str):
                    cols = [key]
                else:
                    raise ValueError(
                        f"Invalid key type in return_type dict: {type(key)} for key {key}"
                    )

                returns_subset = returns[cols]
                reference_prices_subset = reference_prices[cols]
                prices_subset = self.returns_to_prices(
                    returns_subset, reference_prices_subset, return_type
                )

                prices_list.append(
                    pd.DataFrame(
                        prices_subset,
                        index=returns_subset.index,
                        columns=returns_subset.columns,
                    )
                )

            return pd.concat(prices_list, axis=1)
        else:
            raise ValueError(f"Invalid type for return_type: {type(self.return_types)}")


def calculate_sensis(
    market_context: MarketContext,
    portfolio_instruments: PortfolioInstruments,
    keys: list[str] | None = None,
    bump_size: float = 1e-4,
    mode: Literal["central", "forward", "backward"] = "central",
    percent: bool = True,
) -> pd.DataFrame:
    """Calculate the sensitivities (Greeks) of the portfolio instruments to the market context parameters.

    Parameters
    ----------
    market_context : MarketContext
        The market context containing pricing parameters.
    portfolio_instruments : PortfolioInstruments
        The portfolio of instruments to calculate sensitivities for.
    keys : list[str]
        The list of market context parameter keys to calculate sensitivities for.
    bump_size : float, optional
        The size of the bump to apply to each market parameter, by default 1e-4 (1bp).
    percent : bool, optional
        Whether to express sensitivities as percentage changes, by default True.
        For example, setting this to True would correspond to calculating modified duration,
        while setting it False would correspond to DV01.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the sensitivities of each instrument to each market parameter.
    """
    keys = keys if keys is not None else list(market_context.keys())
    sensitivities = pd.DataFrame(index=portfolio_instruments.keys(), columns=keys)

    original_prices = portfolio_instruments.price(market_context)
    denominator = bump_size * original_prices if percent else bump_size

    for param in keys:
        original_value = market_context[param]

        match mode:
            case "forward":
                # Bump up
                market_context[param] = original_value + bump_size
                bumped_up_prices = portfolio_instruments.price(market_context)

                # Restore original value
                market_context[param] = original_value

                # Calculate sensitivity using forward difference
                sensitivity = (bumped_up_prices - original_prices) / denominator
                sensitivities[param] = sensitivity
            case "backward":
                # Bump down
                market_context[param] = original_value - bump_size
                bumped_down_prices = portfolio_instruments.price(market_context)

                # Restore original value
                market_context[param] = original_value

                # Calculate sensitivity using backward difference
                sensitivity = (original_prices - bumped_down_prices) / denominator
                sensitivities[param] = sensitivity
            case "central":
                # Bump up
                market_context[param] = original_value + bump_size
                bumped_up_prices = portfolio_instruments.price(market_context)

                # Bump down
                market_context[param] = original_value - bump_size
                bumped_down_prices = portfolio_instruments.price(market_context)

                # Restore original value
                market_context[param] = original_value

                # Calculate sensitivity using central difference
                sensitivity = (bumped_up_prices - bumped_down_prices) / (
                    2 * denominator
                )
                sensitivities[param] = sensitivity

    return sensitivities.T


class NonLinearPrior(BasePrior):
    """The NonLinearPrior class creates a distribution of returns for a portfolio of non-linear instruments.
    Instead of the usual design pattern of fitting a prior using the historical asset returns, instead this
    prior is fitted using a history of market quotes used to price the given portfolio. These market quotes
    could be, for example, implied volatilities for options, z-scores for bonds, swap-rates for rates
    instruments, etc.
    """

    market_quotes_prior_: BasePrior

    def __init__(
        self,
        portfolio_instruments: PortfolioInstruments,
        market_quotes_prior: BasePrior | None = None,
        reference_market_context: MarketContext | None = None,
        market_data_parser: Callable[[pd.Series], dict] | None = None,
        reference_index=-1,
        pricing_date_offset: str = "1B",
        returns_processor: ReturnsProcessor | None = None,
        transform_quotes_prior_moments: bool = True,
    ):
        self.portfolio_instruments = portfolio_instruments
        self.market_quotes_prior = market_quotes_prior or EmpiricalPrior()
        self.reference_index = reference_index
        self.reference_market_context = (
            reference_market_context
            if reference_market_context is not None
            else MarketContext()
        )
        self.market_data_parser = market_data_parser
        self.pricing_date_offset = pricing_date_offset
        self.returns_processor = returns_processor or ReturnsProcessor()
        self.transform_quotes_prior_moments = (
            transform_quotes_prior_moments if market_quotes_prior else False
        )  # Only transform is a prior has been provided (which may have different moments than empirical)

    def get_metadata_routing(self):
        # noinspection PyTypeChecker
        router = (
            skm.MetadataRouter(owner=self.__class__.__name__)
            .add_self_request(self)
            .add(
                market_quotes_prior=self.market_quotes_prior,
                method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
            )
        )
        return router

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
        market_quotes: npt.ArrayLike | None = None,
        **fit_params,
    ):
        if market_quotes is not None:
            # noinspection PyTypeChecker
            fit_params["market_quotes"] = market_quotes

        routed_params = skm.process_routing(self, "fit", **fit_params)

        # Validation
        skv.validate_data(self, market_quotes)

        self.market_quotes_prior_ = check_estimator(
            self.market_quotes_prior,
            default=EmpiricalPrior(),
            check_type=BasePrior,
        )

        quote_returns = self.returns_processor.df_to_returns(market_quotes)

        # Fitting prior estimator
        self.market_quotes_prior_.fit(
            quote_returns, y, **routed_params.market_quotes_prior.fit
        )
        # Prior distribution
        prior_quote_returns = pd.DataFrame(
            self.market_quotes_prior_.return_distribution_.returns,
            columns=self.market_quotes_prior_.feature_names_in_,
        )
        sample_weight = self.market_quotes_prior_.return_distribution_.sample_weight
        n_observations = len(prior_quote_returns)
        if sample_weight is None:
            sample_weight = np.ones(n_observations) / n_observations

        reference_prices = self.portfolio_instruments.price(
            self.reference_market_context
        )

        reference_date = self.reference_market_context.date
        pricing_date = (
            (reference_date + to_offset(self.pricing_date_offset))
            if self.pricing_date_offset
            else reference_date
        )

        reference_quotes = pd.Series(
            {
                key: self.reference_market_context[key]
                for key in self.market_quotes_prior_.feature_names_in_
            }
        )

        market_quote_distribution = self.returns_processor.returns_to_df(
            prior_quote_returns, reference_quotes
        )

        self.reference_market_context.update_date(pricing_date)
        portfolio_price_distribution = price_df(
            market_quote_distribution,
            self.portfolio_instruments,
            reference_market_context=self.reference_market_context,
            market_data_parser=self.market_data_parser,
        )

        returns = (
            portfolio_price_distribution - reference_prices.values
        ) / reference_prices.values

        if self.transform_quotes_prior_moments:
            sensis = calculate_sensis(
                self.reference_market_context,
                self.portfolio_instruments,
                keys=self.market_quotes_prior_.feature_names_in_,
            )
            mu = (
                reference_prices.values
                + sensis.values @ self.market_quotes_prior_.return_distribution_.mu
            ) / reference_prices.values - 1

            cov = (
                sensis.values
                @ self.market_quotes_prior_.return_distribution_.covariance
                @ sensis.values.T
            )
            # mu = sm.mean(returns, sample_weight=sample_weight).values
            # cov = np.cov(returns, rowvar=False, aweights=sample_weight)
        else:
            mu = sm.mean(returns, sample_weight=sample_weight).values
            cov = np.cov(returns, rowvar=False, aweights=sample_weight)

        self.return_distribution_ = ReturnDistribution(
            mu=mu,
            covariance=cov,
            returns=returns.values,
            sample_weight=sample_weight,
        )

        return self
