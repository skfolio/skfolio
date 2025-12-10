import datetime as dt
from abc import ABC, abstractmethod
from collections.abc import Callable, ItemsView, Iterable
from typing import Any, Literal, Self, overload

import numpy as np
import pandas as pd
import sklearn.base as skb
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv
from sklearn.compose import make_column_selector

import skfolio.measures as sm
import skfolio.typing as skt
from skfolio.moments import BaseCovariance, BaseMu, EmpiricalCovariance, EmpiricalMu
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
        """Initialize MarketContext.

        Parameters
        ----------
        date : dt.date, optional
            The pricing date for the market context. If not provided uses the current date.
        **kwargs : Any
            Additional market parameters as key-value pairs.

        Raises
        ------
        ValueError
            If date is not None and not of type datetime.date.
        """
        if date is None:
            date = dt.date.today()
        elif not isinstance(date, dt.date):
            raise ValueError(
                f"MarketContext requires a valid date of type datetime.date, not {type(date)}"
            )

        self.date = date
        self.data = dict(**kwargs)

    @classmethod
    def from_series(cls, series: pd.Series, **kwargs) -> Self:
        """Create a MarketContext from a pandas Series.

        Parameters
        ----------
        series : pd.Series
            A pandas Series where the index name is used as the date and
            the values become market parameters.
        **kwargs : Any
            Additional market parameters as key-value pairs.

        Returns
        -------
        MarketContext
            A new MarketContext instance.
        """
        date = series.name
        if isinstance(date, pd.Timestamp):
            date = date.date()
        elif not isinstance(date, dt.date):
            raise ValueError(
                f"Series name must be of type datetime.date or pd.Timestamp, not {type(date)}"
            )

        return cls(date=date, **series, **kwargs)

    def update_date(self, date: dt.date) -> Self:
        """Update the date of the market context.

        Parameters
        ----------
        date : dt.date
            The new date to set.

        Returns
        -------
        MarketContext
            Self for method chaining.
        """
        self.date = date
        return self

    def update_from_series(self, series: pd.Series, overwrite=True) -> Self:
        """Update the market context from a pandas Series.

        Parameters
        ----------
        series : pd.Series
            A pandas Series containing market parameters.
        overwrite : bool, default=True
            If True, existing parameters are overwritten. If False, existing
            parameters are preserved.

        Returns
        -------
        MarketContext
            Self for method chaining.
        """
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
        """Get the market parameters. Necessary for sklearn cloning.

        Parameters
        ----------
        deep : bool, default=True
            If True, performs a deep copy (not currently used).

        Returns
        -------
        dict[str, Any]
            Dictionary of market parameters.
        """
        return self.data

    def __len__(self) -> int:
        """Return the number of market parameters.

        Returns
        -------
        int
            Number of parameters in the market context.
        """
        return self.data.__len__()

    def __getitem__(self, key):
        """Get a market parameter by key.

        Parameters
        ----------
        key : str
            The parameter key.

        Returns
        -------
        Any
            The parameter value.
        """
        return self.data.__getitem__(key)

    def __setitem__(self, key, value) -> None:
        """Set a market parameter.

        Parameters
        ----------
        key : str
            The parameter key.
        value : Any
            The parameter value.
        """
        self.data.__setitem__(key, value)

    def __delitem__(self, key) -> None:
        """Delete a market parameter.

        Parameters
        ----------
        key : str
            The parameter key to delete.
        """
        self.data.__delitem__(key)

    def __iter__(self):
        """Iterate over market parameter keys.

        Returns
        -------
        iterator
            Iterator over parameter keys.
        """
        return self.data.__iter__()

    def __getattr__(self, attr):
        """Get an attribute from the underlying data dictionary.

        Parameters
        ----------
        attr : str
            The attribute name.

        Returns
        -------
        Any
            The attribute value.
        """
        return getattr(self.data, attr)


class Instrument(ABC):
    @abstractmethod
    def price(self, market_context: MarketContext) -> float:
        """Price the instrument given a market context.

        Parameters
        ----------
        market_context : MarketContext
            The market context containing pricing parameters.

        Returns
        -------
        float
            The price of the instrument.
        """
        pass

    def cashflow(self, market_context: MarketContext) -> float:
        """Calculate the cashflow of the instrument given a market context.
        The default implementation returns zero cashflows but this should be
        overridden by instruments that pay coupons or dividends.

        Although many instruments' cashflows may depend only on time (e.g.,
        coupons paid at specific dates), many more exotic instruments have
        conditional coupons. Therefore we require a full MarketContext to be
        passed instead of only a date.

        Parameters
        ----------
        market_context : MarketContext
            The market context containing pricing parameters.

        Returns
        -------
        float
            The cashflow of the instrument.
        """
        return 0.


class InstrumentAdapter(Instrument):
    def __init__(self, instrument):
        """Initialize InstrumentAdapter.

        Parameters
        ----------
        instrument : object
            The instrument to wrap.
        """
        self.instrument = instrument

    def __getattr__(self, attr):
        """Get an attribute from the wrapped instrument.

        All non-adapted calls are passed to the original instrument object.

        Parameters
        ----------
        attr : str
            The attribute name.

        Returns
        -------
        Any
            The attribute value from the wrapped instrument.
        """
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
        """Initialize PortfolioInstruments.

        Parameters
        ----------
        **kwargs : dict[str, Instrument]
            Instrument IDs mapped to Instrument objects.

        Raises
        ------
        ValueError
            If any value is not an Instrument or any key is not a string.
        """
        if not all(isinstance(instr, Instrument) for instr in kwargs.values()):
            raise ValueError("All portfolio items must be of type Instrument")

        if not all(isinstance(instr_id, str) for instr_id in kwargs.keys()):
            raise ValueError("All portfolio instrument ids must be of type str")

        super().__init__(**kwargs)

    def price(self, market_context: MarketContext) -> pd.Series:
        """Price all instruments in the portfolio.

        Parameters
        ----------
        market_context : MarketContext
            The market context containing pricing parameters.

        Returns
        -------
        pd.Series
            Series of prices for each instrument, indexed by instrument ID.
        """
        return pd.Series(
            {instr_id: instr.price(market_context) for instr_id, instr in self.items()},
            name=market_context.date,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get the portfolio instruments. Necessary for sklearn cloning.

        Parameters
        ----------
        deep : bool, default=True
            If True, performs a deep copy (not currently used).

        Returns
        -------
        dict[str, Any]
            The portfolio instruments dictionary.
        """
        return self

    @overload
    def __getitem__(self, key: str) -> Instrument: ...

    @overload
    def __getitem__(self, key: Iterable) -> Self: ...

    def __getitem__(self, key: str | Iterable) -> Instrument | Self:
        """Get instrument(s) by key(s).

        Parameters
        ----------
        key : str or Iterable
            The instrument ID or an iterable of instrument IDs.

        Returns
        -------
        Instrument or List[Instrument]
            The requested instrument(s).
        """
        if isinstance(key, str):
            return dict.__getitem__(self, key)
        elif isinstance(key, Iterable):
            return self.__class__(**{k: dict.__getitem__(self, k) for k in key})
        else:
            raise KeyError(f"Invalid key type: {type(key)}")

    def items(self) -> ItemsView[str, Instrument]:
        """Get the items of the portfolio instruments.

        Explicitly defined to provide proper type hints.

        Returns
        -------
        ItemsView
            The items of the portfolio instruments.
        """
        return dict.items(self)


def _default_market_data_parser(
    row: pd.Series,
    reference_market_context: MarketContext,
    portfolio_instruments: PortfolioInstruments,
):
    return reference_market_context.update_from_series(row)


def price_df(
    X: pd.DataFrame,
    portfolio_instruments: PortfolioInstruments,
    reference_market_context: MarketContext | None = None,
    market_data_parser: Callable[
        [pd.Series, MarketContext, PortfolioInstruments], MarketContext
    ] = _default_market_data_parser,
    use_date_index: bool = True,
) -> pd.DataFrame:
    """Price portfolio instruments over a time series of market data.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing market data over time.
    portfolio_instruments : PortfolioInstruments
        Portfolio of instruments to price.
    reference_market_context : MarketContext, optional
        Reference market context to update with data from X.
        Defaults to empty MarketContext.
    market_data_parser : Callable[[pd.Series], dict], optional
        Function to parse each row of market data.
    use_date_index : bool, default=True
        Whether to use the DataFrame index as the pricing date.

    Returns
    -------
    pd.DataFrame
        DataFrame of prices for each instrument over time.
    """
    reference_market_context = skb.clone(reference_market_context)
    if reference_market_context is None:
        reference_market_context = MarketContext()

    if not use_date_index:
        X.index = [reference_market_context.date] * len(X)

    rows = []
    for _, market_data in X.iterrows():
        reference_market_context = market_data_parser(
            market_data, reference_market_context, portfolio_instruments
        )

        rows.append(portfolio_instruments.price(reference_market_context))

    return pd.concat(rows, axis=1).T


def adjust_prices_for_cashflows(
    prices: pd.DataFrame,
    portfolio_instruments: PortfolioInstruments,
    method: Literal["simple", "reinvested"] = "simple",
    include_redemption: bool = False,
) -> pd.DataFrame:
    """Adjust prices for cashflows from the instruments.
    The resulting price series can be passed to prices_to_returns functions
    to obtain total returns.

    When using method "simple", cashflows are added to the price series.
    When using method "reinvested", cashflows are assumed to be reinvested
    on the same day the cashflow is received.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of prices for each instrument over time.
    portfolio_instruments : PortfolioInstruments
        Portfolio of instruments.

    Returns
    -------
    pd.DataFrame
        DataFrame of adjusted prices.
    """
    adjusted_prices = prices.copy()
    cashflows = []
    for date in adjusted_prices.index:
        market_context = MarketContext(date=date)
        cashflows_row = []
        for instr_id in adjusted_prices.columns:
            instr = portfolio_instruments[instr_id]
            cashflow = instr.cashflow(market_context)
            cashflows_row.append(cashflow)
        cashflows.append(cashflows_row)

    cashflows_df = pd.DataFrame(
        cashflows, index=adjusted_prices.index, columns=adjusted_prices.columns
    )

    match method:
        case "simple":
            adjusted_prices += cashflows_df.cumsum()
        case "reinvested":
            adjusted_prices *= (1 + cashflows_df / prices).cumprod()
        case _:
            raise ValueError(f"Unknown adjustment method: {method}")

    return adjusted_prices


class ReturnsProcessor:
    def __init__(
        self,
        periods: int = 1,
        freq: str | None = None,
        return_types: skt.ReturnType
        | dict[str, skt.ReturnType]
        | dict[make_column_selector, skt.ReturnType] = "linear",
        drop_nan: bool = True,
    ):
        """Initialize ReturnsProcessor.

        Parameters
        ----------
        periods : int, default=1
            Number of periods for return calculation.
        freq : str, optional
            Frequency string for return calculation.
        return_types : skt.ReturnType or dict, default="linear"
            Type of returns to calculate. Can be a single return type or a
            dictionary mapping column selectors to return types.
        """
        self.periods = periods
        self.freq = freq
        self.return_types = return_types
        self.drop_nan = drop_nan

    def linear_returns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate linear (percentage) returns.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame of prices.

        Returns
        -------
        pd.DataFrame
            DataFrame of linear returns.

        Raises
        ------
        ValueError
            If any prices are non-positive.
        """
        if np.any(X <= 0):
            raise ValueError("Prices must be positive to compute returns.")

        return X.pct_change(freq=self.freq, periods=self.periods).iloc[self.periods :]

    def log_returns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate logarithmic returns.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame of prices.

        Returns
        -------
        pd.DataFrame
            DataFrame of log returns.
        """
        return np.log1p(self.linear_returns(X)).iloc[self.periods :]

    def arithmetic_returns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate arithmetic (simple difference) returns.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame of prices.

        Returns
        -------
        pd.DataFrame
            DataFrame of arithmetic returns.
        """
        return X.diff(periods=self.periods).iloc[self.periods :]

    def prices_to_returns(
        self, X: pd.DataFrame, return_type: skt.ReturnType
    ) -> pd.DataFrame:
        """Convert prices to returns based on return type.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame of prices.
        return_type : skt.ReturnType
            Type of returns: "linear", "log", or "arithmetic".

        Returns
        -------
        pd.DataFrame
            DataFrame of returns.

        Raises
        ------
        ValueError
            If return_type is not recognized.
        """
        match return_type:
            case "linear":
                df = self.linear_returns(X)
            case "log":
                df = self.log_returns(X)
            case "arithmetic":
                df = self.arithmetic_returns(X)
            case _:
                raise ValueError(f"Unknown return_type: {return_type}")

        if self.drop_nan:
            df = df.dropna(axis=1)

        return df

    def returns_to_prices(
        self,
        returns: pd.DataFrame,
        reference_prices: pd.Series,
        return_type: skt.ReturnType,
    ) -> pd.DataFrame:
        """Convert returns to prices based on return type.

        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame of returns.
        reference_prices : pd.Series
            Series of reference prices to use as the base.
        return_type : skt.ReturnType
            Type of returns: "linear", "log", or "arithmetic".

        Returns
        -------
        pd.DataFrame
            DataFrame of prices.

        Raises
        ------
        ValueError
            If return_type is not recognized.
        """
        match return_type:
            case "linear":
                prices = reference_prices.values * (1 + returns)
            case "log":
                prices = reference_prices.values * np.exp(returns)
            case "arithmetic":
                prices = reference_prices.values + returns
            case _:
                raise ValueError(f"Unknown return_type: {return_type}")

        df = pd.DataFrame(prices, index=returns.index, columns=returns.columns)
        if self.drop_nan:
            df = df.dropna(how="all")

        return df

    def df_to_returns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert a DataFrame of prices to returns.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame of prices.

        Returns
        -------
        pd.DataFrame
            DataFrame of returns.

        Raises
        ------
        ValueError
            If return_types configuration is invalid.
        """
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
        """Convert a DataFrame of returns to prices.

        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame of returns.
        reference_prices : pd.Series
            Series of reference prices to use as the base.

        Returns
        -------
        pd.DataFrame
            DataFrame of prices.

        Raises
        ------
        ValueError
            If return_types configuration is invalid.
        """
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
    reference_market_quotes: pd.Series,
    reference_market_context: MarketContext,
    portfolio_instruments: PortfolioInstruments,
    market_data_parser: Callable[
        [pd.Series, MarketContext, PortfolioInstruments], MarketContext
    ]
    | None = None,
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
    keys : list[str], optional
        The list of market context parameter keys to calculate sensitivities for.
        If None, all keys in the market context are used.
    bump_size : float, default=1e-4
        The size of the bump to apply to each market parameter (1 basis point by default).
    mode : {"central", "forward", "backward"}, default="central"
        The finite difference method to use:
        - "central": Central difference (most accurate)
        - "forward": Forward difference
        - "backward": Backward difference
    percent : bool, default=True
        Whether to express sensitivities as percentage changes.
        For example, setting this to True would correspond to calculating modified duration,
        while setting it False would correspond to DV01.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the sensitivities of each instrument to each market parameter.
    """
    keys = keys if keys is not None else list(reference_market_quotes.index)
    sensitivities = pd.DataFrame(index=portfolio_instruments.keys(), columns=keys)

    if market_data_parser is None:
        market_data_parser = _default_market_data_parser

    pricing_context = market_data_parser(
        reference_market_quotes, reference_market_context, portfolio_instruments
    )
    original_prices = portfolio_instruments.price(pricing_context)
    denominator = bump_size * original_prices if percent else bump_size

    for param in keys:
        original_value = reference_market_quotes[param]

        match mode:
            case "forward":
                # Bump up
                reference_market_quotes[param] = original_value + bump_size
                bumped_up_prices = portfolio_instruments.price(
                    market_data_parser(
                        reference_market_quotes, pricing_context, portfolio_instruments
                    )
                )

                # Restore original value
                reference_market_quotes[param] = original_value

                # Calculate sensitivity using forward difference
                sensitivity = (bumped_up_prices - original_prices) / denominator
                sensitivities[param] = sensitivity
            case "backward":
                # Bump down
                reference_market_quotes[param] = original_value - bump_size
                bumped_down_prices = portfolio_instruments.price(
                    market_data_parser(
                        reference_market_quotes, pricing_context, portfolio_instruments
                    )
                )

                # Restore original value
                reference_market_quotes[param] = original_value

                # Calculate sensitivity using backward difference
                sensitivity = (original_prices - bumped_down_prices) / denominator
                sensitivities[param] = sensitivity
            case "central":
                # Bump up
                reference_market_quotes[param] = original_value + bump_size
                bumped_up_prices = portfolio_instruments.price(
                    market_data_parser(
                        reference_market_quotes, pricing_context, portfolio_instruments
                    )
                )

                # Bump down
                reference_market_quotes[param] = original_value - bump_size
                bumped_down_prices = portfolio_instruments.price(
                    market_data_parser(
                        reference_market_quotes, pricing_context, portfolio_instruments
                    )
                )
                # Restore original value
                reference_market_quotes[param] = original_value

                # Calculate sensitivity using central difference
                sensitivity = (bumped_up_prices - bumped_down_prices) / (
                    2 * denominator
                )
                sensitivities[param] = sensitivity

    return sensitivities


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
        market_data_parser: Callable[
            [pd.Series, MarketContext, PortfolioInstruments], MarketContext
        ] = _default_market_data_parser,
        reference_index=-1,
        pricing_date_offset: pd.offsets.BaseOffset | None = None,
        returns_processor: ReturnsProcessor | None = None,
        transform_quotes_prior_moments: bool = False,
        mu_estimator: BaseMu | None = None,
        covariance_estimator: BaseCovariance | None = None,
    ):
        """Initialize NonLinearPrior.

        Parameters
        ----------
        portfolio_instruments : PortfolioInstruments
            The portfolio of instruments to price.
        market_quotes_prior : BasePrior, optional
            Prior estimator for market quotes. Defaults to EmpiricalPrior.
        reference_market_context : MarketContext, optional
            Reference market context for pricing. Defaults to empty MarketContext.
        market_data_parser : Callable[[pd.Series], dict], optional
            Function to parse market data from a Series.
        reference_index : int, default=-1
            Index to use as reference point.
        pricing_date_offset : str, default="1B"
            Date offset for pricing (e.g., "1B" for one business day).
        returns_processor : ReturnsProcessor, optional
            Processor for handling returns calculations. Defaults to ReturnsProcessor().
        transform_quotes_prior_moments : bool, default=True
            Whether to transform moments from the quotes prior using sensitivities.
        """
        self.portfolio_instruments = portfolio_instruments
        self.market_quotes_prior = market_quotes_prior or EmpiricalPrior()
        self.reference_index = reference_index
        self.reference_market_context = (
            reference_market_context
            if reference_market_context is not None
            else MarketContext()
        )
        self.market_data_parser = market_data_parser
        self.pricing_date_offset = (
            pricing_date_offset if pricing_date_offset else pd.offsets.BDay(1)
        )
        self.returns_processor = returns_processor or ReturnsProcessor()
        self.transform_quotes_prior_moments = (
            transform_quotes_prior_moments if market_quotes_prior else False
        )  # Only transform is a prior has been provided (which may have different moments than empirical)
        self.mu_estimator = mu_estimator
        self.covariance_estimator = covariance_estimator

    def get_metadata_routing(self):
        """Get metadata routing for this estimator.

        Returns
        -------
        MetadataRouter
            The metadata router for routing fit parameters.
        """
        # noinspection PyTypeChecker
        router = (
            skm.MetadataRouter(owner=self.__class__.__name__)
            .add_self_request(self)
            .add(
                market_quotes_prior=self.market_quotes_prior,
                method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
            )
            .add(
                mu_estimator=self.mu_estimator,
                method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
            )
            .add(
                covariance_estimator=self.covariance_estimator,
                method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
            )
        )
        return router

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
        market_quotes: pd.DataFrame | None = None,
        **fit_params,
    ):
        """Fit the NonLinearPrior model.

        Parameters
        ----------
        X : pd.DataFrame
            Historical asset prices (not used directly).
        y : Ignored
            Not used, present for API consistency.
        market_quotes : npt.ArrayLike, optional
            Historical market quotes used for pricing the instruments.
        **fit_params : dict
            Additional fit parameters to route to sub-estimators.

        Returns
        -------
        self : NonLinearPrior
            Fitted estimator.
        """
        if market_quotes is not None:
            # noinspection PyTypeChecker
            fit_params["market_quotes"] = market_quotes
        else:
            raise ValueError(
                "market_quotes metadata must be provided to fit NonLinearPrior."
            )

        routed_params = skm.process_routing(self, "fit", **fit_params)

        # Validate X and get feature_names_in_
        skv.validate_data(self, X, ensure_all_finite="allow-nan")
        # Only keep instruments that are in X
        self.portfolio_instruments_ = self.portfolio_instruments[self.feature_names_in_]

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

        reference_date = market_quotes.index[-1]
        self.pricing_date_ = reference_date + self.pricing_date_offset
        reference_quotes = market_quotes.loc[
            reference_date,
            quote_returns.columns,  # Remove any columns that were dropped due to NaNs
        ]
        self.reference_market_context = self.market_data_parser(
            reference_quotes,
            self.reference_market_context,
            self.portfolio_instruments_,
        )

        self.market_quote_distribution_ = self.returns_processor.returns_to_df(
            prior_quote_returns, reference_quotes
        )
        self.reference_prices_ = self.portfolio_instruments_.price(
            self.reference_market_context
        )
        self.reference_market_context.update_date(self.pricing_date_)
        portfolio_price_distribution = price_df(
            self.market_quote_distribution_,
            self.portfolio_instruments_,
            reference_market_context=self.reference_market_context,
            market_data_parser=self.market_data_parser,
            use_date_index=False,
        )

        returns = (
            portfolio_price_distribution - self.reference_prices_.values
        ) / self.reference_prices_.values
        # print(portfolio_price_distribution, reference_prices, returns)

        moment_estimators_given = (self.mu_estimator is not None) or (
            self.covariance_estimator is not None
        )

        if sample_weight is None:
            sample_weight = np.ones(n_observations) / n_observations
        elif moment_estimators_given:
            raise ValueError(
                "When mu_estimator or covariance_estimator are provided, "
                "the sample_weight of the market_quotes_prior must be None."
                f"Got sample_weight={sample_weight}"
            )

        if moment_estimators_given:
            self.mu_estimator_ = check_estimator(
                self.mu_estimator,
                default=EmpiricalMu(),
                check_type=BaseMu,
            )
            self.covariance_estimator_ = check_estimator(
                self.covariance_estimator,
                default=EmpiricalCovariance(),
                check_type=BaseCovariance,
            )
            # fitting estimators
            # Expected returns
            # noinspection PyArgumentList
            self.mu_estimator_.fit(returns, y, **routed_params.mu_estimator.fit)
            mu = self.mu_estimator_.mu_

            # Covariance
            # noinspection PyArgumentList
            self.covariance_estimator_.fit(
                returns, y, **routed_params.covariance_estimator.fit
            )
            covariance = self.covariance_estimator_.covariance_
        else:
            mu = sm.mean(returns, sample_weight=sample_weight).values
            covariance = np.cov(returns, rowvar=False, aweights=sample_weight)

        if self.transform_quotes_prior_moments:
            sensis = calculate_sensis(
                reference_quotes,
                self.reference_market_context,
                self.portfolio_instruments_,
                market_data_parser=self.market_data_parser,
                keys=self.market_quotes_prior_.feature_names_in_,
            )
            market_quotes_mu = self.market_quotes_prior_.return_distribution_.mu
            market_quotes_covariance = (
                self.market_quotes_prior_.return_distribution_.covariance
            )

            mu = (
                self.reference_prices_.values + sensis.values @ market_quotes_mu
            ) / self.reference_prices_.values - 1

            covariance = sensis.values @ market_quotes_covariance @ sensis.values.T

        self.return_distribution_ = ReturnDistribution(
            mu=mu,
            covariance=covariance,
            returns=returns.values,
            sample_weight=sample_weight,
        )

        return self
