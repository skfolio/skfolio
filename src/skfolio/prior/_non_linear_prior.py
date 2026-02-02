import datetime as dt
from abc import ABC, abstractmethod
from collections.abc import ItemsView, Iterable
from itertools import product
from typing import Any, Literal, Self, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.base as skb
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv
from scipy.differentiate import hessian, jacobian
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
        return 0.0


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

    def keys(self) -> list[str]:
        """Get the items of the portfolio instruments.

        Explicitly defined to provide proper type hints.

        Returns
        -------
        ItemsView
            The items of the portfolio instruments.
        """
        return list(dict.keys(self))

    def values(self) -> list[Instrument]:
        """Get the items of the portfolio instruments.

        Explicitly defined to provide proper type hints.

        Returns
        -------
        ItemsView
            The items of the portfolio instruments.
        """
        return list(dict.values(self))


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
    market_data_parser: skt.MarketDataParser = _default_market_data_parser,
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


def get_cashflows(
    portfolio_instruments: PortfolioInstruments,
    dates: Iterable[dt.date],
    reference_market_context: MarketContext | None = None,
) -> pd.DataFrame:
    """Get cashflows from portfolio instruments over a series of dates.

    Parameters
    ----------
    portfolio_instruments : PortfolioInstruments
        The portfolio of instruments to get cashflows from.
    dates : Iterable[dt.date]
        An iterable of dates for which to calculate cashflows.
    reference_market_context : MarketContext, optional
        Reference market context to use. If None, an empty MarketContext is created.

    Returns
    -------
    pd.DataFrame
        DataFrame of cashflows indexed by dates with columns for each instrument.
    """
    if reference_market_context is None:
        reference_market_context = MarketContext()

    cashflows = []
    for date in dates:
        market_context = reference_market_context.update_date(date)
        cashflows_row = []
        for instr_id in portfolio_instruments.keys():
            instr = portfolio_instruments[instr_id]
            cashflow = instr.cashflow(market_context)
            cashflows_row.append(cashflow)
        cashflows.append(cashflows_row)

    return pd.DataFrame(
        cashflows, index=list(dates), columns=portfolio_instruments.keys()
    )


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
    cashflows_df = get_cashflows(portfolio_instruments, adjusted_prices.index)

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

    @overload
    def returns_to_df(
        self,
        returns: pd.DataFrame,
        reference_prices: pd.Series,
    ) -> pd.DataFrame: ...

    @overload
    def returns_to_df(
        self,
        returns: pd.Series,
        reference_prices: pd.Series,
    ) -> pd.Series: ...

    def returns_to_df(
        self, returns: pd.DataFrame | pd.Series, reference_prices: pd.Series
    ) -> pd.DataFrame | pd.Series:
        """Convert a DataFrame of returns to prices.

        Parameters
        ----------
        returns : pd.DataFrame | pd.Series
            DataFrame or Series of returns.
        reference_prices : pd.Series
            Series of reference prices to use as the base.

        Returns
        -------
        pd.DataFrame | pd.Series
            DataFrame or Series of prices.

        Raises
        ------
        ValueError
            If return_types configuration is invalid.
        """
        if isinstance(returns, pd.Series):
            returns = pd.DataFrame(returns).T

        if isinstance(self.return_types, str):
            out_df = self.returns_to_prices(
                returns, reference_prices, self.return_types
            )
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

            out_df = pd.concat(prices_list, axis=1)
        else:
            raise ValueError(f"Invalid type for return_type: {type(self.return_types)}")

        if isinstance(returns, pd.Series):
            return out_df.iloc[0]
        return out_df


@overload
def calculate_sensis(
    reference_market_quotes: pd.Series,
    reference_market_context: MarketContext,
    portfolio_instruments: PortfolioInstruments,
    market_data_parser: skt.MarketDataParser = _default_market_data_parser,
    keys: list[str] | None = None,
    differentiation_order: Literal[1, 2] = 1,
    percent: bool = True,
    *,
    pandas_output: Literal[False] = False,
    **kwargs,
) -> npt.NDArray[np.float64]: ...


@overload
def calculate_sensis(
    reference_market_quotes: pd.Series,
    reference_market_context: MarketContext,
    portfolio_instruments: PortfolioInstruments,
    market_data_parser: skt.MarketDataParser = _default_market_data_parser,
    keys: list[str] | None = None,
    differentiation_order: Literal[1, 2] = 1,
    percent: bool = True,
    *,
    pandas_output: Literal[True],
    **kwargs,
) -> pd.DataFrame: ...


def calculate_sensis(
    reference_market_quotes: pd.Series,
    reference_market_context: MarketContext,
    portfolio_instruments: PortfolioInstruments,
    market_data_parser: skt.MarketDataParser = _default_market_data_parser,
    keys: list[str] | None = None,
    differentiation_order: Literal[1, 2] = 1,
    percent: bool = True,
    *,
    pandas_output: bool = True,
    **kwargs,
) -> npt.NDArray[np.float64] | pd.DataFrame:
    """Calculate the sensitivities (Greeks) of the portfolio instruments to the market context parameters.
    Second-order differentiation is also supported but appears to be quite unstable. Use with caution.

    Parameters
    ----------
    reference_market_quotes : pd.Series
        The market quotes serving as the reference point for sensitivity calculations.

    reference_market_context : MarketContext
        The market context containing pricing parameters.

    portfolio_instruments : PortfolioInstruments
        The portfolio of instruments to calculate sensitivities for.

    market_data_parser : skt.MarketDataParser, optional
        Function to build a MarketContext from market quotes. Defaults to _default_market_data_parser.

    keys : list[str], optional
        The list of market context parameter keys to calculate sensitivities for.
        If None, all keys in the market context are used.

    differentiation_order : {1, 2}, default=1
        The order of the derivative to compute.

    percent : bool, default=False
        Whether to express sensitivities as percentage changes.

    pandas_output : bool, default=True
        Whether to return the result as a pandas DataFrame. Only supported for first-order derivatives.

    kwargs: dict, optional
        Arguments to be passed to scipy's finite differencing functions.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the sensitivities of each instrument to each market parameter.
    """
    # Some default parameters that worked well during tests
    finite_difference_params = {
        "initial_step": 1e-5,
        "order": 2,
        "maxiter": 2,
    } | kwargs

    if differentiation_order not in (1, 2):
        raise ValueError("Order must be 1 or 2.")

    keys = keys if keys is not None else list(reference_market_quotes.index)

    pricing_context = market_data_parser(
        reference_market_quotes, reference_market_context, portfolio_instruments
    )

    def price_function(market_quotes):
        temp_series = pd.Series(market_quotes, index=keys)
        temp_context = market_data_parser(
            temp_series, pricing_context, portfolio_instruments
        )
        return portfolio_instruments.price(temp_context).values

    def vectorized_price_function(x):
        return np.apply_along_axis(price_function, axis=0, arr=x)

    if differentiation_order == 1:
        res = jacobian(
            vectorized_price_function,
            reference_market_quotes[keys],
            **finite_difference_params,
        )
        df = res.df.T
        if pandas_output:
            df = pd.DataFrame(
                df,
                index=keys,
                columns=list(portfolio_instruments.keys()),
            )
    elif differentiation_order == 2:
        res = hessian(
            vectorized_price_function,
            reference_market_quotes[keys],
            **finite_difference_params,
        )
        df = res.ddf.T
        if pandas_output:
            m, r, n = df.shape
            df = pd.DataFrame(
                df.reshape(m * r, n),
                index=pd.MultiIndex.from_tuples(product(keys, repeat=2)),
                columns=list(portfolio_instruments.keys()),
            )

    if percent:
        df /= portfolio_instruments.price(pricing_context).values

    return df


class BaseSensiEstimator(skb.BaseEstimator, ABC):
    """Base class for sensitivity estimators."""

    sensi_: npt.NDArray[np.float64]

    @abstractmethod
    def __init__(self):
        pass

    def get_metadata_routing(self):
        # noinspection PyTypeChecker
        router = skm.MetadataRouter(owner=self.__class__.__name__).add_self_request(
            self
        )
        return router

    @abstractmethod
    def fit(
        self, X: npt.ArrayLike, y=None, market_quotes: pd.DataFrame | None = None
    ) -> Self:
        pass


class FiniteDifferenceSensiEstimator(BaseSensiEstimator):
    """Estimator that calculates sensitivities using finite differencing.

    Parameters
    ----------
    sensi_calculator : Callable[[pd.Series, Literal[1, 2]], npt.NDArray[np.float64]]
        Function to calculate sensitivities.

    finite_difference_kwargs : dict
        Additional keyword arguments to pass to the finite differencing functions.
    """

    def __init__(
        self,
        portfolio_instruments: PortfolioInstruments,
        reference_market_context: MarketContext,
        market_data_parser: skt.MarketDataParser = _default_market_data_parser,
        differentiation_order: Literal[1, 2] = 1,
        **finite_difference_kwargs,
    ):
        self.portfolio_instruments = portfolio_instruments
        self.reference_market_context = (
            reference_market_context
            if reference_market_context is not None
            else MarketContext()
        )
        self.market_data_parser: skt.MarketDataParser = market_data_parser
        self.differentiation_order: Literal[1, 2] = differentiation_order
        self.finite_difference_kwargs = finite_difference_kwargs

    def fit(
        self, X: npt.ArrayLike, y=None, market_quotes: pd.DataFrame | None = None
    ) -> Self:
        """Fit the estimator.

        Parameters
        ----------
        X : npt.ArrayLike
            Market quotes used for pricing the instruments.
        y : Ignored
            Not used, present for API consistency.
        market_quotes : pd.DataFrame | None
            DataFrame of market quotes. The last row is used as the reference
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        self : FiniteDifferenceSensiEstimator
            Fitted estimator.
        """
        if market_quotes is None:
            raise ValueError(
                "market_quotes metadata must be provided to fit FiniteDifferenceSensiEstimator."
            )
        self.sensi_ = calculate_sensis(
            market_quotes.iloc[-1],
            self.reference_market_context,
            self.portfolio_instruments,
            market_data_parser=self.market_data_parser,
            differentiation_order=self.differentiation_order,
            percent=True,
            pandas_output=False,
            **self.finite_difference_kwargs,
        )
        return self


class NonLinearPrior(BasePrior):
    """A base class for all priors designed to generate return distributions whose prices relate non-linearly to a set of market quotes.

    Parameters
    ----------
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

    adjust_for_cashflows : bool, default=False
        Whether to adjust prices for cashflows before calculating returns.

    Attributes
    ----------
    return_distribution_ : ReturnDistribution
        Fitted :class:`~skfolio.prior.ReturnDistribution` to be used by the optimization
        estimators, containing the assets distribution, moments estimation and the EP
        posterior probabilities (sample weights).

    market_quotes_prior_ : BasePrior
        Fitted `prior_estimator`.

    reference_date_ : dt.date
        The reference date for pricing. Taken as the last date in the market quotes data.

    pricing_date_ : dt.date
        The pricing date after applying the offset to the reference date.

    mu_estimator_ : BaseMu | None
        Fitted mean estimator for market quotes, or None if not provided.

    covariance_estimator_ : BaseCovariance | None
        Fitted covariance estimator for market quotes, or None if not provided.

    n_features_in_ : int
       Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
       Names of features seen during `fit`. Defined only when `X`
       has feature names that are all strings.
    """

    market_quotes_prior_: BasePrior
    reference_date_: dt.date
    pricing_date_: dt.date
    mu_estimator_: BaseMu | None
    covariance_estimator_: BaseCovariance | None
    n_features_in_: int
    feature_names_in_: np.ndarray
    reference_quotes_: pd.Series

    def __init__(
        self,
        market_quotes_prior: BasePrior | None = None,
        pricing_date_offset: pd.offsets.BaseOffset | None = None,
        returns_processor: ReturnsProcessor | None = None,
        mu_estimator: BaseMu | None = None,
        covariance_estimator: BaseCovariance | None = None,
    ):
        self.market_quotes_prior = market_quotes_prior or EmpiricalPrior()
        self.pricing_date_offset = (
            pricing_date_offset if pricing_date_offset else pd.offsets.BDay(1)
        )
        self.returns_processor = returns_processor or ReturnsProcessor()
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

    @abstractmethod
    def _generate_return_distribution(
        self,
        market_quote_distribution: pd.DataFrame,
        X: pd.DataFrame,
        y=None,
        market_quotes: pd.DataFrame | None = None,
        **fit_params,
    ) -> pd.DataFrame:
        """Generate the return distribution for the portfolio instruments.

        This method should be implemented by subclasses to generate
        the return distribution based on the fitted prior and other parameters.
        """
        pass

    def _calc_mu(self, returns, sample_weight, routed_params) -> np.ndarray:
        """Calculate the mean vector of the return distribution.

        Returns
        -------
        np.ndarray
            Mean vector of the return distribution.
        """
        if self.mu_estimator is not None:
            self.mu_estimator_ = check_estimator(
                self.mu_estimator,
                default=EmpiricalMu(),
                check_type=BaseMu,
            )
            self.mu_estimator_.fit(returns, **routed_params.mu_estimator.fit)
            return self.mu_estimator_.mu_
        else:
            return np.array(sm.mean(returns, sample_weight=sample_weight))

    def _calc_covariance(self, returns, sample_weight, routed_params) -> np.ndarray:
        """Calculate the covariance matrix of the return distribution.

        Returns
        -------
        np.ndarray
            Covariance matrix of the return distribution.
        """
        if self.covariance_estimator is not None:
            self.covariance_estimator_ = check_estimator(
                self.covariance_estimator,
                default=EmpiricalCovariance(),
                check_type=BaseCovariance,
            )
            self.covariance_estimator_.fit(
                returns, **routed_params.covariance_estimator.fit
            )
            return self.covariance_estimator_.covariance_
        else:
            return np.cov(returns, rowvar=False, aweights=sample_weight)

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

        self.market_quotes_prior_ = check_estimator(
            self.market_quotes_prior,
            default=EmpiricalPrior(),
            check_type=BasePrior,
        )

        # Fitting prior estimator
        quote_returns = self.returns_processor.df_to_returns(market_quotes)
        self.market_quotes_prior_.fit(
            quote_returns, y, **routed_params.market_quotes_prior.fit
        )
        prior_quote_returns = pd.DataFrame(
            self.market_quotes_prior_.return_distribution_.returns,
            columns=self.market_quotes_prior_.feature_names_in_,
        )
        sample_weight = self.market_quotes_prior_.return_distribution_.sample_weight
        if sample_weight is None:
            n_observations = len(prior_quote_returns)
            sample_weight = np.ones(n_observations) / n_observations
        elif (self.mu_estimator is not None) or (self.covariance_estimator is not None):
            raise ValueError(
                "When mu_estimator or covariance_estimator are provided, "
                "the sample_weight of the market_quotes_prior must be None."
                f"Got sample_weight={sample_weight}"
            )

        self.reference_date_ = market_quotes.index[-1]
        self.pricing_date_ = self.reference_date_ + self.pricing_date_offset
        self.reference_quotes_ = market_quotes.loc[
            self.reference_date_,
            quote_returns.columns,  # Remove any columns that were dropped due to NaNs
        ]

        market_quote_distribution = self.returns_processor.returns_to_df(
            prior_quote_returns, self.reference_quotes_
        )
        returns = self._generate_return_distribution(
            market_quote_distribution, X, y, **fit_params
        )

        self.return_distribution_ = ReturnDistribution(
            mu=self._calc_mu(returns, sample_weight, routed_params),
            covariance=self._calc_covariance(returns, sample_weight, routed_params),
            returns=returns.values,
            sample_weight=sample_weight,
        )

        return self


class RepricingPrior(NonLinearPrior):
    """The RepricingPrior class creates a distribution of returns for a portfolio of non-linear instruments.
    Instead of using the historical returns of the instruments directly, it uses arbitrary market quotes,
    provided these market quotes can be used to reprice the instruments in the portfolio. The RepricingPrior
    assumes the returns of the market quotes are independent and identically distributed, and based on this
    assumption builds a distribution of returns for the instruments in the portfolio. The process is as follows.

    1. The user fits the RepricingPrior with historical market quotes data (e.g. implied vols or credit spreads)
        whose returns are approximately i.i.d.
    2. The RepricingPrior builds a distribution of market quotes returns using the market_quotes_prior (e.g. EmpiricalPrior).
    3. Using the distribution of market quotes returns, the RepricingPrior generates a distribution of market quotes
      at the investment horizon.
    4. The RepricingPrior maps the distribution of invariants into the distribution of security prices at the investment
      horizon through the pricing functions provided in the portfolio_instruments.
    5. Finally, it computes the linear returns of the instruments in the portfolio.

    Parameters
    ----------
    portfolio_instruments : PortfolioInstruments
        The portfolio of instruments to price.

    reference_market_context : MarketContext, optional
        Reference market context for pricing. Defaults to empty MarketContext.

    market_data_parser : Callable[[pd.Series], dict], optional
        Function to parse market data from a Series.

    adjust_for_cashflows : bool, default=False
        Whether to adjust prices for cashflows before calculating returns.

    Attributes
    ----------
    portfolio_instruments_ : PortfolioInstruments
        The portfolio of instruments to price, filtered to those present in the training data.

    reference_market_context_ : MarketContext
        The reference market context used for pricing. Created by calling the market_data_parser
        with the last market quotes and the reference_market_context passed on instantiation.

    reference_prices_ : pd.Series
        The reference prices of the portfolio instruments at the reference date.

    """

    reference_market_context_: MarketContext
    reference_prices_: pd.Series

    def __init__(
        self,
        portfolio_instruments: PortfolioInstruments,
        reference_market_context: MarketContext | None = None,
        market_data_parser: skt.MarketDataParser = _default_market_data_parser,
        adjust_for_cashflows: bool = False,
        **kwargs,
    ):
        self.portfolio_instruments = portfolio_instruments
        self.reference_market_context = (
            reference_market_context
            if reference_market_context is not None
            else MarketContext()
        )
        self.market_data_parser = market_data_parser
        self.adjust_for_cashflows = adjust_for_cashflows
        super().__init__(**kwargs)

    def _generate_return_distribution(
        self,
        market_quote_distribution: pd.DataFrame,
        X: pd.DataFrame,
        y=None,
        market_quotes: pd.DataFrame | None = None,
        **fit_params,
    ) -> pd.DataFrame:
        """Generate the return distribution for the portfolio instruments.

        This method uses the pricing functions provided in the portfolio_instruments
        to reprice the instruments based on the market quote distribution.

        Parameters
        ----------
        market_quote_distribution : pd.DataFrame
            DataFrame of market quotes at the investment horizon.

        Returns
        -------
        pd.DataFrame
            DataFrame of prices for the portfolio instruments at the investment horizon.
        """
        # Only keep instruments that are in X
        self.portfolio_instruments_ = self.portfolio_instruments[self.feature_names_in_]

        self.reference_market_context_ = self.market_data_parser(
            self.reference_quotes_,
            self.reference_market_context,
            self.portfolio_instruments_,
        )
        self.reference_prices_ = self.portfolio_instruments_.price(
            self.reference_market_context_
        )

        self.reference_market_context_.update_date(self.pricing_date_)

        portfolio_price_distribution = price_df(
            market_quote_distribution,
            self.portfolio_instruments_,
            reference_market_context=self.reference_market_context_,
            market_data_parser=self.market_data_parser,
            use_date_index=False,
        )

        if self.adjust_for_cashflows:
            portfolio_price_distribution += (
                get_cashflows(
                    self.portfolio_instruments_,
                    pd.bdate_range(self.reference_date_, self.pricing_date_),
                    reference_market_context=self.reference_market_context_,
                )
                .cumsum()
                .loc[self.pricing_date_]
                .values
            )

        return (
            portfolio_price_distribution - self.reference_prices_.values
        ) / self.reference_prices_.values


class SensiPrior(NonLinearPrior):
    """The SensiPrior creates a distribution of returns for non-linear instruments using their sensitivities with respect to market quotes.

    Note
    ----
    When this prior mentions sensis, it does not mean the derivative of price with respect to the market quotes directly.
    Instead, it means the derivative of price with respect to the market quotes *divided* by the price (i.e. the percentage change in price
    for an absolute change in the market quote: (dx/dy) / x). This means one does not have to pass reference prices to the SensiPrior.

    Parameters
    ----------
    first_order_sensis : npt.ArrayLike | BaseSensiEstimator
        First order sensitivities of the portfolio instruments to the market quotes.
        Can be provided as a numpy array or as a fitted/unfitted BaseSensiEstimator.

    second_order_sensis : npt.ArrayLike | BaseSensiEstimator | None, optional
        Second order sensitivities of the portfolio instruments to the market quotes.
        Can be provided as a numpy array or as a fitted/unfitted BaseSensiEstimator.
        If None, only first order sensitivities are used. Default is None.

    transform_quotes_prior_moments : bool, default=False
        Whether to transform the moments from the quotes prior using the sensitivities.

    Attributes
    ----------
    first_order_sensis_ : ndarray of shape (`market_quotes_prior.n_features_in_`, `n_features_in_`)
        First order sensitivities of the portfolio instruments to the market quotes.
        If first_order_sensititivities were not provided at instantiation, these are calculated
        using finite differencing.

    first_order_sensis_ : ndarray of shape (`market_quotes_prior.n_features_in_`, `market_quotes_prior.n_features_in_`, `n_features_in_`)
        First order sensitivities of the portfolio instruments to the market quotes.
        If first_order_sensititivities were not provided at instantiation, these are calculated
        using finite differencing.

    transform_quotes_prior_moments : bool
        Whether to transform the moments from the quotes prior using the sensitivities.
        This is useful when using moment estimators other than the empirical moments.
        However, this feature is experimental and may not work for all return types.
    """

    first_order_sensis_: npt.NDArray[np.float64]
    second_order_sensis_: npt.NDArray[np.float64] | None = None

    def __init__(
        self,
        first_order_sensis: npt.ArrayLike | BaseSensiEstimator,
        second_order_sensis: npt.ArrayLike | BaseSensiEstimator | None = None,
        transform_quotes_prior_moments: bool = False,  # This feature is experimental
        **kwargs,
    ):
        self.first_order_sensis = first_order_sensis
        self.second_order_sensis = second_order_sensis
        self.transform_quotes_prior_moments = transform_quotes_prior_moments
        super().__init__(**kwargs)

    def _generate_return_distribution(
        self,
        market_quote_distribution: pd.DataFrame,
        X: pd.DataFrame,
        y=None,
        market_quotes: pd.DataFrame | None = None,
        **fit_params,
    ) -> pd.DataFrame:
        """Generates a return distribution using a first- or second-order taylor series expansion.

        Taylor Series Expansion: f(x + dx) ≈ f(x) + J·dx + 0.5·dx^T·H·dx
        Where:
        - J is the Jacobian matrix of prices with respect to the market quotes (first order sensitivities)
        - H is the Hessian tensor of prices with respect to the market quotes (second order sensitivities)
        - dx is the change in market quotes
        """
        # Calculate the fitted sensis
        n = self.n_features_in_
        m = self.market_quotes_prior_.n_features_in_
        if isinstance(self.first_order_sensis, BaseSensiEstimator):
            self.first_order_sensis_ = self.first_order_sensis.fit(
                X, y, market_quotes
            ).sensi_
        else:
            try:
                self.first_order_sensis_ = np.array(self.first_order_sensis).reshape(
                    m, n
                )
            except Exception as e:
                raise ValueError(
                    f"Cannot reshape first_order_sensis to correct shape ({m}, {n})."
                ) from e

        if self.second_order_sensis is not None:
            if isinstance(self.second_order_sensis, BaseSensiEstimator):
                self.second_order_sensis_ = self.second_order_sensis.fit(
                    X, y, market_quotes
                ).sensi_
            else:
                try:
                    self.second_order_sensis_ = np.array(
                        self.second_order_sensis
                    ).reshape(m, m, n)
                except Exception as e:
                    raise ValueError(
                        f"Cannot reshape second_order_sensis to correct shape ({m}, {m}, {n})."
                    ) from e

        dx = (market_quote_distribution - self.reference_quotes_).values
        return_dist = dx @ self.first_order_sensis_
        if self.second_order_sensis_ is not None:
            return_dist += 0.5 * np.einsum(
                "ik,lkj,il->ij", dx, self.second_order_sensis_, dx
            )

        return pd.DataFrame(
            return_dist,
            index=market_quote_distribution.index,
            columns=self.feature_names_in_,
        )

    def _calc_mu(self, returns, sample_weight, routed_params) -> np.ndarray:
        if self.transform_quotes_prior_moments:
            mu = pd.Series(
                self.market_quotes_prior_.return_distribution_.mu,
                index=self.market_quotes_prior_.feature_names_in_,
            )
            arithmetic_mu = (
                self.returns_processor.returns_to_df(
                    mu,
                    self.reference_quotes_,
                )
                - self.reference_quotes_
            )
            return arithmetic_mu @ self.first_order_sensis_
        else:
            return super()._calc_mu(returns, sample_weight, routed_params)

    def _calc_covariance(self, returns, sample_weight, routed_params) -> np.ndarray:
        if self.transform_quotes_prior_moments:
            # TODO: this one needs some serious soul searching. It won't work for every return-type.
            return (
                self.first_order_sensis_.T
                @ self.market_quotes_prior_.return_distribution_.covariance
                @ self.first_order_sensis_
            )
        else:
            return super()._calc_covariance(returns, sample_weight, routed_params)
