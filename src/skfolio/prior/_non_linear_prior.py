import numpy as np
import pandas as pd
import datetime as dt
from pandas.tseries.frequencies import to_offset
from abc import ABC, abstractmethod
from typing import Callable, Any, Dict, Literal

import sklearn.utils.metadata_routing as skm
from sklearn.compose import make_column_selector
import sklearn.utils.validation as skv
from sklearn.base import BaseEstimator, TransformerMixin

import skfolio.measures as sm
from skfolio.prior import BasePrior, EmpiricalPrior, ReturnDistribution
import skfolio.typing as skt
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
    data: Dict[str, Any]
    date: dt.date

    def __init__(self, date: dt.date | None = None, **kwargs: Any):
        if (date is not None) and (not isinstance(date, dt.date)):
            raise ValueError(f"MarketContext requires a valid date of type datetime.date, not {type(date)}")

        self.date = date
        self.data = dict(**kwargs)

    @classmethod
    def from_series(cls, series: pd.Series) -> "MarketContext":
        date = series.name
        return cls(date=date, **data)
    
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
        """All non-adapted calls are passed to the original instrument object"""
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

    def __init__(self, **kwargs: Dict[str, Instrument]):
        if not all(isinstance(instr, Instrument) for instr in kwargs.values()):
            raise ValueError("All portfolio items must be of type Instrument")

        if not all(isinstance(instr_id, str) for instr_id in kwargs.keys()):
            raise ValueError("All portfolio instrument ids must be of type str")

        super().__init__(**kwargs)
    
    def price(self, market_context: MarketContext) -> pd.Series:
        return pd.Series({instr_id: instr.price(market_context) for instr_id, instr in self.items()}, 
                         name=market_context.date)
    

def price_df(X: pd.DataFrame,
            portfolio_instruments: PortfolioInstruments,
            reference_market_context = MarketContext(),
            market_data_parser: Callable[[pd.Series], dict] | None = None,
            use_date_index: bool = True
            ) -> pd.DataFrame:

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
    def __init__(self,
                periods: int = 1,
                freq: str | None = None,
                return_type: skt.ReturnType | Dict[str, skt.ReturnType] | Dict[make_column_selector, skt.ReturnType] = "linear"
                ):
        self.periods = periods
        self.freq = freq
        self.return_type = return_type

    def linear_returns(self, X: pd.DataFrame) -> pd.DataFrame:
        if np.any(X <= 0):
            raise ValueError("Prices must be positive to compute returns.")

        return X.pct_change(freq=self.freq, periods=self.periods).iloc[self.periods:]
    
    def log_returns(self, X: pd.DataFrame) -> pd.DataFrame:
        return np.log1p(self.linear_returns(X))
    
    def arithmetic_returns(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.diff(periods=self.periods)
    
    def prices_to_returns(self, X: pd.DataFrame, return_type: skt.ReturnType) -> pd.DataFrame:
        match return_type:
            case "linear":
                return self.linear_returns(X)
            case "log":
                return self.log_returns(X)
            case "arithmetic":
                return self.arithmetic_returns(X)
            case _:
                raise ValueError(f"Unknown return_type: {return_type}")
            
    def returns_to_prices(self,
                           returns: pd.DataFrame, 
                           reference_prices: pd.Series, 
                           return_type: skt.ReturnType) -> pd.DataFrame:
        match self.return_type:
            case "linear":
                prices = reference_prices.values * (1 + returns)
            case "log":
                prices = reference_prices.values * np.exp(returns)
            case "arithmetic":
                prices = reference_prices.values + returns
            case _:
                raise ValueError(f"Unknown return_type: {self.return_type}")
        return pd.DataFrame(prices, index=returns.index, columns=returns.columns)
            
    def df_to_returns(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(self.return_type, str):
            return self.prices_to_returns(X, self.return_type)
        elif isinstance(self.return_type, dict):
            returns_list = []
            for key, ret_type in self.return_type.items():
                if callable(key):
                    cols = key(X)
                elif isinstance(key, str):
                    cols = [key]
                else:
                    raise ValueError(f"Invalid key type in return_type dict: {type(key)} for key {key}")
                
                X_subset = X[cols]
                returns_subset = self.prices_to_returns(X_subset, ret_type)
                returns_list.append(returns_subset)
            
            return pd.concat(returns_list, axis=1)
        else:
            raise ValueError(f"Invalid type for return_type: {type(self.return_type)}")
    
    def returns_to_df(self, returns: pd.DataFrame, reference_prices: pd.Series) -> pd.DataFrame:
        if isinstance(self.return_type, str):
            return self.returns_to_prices(returns, reference_prices, self.return_type)
        elif isinstance(self.return_type, dict):
            prices_list = []
            for key, return_type in self.return_type.items():
                if callable(key):
                    cols = key(returns)
                elif isinstance(key, str):
                    cols = [key]
                else:
                    raise ValueError(f"Invalid key type in return_type dict: {type(key)} for key {key}")
                
                returns_subset = returns[cols]
                reference_prices_subset = reference_prices[cols]
                prices_subset = self.returns_to_prices(returns_subset, reference_prices_subset, return_type)
                
                prices_list.append(pd.DataFrame(prices_subset, index=returns_subset.index, columns=returns_subset.columns))
            
            return pd.concat(prices_list, axis=1)
        else:
            raise ValueError(f"Invalid type for return_type: {type(self.return_type)}")


class NonLinearPrior(BasePrior):

    """The NonLinearPrior class creates a distribution of returns for a portfolio of non-linear instruments.
    Instead of the usual design pattern of fitting a prior using the historical asset returns, instead this
    prior is fitted using a history of market quotes used to price the given portfolio. These market quotes
    could be, for example, implied volatilities for options, z-scores for bonds, swap-rates for rates 
    instruments, etc.
    """

    prior_estimator_: BasePrior

    def __init__(self,
                 portfolio_instruments: PortfolioInstruments,
                 return_type: skt.ReturnType | Dict[str, skt.ReturnType] | Dict[make_column_selector, skt.ReturnType] = "linear",
                 prior_estimator: BasePrior = EmpiricalPrior(),
                 reference_market_context: MarketContext = MarketContext(),
                 market_data_parser: Callable[[pd.Series], dict] | None = None,
                 reference_index=-1,
                 pricing_date_offset: str | None = "1B",
                 returns_processor: ReturnsProcessor = ReturnsProcessor()
                 ):
        self.portfolio_instruments = portfolio_instruments
        self.return_type = return_type  
        self.prior_estimator = prior_estimator
        self.reference_index = reference_index
        self.reference_market_context = reference_market_context
        self.market_data_parser = market_data_parser
        self.pricing_date_offset = pricing_date_offset
        self.returns_processor = returns_processor

    def fit(self, X: pd.DataFrame, y=None, **fit_params):

        routed_params = skm.process_routing(self, "fit", **fit_params)

        # Validation
        skv.validate_data(self, X)

        prior_estimator_ = check_estimator(
            self.prior_estimator,
            default=EmpiricalPrior(),
            check_type=BasePrior,
        )

        quote_returns = self.returns_processor.df_to_returns(X)

        # Fitting prior estimator
        prior_estimator_.fit(quote_returns, y, **routed_params.prior_estimator.fit)
        # Prior distribution
        prior_quote_returns = pd.DataFrame(prior_estimator_.return_distribution_.returns,
                                             columns=prior_estimator_.feature_names_in_)
        sample_weight = (
            prior_estimator_.return_distribution_.sample_weight
        )
        n_observations = len(prior_quote_returns)
        if sample_weight is None:
            sample_weight = np.ones(n_observations) / n_observations

        # Get reference quotes and prices
        if self.reference_index in X.index:
            reference_quotes = X.loc[self.reference_index]
        elif isinstance(self.reference_index, int):
            reference_quotes = X.iloc[self.reference_index]
        else:
            raise ValueError("reference_index must be an index label or integer position")
        
        if self.market_data_parser:
            reference_quotes = self.market_data_parser(reference_quotes)
        self.reference_market_context.update(**reference_quotes)

        # Determine reference date used for computing the reference portfolio prices
        pricing_date = self.reference_market_context.date
        if isinstance(reference_quotes.name, dt.date):
            reference_date = reference_quotes.name
        elif isinstance(reference_quotes.name, pd.Timestamp):
            reference_date = reference_quotes.name.date()
        elif pricing_date:
            reference_date = pricing_date
        else:
            raise ValueError("""Could not determine reference date for pricing instruments.
                             Please ensure that either X has a date index or the 
                             reference_market_context has a valid date.""")
        self.reference_market_context.update_date(reference_date)
        reference_prices = self.portfolio_instruments.price(self.reference_market_context)

        if not pricing_date:
            pricing_date = (reference_date + to_offset(self.pricing_date_offset)) if self.pricing_date_offset else reference_date
        # TODO: update the below to handle different return types
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

        returns = (portfolio_price_distribution - reference_prices.values) / reference_prices.values

        self.return_distribution_ = ReturnDistribution(
            mu=sm.mean(returns, sample_weight=sample_weight).values,
            covariance=np.cov(returns, rowvar=False, aweights=sample_weight),
            returns=returns.values,
            sample_weight=sample_weight,
        )

        return self

    def __sklearn_clone__(self):
        """Custom clone method to avoid cloning the portfolio instruments."""
        return self


class MarketDataProcessor(ABC, BaseEstimator, TransformerMixin):
    # TODO: add caching logic

    def parse_market_quotes(self, market_quotes: pd.Series) -> MarketContext:
        """A default implentation is provided for parsing market quotes that
        simply converts the pd.Series directly into a MarketContext. However,
        subclasses may override this method to provide custom parsing logic.
        """
        return MarketContext.from_series(market_quotes)

    @abstractmethod
    def transform_market_context(self, market_context: MarketContext) -> pd.Series:
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        return pd.concat([
            self.transform_market_context(self.parse_market_quotes(X.iloc[i]))
            for i in range(len(X))
        ], axis=1)

class SaveMarketContext(BaseEstimator, TransformerMixin):
    """A simple transformer that saves a market context from a given row in the input DataFrame to a target MarketContext object.
    This transformer does not modify the input DataFrame, it simply updates the target MarketContext.
    The row from the DataFrame to be used is set as the last row by default, but any index label or integer position may be specified.

    By default, the given dataframe row is converted directly to a market_context, but more complicated logic may be provided by setting
    the market_context_parser argument.
    """

    def __init__(self,
                 target_market_context: MarketContext,
                 df_index=-1,
                 market_context_parser: Callable[[pd.Series], MarketContext]=MarketContext.from_series):
        self.target_market_context = target_market_context
        self.df_index = df_index
        self.market_context_parser = market_context_parser

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """The transform function does not actually transform the data at all, 
        instead it simply grabs the correct market context and saves it in the given market_context object"""

        # Get the correct row in X. If df_index is an integer, first check if it's in the index. Otherwise, treat it as a positional index.
        if self.df_index in X.index:
            row = X.loc[self.df_index]
        elif isinstance(self.df_index, int):
            row = X.iloc[self.df_index]
        else:
            raise ValueError("df_index must be an index label or integer position")

        # Update the target market context in-place to ensure the reference is preserved
        self.target_market_context.update({label: value for label, value in self.market_context_parser(row).items()})

        if isinstance(row.name, dt.date):
            self.target_market_context.date = row.name

        return X


class ExtrapolateReturns(TransformerMixin, BaseEstimator):
    def __init__(self,
                 periods: int = 1,
                 freq: str = "D",
                 return_type: Literal["linear", "log", "arithmetic"] = "linear"
                 ):
        self.periods = periods
        self.freq = freq
        self.return_type = return_type

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        returns = PricesToReturns(
            periods=self.periods,
        ).transform(X)

        output_index = None
        if isinstance(X.index, pd.DatetimeIndex):
            last_date = max(X.index)
            reference_prices = X.loc[last_date]
            output_date = last_date + to_offset(self.freq) * self.periods
            output_index = pd.Index([output_date] * len(returns), name=returns.index.name)
        else:
            reference_prices = X.iloc[-1]

        if self.return_type == "arithmetic":
            prices = reference_prices.values + returns.values
        else:
            prices = reference_prices.values * (1 + returns.values)

        return pd.DataFrame(prices, index=output_index, columns=X.columns)