r"""
================================
Market Sentiment Views by Adanos
================================

This tutorial shows how an optional external market sentiment signal can be
translated into views for :class:`~skfolio.prior.BlackLitterman` and
:class:`~skfolio.prior.EntropyPooling`.

The example uses the Adanos Market Sentiment API as a possible source of Reddit
stock sentiment, but it does not add Adanos as a dependency of `skfolio`. By default,
the tutorial runs on a small embedded sample so that examples and tests remain fully
reproducible without network access or an API key.

To try live sentiment data, set:

.. code-block:: shell

   export ADANOS_USE_LIVE_API=1
   export ADANOS_API_KEY=...

The important pattern is provider-agnostic: convert an external signal into
well-scaled views, then pass those views to the existing prior estimators.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` and keep a compact universe of seven
# liquid stocks. The sentiment sample below intentionally covers the same universe so
# the tutorial runs without a live API call.

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import BlackLitterman, EntropyPooling

DEFAULT_ADANOS_URL = "https://api.adanos.org/reddit/stocks/v1/trending?limit=25"

SAMPLE_ADANOS_SENTIMENT = [
    {"ticker": "AMD", "sentiment_score": 0.72, "buzz_score": 88},
    {"ticker": "JPM", "sentiment_score": 0.41, "buzz_score": 64},
    {"ticker": "LLY", "sentiment_score": 0.24, "buzz_score": 51},
    {"ticker": "JNJ", "sentiment_score": 0.05, "buzz_score": 35},
    {"ticker": "PG", "sentiment_score": -0.08, "buzz_score": 31},
    {"ticker": "BAC", "sentiment_score": -0.33, "buzz_score": 58},
    {"ticker": "GE", "sentiment_score": -0.49, "buzz_score": 69},
]


def sentiment_frame(records: list[dict], assets: pd.Index) -> pd.DataFrame:
    """Normalize Adanos-style sentiment records to the assets used by the model."""
    asset_set = set(assets)
    rows = []
    for record in records:
        ticker = record.get("ticker") or record.get("symbol")
        if ticker is not None:
            ticker = str(ticker).upper()
        if ticker not in asset_set:
            continue
        sentiment = record.get("sentiment_score", record.get("sentiment"))
        if sentiment is None:
            continue
        buzz = record.get("buzz_score", record.get("buzz", 50.0))
        try:
            sentiment_score = float(sentiment)
            buzz_score = float(buzz)
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "ticker": ticker,
                "sentiment_score": sentiment_score,
                "buzz_score": buzz_score,
            }
        )

    if not rows:
        rows = [row for row in SAMPLE_ADANOS_SENTIMENT if row["ticker"] in asset_set]

    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset="ticker", keep="first")
        .set_index("ticker")
        .reindex(assets)
        .dropna()
    )


def extract_sentiment_records(payload: object) -> list[dict]:
    """Extract record lists from common API response envelopes."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "results", "items", "trending"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def load_adanos_sentiment(assets: pd.Index) -> pd.DataFrame:
    """Load live Adanos sentiment only when explicitly enabled, else use a sample."""
    use_live_api = os.environ.get("ADANOS_USE_LIVE_API") == "1"
    api_key = os.environ.get("ADANOS_API_KEY")

    if use_live_api and api_key:
        request = Request(
            os.environ.get("ADANOS_SENTIMENT_URL", DEFAULT_ADANOS_URL),
            headers={"X-API-Key": api_key},
        )
        try:
            with urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            records = extract_sentiment_records(payload)
            sentiment = sentiment_frame(records, assets)
            if not sentiment.empty:
                return sentiment
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            pass

    return sentiment_frame(SAMPLE_ADANOS_SENTIMENT, assets)


def sentiment_to_black_litterman_views(
    sentiment: pd.DataFrame,
    *,
    annual_view_scale: float = 0.20,
    trading_days: int = 252,
) -> list[str]:
    """Convert sentiment into absolute daily expected-return views."""
    views = []
    for ticker, row in sentiment.iterrows():
        confidence = min(max(row["buzz_score"], 0.0), 100.0) / 100.0
        annual_view = row["sentiment_score"] * confidence * annual_view_scale
        daily_view = annual_view / trading_days
        views.append(f"{ticker} == {daily_view:.6f}")
    return views


def sentiment_to_entropy_pooling_views(sentiment: pd.DataFrame) -> list[str]:
    """Convert the strongest positive and negative sentiment into a ranking view."""
    sorted_sentiment = sentiment.sort_values("sentiment_score")
    bearish_ticker = sorted_sentiment.index[0]
    bullish_ticker = sorted_sentiment.index[-1]
    return [f"{bullish_ticker} >= {bearish_ticker}"]


prices = load_sp500_dataset()
prices = prices[["AMD", "BAC", "GE", "JNJ", "JPM", "LLY", "PG"]]
X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

# %%
# Sentiment as views
# ==================
# Adanos sentiment is treated as an optional external view layer. The `sentiment_score`
# is directional, while `buzz_score` is used as a simple confidence scalar so that
# high-conviction assets receive larger expected-return views.

sentiment = load_adanos_sentiment(X.columns)
sentiment

# %%
# The sentiment records can be represented as absolute expected-return views for
# `BlackLitterman`. The values are converted from annualized assumptions to daily
# returns so they match the frequency of `X_train`.

black_litterman_views = sentiment_to_black_litterman_views(sentiment)
black_litterman_views

# %%
# Black-Litterman portfolio
# =========================
# We now fit a Maximum Sharpe Ratio portfolio with those views:

model_bl = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    prior_estimator=BlackLitterman(views=black_litterman_views),
    portfolio_params=dict(name="Adanos Black-Litterman"),
)
model_bl.fit(X_train)
model_bl.weights_

# %%
# Entropy-Pooling alternative
# ===========================
# The same sentiment source can also produce a softer ranking view for
# `EntropyPooling`. Here the most positive asset is constrained to have a mean return
# at least as high as the most negative asset.

entropy_pooling_views = sentiment_to_entropy_pooling_views(sentiment)
entropy_pooling_views

model_ep = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    prior_estimator=EntropyPooling(mean_views=entropy_pooling_views),
    portfolio_params=dict(name="Adanos Entropy Pooling"),
)
model_ep.fit(X_train)
model_ep.weights_

# %%
# Benchmark
# =========
# Finally, we compare both sentiment-driven priors with the empirical baseline:

model_empirical = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    portfolio_params=dict(name="Empirical"),
)
model_empirical.fit(X_train)

population = Population(
    [
        model_bl.predict(X_test),
        model_ep.predict(X_test),
        model_empirical.predict(X_test),
    ]
)

population.plot_cumulative_returns()

# %%
# This example is not meant to claim that social sentiment is predictive on its own.
# It shows a maintainable integration boundary: keep external data access optional,
# normalize the signal outside the optimizer, and express the result through skfolio's
# existing prior-estimator APIs.

weights = pd.DataFrame(
    {
        "Black-Litterman": model_bl.weights_,
        "Entropy Pooling": model_ep.weights_,
        "Empirical": model_empirical.weights_,
    },
    index=X.columns,
)
weights
