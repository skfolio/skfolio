"""
==================================================
Using Implied Volatility in Portfolio Optimization
==================================================

This tutorial explores the difference between the general
procedure using different investment horizon and the simplified procedure as explained
in :ref:`data preparation <data_preparation>`.
"""
# %%
# Prices
# ======
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition starting from 1990-01-02 up to 2022-12-28:
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.io import show
from sklearn.model_selection import train_test_split
from sklearn import set_config

from skfolio import PerfMeasure, Population, RiskMeasure
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.datasets import load_sp500_dataset, load_sp500_implied_vol_dataset
from skfolio.optimization import MeanRisk, InverseVolatility
from skfolio.moments   import ImpliedCovariance, EmpiricalCovariance
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior


# %%
# Load Datasets
# =============
prices = load_sp500_dataset()

# %%
implied_vol = load_sp500_implied_vol_dataset()
implied_vol.head()

# %%
# Linear Returns
# ==============
# We transform the daily prices into daily linear returns:
X = prices_to_returns(prices)
X = X.loc["2010":]

# %%
model = ImpliedCovariance()
model.fit(X, implied_vol=implied_vol)

model.coefs_
model.r2_scores_


#
coefs={}
for window_size in [10, 20, 60, 100]:
    model = ImpliedCovariance(window_size=window_size)
    model.fit(X, implied_vol=implied_vol)
    coefs[window_size]=model.r2_scores_

df = pd.DataFrame(coefs, index=X.columns).unstack().reset_index().rename(
    columns={'level_0': 'window_size', "level_1": "asset", 0: "R2"})
df['window_size'] = df['window_size'].astype(str)
fig = px.bar(df, x="asset", y="R2",
                 color="window_size", barmode="group")
fig.show()

{k: np.mean(v) for k, v in coefs.items()}

#
set_config(enable_metadata_routing=True)

model=InverseVolatility(
    prior_estimator=EmpiricalPrior(
        covariance_estimator=ImpliedCovariance().set_fit_request(implied_vol=True)
    )
)

model.fit(X, implied_vol=implied_vol)
model.weights_

#
cv = WalkForward(train_size=300, test_size=20)

benchmark_1 = InverseVolatility()

benchmark_2 = InverseVolatility(
prior_estimator=EmpiricalPrior(
        covariance_estimator=EmpiricalCovariance(window_size=20)
    )
)


pred_bench_1 = cross_val_predict(benchmark_1, X, cv=cv)
pred_bench_1.name = "Benchmark 300 RV"

pred_bench_2 = cross_val_predict(benchmark_2, X, cv=cv)
pred_bench_2.name = "Benchmark 20 RV"

pred_model = cross_val_predict(model, X, cv=cv, params={"implied_vol":implied_vol})
pred_model.name = "Implied Vol"

population = Population([pred_bench_1,pred_bench_2, pred_model])
population.plot_cumulative_returns().show()
population.plot_composition().show()
population.summary().loc[["Annualized Standard Deviation", "Annualized Sharpe Ratio"]]



#
model=MeanRisk(
    prior_estimator=EmpiricalPrior(
        covariance_estimator=ImpliedCovariance().set_fit_request(implied_vol=True)
    )
)
benchmark_1 = MeanRisk()

benchmark_2 = MeanRisk(
prior_estimator=EmpiricalPrior(
        covariance_estimator=EmpiricalCovariance(window_size=20)
    )
)


pred_bench_1 = cross_val_predict(benchmark_1, X, cv=cv)
pred_bench_1.name = "Benchmark 300 RV"

pred_bench_2 = cross_val_predict(benchmark_2, X, cv=cv)
pred_bench_2.name = "Benchmark 20 RV"

pred_model = cross_val_predict(model, X, cv=cv, params={"implied_vol":implied_vol})
pred_model.name = "Implied Vol"

population = Population([pred_bench_1,pred_bench_2, pred_model])
population.plot_cumulative_returns().show()
population.plot_composition().show()
population.summary().loc[["Annualized Standard Deviation", "Annualized Sharpe Ratio"]]



