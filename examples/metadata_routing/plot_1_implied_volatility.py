"""
==============================================
Using Implied Volatility with Metadata Routing
==============================================

This tutorial shows how to use :ref:`metadata routing <metadata_routing>`.

We will use the :class:`~skfolio.moments.ImpliedCovariance` estimator inside
optimization models and grid search procedures to show how the implied volatility
time series can be routed.
"""

# %%
# Load Datasets
# =============
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition and the implied volatility time series
# of these 20 assets starting from 2010-01-04 up to 2022-12-28.
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.io import show
from sklearn import set_config
from sklearn.model_selection import GridSearchCV, train_test_split

from skfolio import Population, RatioMeasure
from skfolio.datasets import load_sp500_dataset, load_sp500_implied_vol_dataset
from skfolio.metrics import make_scorer
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.moments import (
    EmpiricalCovariance,
    GerberCovariance,
    ImpliedCovariance,
    LedoitWolf,
)
from skfolio.optimization import InverseVolatility, MeanRisk
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior

prices = load_sp500_dataset()
implied_vol = load_sp500_implied_vol_dataset()

X = prices_to_returns(prices)
X = X.loc["2010":]

implied_vol.head()

# %%
# Implied Covariance Estimator
# ============================
# We use the :class:`~skfolio.moments.ImpliedCovariance` estimator as an example for
# metadata routing because, in addition to the assets' returns `X`, it also needs
# the assets' implied volatilities passed to its `fit` method.
#
# Below, we give a quick summary of the estimator. The detailed
# documentation and literature references are available in the docstring:
# :class:`~skfolio.moments.ImpliedCovariance`.
#
# For each asset, the implied volatility time series is used to estimate the realised
# volatility using the non-overlapping log-transformed OLS model:
#
# .. math:: \ln(RV_{t}) = \alpha + \beta_{1} \ln(IV_{t-1}) + \beta_{2} \ln(RV_{t-1}) + \epsilon
#
# with :math:`\alpha`, :math:`\beta_{1}` and :math:`\beta_{2}` the intercept and
# coefficients to estimate, :math:`RV` the realised volatility, and :math:`IV` the
# implied volatility. The training set uses non-overlapping data of sample size
# `window_size` to avoid possible regression errors caused by auto-correlation.
# The logarithmic transformation of volatilities is used for its better finite sample
# properties and distribution, which is closer to normality, less skewed and
# leptokurtic.
#
# The final step is the reconstruction of the covariance matrix from the correlation
# and estimated realised volatilities :math:`D`:
#
# .. math:: \Sigma = D \ Corr \ D
#
# With :math:`Corr`, the correlation matrix computed from the prior covariance
# estimator. The default is the `EmpiricalCovariance`. It can be changed to any
# covariance estimator using `prior_covariance_estimator`.

model = ImpliedCovariance()
model.fit(X, implied_vol=implied_vol)
print(model.covariance_.shape)
# %%
# The intercept, coefficients and R2 score are saved in `model.intercepts_`,
# `model.coefs_` and `model.r2_scores_`
#
# Let's analyse the R2 score as a function of the window size:
coefs = {}
for window_size in [10, 20, 60, 100]:
    model = ImpliedCovariance(window_size=window_size)
    model.fit(X, implied_vol=implied_vol)
    coefs[window_size] = model.r2_scores_

df = (
    pd.DataFrame(coefs, index=X.columns)
    .unstack()
    .reset_index()
    .rename(columns={"level_0": "Window Size", "level_1": "Asset", 0: "R2 score"})
)
df["Window Size"] = df["Window Size"].astype(str)
fig = px.bar(
    df,
    x="Asset",
    y="R2 score",
    color="Window Size",
    barmode="group",
    title="R2 score per Window Size",
)
show(fig)

# %%
# |
#
# Let's print the average R2 per window size:
print({k: f"{np.mean(v):0.1%}" for k, v in coefs.items()})

# %%
# The highest R2 is achieved for a window size of 20 observations.

# %%
# Inverse Volatility
# ==================
# To use the `ImpliedCovariance` estimator inside a meta-estimator such as the
# `InverseVolatility`, you must enable metadata routing with `set_config` and
# specify where to route the implied vol using `set_fit_request` as shown below:
set_config(enable_metadata_routing=True)

model = InverseVolatility(
    prior_estimator=EmpiricalPrior(
        covariance_estimator=ImpliedCovariance().set_fit_request(implied_vol=True)
    )
)

# %%
# Then you can pass the implied volatility to the `fit` method of the meta-estimator:
model.fit(X, implied_vol=implied_vol)
print(model.weights_)

# %%
# Cross Validation
# ================
# In this section, we show how to use metadata routing with `cross_val_predict`.
# First, we create a `WalkForward` cross-validator to rebalance our portfolio every 20
# business days by re-fitting the model on the previous 400 business days (~ 1.5 years):
cv = WalkForward(train_size=400, test_size=20)

# %%
# We use the model created above and pass the implied volatility in `params`:
pred_model = cross_val_predict(model, X, cv=cv, params={"implied_vol": implied_vol})
pred_model.name = "Implied Vol"

# %%
# Let's compare the model with a benchmark using `InverseVolatility` with the default
# `EmpiricalCovariance` estimator:
benchmark = InverseVolatility()
pred_bench = cross_val_predict(benchmark, X, cv=cv)
pred_bench.name = "Benchmark"

# %%
# For easier analysis, we add both predicted portfolios into a `Population`:
population = Population([pred_bench, pred_model])
summary = population.summary()
print(summary.loc[["Annualized Standard Deviation", "Annualized Sharpe Ratio"]])

# %%
# Let's plot the Composition and Cumulative returns:
population.plot_composition(display_sub_ptf_name=False)
# %%
population.plot_cumulative_returns()

# %%
# Hyper-Parameters Tuning
# =======================
# In this section, we show how to use metadata routing with `GridSearchCV`.
# First, we split the data into a train and a test set:

X_train, X_test, implied_vol_train, implied_vol_test = train_test_split(
    X, implied_vol, test_size=1 / 2, shuffle=False
)

# %%
# We create a Minimum Variance that uses the `ImpliedCovariance` estimator:
model = MeanRisk(
    prior_estimator=EmpiricalPrior(
        covariance_estimator=ImpliedCovariance().set_fit_request(implied_vol=True)
    )
)

# %%
# Then, we find the hyper-parameters of the `ImpliedCovariance` estimator that
# maximizes the out-of-sample Sharpe Ratio of the Minimum Variance model:
grid_search = GridSearchCV(
    estimator=model,
    param_grid={
        "prior_estimator__covariance_estimator__window_size": np.arange(5, 50, 3),
        "prior_estimator__covariance_estimator__prior_covariance_estimator": [
            LedoitWolf(),
            GerberCovariance(),
            EmpiricalCovariance(),
        ],
    },
    return_train_score=True,
    scoring=make_scorer(RatioMeasure.ANNUALIZED_SHARPE_RATIO),
    n_jobs=-1,
    cv=cv,
)
grid_search.fit(X_train, implied_vol=implied_vol_train)
gs_model = grid_search.best_estimator_
print(gs_model)

# %%
# Let's plot the out-of-sample Sharpe Ratio as a function of the window size and
# the prior covariance estimator used to compute the correlation matrix:
cv_results = grid_search.cv_results_

df = pd.DataFrame(
    {
        "Prior Cov Estimator": [
            str(x)
            for x in cv_results[
                "param_prior_estimator__covariance_estimator__prior_covariance_estimator"
            ]
        ],
        "Window Size": cv_results[
            "param_prior_estimator__covariance_estimator__window_size"
        ],
        "Test Sharpe Ratio": cv_results["mean_test_score"],
        "error": cv_results["std_test_score"] / 10,  # one tenth of std for readability
    }
)
px.line(
    df,
    x="Window Size",
    y="Test Sharpe Ratio",
    color="Prior Cov Estimator",
    error_y="error",
    title="Out-of-Sample Sharpe Ratio",
)

# %%
# Finally, we compare the optimal Grid Search model with a naive Minimum Variance
# benchmark on the **test set**:

pred_gs_model = cross_val_predict(
    gs_model, X_test, params={"implied_vol": implied_vol_test}, cv=cv, n_jobs=-1
)
pred_gs_model.name = "GS Model"

benchmark = MeanRisk()
pred_bench = cross_val_predict(benchmark, X_test, cv=cv)
pred_bench.name = "Benchmark"

population = Population([pred_bench, pred_gs_model])
summary = population.summary()
print(summary.loc[["Annualized Standard Deviation", "Annualized Sharpe Ratio"]])

# %%
population.plot_cumulative_returns()

# %%
# Conclusion
# ==========
# This was a toy example to introduce the metadata routing API.
# For more information, see :ref:`Metadata Routing User Guide <metadata_routing>`.
