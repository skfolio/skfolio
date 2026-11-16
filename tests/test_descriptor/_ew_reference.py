"""Batch references independent of the production EW recurrences."""

import numpy as np

from skfolio.utils.stats import _market_returns


def weighted_regression(history, half_life):
    """Fit an intercept and slopes by weighted least squares on explicit history."""
    values = np.asarray(history)
    weights = 2.0 ** (-np.arange(len(values) - 1, -1, -1) / half_life)
    weights /= weights.sum()
    shifted = values - values[0]
    centered = shifted - weights @ shifted
    predictors = centered[:, :-1]
    scale = np.sqrt(weights @ predictors**2)
    design = np.column_stack((np.ones(len(values)), predictors / scale))
    coefficients = np.linalg.lstsq(
        np.sqrt(weights)[:, None] * design,
        np.sqrt(weights) * centered[:, -1],
        rcond=None,
    )[0]
    slopes = coefficients[1:] / scale
    residuals = centered[:, -1] - design @ coefficients
    information = design.T @ (weights[:, None] * design)
    squared_information = design.T @ (weights[:, None] ** 2 * design)
    inverse = np.linalg.inv(information)
    residual_mass = 1 - np.trace(inverse @ squared_information)
    if len(values) <= design.shape[1] or residual_mass <= 0:
        return slopes, np.full(slopes.shape, np.nan)
    covariance = (
        (weights @ residuals**2)
        / residual_mass
        * inverse
        @ squared_information
        @ inverse
    )
    return slopes, np.diag(covariance)[1:] / scale**2


def regression_path(
    panel, half_life, min_periods, aggregation_period=1, reference=None
):
    """Reconstruct each asset's eligible history and refit it at every boundary."""
    market = _market_returns(
        panel["returns"], panel["market_cap"], panel.estimation_mask
    )
    histories = [[] for _ in range(panel.n_assets)]
    pending = [[] for _ in histories]
    previous = np.full(panel.n_assets, np.nan)
    result = np.full((panel.n_observations, panel.n_assets), np.nan)
    n_predictors = 1 if reference is None else 2
    for t in range(panel.n_observations):
        for i in range(panel.n_assets):
            if not panel.active_mask[t, i]:
                histories[i].clear()
                pending[i].clear()
                previous[i] = np.nan
                continue
            row = [market[t], panel["returns"][t, i]]
            if reference is not None:
                row.insert(1, reference[t])
            if np.isfinite(row).all():
                pending[i].append(row)
        if (t + 1) % aggregation_period == 0:
            for i in range(panel.n_assets):
                if pending[i]:
                    histories[i].append(np.mean(pending[i], axis=0))
                    pending[i].clear()
                    if len(histories[i]) >= max(min_periods, n_predictors + 1):
                        previous[i] = weighted_regression(histories[i], half_life)[0][
                            -1
                        ]
        result[t] = previous
    return result


def residual_path(
    panel, half_life, beta_half_life, min_periods, min_acceptable_return=None
):
    """Refit beta from its complete history and independently weight residual history."""
    market = _market_returns(
        panel["returns"], panel["market_cap"], panel.estimation_mask
    )
    result = np.full((panel.n_observations, panel.n_assets), np.nan)
    for i in range(panel.n_assets):
        history, residuals = [], []
        previous = np.nan
        for t in range(panel.n_observations):
            if not panel.active_mask[t, i]:
                history.clear()
                residuals.clear()
                previous = np.nan
            elif np.isfinite(panel["returns"][t, i]):
                history.append([market[t], panel["returns"][t, i]])
                previous = np.nan
                if len(history) >= 10:
                    beta = weighted_regression(history, beta_half_life)[0][0]
                    residual = panel["returns"][t, i] - beta * market[t]
                    if min_acceptable_return is not None:
                        residual = min(residual - min_acceptable_return, 0)
                    residuals.append(residual)
                    if len(residuals) >= min_periods:
                        weights = 2.0 ** (
                            -np.arange(len(residuals) - 1, -1, -1) / half_life
                        )
                        previous = np.sqrt(
                            np.average(np.square(residuals), weights=weights)
                        )
            result[t, i] = previous
    return result


def downside_path(panel, half_life, min_periods, min_acceptable_return=0):
    """Evaluate paired uncentered lower moments from explicit finite histories."""
    market = _market_returns(
        panel["returns"], panel["market_cap"], panel.estimation_mask
    )
    result = np.full((panel.n_observations, panel.n_assets), np.nan)
    for i in range(panel.n_assets):
        history = []
        previous = np.nan
        for t in range(panel.n_observations):
            if not panel.active_mask[t, i]:
                history.clear()
                previous = np.nan
            elif np.isfinite(panel["returns"][t, i]):
                history.append([market[t], panel["returns"][t, i]])
                if len(history) >= min_periods:
                    downside = np.minimum(np.array(history) - min_acceptable_return, 0)
                    weights = 2.0 ** (-np.arange(len(history) - 1, -1, -1) / half_life)
                    denominator = weights @ downside[:, 0] ** 2
                    previous = (
                        (weights @ (downside[:, 0] * downside[:, 1])) / denominator
                        if denominator > 0
                        else np.nan
                    )
            result[t, i] = previous
    return result
