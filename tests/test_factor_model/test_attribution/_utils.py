import numpy as np


def _create_realized_model(
    n_obs,
    n_assets,
    n_factors,
    static_exposures=True,
    static_weights=True,
    seed=42,
    include_observations=False,
):
    """Helper to create realized attribution test data.

    Parameters
    ----------
    n_obs : int
        Number of observations.
    n_assets : int
        Number of assets.
    n_factors : int
        Number of factors.
    static_exposures : bool
        If True, exposures are static (N, K). If False, time-varying (T, N, K).
    static_weights : bool
        If True, weights are static (N,). If False, time-varying (T, N).
    seed : int
        Random seed for reproducibility.
    include_observations : bool
        If True, include observations array (for rolling attribution).

    Returns
    -------
    dict
        Dictionary with all inputs for realized_factor_attribution.
    """
    np.random.seed(seed)

    factor_returns = np.random.randn(n_obs, n_factors) * 0.01
    idio_returns = np.random.randn(n_obs, n_assets) * 0.005

    if static_exposures:
        exposures = np.random.randn(n_assets, n_factors)
    else:
        exposures = np.random.randn(n_obs, n_assets, n_factors)

    if static_weights:
        weights_raw = np.abs(np.random.randn(n_assets)) + 0.1
        weights = weights_raw / weights_raw.sum()
    else:
        weights_raw = np.abs(np.random.randn(n_obs, n_assets)) + 0.1
        weights = weights_raw / weights_raw.sum(axis=1, keepdims=True)

    # Compute portfolio returns from decomposition.
    # For time-varying exposures, follow the as-of convention:
    # R(t) = B(t-1) f(t) + eps(t), so use lagged exposures.
    # portfolio_returns[0] is arbitrary (trimmed by exposure_lag).
    if static_exposures and static_weights:
        x_t = exposures.T @ weights
        s = factor_returns * x_t
        u_P = idio_returns @ weights
        portfolio_returns = np.sum(s, axis=1) + u_P
    elif static_exposures and not static_weights:
        x_t = np.einsum("ta,ak->tk", weights, exposures)
        s = x_t * factor_returns
        u_P = np.sum(weights * idio_returns, axis=1)
        portfolio_returns = np.sum(s, axis=1) + u_P
    elif not static_exposures and static_weights:
        lagged_exp = exposures[:-1]
        x_t = np.einsum("a,tak->tk", weights, lagged_exp)
        s = x_t * factor_returns[1:]
        u_P = np.sum(weights * idio_returns[1:], axis=1)
        portfolio_returns = np.zeros(n_obs)
        portfolio_returns[1:] = np.sum(s, axis=1) + u_P
    else:  # both time-varying
        lagged_exp = exposures[:-1]
        x_t = np.einsum("tik,ti->tk", lagged_exp, weights[1:])
        s = x_t * factor_returns[1:]
        u_P = np.sum(weights[1:] * idio_returns[1:], axis=1)
        portfolio_returns = np.zeros(n_obs)
        portfolio_returns[1:] = np.sum(s, axis=1) + u_P

    factor_names = np.array([f"Factor_{i}" for i in range(n_factors)])
    asset_names = np.array([f"Asset_{i}" for i in range(n_assets)])

    result = {
        "factor_returns": factor_returns,
        "portfolio_returns": portfolio_returns,
        "exposures": exposures,
        "weights": weights,
        "idio_returns": idio_returns,
        "factor_names": factor_names,
        "asset_names": asset_names,
    }

    if include_observations:
        result["observations"] = np.arange(n_obs)

    return result
