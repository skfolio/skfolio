"""Bivariate Copula Utils."""

# Copyright (c) 2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# Credits: Matteo Manzi, Vincent Maladière, Carlo Nicolini
# SPDX-License-Identifier: BSD-3-Clause

import operator
import warnings
from collections.abc import Callable
from enum import Enum

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import scipy.optimize as so
import scipy.stats as st


class CopulaRotation(Enum):
    r"""Enum representing the rotation (in degrees) to apply to a bivariate copula.

    It follows the standard clockwise convention:

    - `CopulaRotation.R0` (0°): :math:`(u, v) \mapsto (u, v)`
    - `CopulaRotation.R90` (90°): :math:`(u, v) \mapsto (v,\, 1 - u)`
    - `CopulaRotation.R180` (180°): :math:`(u, v) \mapsto (1 - u,\, 1 - v)`
    - `CopulaRotation.R270` (270°): :math:`(u, v) \mapsto (1 - v,\, u)`

    Attributes
    ----------
    R0 : int
        No rotation (0°).
    R90 : int
        90° rotation.
    R180 : int
        180° rotation.
    R270 : int
        270° rotation.
    """

    R0 = "0°"
    R90 = "90°"
    R180 = "180°"
    R270 = "270°"

    def __str__(self) -> str:
        """String representation."""
        return self.value


def compute_pseudo_observations(X: npt.ArrayLike) -> np.ndarray:
    """
    Compute pseudo-observations by ranking each column of the data and scaling the
    ranks.

    The goal of computing pseudo-observations is to transform your raw data into a
    form that has uniform marginal distributions on the open interval (0, 1). This is
    particularly useful in copula modeling and other statistical methods where the
    dependence structure is of primary interest, independent of the marginal
    distributions.

    This function transforms each column of the input data into pseudo-observations
    on the (0, 1) interval. For each column, the ranks (starting at 1) are divided by
    (n_samples + 1) to avoid 0 and 1 values, which are problematic for many copula
    methods.

    Parameters
    ----------
    X : array-like of shape (n_observations, n_assets)
        Input data.

    Returns
    -------
    pseudo_observations: ndarray of shape (n_observations, n_assets)
        An array of pseudo-observations corresponding to the ranks scaled to (0, 1).
    """
    X = np.asarray(X)
    # Compute column-wise ranks; rankdata returns ranks starting at 1.
    ranks = st.rankdata(X, axis=0)
    n_samples = X.shape[0]
    pseudo_observations = ranks / (n_samples + 1)
    return pseudo_observations


def empirical_tail_concentration(X: npt.ArrayLike, quantiles: np.ndarray) -> np.ndarray:
    """
    Compute empirical tail concentration for the two variables in X.
    This function computes the concentration at each quantile provided.

    The tail concentration are estimated as:
      - Lower tail: λ_L(q) = P(U₂ ≤ q | U₁ ≤ q)
      - Upper tail: λ_U(q) = P(U₂ ≥ q | U₁ ≥ q)

    where U₁ and U₂ are the pseudo-observations.

    Parameters
    ----------
    X : array-like of shape (n_observations, 2)
        A 2D array with exactly 2 columns representing the pseudo-observations.

    quantiles : array-like of shape (n_quantiles,)
        A 1D array of quantile levels (values between 0 and 1) at which to compute the
        concentration.

    Returns
    -------
    concentration : ndarray of shape (n_quantiles,)
        An array of empirical tail concentration values for the given quantiles.

    References
    ----------
    .. [1] "Quantitative Risk Management: Concepts, Techniques, and Tools",
        McNeil, Frey, Embrechts (2005)

    Raises
    ------
    ValueError
        If X is not a 2D array with exactly 2 columns or if quantiles are not in [0, 1].
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must be a 2D array with exactly 2 columns.")
    if not np.all((X >= 0) & (X <= 1)):
        raise ValueError("X must be pseudo-observation in the interval `[0, 1]`")
    quantiles = np.asarray(quantiles)
    if not np.all((quantiles >= 0) & (quantiles <= 1)):
        raise ValueError("quantiles must be between 0.0 and 1.0.")

    def func(q: np.ndarray, is_lower: bool) -> np.ndarray:
        op = operator.le if is_lower else operator.ge
        cond = op(X[:, 0, np.newaxis], q)
        count = np.count_nonzero(cond, axis=0).astype(float)
        mask = count != 0
        count[mask] = (
            np.count_nonzero(cond & op(X[:, 1, np.newaxis], q), axis=0)[mask]
            / count[mask]
        )
        return count

    concentration = np.where(
        quantiles <= 0.5, func(quantiles, True), func(quantiles, False)
    )
    return concentration


def plot_tail_concentration(
    tail_concentration_dict: dict[str, npt.ArrayLike],
    quantiles: np.ndarray,
    title: str = "Empirical Tail Dependencies",
    smoothing: float | None = 0.5,
) -> go.Figure:
    """
    Plot the empirical tail concentration curves.

    This function takes a dictionary where keys are dataset names and values are the
    corresponding tail concentration arrays computed at the given quantiles. It then
    creates a Plotly figure with the tail concentration curves. The x-axis (quantiles)
    and y-axis (tail concentration) are both formatted as percentages.

    Parameters
    ----------
    tail_concentration_dict : dict[str, ArrayLike]
        A dictionary mapping dataset names to their tail concentration values.

    quantiles : array-like of shape (n_quantiles,)
        The quantile levels at which the tail concentration has been computed.

    title : str, default="Empirical Tail Dependencies"
        The title for the plot.

    smoothing : float or None, default=0.5
        Smoothing parameter for the spline line shape. If provided, the curves will be
        smoothed using a spline interpolation.

    Returns
    -------
    fig : go.Figure
        A Plotly figure object containing the tail concentration curves.

    Raises
    ------
    ValueError
        If the smoothing parameter is not in the allowed range.
    """
    if smoothing is not None and not (0 <= smoothing <= 1.3):
        raise ValueError("The smoothing parameter must be between 0 and 1.3.")

    quantiles = np.asarray(quantiles)
    traces = []
    # Determine the line shape and include the smoothing parameter if applicable.
    if smoothing is not None:
        line_dict = {"shape": "spline", "smoothing": smoothing}
    else:
        line_dict = {"shape": "linear"}

    # Iterate over each dataset name and its corresponding data array.
    for name, concentration in tail_concentration_dict.items():
        concentration = np.asarray(concentration)
        trace = go.Scatter(
            x=quantiles,
            y=np.asarray(concentration),
            mode="lines",
            name=name,
            line=line_dict,
        )
        traces.append(trace)

    # Create the Plotly figure.
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        xaxis_title="Quantile",
        yaxis_title="Tail Concentration",
    )
    # Update both axes to show percentages with an enhanced grid.
    fig.update_xaxes(
        range=[0, 1],
        tickformat=".0%",
        dtick=0.05,
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgrey",
    )
    fig.update_yaxes(
        range=[0, 1],
        tickformat=".0%",
        dtick=0.05,
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgrey",
    )

    return fig


def _select_rotation_itau(
    func: Callable, X: np.ndarray, theta: float
) -> CopulaRotation:
    """
    Select the optimal copula rotation based on a provided function.

    This helper function applies each rotation defined in CopulaRotation to the data X,
    computes a criterion value using the provided function (which takes X and theta as
    arguments), and returns the rotation that minimizes this value.

    Parameters
    ----------
    func : Callable
       A function that computes a criterion (e.g., a negative log-likelihood) for a given
       rotated dataset and copula parameter theta.

    X : ndarray of shape (n_observations, 2)
       A 2D array of bivariate inputs.

    theta : float
       The copula parameter to be used in the criterion function.

    Returns
    -------
    CopulaRotation
       The rotation (an element of CopulaRotation) that minimizes the criterion value.
    """
    results = {}
    for rotation in CopulaRotation:
        X_rotated = _apply_copula_rotation(X, rotation=rotation)
        results[rotation] = func(X=X_rotated, theta=theta)
    best_rotation = min(results, key=results.get)
    return best_rotation


def _select_theta_and_rotation_mle(
    func: Callable, X: np.ndarray, bounds: tuple[float, float], tolerance: float = 1e-4
) -> tuple[float, CopulaRotation]:
    """
    Select the optimal copula parameter theta and rotation using maximum likelihood
    estimation.

    For each rotation defined in CopulaRotation, this function applies the rotation to
    X, then minimizes the negative log-likelihood over theta using a bounded scalar
    optimization. It returns the theta and rotation that yield the minimum criterion
    value.

    Parameters
    ----------
    func : Callable
        A function that computes the negative log-likelihood (or similar criterion) for a
        given value of theta and rotated data X.

    X : ndarray of shape (n_observations, 2)
        A 2D array of bivariate inputs.

    bounds : tuple[float, float]
        The lower and upper bounds for the copula parameter theta.

    tolerance : float, default=1e-4
        The tolerance for the scalar minimization optimization.

    Returns
    -------
    tuple
        A tuple (theta, rotation) where theta is the optimal copula parameter and
        rotation is the corresponding CopulaRotation.

    Raises
    ------
    RuntimeError
        If the optimization fails for all rotations.
    """
    results = []
    for rotation in CopulaRotation:
        X_rotated = _apply_copula_rotation(X, rotation=rotation)
        result = so.minimize_scalar(
            func,
            args=(X_rotated,),
            bounds=bounds,
            method="bounded",
            options={"xatol": tolerance},
        )
        if result.success:
            results.append(
                {
                    "neg_log_likelihood": result.fun,
                    "theta": result.x,
                    "rotation": rotation,
                }
            )
        else:
            warnings.warn(
                f"Optimization failed for rotation {rotation}: {result.message}",
                RuntimeWarning,
                stacklevel=2,
            )
    if len(results) == 0:
        raise RuntimeError("Optimization failed for all rotations")

    best = min(results, key=lambda d: d["neg_log_likelihood"])
    return best["theta"], best["rotation"]


def _apply_copula_rotation(X: npt.ArrayLike, rotation: CopulaRotation) -> np.ndarray:
    r"""Apply a bivariate copula rotation using the standard (clockwise) convention.

    The transformations are defined as follows:

    - `CopulaRotation.R0` (0°): :math:`(u, v) \mapsto (u, v)`
    - `CopulaRotation.R90` (90°): :math:`(u, v) \mapsto (v,\, 1 - u)`
    - `CopulaRotation.R180` (180°): :math:`(u, v) \mapsto (1 - u,\, 1 - v)`
    - `CopulaRotation.R270` (270°): :math:`(u, v) \mapsto (1 - v,\, u)`

    Parameters
    ----------
    X : array-like of shape (n_observations, 2)
        An array of bivariate inputs `(u, v)` where each row represents a
        bivariate observation.

    rotation : CopulaRotation
        The rotation to apply to the copula (default is no rotation).

    Returns
    -------
    rotated_X: ndarray of shape (n_observations, 2)
        The rotated data array.
    """
    match rotation:
        case CopulaRotation.R0:
            # No rotation
            pass
        case CopulaRotation.R90:
            # (u, v) -> (v, 1 - u)
            X = np.column_stack([X[:, 1], 1.0 - X[:, 0]])
        case CopulaRotation.R180:
            # (u, v) -> (1 - u, 1 - v)
            X = 1.0 - X
        case CopulaRotation.R270:
            # (u, v) -> (1 - v, u)
            X = np.column_stack([1.0 - X[:, 1], X[:, 0]])
        case _:
            raise ValueError(f"Unsupported rotation: {rotation}")
    return X


def _apply_margin_swap(X: np.ndarray, first_margin: bool) -> np.ndarray:
    """
    Swap the columns of X if first_margin is False.

    If first_margin is True, X is returned unchanged; otherwise, the columns
    of X are swapped.

    Parameters
    ----------
    X : ndarray of shape (n_observations, 2)
        A 2D array of bivariate inputs (u, v).
    first_margin : bool
        If True, no swap is performed; if False, the columns of X are swapped.

    Returns
    -------
    X_swapped : ndarray of shape (n_observations, 2)
        The data array with columns swapped if first_margin is False.
    """
    assert X.ndim == 2
    assert X.shape[1] == 2
    if first_margin:
        return X[:, [1, 0]]
    return X


def _apply_rotation_cdf(
    func: Callable, X: np.ndarray, rotation: CopulaRotation, **kwargs
) -> np.ndarray:
    """
    Apply a copula rotation to X and compute the corresponding CDF values.

    Parameters
    ----------
    func : Callable
       A function that computes the CDF given data X and additional keyword arguments.

    X : ndarray of shape (n_observations, 2)
       A 2D array of bivariate inputs.

    rotation : CopulaRotation
       The rotation to apply.

    **kwargs
       Additional keyword arguments to pass to the CDF function.

    Returns
    -------
    rotated_cdf : ndarray of shape (n_observations,)
       The transformed CDF values after applying the rotation.
    """
    rotated_X = _apply_copula_rotation(X, rotation=rotation)
    cdf = func(X=rotated_X, **kwargs)

    match rotation:
        case CopulaRotation.R0:
            pass
        case CopulaRotation.R90:
            cdf = X[:, 1] - cdf
        case CopulaRotation.R180:
            cdf = np.sum(X, axis=1) - 1 + cdf
        case CopulaRotation.R270:
            cdf = X[:, 0] - cdf
        case _:
            raise ValueError(f"Unsupported rotation: {rotation}")

    return cdf


def _apply_rotation_partial_derivatives(
    func: Callable,
    X: np.ndarray,
    rotation: CopulaRotation,
    first_margin: bool,
    **kwargs,
) -> np.ndarray:
    """
    Apply a copula rotation to X and compute the corresponding partial derivatives.

    This function rotates the data X using the specified rotation and then computes
    the partial derivative (h-function) using the provided function. The result is then
    adjusted according to the rotation and the margin of interest.

    Parameters
    ----------
    func : Callable
        A function that computes the partial derivative (h-function) given X, the
        margin, and any additional keyword arguments.

    X : ndarray of shape (n_observations, 2)
        A 2D array of bivariate inputs.

    rotation : CopulaRotation
        The rotation to apply.

    first_margin : bool
        If True, compute the partial derivative with respect to the first margin;
        otherwise, compute it with respect to the second margin.

    **kwargs
        Additional keyword arguments to pass to the partial derivative function.

    Returns
    -------
    z : ndarray of shape (n_observations,)
        The transformed partial derivative values after applying the rotation.
    """
    rotated_X = _apply_copula_rotation(X, rotation=rotation)

    match rotation:
        case CopulaRotation.R0:
            z = func(X=rotated_X, first_margin=first_margin, **kwargs)
        case CopulaRotation.R90:
            if first_margin:
                z = func(X=rotated_X, first_margin=not first_margin, **kwargs)
            else:
                z = 1 - func(X=rotated_X, first_margin=not first_margin, **kwargs)
        case CopulaRotation.R180:
            z = 1 - func(X=rotated_X, first_margin=first_margin, **kwargs)
        case CopulaRotation.R270:
            if first_margin:
                z = 1 - func(X=rotated_X, first_margin=not first_margin, **kwargs)
            else:
                z = func(X=rotated_X, first_margin=not first_margin, **kwargs)
        case _:
            raise ValueError(f"Unsupported rotation: {rotation}")
    return z
