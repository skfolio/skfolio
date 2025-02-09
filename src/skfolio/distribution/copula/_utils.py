import warnings
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import scipy.optimize as so
import scipy.stats as st
import sklearn as sk

from skfolio.distribution.copula._base import (
    BaseBivariateCopula,
    CopulaRotation,
)
from skfolio.distribution.copula._independent import IndependentCopula


def best_bivariate_copula_and_fit(
    X: np.ndarray,
    copula_candidates: list[BaseBivariateCopula],
    aic: bool = True,
    independence_level: float = 0.05,
) -> BaseBivariateCopula:
    """Find the best bivariate copula that minimize either the AIC or BIC
    criterion and returns the fitted model.

    Parameters
    ----------

    independence_level : float, default=0.05
        Significance level of the Kendall tau independence test. A p-value below this
        level means that the independence hypothesis is rejected and non-independent
        copula will be search and fitted.
    """

    kendall_tau, p_value = st.kendalltau(X[:, 0], X[:, 1])
    if p_value >= independence_level:
        return IndependentCopula().fit(X)

    results = {}
    for copula in copula_candidates:
        if not isinstance(copula, BaseBivariateCopula):
            raise ValueError(
                "The candidate copula must inherit from BaseBivariateCopula"
            )
        copula = sk.clone(copula)
        if copula.itau and copula.kendall_tau is None:
            # Faster computation by reusing kendall tau if itau
            copula.kendall_tau = kendall_tau
        copula.fit(X)
        results[copula] = copula.aic(X) if aic else copula.bic(X)
    best_copula = min(results, key=results.get)
    return best_copula


def _find_best_rotation_kendall_tau_inversion(
    func: Callable, X: np.ndarray, theta: float
) -> CopulaRotation:
    results = {}
    for rotation in CopulaRotation:
        X_rotated = _apply_copula_rotation(X, rotation=rotation)
        results[rotation] = func(X=X_rotated, theta=theta)
    best_rotation = min(results, key=results.get)
    return best_rotation


def _find_best_theta_and_rotation_mle(
    func: Callable, X: np.ndarray, bounds: tuple[float, float], tolerance: float = 1e-4
) -> tuple[float, CopulaRotation]:
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
    """Swap u with v if first_margin is False.

    Parameters
    ----------
    X : array-like of shape (n_observations, 2)
       An array of bivariate inputs `(u, v)` where each row represents a
       bivariate observation.

    first_margin : bool, default False
        If True, X is returned without modification; otherwise, X coluns are
        swapped.

    Returns
    -------
    X: ndarray of shape (n_observations, 2)
       The swapped data array if first_margin is False.
    """
    assert X.ndim == 2
    assert X.shape[1] == 2
    if first_margin:
        return np.hstack((X[:, [1]], X[:, [0]]))
    return X


def _apply_rotation_cdf(
    func: Callable, X: np.ndarray, rotation: CopulaRotation, **kwargs
) -> np.ndarray:
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
