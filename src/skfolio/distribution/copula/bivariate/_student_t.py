"""
Bivariate Student's t Copula Estimation
---------------------------------------
"""

# Authors: The skfolio developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numpy.typing as npt
import scipy.optimize as so
import scipy.special as sp
import scipy.stats as st
import sklearn.utils.validation as skv

from skfolio.distribution.copula.bivariate._base import (
    BaseBivariateCopula,
    CopulaRotation,
)


class StudentTCopula(BaseBivariateCopula):
    r"""
    Bivariate Student's :math:`t` Copula Estimation.

    This class implements the estimation of a bivariate Student's :math:`t` copula
    using a Kendall-based two-step approach for robustness and efficiency.
    It allows for different copula rotations to capture various dependence
    structures.

    The bivariate Student's :math:`t` copula density is defined as:

    .. math::
        c_{\nu,\rho}(u, v) =
            \frac{\Gamma\left(\frac{\nu + 2}{2}\right)\Gamma\left(\frac{\nu}{2}\right)}{\Gamma\left(\frac{\nu + 1}{2}\right)^2}
            \cdot \frac{1}{\sqrt{1 - \rho^2}}
            \cdot \frac{\left(1 + \frac{x^2}{\nu}\right)^{\frac{\nu + 1}{2}}
                    \left(1 + \frac{y^2}{\nu}\right)^{\frac{\nu + 1}{2}}}
                   {\left(1 + \frac{x^2 - 2\rho x y + y^2}{\nu (1 - \rho^2)}\right)^{\frac{\nu + 2}{2}}},

    where:
    - :math:`\nu > 0` is the degrees of freedom.
    - :math:`\rho \in (-1, 1)` is the correlation coefficient.
    - :math:`x = t_{\nu}^{-1}(u)` and :math:`y = t_{\nu}^{-1}(v)` are the inverse CDF (percent point function) transformations.

    Parameters
    ----------
    rotation : CopulaRotation, optional
        The rotation to apply to the copula (default is no rotation).

    Attributes
    ----------
    params_ : dict[str, float]
        Dictionary containing the estimated copula parameters:
        - 'rho': Correlation coefficient (:math:`\rho`) in [-1, 1].
        - 'dof': Degrees of freedom (:math:`\nu`) > 0.
    """

    params_: dict[str, float]

    def __init__(self, rotation: CopulaRotation = CopulaRotation.R0):
        super().__init__(rotation=rotation)

    def fit(self, X, y=None):
        """
        Fit the Bivariate Student's :math:`t` Copula.

        This method uses a Kendall-based two-step approach:
        1. Estimates the correlation parameter (:math:`\rho`) from Kendall's :math:`\tau`.
        2. Optimizes the degrees of freedom (:math:`\nu`) by minimizing the negative log-likelihood.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            The input data where each row represents a bivariate observation.
            The data should be transformed to uniform marginals in [0, 1].

        y : None
            Ignored. This parameter exists only with scikit-learn's API.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_and_rotate(X, reset=True)

        # Estimate correlation from Kendall's tau
        tau = st.kendalltau(X[:, 0], X[:, 1]).statistic
        rho = np.sin((np.pi * tau) / 2.0)

        def _neg_log_likelihood(log_dof: float) -> float:
            """
            Negative log-likelihood function for optimization.

            Parameters
            ----------
            log_dof : float
                Log-transformed degrees of freedom.

            Returns
            -------
            float
                The negative log-likelihood value.
            """
            # Log-transform to map from x ∈ (-inf, +inf) to theta ∈ (0, +inf)
            _dof = np.exp(log_dof)
            return -np.sum(_sample_scores(X=X, rho=rho, dof=_dof))

        res = so.minimize_scalar(_neg_log_likelihood, method="brent")
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")
        dof = np.exp(res.x)

        self.params_ = {
            "rho": rho,
            "dof": dof,
        }
        return self

    def score_samples(self, X: npt.ArrayLike) -> np.ndarray:
        """Compute the log-likelihood of each sample under the model.

        Parameters
        ----------
         X : array-like of shape (n_samples, 2)
            The input data where each row represents a bivariate observation.
            The data should be transformed to uniform marginals in [0, 1].

        Returns
        -------
        density : ndarray of shape (n_samples,)
            The log-likelihood of each sample under the fitted copula.
        """
        skv.check_is_fitted(self)
        X = self._validate_and_rotate(X, reset=False)
        log_density = _sample_scores(
            X=X, rho=self.params_["rho"], dof=self.params_["dof"]
        )
        return log_density


def _sample_scores(X: np.ndarray, rho: float, dof: float) -> np.ndarray:
    """
    Compute the log-likelihood for each sample under a bivariate Student's :math:`t` copula.

    This function calculates the log-density of each observation in X given the
    copula parameters rho and dof.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 2)
        The input data where each row represents a bivariate observation.
        The data should be transformed to uniform marginals in [0, 1].
    rho : float
        The correlation coefficient parameter (:math:`\rho`) of the copula. Must be in [-1, 1].
    dof : float
        The degrees of freedom parameter (:math:`\nu`) of the copula. Must be > 0.

    Returns
    -------
    log_likelihood : np.ndarray of shape (n_samples,)
        The log-likelihood values for each sample.

    Raises
    ------
    ValueError
        If rho is not in (-1, 1) or dof is not positive.
    """
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must be between -1 and 1 (exclusive).")
    if dof <= 0.0:
        raise ValueError("Degrees of freedom (dof) must be positive.")

    # Inverse CDF (ppf) to transform uniform (0,1) data to t-values
    x, y = st.t.ppf(X, dof).T  # shape (T,2)
    a = 1.0 - rho**2
    ll = (
        sp.gammaln((dof + 2.0) / 2.0)
        + sp.gammaln(dof / 2.0)
        - 2.0 * sp.gammaln((dof + 1.0) / 2.0)
        - np.log(a) / 2
        + (dof + 1.0) / 2.0 * (np.log(1.0 + x**2 / dof) + np.log(1.0 + y**2 / dof))
        - (dof + 2.0) / 2.0 * np.log(1.0 + (x**2 - 2 * rho * x * y + y**2) / a / dof)
    )
    return ll
