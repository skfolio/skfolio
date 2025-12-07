"""Johnson SU Estimator."""

# Copyright (c) 2025
# Authors: The skfolio developers
# Credits: Matteo Manzi, Vincent Maladière, Carlo Nicolini
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numpy.typing as npt
import scipy.stats as st
from scipy.optimize import root
import torch

from skfolio.distribution.univariate._base import BaseUnivariateDist


class JohnsonSU(BaseUnivariateDist):
    r"""Johnson SU Distribution Estimation.

    This estimator fits a univariate Johnson SU distribution to the input data.
    The Johnson SU distribution is flexible and can capture both skewness and fat tails,
    making it appropriate for financial time series modeling.

    The probability density function is:

    .. math::

        f(x, a, b) = \frac{b}{\sqrt{x^2 + 1}}
                     \phi(a + b \log(x + \sqrt{x^2 + 1}))

    where :math:`x`, :math:`a`, and :math:`b` are real scalars; :math:`b > 0`.
    :math:`\phi` is the pdf of the normal distribution.

    The probability density above is defined in the "standardized" form. To shift
    and/or scale the distribution use the loc and scale parameters. Specifically,
    `pdf(x, a, b, loc, scale)` is equivalent to `pdf(y, a, b) / scale` with
    `y = (x - loc) / scale`.

    For more information, you can refer to the `scipy documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.johnsonsu.html#scipy.stats.johnsonsu>`_

    Parameters
    ----------
    loc : float, optional
        If provided, the location parameter is fixed to this value during fitting.
        Otherwise, it is estimated from the data.

    scale : float, optional
        If provided, the scale parameter is fixed to this value during fitting.
        Otherwise, it is estimated from the data.

    random_state : int, RandomState instance or None, default=None
        Seed or random state to ensure reproducibility.

    Attributes
    ----------
    a_ : float
        The fitted first shape parameter of the Johnson SU distribution.

    b_ : float
        The fitted second shape parameter of the Johnson SU distribution.

    loc_ : float
        The fitted location parameter.

    scale_ : float
        The fitted scale parameter.

    Examples
    --------
    >>> from skfolio.datasets import load_sp500_index
    >>> from skfolio.preprocessing import prices_to_returns
    >>> from skfolio.distribution.univariate import JohnsonSU
    >>>
    >>> # Load historical prices and convert them to returns
    >>> prices = load_sp500_index()
    >>> X = prices_to_returns(prices)
    >>>
    >>> # Initialize the estimator.
    >>> model = JohnsonSU()
    >>>
    >>> # Fit the model to the data.
    >>> model.fit(X)
    >>>
    >>> # Display the fitted parameters.
    >>> print(model.fitted_repr)
    JohnsonSU(0.0742, 1.08, 0.00115, 0.00774)
    >>>
    >>> # Compute the log-likelihood, total log-likelihood, CDF, PPF, AIC, and BIC
    >>> log_likelihood = model.score_samples(X)
    >>> score = model.score(X)
    >>> cdf = model.cdf(X)
    >>> ppf = model.ppf(X)
    >>> aic = model.aic(X)
    >>> bic = model.bic(X)
    >>>
    >>> # Generate 5 new samples from the fitted distribution.
    >>> samples = model.sample(n_samples=5)
    >>>
    >>> # Plot the estimated probability density function (PDF).
    >>> fig = model.plot_pdf()
    >>> fig.show()
    """

    a_: float
    b_: float
    loc_: float
    scale_: float
    _scipy_model = st.johnsonsu

    def __init__(
        self,
        loc: float | None = None,
        scale: float | None = None,
        random_state: int | None = None,
    ):
        super().__init__(random_state=random_state)
        self.loc = loc
        self.scale = scale

    @property
    def _scipy_params(self) -> dict[str, float]:
        """Dictionary of parameters to pass to the underlying SciPy distribution."""
        return {"a": self.a_, "b": self.b_, "loc": self.loc_, "scale": self.scale_}

    def fit(self, X: npt.ArrayLike, y=None) -> "JohnsonSU":
        """Fit the univariate Johnson SU distribution model.

        Parameters
        ----------
        X : array-like of shape (n_observations, 1)
            The input data. X must contain a single column.

        y : None
            Ignored. Provided for compatibility with scikit-learn's API.

        Returns
        -------
        self : JohnsonSU
            Returns the instance itself.
        """
        X = self._validate_X(X, reset=True)

        if self.loc is not None and self.scale is not None:
            raise ValueError("Either loc or scale must be None to be fitted")

        fixed_params = {}
        if self.loc is not None:
            fixed_params["floc"] = self.loc
        if self.scale is not None:
            fixed_params["fscale"] = self.scale

        self.a_, self.b_, self.loc_, self.scale_ = self._scipy_model.fit(
            X, **fixed_params
        )
        return self

    @staticmethod
    def _find_johnson_su_params(skew_target: float, excess_kurt_target: float) -> tuple[float, float]:
        """Find the a (gamma) and b (delta) parameters for a Johnson SU distribution.

        Finds parameters that match the target skewness and excess kurtosis.

        Parameters
        ----------
        skew_target : float
            Target skewness value.
        excess_kurt_target : float
            Target excess kurtosis value.

        Returns
        -------
        tuple[float, float]
            The (a, b) parameters for the Johnson SU distribution.

        Raises
        ------
        RuntimeError
            If the optimization fails to find parameters matching the target moments.
        """
        def equations(p):
            a, b = p
            if b <= 0:
                return [1e9, 1e9]
            try:
                skew, kurt = st.johnsonsu.stats(a, b, moments="sk")
            except (ValueError, ZeroDivisionError):
                return [1e9, 1e9]

            return [skew - skew_target, kurt - excess_kurt_target]

        initial_guess = [0.0, 1.0]
        solution = root(equations, initial_guess, method="lm")

        if not solution.success:
            raise RuntimeError(
                f"Failed to find Johnson SU parameters: {solution.message}"
            )

        a, b = solution.x
        return a, b

    @staticmethod
    def sample_from_param(
        mu: float,
        var: float,
        skew: float,
        ex_kurt: float,
        n_samples: int = 1,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Generate random samples from Johnson SU distribution specified by moments.

        This method generates samples from a Johnson SU distribution that matches
        the specified four moments (mean, variance, skewness, excess kurtosis)
        without requiring fitting to data.

        Parameters
        ----------
        mu : float
            Mean (first moment) of the distribution.
        var : float
            Variance (second central moment) of the distribution.
        skew : float
            Skewness (third standardized moment) of the distribution.
        ex_kurt : float
            Excess kurtosis (fourth standardized moment minus 3) of the distribution.
        n_samples : int, default=1
            Number of samples to generate.
        random_state : int, RandomState instance or None, default=None
            Seed or random state to ensure reproducibility.

        Returns
        -------
        X : ndarray of shape (n_samples, 1)
            Random samples from the Johnson SU distribution matching the specified moments.

        Raises
        ------
        ValueError
            If variance is non-positive.
        RuntimeError
            If the optimization fails to find parameters matching the target moments.
        """
        if var <= 0:
            raise ValueError("Variance must be positive")

        std = np.sqrt(var)

        # Find a and b parameters from skewness and excess kurtosis
        a, b = JohnsonSU._find_johnson_su_params(skew, ex_kurt)

        # Calculate loc and scale to match mean and std
        w = np.exp(1 / (b**2))
        mu_y = -np.sqrt(w) * np.sinh(a / b)
        var_y = 0.5 * (w - 1) * (w * np.cosh(2 * a / b) + 1)
        std_y = np.sqrt(var_y)

        scale = std / std_y
        loc = mu - scale * mu_y

        # Generate samples
        samples = st.johnsonsu.rvs(
            a, b, loc=loc, scale=scale, size=n_samples, random_state=random_state
        )

        # Reshape to match existing API (n_samples, 1)
        return samples.reshape(-1, 1)

    @staticmethod
    def sample_from_param_torch(
        batch_size: int,
        n_samples: int,
        mean: float,
        std: float,
        skew: float,
        ex_kurt: float,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Generate random samples from Johnson SU distribution using PyTorch.

        This method generates samples in parallel batches using PyTorch, enabling
        GPU acceleration. The distribution matches the specified four moments
        (mean, standard deviation, skewness, excess kurtosis) without requiring
        fitting to data.

        Parameters
        ----------
        batch_size : int
            Number of parallel batches to generate.
        n_samples : int
            Number of samples per batch.
        mean : float
            Mean (first moment) of the distribution.
        std : float
            Standard deviation (square root of variance) of the distribution.
        skew : float
            Skewness (third standardized moment) of the distribution.
        ex_kurt : float
            Excess kurtosis (fourth standardized moment minus 3) of the distribution.
        device : str, default='cpu'
            PyTorch device on which to generate samples. Options include 'cpu',
            'cuda', 'mps', etc.

        Returns
        -------
        X : torch.Tensor of shape (batch_size, n_samples)
            Random samples from the Johnson SU distribution matching the specified
            moments. The tensor is on the specified device.

        Raises
        ------
        ValueError
            If standard deviation is non-positive.
        RuntimeError
            If the optimization fails to find parameters matching the target moments.
        """
        if std <= 1e-9:
            # Handle case with zero standard deviation
            return torch.full((batch_size, n_samples), mean, device=device)

        if std <= 0:
            raise ValueError("Standard deviation must be positive")

        # Find a and b parameters from skewness and excess kurtosis
        a, b = JohnsonSU._find_johnson_su_params(skew, ex_kurt)

        # Calculate loc and scale to match mean and std
        w = np.exp(1 / (b**2))
        mu_y = -np.sqrt(w) * np.sinh(a / b)
        var_y = 0.5 * (w - 1) * (w * np.cosh(2 * a / b) + 1)
        std_y = np.sqrt(var_y)

        scale = std / std_y
        loc = mean - scale * mu_y

        # Generate standard normal variables
        z = torch.randn((batch_size, n_samples), device=device)

        # Apply inverse Johnson SU transformation
        # Y = loc + scale * sinh((Z - a) / b)
        y = loc + scale * torch.sinh((z - a) / b)

        return y
