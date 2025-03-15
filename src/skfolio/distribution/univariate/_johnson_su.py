"""Johnson SU Estimator."""

# Copyright (c) 2025
# Authors: The skfolio developers
# Credits: Matteo Manzi, Vincent Maladière, Carlo Nicolini
# SPDX-License-Identifier: BSD-3-Clause

import numpy.typing as npt
import scipy.stats as st

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
