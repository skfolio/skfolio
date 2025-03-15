"""Univariate Student's t Estimation."""

# Copyright (c) 2025
# Authors: The skfolio developers
# Credits: Matteo Manzi, Vincent Maladière, Carlo Nicolini
# SPDX-License-Identifier: BSD-3-Clause

import numpy.typing as npt
import scipy.stats as st

from skfolio.distribution.univariate._base import BaseUnivariateDist


class StudentT(BaseUnivariateDist):
    r"""Student's t Distribution Estimation.

    This estimator fits a univariate Student's t distribution to the input data.

    The probability density function is:

    .. math::

        f(x, \nu) = \frac{\Gamma((\nu+1)/2)}
                        {\sqrt{\pi \nu} \Gamma(\nu/2)}
                    (1+x^2/\nu)^{-(\nu+1)/2}

    where :math:`x` is a real number and the degrees of freedom parameter :math:`\nu`
    (denoted `dof` in the implementation) satisfies :math:`\nu > 0`. :math:`\Gamma` is
    the gamma function (`scipy.special.gamma`).

    The probability density above is defined in the "standardized" form. To shift
    and/or scale the distribution use the loc and scale parameters. Specifically,
    `pdf(x, df, loc, scale)` is equivalent to `pdf(y, df) / scale` with
    `y = (x - loc) / scale`.

    For more information, you can refer to the `scipy documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html#scipy.stats.t>`_


    Parameters
    ----------
    loc : float or None, default=None
        If provided, the location parameter is fixed to this value during fitting.
        Otherwise, it is estimated from the data.

    scale : float or None, default=None
        If provided, the scale parameter is fixed to this value during fitting.
        Otherwise, it is estimated from the data.

    random_state : int, RandomState instance or None, default=None
        Seed or random state to ensure reproducibility.

    Attributes
    ----------
    dof_ : float
        The fitted degrees of freedom for the Student's t distribution.

    loc_ : float
        The fitted location parameter.

    scale_ : float
        The fitted scale parameter.

    Examples
    --------
    >>> from skfolio.datasets import load_sp500_index
    >>> from skfolio.preprocessing import prices_to_returns
    >>> from skfolio.distribution.univariate import StudentT
    >>>
    >>> # Load historical prices and convert them to returns
    >>> prices = load_sp500_index()
    >>> X = prices_to_returns(prices)
    >>>
    >>> # Initialize the estimator.
    >>> model = StudentT()
    >>>
    >>> # Fit the model to the data.
    >>> model.fit(X)
    >>>
    >>> # Display the fitted parameters.
    >>> print(model.fitted_repr)
    StudentT(2.75, 0.000618, 0.00681)
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

    dof_: float
    loc_: float
    scale_: float
    _scipy_model = st.t

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
        return {"loc": self.loc_, "scale": self.scale_, "df": self.dof_}

    def fit(self, X: npt.ArrayLike, y=None) -> "StudentT":
        """Fit the univariate Student's t distribution model.

        Parameters
        ----------
        X : array-like of shape (n_observations, 1)
            The input data. X must contain a single column.

        y : None
            Ignored. Provided for compatibility with scikit-learn's API.

        Returns
        -------
        self : StudentT
            Returns the instance itself.
        """
        X = self._validate_X(X, reset=True)

        fixed_params = {}
        if self.loc is not None:
            fixed_params["floc"] = self.loc
        if self.scale is not None:
            fixed_params["fscale"] = self.scale

        self.dof_, self.loc_, self.scale_ = self._scipy_model.fit(X, **fixed_params)
        return self
