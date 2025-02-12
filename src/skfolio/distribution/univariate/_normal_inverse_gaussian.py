"""Normal Inverse Gaussian Estimator"""

# Copyright (c) 2025
# Authors: The skfolio developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy.typing as npt
import scipy.stats as st

from skfolio.distribution.univariate._base import BaseUnivariateDist


class NormalInverseGaussian(BaseUnivariateDist):
    """Normal Inverse Gaussian Distribution Estimation.

    This estimator fits a univariate Normal Inverse Gaussian (NIG) distribution
    to the input data.

    Parameters
    ----------
    loc : float, optional
        If provided, the location parameter is fixed to this value during fitting.
        Otherwise, it is estimated from the data.

    scale : float, optional
        If provided, the scale parameter is fixed to this value during fitting.
        Otherwise, it is estimated from the data.

    Attributes
    ----------
    a_ : float
        The fitted shape parameter a of the NIG distribution.

    b_ : float
        The fitted shape parameter b of the NIG distribution.

    loc_ : float
        The fitted location parameter.

    scale_ : float
        The fitted scale parameter.
    """

    a_: float
    b_: float
    loc_: float
    scale_: float
    _scipy_model = st.norminvgauss

    def __init__(self, loc: float | None = None, scale: float | None = None):
        self.loc = loc
        self.scale = scale

    @property
    def scipy_params(self) -> dict[str, float]:
        """Dictionary of parameters to pass to the underlying SciPy distribution"""
        return {"a": self.a_, "b": self.b_, "loc": self.loc_, "scale": self.scale_}

    @property
    def fitted_repr(self) -> str:
        """String representation of the fitted univariate distribution"""
        return f"{self.__class__.__name__}({self.a_:0.3g}, {self.b_:0.3g}, {self.loc_:0.3g}, {self.scale_:0.3g})"

    def fit(self, X: npt.ArrayLike, y=None) -> "NormalInverseGaussian":
        """Fit the univariate Normal Inverse Gaussian distribution model.

        Parameters
        ----------
        X : array-like of shape (n_observations, 1)
            The input data. X must contain a single column.

        y : None
            Ignored. Provided for compatibility with scikit-learn's API.

        Returns
        -------
        self : NormalInverseGaussian
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
