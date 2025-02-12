"""Univariate Student's t Estimation"""

# Copyright (c) 2025
# Authors: The skfolio developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy.typing as npt
import scipy.stats as st

from skfolio.distribution.univariate._base import BaseUnivariateDist


class StudentT(BaseUnivariateDist):
    """Student's t Distribution Estimation.

    This estimator fits a univariate Student's t distribution to the input data.

    Parameters
    ----------
    loc : float or None, default=None
        If provided, the location parameter is fixed to this value during fitting.
        Otherwise, it is estimated from the data.

    scale : float or None, default=None
        If provided, the scale parameter is fixed to this value during fitting.
        Otherwise, it is estimated from the data.

    Attributes
    ----------
    dof_ : float
        The fitted degrees of freedom for the Student's t distribution.

    loc_ : float
        The fitted location parameter.

    scale_ : float
        The fitted scale parameter.
    """

    dof_: float
    loc_: float
    scale_: float
    _scipy_model = st.t

    def __init__(self, loc: float | None = None, scale: float | None = None):
        self.loc = loc
        self.scale = scale

    @property
    def scipy_params(self) -> dict[str, float]:
        """Dictionary of parameters to pass to the underlying SciPy distribution"""
        return {"loc": self.loc_, "scale": self.scale_, "df": self.dof_}

    @property
    def fitted_repr(self) -> str:
        """String representation of the fitted univariate distribution"""
        return f"{self.__class__.__name__}({self.dof_:0.3g}, {self.loc_:0.3g}, {self.scale_:0.3g})"

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
