"""
Univariate Student Estimation
-----------------------------
"""

# Authors: The skfolio developers
# SPDX-License-Identifier: BSD-3-Clause

import scipy.stats as st

from skfolio.distribution._univariate._base import BaseUnivariate


class StudentT(BaseUnivariate):
    """Student's t Distribution Estimation.

    Parameters
    ----------
    """

    loc_: float
    scale_: float
    dof_: float

    _scipy_model = st.t

    def __init__(self, loc: float | None = None, scale: float | None = None):
        self.loc = loc
        self.scale = scale

    @property
    def scipy_params(self) -> dict[str, float]:
        return {"loc": self.loc_, "scale": self.scale_, "df": self.dof_}

    def fit(self, X, y=None):
        """Fit the Kernel Density model on the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        y : None
            Ignored. This parameter exists only for compatibility

        Returns
        -------
        self : object
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
