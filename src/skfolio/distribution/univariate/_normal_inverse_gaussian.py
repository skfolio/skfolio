"""
Normal Inverse Gaussian Estimator
---------------------------------
"""

# Authors: The skfolio developers
# SPDX-License-Identifier: BSD-3-Clause

import scipy.stats as st

from skfolio.distribution.univariate._base import BaseUnivariate


class NormalInverseGaussian(BaseUnivariate):
    """Normal Inverse Gaussian Distribution Estimator.

    Parameters
    ----------
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
        return {"a": self.a_, "b": self.b_, "loc": self.loc_, "scale": self.scale_}

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
