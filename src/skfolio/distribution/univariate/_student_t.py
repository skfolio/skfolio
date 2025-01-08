"""
Univariate Student Estimation
-----------------------------
"""

# Authors: The skfolio developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import scipy.stats as st
from sklearn.utils.validation import validate_data

from skfolio.distribution.univariate._base import BaseUnivariate


class StudentT(BaseUnivariate):
    """Student's t Distribution Estimation.

    Parameters
    ----------
    """

    params_: dict[str, float]
    _scipy_model = st.t

    def __init__(self):
        pass

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
        X = validate_data(self, X, dtype=np.float64)
        if X.shape[1] != 1:
            raise ValueError(
                "X should should contain a single column for Univariate Distribution"
            )

        df, loc, scale = self._scipy_model.fit(X)

        self.params_ = {
            "df": df,
            "loc": loc,
            "scale": scale,
        }
        return self
