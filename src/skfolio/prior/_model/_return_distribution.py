"""Return Distribution Dataclass."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from skfolio.prior._model._covariance_sqrt import CovarianceSqrt
from skfolio.prior._model._factor_model import FactorModel
from skfolio.typing import BoolArray, FloatArray
from skfolio.utils.stats import safe_cholesky

__all__ = ["ReturnDistribution"]


# frozen=True with eq=False will lead to an id-based hashing which is needed for
# caching CVX models in Optimization without impacting performance
@dataclass(frozen=True, eq=False)
class ReturnDistribution:
    """Return distribution estimated by a prior estimator.

    Prior estimators always return the **full universe** (all assets that have ever been
    part of the investment universe). Assets that are not investable at the current
    point in time (e.g. delisted, not yet listed, warm-up period) are represented with
    `NaN` in `mu`, `covariance`, and/or `returns`.

    An asset is considered investable when both `mu[i]` and `covariance[i, i]` are
    finite. The `investable_mask` property infers this condition on first access
    and reconciles warm-up periods across independent moment estimators.

    Use `investable_subset` before passing the distribution to downstream routines that
    operate only on the investable universe.

    Attributes
    ----------
    mu : ndarray of shape (n_assets,)
        Estimation of the assets expected returns.

    covariance : ndarray of shape (n_assets, n_assets)
        Estimation of the assets covariance matrix.

    returns : ndarray of shape (n_observations, n_assets)
        Estimation of the assets returns.

    sample_weight : ndarray of shape (n_observations,), optional
        Sample weights for each observation. If `None`, equal weights are assumed.

    factor_model : FactorModel, optional
        Factor model decomposition and diagnostics. The default is `None`.
    """

    mu: FloatArray
    covariance: FloatArray
    returns: FloatArray
    sample_weight: FloatArray | None = None
    factor_model: FactorModel | None = None

    def __post_init__(self) -> None:
        """Validate array shapes."""
        if self.mu.ndim != 1:
            raise ValueError("`mu` must be a 1D array of shape (n_assets,).")

        n_assets = self.mu.shape[0]

        if self.covariance.shape != (n_assets, n_assets):
            raise ValueError(
                "`covariance` must be a 2D array of shape "
                f"({n_assets}, {n_assets}), got {self.covariance.shape}."
            )

        if self.returns.ndim != 2 or self.returns.shape[1] != n_assets:
            raise ValueError(
                "`returns` must be a 2D array of shape "
                f"(n_observations, {n_assets}), got {self.returns.shape}."
            )

        n_observations = self.returns.shape[0]
        if self.sample_weight is not None and self.sample_weight.shape != (
            n_observations,
        ):
            raise ValueError(
                "`sample_weight` must be a 1D array of shape "
                f"({n_observations},), got {self.sample_weight.shape}."
            )

        if (
            self.factor_model is not None
            and self.factor_model.loading_matrix.shape[0] != n_assets
        ):
            raise ValueError(
                "`factor_model` must be defined on the same asset universe as "
                "`mu` and `covariance`."
            )

    @cached_property
    def investable_mask(self) -> BoolArray | None:
        """Boolean mask where `True` marks investable assets.

        The mask is inferred as the intersection of finite `mu` values and a finite
        diagonal in `covariance`. Returns `None` when all assets are investable.

        Raises
        ------
        ValueError
            If no asset is investable (all `NaN` in `mu` and/or `covariance`).
        """
        mask = np.isfinite(self.mu) & np.isfinite(np.diag(self.covariance))
        if not mask.any():
            raise ValueError(
                "All assets are non-investable (NaN in `mu` and/or `covariance` for "
                "every asset). Ensure that the moment estimators have received enough "
                "observations or that the universe contains at least one active asset."
            )
        if mask.all():
            return None
        return mask

    @property
    def n_assets(self) -> int:
        """Total number of assets in the full universe."""
        return len(self.mu)

    @property
    def n_investable_assets(self) -> int:
        """Number of investable assets."""
        if self.investable_mask is None:
            return self.n_assets
        return int(np.count_nonzero(self.investable_mask))

    @property
    def covariance_sqrt(self) -> CovarianceSqrt:
        r"""Covariance square root for SOC-based optimization.

        When a :class:`~skfolio.prior.FactorModel` is available, delegates to
        :attr:`FactorModel.covariance_sqrt` to exploit the low-rank factor structure.
        Otherwise, falls back to the Cholesky decomposition of `covariance`.

        When non-investable assets are represented with `NaN` entries and no factor
        model is available, callers should apply `investable_subset` first.

        Returns
        -------
        CovarianceSqrt
        """
        if self.factor_model is not None:
            return self.factor_model.covariance_sqrt
        return self._cholesky_covariance_sqrt

    def investable_subset(self, slim: bool = False) -> ReturnDistribution:
        """Return a `ReturnDistribution` restricted to investable assets.

        Parameters
        ----------
        slim : bool, default=False
            When `True`, heavy diagnostic fields on the nested `FactorModel` (e.g.
            `exposures`, `idio_returns`, `idio_variances`, `benchmark_weights`) are set
            to `None` to reduce memory usage. This is typically used by optimization
            estimators that only need `loading_matrix`, covariance, and return series.

        Returns
        -------
        subset : ReturnDistribution
            Distribution over the investable assets only. If all assets are already
            investable and `slim=False`, `self` is returned.
        """
        mask = self.investable_mask
        if mask is None or np.all(mask):
            if not slim or self.factor_model is None:
                return self

            return ReturnDistribution(
                mu=self.mu,
                covariance=self.covariance,
                returns=self.returns,
                sample_weight=self.sample_weight,
                factor_model=self.factor_model.select_assets(slim=True),
            )

        factor_model = self.factor_model
        if factor_model is not None:
            factor_model = factor_model.select_assets(assets=mask, slim=slim)

        return ReturnDistribution(
            mu=self.mu[mask],
            covariance=self.covariance[np.ix_(mask, mask)],
            returns=self.returns[:, mask],
            sample_weight=self.sample_weight,
            factor_model=factor_model,
        )

    @cached_property
    def _cholesky_covariance_sqrt(self) -> CovarianceSqrt:
        """Cholesky-based fallback square root, cached per instance."""
        return CovarianceSqrt(components=(safe_cholesky(self.covariance),))
