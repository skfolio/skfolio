"""Derived factor exposure computed from another factor's exposure."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable

import numpy as np

import skfolio.typing as skt
from skfolio._constants import _BENCHMARK_WEIGHTS, _PASSTHROUGH
from skfolio.containers import AssetPanel
from skfolio.factor_model.factor_exposure._base import BaseFactorExposure
from skfolio.preprocessing import BaseCSTransformer, CSStandardScaler, CSWinsorizer
from skfolio.typing import FloatArray
from skfolio.utils.tools import check_estimator
from skfolio.utils.validation import validate_asset_panel

__all__ = ["DerivedFactor"]


class DerivedFactor(BaseFactorExposure, stateless=True):
    """Factor exposure derived from another factor's computed exposure.

    The derived exposure is computed by applying `func` to the source factor's
    exposure, then optionally applying outlier and scoring transformations.

    Parameters
    ----------
    source : str
        Name of the source factor whose exposure will be transformed. The source factor
        must be defined in the factors list of `CharacteristicsFactorModel`. Dependency
        ordering is handled automatically via topological sorting.

    func : Callable[[np.ndarray], np.ndarray]
        Function to apply to the source exposure. Receives a 2D array of shape
        (n_observations, n_assets) and should return an array of the same shape.
        The source exposure is passed directly. If `func` uses in-place operations,
        it should copy the input first unless mutating the source exposure is intended.

    family : str, default="style"
        The factor family this exposure belongs to (e.g., "market", "style", "industry",
        "country"). Factor families group related factors for basket-neutral constraints,
        neutralization, attribution and reporting. The default is `"style"`.

    outlier_transformer : BaseCSTransformer or "passthrough" or None, default="passthrough"
        Cross-sectional transformer for outlier handling applied after `func`. If None,
        defaults to `CSWinsorizer()`. Use "passthrough" to skip.

    scoring_transformer : BaseCSTransformer or "passthrough" or None, default=None
        Cross-sectional transformer for scoring applied after outlier handling.
        If None, defaults to `CSStandardScaler()`. Use "passthrough" to skip.

    transform_by_group : str, optional
        Name of a categorical characteristic in the AssetPanel to use for group-wise
        transformations. If provided, outlier and scoring transformations are applied
        within each group separately.

    Attributes
    ----------
    outlier_transformer_ : BaseCSTransformer or str
        The fitted outlier transformer.

    scoring_transformer_ : BaseCSTransformer or str
        The fitted scoring transformer.

    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    Examples
    --------
    >>> from skfolio.factor_model.factor_exposure import DerivedFactor, FixedWeightedFactor
    >>> from skfolio.factor_model.descriptor import LogMarketCap
    >>> from skfolio.prior import CharacteristicsFactorModel
    >>>
    >>> # Non-linear size factor
    >>> factors = [
    ...     ("size", FixedWeightedFactor(descriptors=[("log_mcap", LogMarketCap())])),
    ...     ("non_linear_size", DerivedFactor(source="size", func=lambda x: x**3)),
    ... ]
    >>>
    >>> # Orthogonalize non_linear_size vs size
    >>> model = CharacteristicsFactorModel(
    ...     factors=factors,
    ...     neutralize_against={"non_linear_size": ["size"]},
    ... )
    """

    outlier_transformer_: BaseCSTransformer | str
    scoring_transformer_: BaseCSTransformer | str

    def __init__(
        self,
        *,
        source: str,
        func: Callable[[FloatArray], FloatArray],
        family: str = "style",
        outlier_transformer: skt.CSTransformer = "passthrough",
        scoring_transformer: skt.CSTransformer = None,
        transform_by_group: str | None = None,
    ):
        super().__init__(family=family)
        self.source = source
        self.func = func
        self.outlier_transformer = outlier_transformer
        self.scoring_transformer = scoring_transformer
        self.transform_by_group = transform_by_group

    def fit_transform(
        self,
        X: AssetPanel,
        y=None,
        source_exposure: FloatArray | None = None,
        **fit_params,
    ) -> FloatArray:
        """Fit and transform the source exposure.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing benchmark weights and optional grouping.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        source_exposure : ndarray of shape (n_observations, n_assets)
            The computed exposure from the source factor. This is passed automatically
            by `CharacteristicsFactorModel`.

        **fit_params : dict
            Additional fit parameters (unused).

        Returns
        -------
        exposure : ndarray of shape (n_observations, n_assets)
            The derived factor exposure.
        """
        required_fields = [_BENCHMARK_WEIGHTS]
        if self.transform_by_group is not None:
            required_fields.append(self.transform_by_group)

        validate_asset_panel(self, X, required_fields=required_fields)

        if source_exposure is None:
            raise ValueError(
                f"DerivedFactor '{self.source}' requires 'source_exposure' to be passed. "
                "This should be handled automatically by CharacteristicsFactorModel."
            )

        source_exposure = np.asarray(source_exposure)
        expected_shape = (X.n_observations, X.n_assets)
        if source_exposure.shape != expected_shape:
            raise ValueError(
                "`source_exposure` must be a 2D array with shape "
                f"(n_observations, n_assets)={expected_shape}; "
                f"got {source_exposure.shape}."
            )

        self.outlier_transformer_ = check_estimator(
            self.outlier_transformer,
            default=CSWinsorizer(),
            check_type=BaseCSTransformer,
        )

        self.scoring_transformer_ = check_estimator(
            self.scoring_transformer,
            default=CSStandardScaler(),
            check_type=BaseCSTransformer,
        )

        # Apply the transformation function
        derived_exposure = np.asarray(self.func(source_exposure))
        if derived_exposure.shape != source_exposure.shape:
            raise ValueError(
                "`func` must return an array with the same shape as `source_exposure`; "
                f"got {derived_exposure.shape}, expected {source_exposure.shape}."
            )

        # Get cross-section weights and groups
        cs_weight = X[_BENCHMARK_WEIGHTS]
        cs_group = (
            X[self.transform_by_group] if self.transform_by_group is not None else None
        )

        # Apply outlier transformer
        if self.outlier_transformer_ != _PASSTHROUGH:
            derived_exposure = self.outlier_transformer_.fit_transform(
                derived_exposure, cs_weights=cs_weight, cs_groups=cs_group
            )

        # Apply scoring transformer
        if self.scoring_transformer_ != _PASSTHROUGH:
            derived_exposure = self.scoring_transformer_.fit_transform(
                derived_exposure, cs_weights=cs_weight, cs_groups=cs_group
            )

        return derived_exposure
