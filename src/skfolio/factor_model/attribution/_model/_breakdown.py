"""Breakdown Dataclass."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats as st

from skfolio.factor_model.attribution._utils import (
    _format_contrib_with_ci_margin,
    _format_decimal,
    _format_percent,
)
from skfolio.typing import AnyArray, FloatArray, StrArray

__all__ = ["AssetBreakdown", "BaseBreakdown", "FactorBreakdown", "FamilyBreakdown"]


@dataclass(frozen=True)
class BaseBreakdown(ABC):
    r"""Base class for attribution breakdowns.

    Stores common per-item volatility and return contributions for factor, family and
    asset attribution breakdowns.

    For single-point attribution, numeric fields are 1D arrays of shape `(n_items,)`.
    For rolling attribution (from :func:`rolling_realized_factor_attribution`), numeric
    fields are 2D arrays of shape `(n_windows, n_items)`.

    Attributes
    ----------
    names : ndarray of shape (n_items,)
        Item names: factors, families, or assets.

    vol_contrib : ndarray of shape (n_items,) or (n_windows, n_items)
        Volatility contribution to total portfolio volatility.

    pct_total_variance : ndarray of shape (n_items,) or (n_windows, n_items)
        Percentage of total portfolio variance.

    mu_contrib : ndarray of shape (n_items,) or (n_windows, n_items)
        Return contribution to total portfolio return.

    pct_total_mu : ndarray of shape (n_items,) or (n_windows, n_items)
        Percentage of total portfolio return.
    """

    names: StrArray
    vol_contrib: FloatArray
    pct_total_variance: FloatArray
    mu_contrib: FloatArray
    pct_total_mu: FloatArray

    @property
    def _is_rolling(self) -> bool:
        """Whether this breakdown is from rolling attribution."""
        return self.vol_contrib.ndim == 2

    @property
    def _n_rolling_windows(self) -> int:
        """Number of rolling windows (1 for single-point attribution)."""
        if self._is_rolling:
            return self.vol_contrib.shape[0]
        return 1

    def _build_df(
        self,
        data: dict,
        is_realized: bool,
        formatted: bool = False,
        observations: AnyArray | None = None,
        confidence_level: float = 0.95,
    ) -> pd.DataFrame:
        r"""Build a pandas DataFrame from breakdown data.

        Parameters
        ----------
        data : dict
            Dictionary mapping column names to arrays.

        is_realized : bool
            Whether this is realized (ex-post) attribution.

        formatted : bool, default=False
            If True, format numeric columns for display.

        observations : ndarray or None, default=None
            Observation labels for rolling attribution.

        confidence_level : float, default=0.95
            When `formatted=True` and uncertainty data are present, labels the
            merged mean return contribution column `(N% CI)` and formats
            values as :math:`\mu \pm z \cdot SE`.

        Returns
        -------
        df : pandas.DataFrame
            Formatted DataFrame with attribution data.
        """
        mu_label = "Mean Return" if is_realized else "Expected Return"
        mu_contrib_col = f"{mu_label} Contribution"
        merged_mu_ci_col = f"{mu_contrib_col} ({int(confidence_level * 100)}% CI)"

        df = pd.DataFrame(data)

        if self._is_rolling:
            if observations is None:
                raise ValueError(
                    "observations must be provided for rolling attribution"
                )
            # For rolling window, we build a MultiIndex DataFrame
            # Drop the names because will be in Multi Index
            names_col = df.columns[0]
            df.drop(columns=df.columns[0], inplace=True)
            n_items = len(self.names)
            df.index = pd.MultiIndex.from_arrays(
                [
                    np.repeat(observations, n_items),
                    np.tile(self.names, len(observations)),
                ],
                names=["Observation", names_col],
            )
        else:
            # Single-point attribution
            # Sort by absolute percentage of total variance
            df = df.iloc[np.argsort(-np.abs(self.pct_total_variance))].reset_index(
                drop=True
            )

        has_uncertainty = "Mean Return Uncertainty" in df.columns

        if formatted and has_uncertainty:
            z = float(st.norm.ppf((1 + confidence_level) / 2))
            mu_vals = df[mu_contrib_col].to_numpy(dtype=float)
            se_vals = df["Mean Return Uncertainty"].to_numpy(dtype=float)
            df[merged_mu_ci_col] = [
                _format_contrib_with_ci_margin(m, s, z)
                for m, s in zip(mu_vals, se_vals, strict=True)
            ]
            df.drop(
                columns=[mu_contrib_col, "Mean Return Uncertainty"],
                inplace=True,
            )

        # Apply formatting if requested
        if formatted:
            pct_cols = [
                "Standalone Volatility",
                f"Standalone {mu_label}",
                "Volatility Contribution",
                "% of Total Variance",
                mu_contrib_col,
                f"% of Total {mu_label}",
                "Systematic Vol Contribution",
                "Idiosyncratic Vol Contribution",
                f"Systematic {mu_label} Contribution",
                f"Idiosyncratic {mu_label} Contribution",
            ]
            decimal_cols = [
                "Exposure Mean",
                "Exposure Std",
                "Exposure",
                "Correlation with Portfolio",
            ]

            for col in pct_cols:
                if col in df.columns:
                    df[col] = df[col].map(_format_percent)

            for col in decimal_cols:
                if col in df.columns:
                    df[col] = df[col].map(_format_decimal)

        # Reorder
        order = [
            "Asset",
            "Factor",
            "Family",
            "Exposure",
            "Exposure Mean",
            "Exposure Std",
            "Volatility Contribution",
            "Systematic Vol Contribution",
            "Idiosyncratic Vol Contribution",
            "% of Total Variance",
            merged_mu_ci_col,
            mu_contrib_col,
            "Mean Return Uncertainty",
            f"Systematic {mu_label} Contribution",
            f"Idiosyncratic {mu_label} Contribution",
            f"% of Total {mu_label}",
            "Standalone Volatility",
            f"Standalone {mu_label}",
        ]
        df = df[
            [c for c in order if c in df.columns]
            + [c for c in df.columns if c not in order]
        ]

        return df

    def _to_dict(
        self, is_realized: bool, name: str
    ) -> dict[str, FloatArray | StrArray]:
        """Convert breakdown to dictionary for DataFrame construction.

        Parameters
        ----------
        is_realized : bool
            Whether this is realized (ex-post) attribution.

        name : str
            Name for the identifier column (e.g., "Factor", "Asset", "Family").

        Returns
        -------
        data : dict
            Dictionary mapping column names to arrays.
        """
        mu_label = "Mean Return" if is_realized else "Expected Return"

        data = {
            name: np.tile(self.names, self._n_rolling_windows),
            "Volatility Contribution": self.vol_contrib.ravel(),
            "% of Total Variance": self.pct_total_variance.ravel(),
            f"{mu_label} Contribution": self.mu_contrib.ravel(),
            f"% of Total {mu_label}": self.pct_total_mu.ravel(),
        }

        return data


@dataclass(frozen=True)
class FactorBreakdown(BaseBreakdown):
    r"""Per-factor attribution breakdown.

    Contains per-factor attribution with exposures, standalone factor statistics and
    volatility/return contributions.

    For single-point attribution, arrays have shape `(n_factors,)`. For rolling
    attribution, arrays have shape `(n_windows, n_factors)`.

    Attributes
    ----------
    names : ndarray of shape (n_factors,)
        Factor names. Always 1D.

    family : ndarray of shape (n_factors,) or None
        Factor family/category labels (e.g., "Style", "Industry"). `None` if families
        were not provided.

    exposure : ndarray of shape (n_factors,) or (n_windows, n_factors)
        Portfolio exposure to each factor. For realized attribution with time-varying
        inputs, this is the mean exposure over time.

    exposure_std : ndarray or None
        Standard deviation of portfolio factor exposures over time. `None` for predicted
        attribution.

    vol_contrib : ndarray of shape (n_factors,) or (n_windows, n_factors)
        Factor volatility contribution to total portfolio volatility.

    pct_total_variance : ndarray of shape (n_factors,) or (n_windows, n_factors)
        Percentage of total portfolio variance.

    mu_contrib : ndarray of shape (n_factors,) or (n_windows, n_factors)
        Factor return contribution to total portfolio return.

    pct_total_mu : ndarray of shape (n_factors,) or (n_windows, n_factors)
        Percentage of total portfolio return.

    vol : ndarray of shape (n_factors,) or (n_windows, n_factors)
        Standalone factor volatility.

    mu : ndarray of shape (n_factors,) or (n_windows, n_factors)
        Standalone factor return: expected return for predicted attribution and mean
        return for realized attribution.

    corr_with_ptf : ndarray of shape (n_factors,) or (n_windows, n_factors)
        Correlation between each factor return and portfolio returns.

    mu_contrib_uncertainty : ndarray of shape (n_factors,) or (n_windows, n_factors) or None
        Per-factor standard error of the mean return contribution, reflecting factor
        return estimation uncertainty. `None` when uncertainty is not computed.
    """

    family: StrArray | None

    # Exposure
    exposure: FloatArray
    exposure_std: FloatArray | None  # For realized attrib with time-varying exposure

    # Standalone stats
    vol: FloatArray
    mu: FloatArray
    corr_with_ptf: FloatArray

    mu_contrib_uncertainty: FloatArray | None = None

    def _to_df(
        self,
        is_realized: bool,
        formatted: bool,
        observations: AnyArray | None = None,
        confidence_level: float = 0.95,
    ) -> pd.DataFrame:
        r"""Convert breakdown dataclass to DataFrame.

        Parameters
        ----------
        is_realized : bool
            Whether this is realized (ex-post) attribution.

        formatted : bool, default=False
            If True, format numeric columns for display: volatility/return and
            percentage columns as "XX.XX%", other numeric columns rounded to 4 decimal
            places.

        observations : ndarray or None, default=None
            Observation labels for rolling attribution. If provided, returns a
             MultiIndex DataFrame with (observation, factor/family) index.

        confidence_level : float, default=0.95
            When `formatted=True` and uncertainty data are present, labels the
            merged mean return contribution column `(N% CI)` and formats
            values as :math:`\mu \pm z \cdot SE`.

        Returns
        -------
        df : pandas.DataFrame
            Pandas DataFrame with columns for exposures, volatilities, correlations,
            contributions, and percentage shares. For single-point attribution,
            sorted by absolute variance contribution (descending). For rolling
            attribution, returns a MultiIndex DataFrame.
        """
        data = self._to_dict(is_realized=is_realized, name="Factor")
        data.update(
            {
                "Standalone Volatility": self.vol.ravel(),
                "Correlation with Portfolio": self.corr_with_ptf.ravel(),
            }
        )
        if self.family is not None:
            data["Family"] = np.tile(self.family, self._n_rolling_windows)

        if is_realized:
            data.update(
                {
                    "Exposure Mean": self.exposure.ravel(),
                    "Exposure Std": self.exposure_std.ravel(),
                    "Standalone Mean Return": self.mu.ravel(),
                }
            )
        else:
            data.update(
                {
                    "Exposure": self.exposure.ravel(),
                    "Standalone Expected Return": self.mu.ravel(),
                }
            )

        if self.mu_contrib_uncertainty is not None:
            data["Mean Return Uncertainty"] = self.mu_contrib_uncertainty.ravel()

        df = self._build_df(
            data=data,
            is_realized=is_realized,
            formatted=formatted,
            observations=observations,
            confidence_level=confidence_level,
        )

        return df


@dataclass(frozen=True)
class FamilyBreakdown(BaseBreakdown):
    r"""Family-level attribution breakdown.

    Aggregates factor attribution by factor family.

    For single-point attribution, arrays have shape `(n_families,)`. For rolling
    attribution, arrays have shape `(n_windows, n_families)`.

    Attributes
    ----------
    names : ndarray of shape (n_families,)
        Family names. Always 1D.

    exposure : ndarray of shape (n_families,) or (n_windows, n_families)
        Sum of portfolio factor exposures within each family.

    exposure_std : ndarray or None
        Standard deviation of family exposures over time. `None` for predicted
         attribution.

    vol_contrib : ndarray of shape (n_families,) or (n_windows, n_families)
        Family volatility contribution, equal to the sum of its factor volatility
        contributions.

    pct_total_variance : ndarray of shape (n_families,) or (n_windows, n_families)
        Percentage of total portfolio variance.

    mu_contrib : ndarray of shape (n_families,) or (n_windows, n_families)
        Family return contribution, equal to the sum of its factor return contributions.

    pct_total_mu : ndarray of shape (n_families,) or (n_windows, n_families)
        Percentage of total portfolio return.

    mu_contrib_uncertainty : ndarray of shape (n_families,) or (n_windows, n_families) or None
        Standard error of the family mean return contribution, accounting for
        cross-factor estimation correlations within the family. `None` when uncertainty
        is not computed.
    """

    # Exposure
    exposure: FloatArray
    exposure_std: FloatArray | None  # For realized attrib with time-varying exposure

    mu_contrib_uncertainty: FloatArray | None = None

    def _to_df(
        self,
        is_realized: bool,
        formatted: bool,
        observations: AnyArray | None = None,
        confidence_level: float = 0.95,
    ) -> pd.DataFrame:
        r"""Convert breakdown dataclass to DataFrame.

        Parameters
        ----------
        is_realized : bool
            Whether this is realized (ex-post) attribution.

        formatted : bool, default=False
            If True, format numeric columns for display: volatility/return and
            percentage columns as "XX.XX%", other numeric columns rounded to 4 decimal
            places.

        observations : ndarray or None, default=None
            Observation labels for rolling attribution. If provided, returns a
             MultiIndex DataFrame with (observation, factor/family) index.

        confidence_level : float, default=0.95
            When `formatted=True` and uncertainty data are present, labels the merged
            mean return contribution column `(N% CI)` and formats values as
            :math:`\mu \pm z \cdot SE`.

        Returns
        -------
        df : pandas.DataFrame
            Pandas DataFrame with columns for exposures, volatilities, correlations,
            contributions, and percentage shares. For single-point attribution, sorted
            by absolute variance contribution (descending). For rolling attribution,
            returns a MultiIndex DataFrame.
        """
        data = self._to_dict(is_realized=is_realized, name="Family")

        if is_realized:
            data.update(
                {
                    "Exposure Mean": self.exposure.ravel(),
                    "Exposure Std": self.exposure_std.ravel(),
                }
            )
        else:
            data.update(
                {
                    "Exposure": self.exposure,
                }
            )

        if self.mu_contrib_uncertainty is not None:
            data["Mean Return Uncertainty"] = self.mu_contrib_uncertainty.ravel()

        df = self._build_df(
            data=data,
            is_realized=is_realized,
            formatted=formatted,
            observations=observations,
            confidence_level=confidence_level,
        )

        return df


@dataclass(frozen=True)
class AssetBreakdown(BaseBreakdown):
    r"""Per-asset attribution breakdown.

    Decomposes each asset's volatility and return contribution into systematic and
    idiosyncratic components.

    For single-point attribution, arrays have shape `(n_assets,)`. For rolling
    attribution, arrays have shape `(n_windows, n_assets)`.

    Attributes
    ----------
    names : ndarray of shape (n_assets,)
        Asset names. Always 1D.

    weight : ndarray of shape (n_assets,) or (n_windows, n_assets)
        Portfolio asset weights.

    weight_std : ndarray of shape (n_assets,) or (n_windows, n_assets), or None
        Standard deviation of asset weights over time. `None` when weights are not
        time-varying.

    vol_contrib : ndarray of shape (n_assets,) or (n_windows, n_assets)
        Total asset volatility contribution. Sums to `total.vol`.

    systematic_vol_contrib : ndarray of shape (n_assets,) or (n_windows, n_assets)
        Asset volatility contribution attributed to factor exposures. Sums to
        `systematic.vol_contrib`.

    idio_vol_contrib : ndarray of shape (n_assets,) or (n_windows, n_assets)
        Asset volatility contribution not attributed to factor exposures. Sums to
        `idio.vol_contrib`.

    mu_contrib : ndarray of shape (n_assets,) or (n_windows, n_assets)
        Total asset return contribution. Sums to `total.mu`.

    systematic_mu_contrib : ndarray of shape (n_assets,) or (n_windows, n_assets)
        Asset return contribution attributed to factor exposures. Sums to`systematic.mu`.

    idio_mu_contrib : ndarray of shape (n_assets,) or (n_windows, n_assets)
        Asset return contribution not attributed to factor exposures. Sums to
        `idio.mu`.

    pct_total_variance : ndarray of shape (n_assets,) or (n_windows, n_assets)
        Percentage of total portfolio variance.

    pct_total_mu : ndarray of shape (n_assets,) or (n_windows, n_assets)
        Percentage of total portfolio return.

    vol : ndarray of shape (n_assets,) or (n_windows, n_assets)
        Standalone asset volatility: :math:`\sqrt{(B F B^\top + D)_{ii}}`.

    mu : ndarray of shape (n_assets,) or (n_windows, n_assets)
        Standalone asset return: expected return for predicted attribution and
        mean return for realized attribution.

    corr_with_ptf : ndarray of shape (n_assets,) or (n_windows, n_assets)
        Asset correlation with portfolio returns.
    """

    # Weights
    weight: FloatArray
    weight_std: FloatArray | None  # For realized attrib with time-varying weights

    # Systematic vs idio contribs
    systematic_vol_contrib: FloatArray
    systematic_mu_contrib: FloatArray

    # Idio contribs
    idio_vol_contrib: FloatArray
    idio_mu_contrib: FloatArray

    # Standalone asset stats
    vol: FloatArray
    mu: FloatArray
    corr_with_ptf: FloatArray

    def _to_df(
        self,
        is_realized: bool,
        formatted: bool = False,
        observations: AnyArray | None = None,
    ) -> pd.DataFrame:
        """Convert asset breakdown to DataFrame.

        Parameters
        ----------
        is_realized : bool
            Whether this is realized (ex-post) attribution.

        formatted : bool, default=False
            If True, format numeric columns for display.

        observations : ndarray or None, default=None
            Observation labels for rolling attribution.

        Returns
        -------
        df : pandas.DataFrame
            Pandas DataFrame with asset-level attribution data.
        """
        mu_label = "Mean Return" if is_realized else "Expected Return"

        data = self._to_dict(is_realized=is_realized, name="Asset")

        data.update(
            {
                "Weight": self.weight.ravel(),
                "Standalone Volatility": self.vol.ravel(),
                f"Standalone {mu_label}": self.mu.ravel(),
                "Correlation with Portfolio": self.corr_with_ptf.ravel(),
                "Systematic Vol Contribution": self.systematic_vol_contrib.ravel(),
                "Idiosyncratic Vol Contribution": self.idio_vol_contrib.ravel(),
                f"Systematic {mu_label} Contribution": self.systematic_mu_contrib.ravel(),
                f"Idiosyncratic {mu_label} Contribution": self.idio_mu_contrib.ravel(),
            }
        )
        df = self._build_df(
            data=data,
            is_realized=is_realized,
            formatted=formatted,
            observations=observations,
        )

        return df
