"""Attribution Dataclass."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from skfolio.factor_model.attribution._model._asset_by_factor_contribution import (
    AssetByFactorContribution,
)
from skfolio.factor_model.attribution._model._breakdown import (
    AssetBreakdown,
    FactorBreakdown,
    FamilyBreakdown,
)
from skfolio.factor_model.attribution._model._component import Component
from skfolio.factor_model.attribution._utils import (
    _format_contrib_with_ci_margin,
    _format_decimal,
    _format_percent,
)
from skfolio.typing import AnyArray, FloatArray

__all__ = ["Attribution"]


@dataclass(frozen=True)
class Attribution:
    r"""Factor attribution result.

    Result returned by :func:`predicted_factor_attribution`,
     :func:`realized_factor_attribution` or :func:`rolling_realized_factor_attribution`.

    Predicted and realized attribution expose the same decomposition: systematic,
    idiosyncratic, total and per-factor attribution. Realized attribution may also
    include `unexplained`, the residual component between observed portfolio returns
    and model-attributed returns. Rolling attribution uses the same fields with numeric
    values indexed by `observations`.

    Attributes
    ----------
    systematic : Component
        Systematic component. Portfolio risk and return attributed to the portfolio's
        factor exposures.

    idio : Component
        Idiosyncratic component. Portfolio risk and return not attributed to the factor
        exposures.

    unexplained : Component or None
        Residual component between observed portfolio returns and model-attributed
        returns. `None` for predicted attribution.

    total : Component
        Total portfolio risk and return after aggregating all attribution components.

    factors : FactorBreakdown
        Per-factor attribution with exposures, standalone statistics and contributions.

    families : FamilyBreakdown or None
        Family-aggregated attribution. `None` when factor families are not provided.

    assets : AssetBreakdown or None
        Per-asset attribution with systematic/idiosyncratic decomposition. `None` when
        asset attribution is not computed.

    asset_by_factor_contrib : AssetByFactorContribution or None
        Asset-by-factor contribution breakdown. `None` when not computed.

    is_realized : bool
        True for realized (ex-post), False for predicted (ex-ante).

    observations : ndarray or None
        Window end labels for rolling attribution. None for single-point.
    """

    systematic: Component
    idio: Component
    unexplained: Component | None
    total: Component

    factors: FactorBreakdown
    families: FamilyBreakdown | None = None
    assets: AssetBreakdown | None = None
    asset_by_factor_contrib: AssetByFactorContribution | None = None

    is_realized: bool = False
    observations: AnyArray | None = None

    def __getitem__(self, i: int) -> Attribution:
        if self.observations is None:
            raise TypeError(
                "Attribution is not rolling; indexing is only supported for rolling attributions."
            )
        n_observations = len(self.observations)
        if not (0 <= i < n_observations):
            raise IndexError(f"window index {i} out of range [0, {n_observations})")
        return _slice_window(self, i, n_observations)

    def __post_init__(self):
        """Validate Attribution consistency after initialization."""
        # Component fields that must be float (single) or 1D array (rolling)
        component_fields = (
            "vol",
            "vol_contrib",
            "pct_total_variance",
            "mu",
            "pct_total_mu",
            "corr_with_ptf",
        )

        # Common breakdown fields that must be 1D (single) or 2D (rolling)
        breakdown_fields = (
            "exposure",
            "vol",
            "vol_contrib",
            "pct_total_variance",
            "mu",
            "mu_contrib",
            "pct_total_mu",
            "corr_with_ptf",
        )

        # Additional fields for AssetBreakdown
        asset_breakdown_fields = (
            "weight",
            "systematic_vol_contrib",
            "systematic_mu_contrib",
            "idio_vol_contrib",
            "idio_mu_contrib",
        )

        # Collect all components to validate (including optional unexplained)
        components = [
            ("systematic", self.systematic),
            ("idio", self.idio),
            ("total", self.total),
        ]
        if self.unexplained is not None:
            components.append(("unexplained", self.unexplained))

        # Collect all breakdowns to validate
        breakdowns = []
        if self.factors is not None:
            breakdowns.append(("factors", self.factors))
        if self.families is not None:
            breakdowns.append(("families", self.families))
        if self.assets is not None:
            breakdowns.append(("assets", self.assets))

        if self.is_rolling:
            n_windows = len(self.observations)

            # Validate Component fields: must be 1D arrays of shape (n_windows,)
            for comp_name, component in components:
                for field in component_fields:
                    val = getattr(component, field)
                    if not isinstance(val, np.ndarray) or val.ndim != 1:
                        raise ValueError(
                            f"Rolling attribution requires `{comp_name}.{field}` to be "
                            f"a 1D array, got {type(val).__name__}"
                            f"{f' with ndim={val.ndim}' if isinstance(val, np.ndarray) else ''}."
                        )
                    if val.shape[0] != n_windows:
                        raise ValueError(
                            f"`{comp_name}.{field}` has shape {val.shape}, expected "
                            f"({n_windows},) to match `observations`."
                        )

                # Validate mu_uncertainty if present
                if component.mu_uncertainty is not None:
                    val = component.mu_uncertainty
                    if not isinstance(val, np.ndarray) or val.ndim != 1:
                        raise ValueError(
                            f"Rolling attribution requires `{comp_name}.mu_uncertainty` "
                            f"to be a 1D array, got {type(val).__name__}"
                            f"{f' with ndim={val.ndim}' if isinstance(val, np.ndarray) else ''}."
                        )
                    if val.shape[0] != n_windows:
                        raise ValueError(
                            f"`{comp_name}.mu_uncertainty` has shape {val.shape}, "
                            f"expected ({n_windows},) to match `observations`."
                        )

            # Validate Breakdown fields: must be 2D arrays of shape (n_windows, n_items)
            for bd_name, breakdown in breakdowns:
                n_items = len(breakdown.names)

                # Determine which fields to validate based on breakdown type
                fields_to_check = list(breakdown_fields)
                if bd_name == "assets":
                    fields_to_check.extend(asset_breakdown_fields)

                for field in fields_to_check:
                    if not hasattr(breakdown, field):
                        continue
                    val = getattr(breakdown, field)
                    if val.ndim != 2:
                        raise ValueError(
                            f"Rolling attribution requires `{bd_name}.{field}` to be "
                            f"2D (n_windows, n_items), got {val.ndim}D."
                        )
                    if val.shape != (n_windows, n_items):
                        raise ValueError(
                            f"`{bd_name}.{field}` has shape {val.shape}, expected "
                            f"({n_windows}, {n_items})."
                        )

                # Validate exposure_std if present (factors/families)
                if (
                    hasattr(breakdown, "exposure_std")
                    and breakdown.exposure_std is not None
                ):
                    if breakdown.exposure_std.shape != (n_windows, n_items):
                        raise ValueError(
                            f"`{bd_name}.exposure_std` has shape {breakdown.exposure_std.shape}, "
                            f"expected ({n_windows}, {n_items})."
                        )

                # Validate weight_std if present (assets)
                if (
                    hasattr(breakdown, "weight_std")
                    and breakdown.weight_std is not None
                ):
                    val = breakdown.weight_std
                    if isinstance(val, np.ndarray) and val.shape != (
                        n_windows,
                        n_items,
                    ):
                        raise ValueError(
                            f"`{bd_name}.weight_std` has shape {val.shape}, "
                            f"expected ({n_windows}, {n_items})."
                        )

                # Validate mu_contrib_uncertainty if present (factors/families)
                if (
                    hasattr(breakdown, "mu_contrib_uncertainty")
                    and breakdown.mu_contrib_uncertainty is not None
                ):
                    if breakdown.mu_contrib_uncertainty.shape != (
                        n_windows,
                        n_items,
                    ):
                        raise ValueError(
                            f"`{bd_name}.mu_contrib_uncertainty` has shape "
                            f"{breakdown.mu_contrib_uncertainty.shape}, "
                            f"expected ({n_windows}, {n_items})."
                        )

            # Validate asset_factor_contribs: must be 3D (n_windows, n_assets, n_factors)
            if self.asset_by_factor_contrib is not None:
                afc = self.asset_by_factor_contrib
                n_assets = len(afc.asset_names)
                n_factors = len(afc.factor_names)
                expected_shape = (n_windows, n_assets, n_factors)

                for field in ("vol_contrib", "mu_contrib"):
                    val = getattr(afc, field)
                    if val.ndim != 3:
                        raise ValueError(
                            f"Rolling attribution requires `asset_factor_contribs.{field}` "
                            f"to be 3D (n_windows, n_assets, n_factors), got {val.ndim}D."
                        )
                    if val.shape != expected_shape:
                        raise ValueError(
                            f"`asset_factor_contribs.{field}` has shape {val.shape}, "
                            f"expected {expected_shape}."
                        )

                # Validate name consistency
                if self.assets is not None:
                    if not np.array_equal(afc.asset_names, self.assets.names):
                        raise ValueError(
                            "`asset_factor_contribs.asset_names` does not match `assets.names`."
                        )
                if self.factors is not None:
                    if not np.array_equal(afc.factor_names, self.factors.names):
                        raise ValueError(
                            "`asset_factor_contribs.factor_names` does not match `factors.names`."
                        )
        else:
            # Single-point: Component fields must be scalars (float)
            for comp_name, component in components:
                for field in component_fields:
                    val = getattr(component, field)
                    if isinstance(val, np.ndarray):
                        raise ValueError(
                            f"Single-point attribution requires `{comp_name}.{field}` "
                            f"to be a scalar, got array with shape {val.shape}."
                        )

            # Single-point: Breakdown fields must be 1D arrays of shape (n_items,)
            for bd_name, breakdown in breakdowns:
                n_items = len(breakdown.names)

                # Determine which fields to validate based on breakdown type
                fields_to_check = list(breakdown_fields)
                if bd_name == "assets":
                    fields_to_check.extend(asset_breakdown_fields)

                for field in fields_to_check:
                    if not hasattr(breakdown, field):
                        continue
                    val = getattr(breakdown, field)
                    if val.ndim != 1:
                        raise ValueError(
                            f"Single-point attribution requires `{bd_name}.{field}` "
                            f"to be 1D (n_items,), got {val.ndim}D."
                        )
                    if val.shape[0] != n_items:
                        raise ValueError(
                            f"`{bd_name}.{field}` has shape {val.shape}, expected "
                            f"({n_items},) to match `{bd_name}.names`."
                        )

                # Validate exposure_std if present (factors/families)
                if (
                    hasattr(breakdown, "exposure_std")
                    and breakdown.exposure_std is not None
                ):
                    if breakdown.exposure_std.shape != (n_items,):
                        raise ValueError(
                            f"`{bd_name}.exposure_std` has shape {breakdown.exposure_std.shape}, "
                            f"expected ({n_items},)."
                        )

                # Validate weight_std if present (assets) - can be scalar 0.0 for static
                if (
                    hasattr(breakdown, "weight_std")
                    and breakdown.weight_std is not None
                ):
                    val = breakdown.weight_std
                    if isinstance(val, np.ndarray) and val.shape != (n_items,):
                        raise ValueError(
                            f"`{bd_name}.weight_std` has shape {val.shape}, "
                            f"expected ({n_items},)."
                        )

                # Validate mu_contrib_uncertainty if present (factors/families)
                if (
                    hasattr(breakdown, "mu_contrib_uncertainty")
                    and breakdown.mu_contrib_uncertainty is not None
                ):
                    if breakdown.mu_contrib_uncertainty.shape != (n_items,):
                        raise ValueError(
                            f"`{bd_name}.mu_contrib_uncertainty` has shape "
                            f"{breakdown.mu_contrib_uncertainty.shape}, "
                            f"expected ({n_items},)."
                        )

            # Validate asset_factor_contribs: must be 2D (n_assets, n_factors)
            if self.asset_by_factor_contrib is not None:
                afc = self.asset_by_factor_contrib
                n_assets = len(afc.asset_names)
                n_factors = len(afc.factor_names)
                expected_shape = (n_assets, n_factors)

                for field in ("vol_contrib", "mu_contrib"):
                    val = getattr(afc, field)
                    if val.ndim != 2:
                        raise ValueError(
                            f"Single-point attribution requires `asset_factor_contribs.{field}` "
                            f"to be 2D (n_assets, n_factors), got {val.ndim}D."
                        )
                    if val.shape != expected_shape:
                        raise ValueError(
                            f"`asset_factor_contribs.{field}` has shape {val.shape}, "
                            f"expected {expected_shape}."
                        )

                # Validate name consistency
                if self.assets is not None:
                    if not np.array_equal(afc.asset_names, self.assets.names):
                        raise ValueError(
                            "`asset_factor_contribs.asset_names` does not match `assets.names`."
                        )
                if self.factors is not None:
                    if not np.array_equal(afc.factor_names, self.factors.names):
                        raise ValueError(
                            "`asset_factor_contribs.factor_names` does not match `factors.names`."
                        )

    @property
    def is_rolling(self) -> bool:
        """Whether this is rolling attribution (observations is not None)."""
        return self.observations is not None

    @property
    def n_factors(self) -> int:
        """Number of factors."""
        if self.factors is not None:
            return len(self.factors.names)
        return 0

    @property
    def n_families(self) -> int:
        """Number of factor families."""
        if self.families is not None:
            return len(self.families.names)
        return 0

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        if self.assets is not None:
            return len(self.assets.names)
        return 0

    def summary_df(
        self, formatted: bool = True, confidence_level: float = 0.95
    ) -> pd.DataFrame:
        r"""Return component-level attribution as a DataFrame.

        The summary reports volatility contribution, percentage of total variance,
        return contribution and percentage of total return for the systematic,
        idiosyncratic, optional unexplained, and total components.

        Parameters
        ----------
        formatted : bool, default=True
            Format volatility, return, and percentage-of-total columns as percentage
            strings.

        confidence_level : float, default=0.95
            When `formatted=True` and uncertainty data are present, the mean return
            contribution is shown in one `{label} Contribution (N% CI)` column with
            values :math:`\mu \pm z \times SE`.

        Returns
        -------
        DataFrame
            Rows: Systematic, Idiosyncratic, [Unexplained], Total. Rolling attribution
            returns a MultiIndex with levels `Observation` and `Component`.
        """
        mu_label = "Mean Return" if self.is_realized else "Expected Return"
        mu_contrib_col = f"{mu_label} Contribution"
        merged_mu_ci_col = f"{mu_contrib_col} ({int(confidence_level * 100)}% CI)"

        # Build component names and data tuples (vol_contrib, pct_var, mu, pct_mu)
        component_names = ["Systematic", "Idiosyncratic"]
        component_objects = [self.systematic, self.idio]

        if self.unexplained is not None:
            component_names.append("Unexplained")
            component_objects.append(self.unexplained)

        component_names.append("Total")
        component_objects.append(self.total)

        # Extract data from component objects
        vol_contrib = [c.vol_contrib for c in component_objects]
        pct_var = [c.pct_total_variance for c in component_objects]
        mu_contrib = [c.mu for c in component_objects]
        pct_mu = [c.pct_total_mu for c in component_objects]

        has_uncertainty = any(c.mu_uncertainty is not None for c in component_objects)

        if self.is_rolling:
            # Rolling: build MultiIndex DataFrame
            n_components = len(component_names)
            n_windows = len(self.observations)
            obs_repeated = np.repeat(self.observations, n_components)
            components_tiled = np.tile(component_names, n_windows)
            multi_index = pd.MultiIndex.from_arrays(
                [obs_repeated, components_tiled], names=["Observation", "Component"]
            )

            # Stack arrays: each element is shape (n_windows,), stack then ravel
            data = {
                "Volatility Contribution": np.column_stack(vol_contrib).ravel(),
                "% of Total Variance": np.column_stack(pct_var).ravel(),
                mu_contrib_col: np.column_stack(mu_contrib).ravel(),
                f"% of Total {mu_label}": np.column_stack(pct_mu).ravel(),
            }

            if has_uncertainty:
                uncertainty = [
                    c.mu_uncertainty
                    if c.mu_uncertainty is not None
                    else np.full(n_windows, np.nan)
                    for c in component_objects
                ]
                data["Mean Return Uncertainty"] = np.column_stack(uncertainty).ravel()

            df = pd.DataFrame(data, index=multi_index)
        else:
            # Single-point: simple DataFrame
            data = {
                "Component": component_names,
                "Volatility Contribution": vol_contrib,
                "% of Total Variance": pct_var,
                mu_contrib_col: mu_contrib,
                f"% of Total {mu_label}": pct_mu,
            }

            if has_uncertainty:
                data["Mean Return Uncertainty"] = [
                    c.mu_uncertainty if c.mu_uncertainty is not None else np.nan
                    for c in component_objects
                ]

            df = pd.DataFrame(data)

        if formatted and has_uncertainty:
            from scipy.stats import norm as sp_norm

            z = float(sp_norm.ppf((1 + confidence_level) / 2))
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

        if formatted:
            pct_cols = [
                "Volatility Contribution",
                "% of Total Variance",
                mu_contrib_col,
                f"% of Total {mu_label}",
            ]
            for col in pct_cols:
                if col in df.columns:
                    df[col] = df[col].map(_format_percent)

        pct_mu_col = f"% of Total {mu_label}"
        pref: list[str] = []
        if "Component" in df.columns:
            pref.append("Component")
        pref.extend(
            [
                "Volatility Contribution",
                "% of Total Variance",
            ]
        )
        if merged_mu_ci_col in df.columns:
            pref.append(merged_mu_ci_col)
        else:
            pref.append(mu_contrib_col)
        pref.append(pct_mu_col)
        if "Mean Return Uncertainty" in df.columns:
            pref.append("Mean Return Uncertainty")
        rest = [c for c in df.columns if c not in pref]
        df = df[pref + rest]

        return df

    def factors_df(
        self, formatted: bool = True, confidence_level: float = 0.95
    ) -> pd.DataFrame:
        r"""Return per-factor attribution as a DataFrame.

        Parameters
        ----------
        formatted : bool, default=True
            Format volatility, return, and percentage-of-total columns as percentage
            strings.

        confidence_level : float, default=0.95
            When `formatted=True` and uncertainty data are present, merges mean return
            contribution with :math:`\mu \pm z \times SE` into one `(N% CI)` column.

        Returns
        -------
        DataFrame
            Per-factor attribution with exposures, standalone statistics, and
            contributions. For rolling: MultiIndex (Observation, Factor).
        """
        if self.factors is None:
            raise ValueError(
                "`factors_df()` requires factor attribution to be computed."
            )
        return self.factors._to_df(
            is_realized=self.is_realized,
            formatted=formatted,
            observations=self.observations,
            confidence_level=confidence_level,
        )

    def families_df(
        self, formatted: bool = True, confidence_level: float = 0.95
    ) -> pd.DataFrame:
        r"""Return family-aggregated attribution as a DataFrame.

        Parameters
        ----------
        formatted : bool, default=True
            Format volatility, return, and percentage-of-total columns as percentage
            strings.

        confidence_level : float, default=0.95
            When `formatted=True` and uncertainty data are present, merges mean return
            contribution with :math:`\mu \pm z \times SE` into one `(N% CI)` column.

        Returns
        -------
        DataFrame
            Family-aggregated exposures and contributions. For rolling: MultiIndex
            (Observation, Family).

        Raises
        ------
        ValueError
            If `factor_families` was not provided.
        """
        if self.families is None:
            raise ValueError(
                "`families_df()` requires `factor_families` to have been provided "
                "during attribution."
            )
        return self.families._to_df(
            is_realized=self.is_realized,
            formatted=formatted,
            observations=self.observations,
            confidence_level=confidence_level,
        )

    def assets_df(self, formatted: bool = True) -> pd.DataFrame:
        """Return per-asset attribution as a DataFrame.

        Parameters
        ----------
        formatted : bool, default=True
            Format volatility, return, and percentage-of-total columns as percentage
            strings.

        Returns
        -------
        DataFrame
            Per-asset weights, volatility and return contributions (total/systematic/idiosyncratic).
            For rolling: MultiIndex (Observation, Asset).

        Raises
        ------
        ValueError
            If asset attribution was not computed.
        """
        if self.assets is None:
            raise ValueError(
                "`assets_df()` requires asset attribution to be computed. "
                "Set `compute_asset_breakdowns=True`."
            )
        return self.assets._to_df(
            is_realized=self.is_realized,
            formatted=formatted,
            observations=self.observations,
        )

    def asset_factor_df(
        self,
        metric: str = "vol_contrib",
        formatted: bool = True,
        observation_idx: int | None = None,
    ) -> pd.DataFrame:
        """Return the asset-by-factor contribution as a DataFrame.

        Parameters
        ----------
        metric : {"vol_contrib", "mu_contrib"}, default="vol_contrib"
            Contribution metric to display.

        formatted : bool, default=True
            Format values as percentages.

        observation_idx : int or None
            Observation index. Required for rolling attribution.

        Returns
        -------
        DataFrame
            Matrix with assets as rows and factors as columns.
        """
        if self.asset_by_factor_contrib is None:
            raise ValueError(
                "`asset_factor_df()` requires asset-by-factor contribution "
                "to be computed."
            )
        return self.asset_by_factor_contrib._to_df(
            metric=metric, formatted=formatted, observation_idx=observation_idx
        )

    def plot_vol_contrib(
        self,
        by_family: bool = False,
        top_n: int | None = 25,
        include_idio: bool = True,
    ) -> go.Figure:
        """Plot volatility contribution by factor or family.

        Single-point attribution returns one bar trace. Rolling attribution
        returns one trace per displayed component with observations on the x-axis.

        Parameters
        ----------
        by_family : bool, default=False
            Aggregate factors by family.

        top_n : int or None, default=25
            Maximum number of factors or families to show, sorted by absolute
            volatility contribution.
            If there are more factors/families than `top_n`, the remaining
            components are aggregated into `Other`.

        include_idio : bool, default=True
            Include idiosyncratic component.

        Returns
        -------
        go.Figure
            Plotly bar chart.
        """
        data = self._get_breakdown_data(by_family)
        return _plot_bar_chart(
            data=data,
            idio=self.idio,
            top_n=top_n,
            include_idio=include_idio,
            is_rolling=self.is_rolling,
            is_realized=self.is_realized,
            is_risk=True,
            observations=self.observations,
        )

    def plot_return_contrib(
        self,
        by_family: bool = False,
        top_n: int | None = 25,
        include_idio: bool = True,
        confidence_level: float | None = 0.95,
    ) -> go.Figure:
        """Plot return contribution by factor or family.

        When realized attribution includes per-factor standard errors
        (`mu_contrib_uncertainty`), hover text shows the mean return SE and,
        if `confidence_level` is not `None`, the corresponding confidence
        interval. Single-point charts also display vertical error bars. Rolling
        charts include confidence intervals in hover text only.

        Parameters
        ----------
        by_family : bool, default=False
            Aggregate factors by family.

        top_n : int or None, default=25
            Maximum number of factors or families to show, sorted by absolute
            return contribution.
            If there are more factors/families than `top_n`, the remaining
            components are aggregated into `Other`.

        include_idio : bool, default=True
            Include idiosyncratic component.

        confidence_level : float or None, default=0.95
            Confidence level for interval display. If `None`, error bars are
            not drawn and hover text omits the confidence interval line. The
            mean return SE is still shown when uncertainty data exist.

        Returns
        -------
        go.Figure
            Plotly bar chart.
        """
        data = self._get_breakdown_data(by_family)
        return _plot_bar_chart(
            data=data,
            idio=self.idio,
            top_n=top_n,
            include_idio=include_idio,
            is_rolling=self.is_rolling,
            is_realized=self.is_realized,
            is_risk=False,
            observations=self.observations,
            confidence_level=confidence_level,
        )

    def plot_return_vs_vol_contrib(
        self,
        by_family: bool = False,
        top_n: int | None = 25,
        include_idio: bool = True,
        size_max: float = 50,
    ) -> go.Figure:
        """Plot return contribution against volatility contribution.

        X-axis: volatility contribution, Y-axis: return contribution.
        Factor marker sizes are proportional to absolute exposure.
        The idiosyncratic point uses a fixed-size diamond marker when included.
        Rolling attribution returns an animated scatter plot over time.

        Parameters
        ----------
        by_family : bool, default=False
            Aggregate factors by family.

        top_n : int or None, default=25
            Maximum number of factors or families to show, sorted by absolute
            volatility contribution.
            If there are more factors/families than `top_n`, the remaining
            components are aggregated into `Other`.

        include_idio : bool, default=True
            Include idiosyncratic component. Displayed with a diamond marker
            at a fixed size (no exposure-based sizing).

        size_max : float, default=50
            Maximum marker size for the largest absolute exposure. The
            idiosyncratic marker uses `0.4 * size_max`.

        Returns
        -------
        go.Figure
            Plotly scatter plot. Rolling attribution uses observation labels as
            animation frames.
        """
        data = self._get_breakdown_data(by_family)
        mu_label = "Mean Return" if self.is_realized else "Expected Return"
        attrs = ["vol_contrib", "mu_contrib", "exposure"]

        names, values, colors = _prepare_plot_data(
            data=data,
            idio=self.idio,
            attrs=attrs,
            top_n=top_n,
            include_idio=include_idio,
            is_rolling=self.is_rolling,
        )

        df = _scatter_plot_df(
            names=names,
            values=values,
            mu_label=mu_label,
            observations=self.observations if self.is_rolling else None,
        )

        scatter_kwargs: dict[str, Any] = {
            "x": "Volatility Contribution",
            "y": f"{mu_label} Contribution",
            "size": "Abs Exposure",
            "color": "Name",
            "hover_name": "Name",
            "hover_data": {
                "Volatility Contribution": ":.2%",
                f"{mu_label} Contribution": ":.2%",
                "Exposure": ":.4f",
                "Abs Exposure": False,
                "Name": False,
            },
            "size_max": size_max,
            "color_discrete_map": dict(zip(names, colors, strict=True)),
            "category_orders": {"Name": names},
        }
        if self.is_rolling:
            scatter_kwargs["animation_frame"] = "Observation"
            scatter_kwargs["animation_group"] = "Name"
            x_min = df["Volatility Contribution"].min()
            x_max = df["Volatility Contribution"].max()
            y_min = df[f"{mu_label} Contribution"].min()
            y_max = df[f"{mu_label} Contribution"].max()
            x_pad = (x_max - x_min) * 0.1 if x_max != x_min else 0.01
            y_pad = (y_max - y_min) * 0.1 if y_max != y_min else 0.01
            scatter_kwargs["range_x"] = [x_min - x_pad, x_max + x_pad]
            scatter_kwargs["range_y"] = [y_min - y_pad, y_max + y_pad]
            scatter_kwargs["hover_data"] = {
                **scatter_kwargs["hover_data"],
                "Observation": False,
            }

        fig = px.scatter(df, **scatter_kwargs)

        title = "Return vs Vol Contribution" + (" Over Time" if self.is_rolling else "")
        fig.update_layout(
            title=title,
            xaxis_title="Volatility Contribution",
            yaxis_title=f"{mu_label} Contribution",
        )

        if include_idio:
            idio_size = size_max * 0.4
            idio_hover = (
                "<b>%{hovertext}</b><br>"
                "Volatility Contribution=%{x:.2%}<br>"
                f"{mu_label} Contribution=%{{y:.2%}}<br>"
                "<extra></extra>"
            )
            traces = list(fig.data)
            for frame in fig.frames:
                traces.extend(frame.data)
            for trace in traces:
                if trace.name == "Idiosyncratic":
                    trace.marker.symbol = "diamond"
                    trace.marker.size = idio_size
                    trace.hovertemplate = idio_hover

        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
        fig.update_xaxes(tickformat=".2%")
        fig.update_yaxes(tickformat=".2%")
        return fig

    def _get_breakdown_data(self, by_family: bool) -> FactorBreakdown | FamilyBreakdown:
        """Get breakdown data, validating by_family if needed."""
        if by_family:
            if self.families is None:
                raise ValueError(
                    "`by_family=True` requires `factor_families` to have been provided."
                )
            return self.families
        return self.factors


def _concat_qualitative_palettes(*palettes: Sequence[str]) -> tuple[str, ...]:
    """Join Plotly qualitative sequences; drop consecutive duplicate hex values."""
    merged: list[str] = []
    for palette in palettes:
        for hex_color in palette:
            if not merged or merged[-1] != hex_color:
                merged.append(hex_color)
    return tuple(merged)


_STATIC_WINDOW_FIELDS = frozenset({"names", "family", "asset_names", "factor_names"})
_RESERVED_COMPONENT_NAMES = frozenset({"Other", "Idiosyncratic"})
_OTHER_COMPONENT_COLOR = "#718EA0"
_IDIO_COMPONENT_COLOR = "#5C5C5C"
_EXTENDED_QUALITATIVE = _concat_qualitative_palettes(
    px.colors.qualitative.Plotly,
    px.colors.qualitative.Light24,
    px.colors.qualitative.Pastel,
)


def _slice_window(
    obj: Any,
    i: int,
    n_observations: int,
    field_name: str | None = None,
) -> Any:
    """Return one rolling window while preserving static metadata fields."""
    if obj is None:
        return None
    if field_name == "observations":
        return None
    if field_name in _STATIC_WINDOW_FIELDS:
        return obj
    if is_dataclass(obj):
        return replace(
            obj,
            **{
                f.name: _slice_window(
                    getattr(obj, f.name),
                    i,
                    n_observations,
                    field_name=f.name,
                )
                for f in fields(obj)
            },
        )
    if isinstance(obj, np.ndarray) and obj.ndim >= 1 and obj.shape[0] == n_observations:
        return obj[i]
    return obj


def _compute_top_indices(
    sort_values: FloatArray, top_n: int | None
) -> tuple[FloatArray, FloatArray | None]:
    """Compute indices for top_n selection and "Other" aggregation.

    Parameters
    ----------
    sort_values : array
        Values to sort by (absolute values used for sorting).
    top_n : int or None
        Max items to show.

    Returns
    -------
    top_idx : ndarray
        Indices of top items.
    other_idx : ndarray or None
        Indices of remaining items
    """
    idx = np.argsort(-np.abs(sort_values))
    if top_n is not None and top_n < len(sort_values):
        return idx[:top_n], idx[top_n:]
    return idx, None


def _as_2d(a: np.ndarray) -> np.ndarray:
    """Convert 1D array to 2D with shape (1, n)."""
    return a if a.ndim == 2 else a.reshape(1, -1)


def _prepare_plot_data(
    data: FactorBreakdown | FamilyBreakdown,
    idio: Component,
    attrs: list[str],
    top_n: int | None,
    include_idio: bool,
    is_rolling: bool,
) -> tuple[list[str], dict[str, np.ndarray], list[str]]:
    """Prepare ordered component data for attribution plots.

    Selects the top components, aggregates the remainder into `Other`, and
    optionally appends the idiosyncratic component.

    Parameters
    ----------
    data : FactorBreakdown or FamilyBreakdown
        Breakdown data to plot.
    idio : Component
        Idiosyncratic component.
    attrs : list of str
        Attribute names to extract (e.g., ["vol_contrib", "mu_contrib", "exposure"]).
    top_n : int or None
        Maximum number of items to show.
    include_idio : bool
        Whether to include idiosyncratic component.
    is_rolling : bool
        Whether this is rolling attribution.

    Returns
    -------
    names : list of str
        Displayed component names.
    values : dict[str, ndarray]
        Values for each attr. Shape is (n_windows, n_items) for rolling,
        (1, n_items) for single-point.
    colors : list of str
        Colors aligned with `names`.
    """
    _validate_no_reserved_component_names(data.names.tolist())

    # Compute sorting criterion
    sort_attr = getattr(data, attrs[0])
    if is_rolling:
        sort_values = np.abs(sort_attr).mean(axis=0)
    else:
        sort_values = np.abs(sort_attr)

    top_idx, other_idx = _compute_top_indices(sort_values, top_n)

    names = list(data.names[top_idx])
    values = {attr: _as_2d(getattr(data, attr))[:, top_idx] for attr in attrs}

    if other_idx is not None:
        names.append("Other")
        for attr in attrs:
            other_sum = _as_2d(getattr(data, attr))[:, other_idx].sum(
                axis=1, keepdims=True
            )
            values[attr] = np.hstack([values[attr], other_sum])

    if include_idio:
        names.append("Idiosyncratic")
        n_rows = values[attrs[0]].shape[0]
        for attr in attrs:
            v = getattr(idio, attr, None)
            col = _to_column(v, n_rows)
            values[attr] = np.hstack([values[attr], col])

    color_map = _make_component_color_map(data.names.tolist())
    colors = _colors_for_names(names, color_map)

    return names, values, colors


def _prepare_perf_mu_uncertainty_se(
    data: FactorBreakdown | FamilyBreakdown,
    idio: Component,
    top_n: int | None,
    include_idio: bool,
    is_rolling: bool,
) -> np.ndarray | None:
    """Align mean return standard errors with plotted return components.

    The `Other` bucket has no aggregated standard error and is filled with NaN.
    The idiosyncratic column uses :attr:`Component.mu_uncertainty`.

    Returns
    -------
    ndarray of shape (n_windows, n_items) or None
        Standard errors aligned with plotted components. `None` when the
        breakdown has no per-item uncertainty.
    """
    unc = getattr(data, "mu_contrib_uncertainty", None)
    if unc is None:
        return None
    unc = np.asarray(unc, dtype=float)
    mu = data.mu_contrib
    if is_rolling:
        sort_values = np.abs(mu).mean(axis=0)
    else:
        sort_values = np.abs(mu)
    top_idx, other_idx = _compute_top_indices(sort_values, top_n)
    u = _as_2d(unc)[:, top_idx]
    if other_idx is not None:
        u = np.hstack([u, np.full((u.shape[0], 1), np.nan, dtype=float)])
    if include_idio:
        col = _to_column(getattr(idio, "mu_uncertainty", None), u.shape[0])
        u = np.hstack([u, col])
    return u


def _perf_contrib_hover_html(
    name: str,
    contrib_names: list[str],
    mu: float,
    pct_mu: float,
    exposure: float,
    se: float,
    confidence_level: float | None,
    z: float | None,
) -> str:
    r"""Build hover text for return contribution bars.

    When a confidence level and finite standard error are available, the mean
    return line matches formatted DataFrames:
    `Mean Return Contribution (N% CI):` :math:`\mu \pm z \times SE`.
    This string is passed as `hovertext`, not `hovertemplate`.
    """
    mu_hdr = _html_escape(contrib_names[0])
    pct_hdr = _html_escape(contrib_names[1])
    lines = [f"<b>{_html_escape(name)}</b>"]

    if np.isfinite(se) and confidence_level is not None and z is not None:
        ci_pct = round(confidence_level * 100)
        merged_hdr = f"{mu_hdr} ({ci_pct}% CI)"
        merged_val = _format_contrib_with_ci_margin(mu, se, z)
        lines.append(f"{merged_hdr}: {merged_val}")
    else:
        lines.append(f"{mu_hdr}: {_format_percent(mu)}")
        if np.isfinite(se):
            lines.append(f"Mean return SE: {_format_percent(se)}")

    lines.append(f"{pct_hdr}: {_format_percent(pct_mu)}")
    if np.isfinite(exposure):
        lines.append(f"Exposure: {_format_decimal(exposure, 2)}")
    return "<br>".join(lines)


def _html_escape(s: str) -> str:
    """Escape `<`, `>`, `&` for use inside Plotly HTML hover."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _mean_return_ci_error_bars(
    mu: np.ndarray,
    se: np.ndarray,
    z: float,
) -> tuple[list[float | None], list[float | None]]:
    """Asymmetric Plotly `error_y` extents from mean return and SE (return units)."""
    array: list[float | None] = []
    arrayminus: list[float | None] = []
    for k in range(mu.shape[0]):
        m = float(mu[k])
        s = float(se[k])
        if not np.isfinite(s) or s < 0:
            array.append(None)
            arrayminus.append(None)
            continue
        lower = m - z * s
        upper = m + z * s
        array.append(max(0.0, upper - m))
        arrayminus.append(max(0.0, m - lower))
    return array, arrayminus


def _to_column(v, n_rows: int) -> np.ndarray:
    """Convert scalar, 1D array, or None to a 2D column of shape (n_rows, 1)."""
    if v is None:
        return np.full((n_rows, 1), np.nan)
    v = np.asarray(v)
    if v.ndim == 0:  # scalar
        return np.full((n_rows, 1), v.item())
    return v.reshape(-1, 1)


def _hover_template(name: str, contrib_names: list[str], show_exposure: bool) -> str:
    """Plotly `hovertemplate` for bar traces using `customdata`."""
    safe = _html_escape(name)
    lines = [
        f"{contrib_name}: %{{customdata[{i}]:.2%}}"
        for i, contrib_name in enumerate(contrib_names)
    ]
    if show_exposure:
        lines.append("Exposure: %{customdata[2]:.2f}")
    body = "<br>".join(lines)
    return f"<b>{safe}</b><br>{body}<extra></extra>"


def _scatter_plot_df(
    names: list[str],
    values: dict[str, np.ndarray],
    mu_label: str,
    observations: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build DataFrame for risk-return scatter (single-point or rolling)."""
    n_items = len(names)
    vol = values["vol_contrib"]
    mu = values["mu_contrib"]
    exp = values["exposure"]

    if observations is None:
        df = pd.DataFrame(
            {
                "Name": names,
                "Volatility Contribution": vol[0],
                f"{mu_label} Contribution": mu[0],
                "Exposure": exp[0],
            }
        )
    else:
        n_windows = len(observations)
        df = pd.DataFrame(
            {
                "Observation": np.repeat(observations, n_items),
                "Name": np.tile(names, n_windows),
                "Volatility Contribution": vol.ravel(),
                f"{mu_label} Contribution": mu.ravel(),
                "Exposure": exp.ravel(),
            }
        )
        df["Observation"] = df["Observation"].astype(str)

    abs_exp = np.abs(df["Exposure"])
    min_valid = abs_exp[abs_exp > 0].min() if (abs_exp > 0).any() else 0.1
    df["Abs Exposure"] = abs_exp.fillna(min_valid * 0.1).replace(0, min_valid * 0.1)
    return df


def _make_component_color_map(base_names: Sequence[str]) -> dict[str, str]:
    """Return stable component colors from model order.

    Base components cycle through `_EXTENDED_QUALITATIVE`. `Other` and
    `Idiosyncratic` use fixed reserved colors.
    """
    seq = _EXTENDED_QUALITATIVE
    k = len(seq)
    out = {name: seq[i % k] for i, name in enumerate(base_names)}
    out["Other"] = _OTHER_COMPONENT_COLOR
    out["Idiosyncratic"] = _IDIO_COMPONENT_COLOR
    return out


def _validate_no_reserved_component_names(names: Sequence[str]) -> None:
    """Reject component names that conflict with synthetic plot labels."""
    reserved = sorted(_RESERVED_COMPONENT_NAMES.intersection(names))
    if reserved:
        raise ValueError(
            "Plot component names cannot use reserved labels: "
            f"{', '.join(map(repr, reserved))}."
        )


def _fallback_color_for_unknown_name(name: str) -> str:
    """Deterministic color for labels missing from `color_map` (rare)."""
    seq = _EXTENDED_QUALITATIVE
    idx = sum(ord(c) for c in name) % len(seq)
    return seq[idx]


def _colors_for_names(names: list[str], color_map: dict[str, str]) -> list[str]:
    """Colors aligned with `names` (display order) via `color_map` lookup."""
    out: list[str] = []
    for name in names:
        if name in color_map:
            out.append(color_map[name])
        else:
            out.append(_fallback_color_for_unknown_name(name))
    return out


def _plot_bar_chart(
    data,
    idio,
    top_n: int,
    include_idio: bool,
    is_rolling: bool,
    is_realized: bool,
    is_risk: bool,
    observations: np.ndarray | None = None,
    confidence_level: float | None = None,
):
    if is_risk:
        title = "Vol Contribution"
        title_rolling = f"{title} Over Time"
        yaxis_title = "Volatility Contribution (%)"
        contrib_names = ["Volatility Contribution", "% of Total Variance"]
        attrs = ["vol_contrib", "pct_total_variance", "exposure"]
    else:
        mu_label = "Mean Return" if is_realized else "Expected Return"
        title = "Return Contribution"
        title_rolling = f"{title} Over Time"
        yaxis_title = f"{mu_label} Contribution (%)"
        contrib_names = [f"{mu_label} Contribution", f"% of Total {mu_label}"]
        attrs = ["mu_contrib", "pct_total_mu", "exposure"]

    names, values, colors = _prepare_plot_data(
        data=data,
        idio=idio,
        attrs=attrs,
        top_n=top_n,
        include_idio=include_idio,
        is_rolling=is_rolling,
    )

    z: float | None = None
    if confidence_level is not None:
        from scipy.stats import norm as sp_norm

        z = float(sp_norm.ppf((1 + confidence_level) / 2))

    se_matrix: np.ndarray | None = None
    if not is_risk and is_realized:
        se_matrix = _prepare_perf_mu_uncertainty_se(
            data, idio, top_n, include_idio, is_rolling
        )
    has_unc_hover = se_matrix is not None and np.any(np.isfinite(se_matrix))

    fig = go.Figure()

    if is_rolling:
        if observations is None:
            raise ValueError("observations required for rolling bar chart")
        n_obs = len(observations)
        for i in range(len(names)):
            name = names[i]
            custom_data = np.column_stack([values[attr][:, i] for attr in attrs])
            show_exposure = not np.all(np.isnan(custom_data[:, 2]))

            if has_unc_hover and not is_risk:
                hover_texts = [
                    _perf_contrib_hover_html(
                        name,
                        contrib_names,
                        float(custom_data[j, 0]),
                        float(custom_data[j, 1]),
                        float(custom_data[j, 2]),
                        float(se_matrix[j, i]),
                        confidence_level,
                        z,
                    )
                    for j in range(n_obs)
                ]
                fig.add_trace(
                    go.Bar(
                        x=observations,
                        y=values[attrs[0]][:, i],
                        name=name,
                        marker_color=colors[i],
                        hovertext=hover_texts,
                        hoverinfo="text",
                    )
                )
            else:
                fig.add_trace(
                    go.Bar(
                        x=observations,
                        y=values[attrs[0]][:, i],
                        name=name,
                        marker_color=colors[i],
                        customdata=custom_data,
                        hovertemplate=_hover_template(
                            name, contrib_names, show_exposure
                        ),
                    )
                )

        fig.update_layout(
            title=title_rolling,
            xaxis_title="Observation",
            yaxis_title=yaxis_title,
            legend_title_text="Component",
            barmode="group",
            legend=dict(traceorder="normal"),
        )

    elif has_unc_hover and not is_risk:
        mu_row = values[attrs[0]][0]
        se_row = se_matrix[0]
        hover_texts = [
            _perf_contrib_hover_html(
                names[i],
                contrib_names,
                float(mu_row[i]),
                float(values["pct_total_mu"][0][i]),
                float(values["exposure"][0][i]),
                float(se_row[i]),
                confidence_level,
                z,
            )
            for i in range(len(names))
        ]
        bar_kwargs: dict[str, Any] = {
            "x": names,
            "y": mu_row,
            "marker_color": colors,
            "hovertext": hover_texts,
            "hoverinfo": "text",
        }
        if confidence_level is not None and z is not None:
            arr, arrm = _mean_return_ci_error_bars(mu_row, se_row, z)
            bar_kwargs["error_y"] = dict(
                type="data",
                symmetric=False,
                array=arr,
                arrayminus=arrm,
                color="rgba(0,0,0,0.35)",
                thickness=1.5,
                width=5,
            )
            ci_pct = round(confidence_level * 100)
            layout_title: str | dict[str, Any] = {
                "text": (
                    f"{title}<br><sup style='font-size:0.7em'>"
                    f"Error bars: {ci_pct}% CI for mean return "
                    f"contribution</sup>"
                )
            }
        else:
            layout_title = title

        fig.add_trace(go.Bar(**bar_kwargs))
        fig.update_layout(
            title=layout_title,
            xaxis_title="",
            yaxis_title=yaxis_title,
            barmode="group",
        )

    else:
        custom_data = np.column_stack([values[attr][0] for attr in attrs])
        hover_template = [
            _hover_template(
                names[i],
                contrib_names,
                show_exposure=not np.isnan(values["exposure"][0][i]),
            )
            for i in range(len(names))
        ]

        fig.add_trace(
            go.Bar(
                x=names,
                y=values[attrs[0]][0],
                marker_color=colors,
                customdata=custom_data,
                hovertemplate=hover_template,
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="",
            yaxis_title=yaxis_title,
            barmode="group",
        )

    fig.update_yaxes(tickformat=".1%")
    return fig
