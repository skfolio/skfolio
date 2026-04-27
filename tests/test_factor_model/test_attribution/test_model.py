"""Tests for skfolio.factor_model.attribution module."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from skfolio.factor_model.attribution import (
    AssetBreakdown,
    AssetByFactorContribution,
    Attribution,
    predicted_factor_attribution,
    realized_factor_attribution,
    rolling_realized_factor_attribution,
)

from ._utils import _create_realized_model


class TestAttributionProperties:
    """Tests for Attribution dataclass properties."""

    def test_n_factors_property(self, simple_factor_model):
        """Test n_factors property returns correct count."""
        result = predicted_factor_attribution(**simple_factor_model)
        assert result.n_factors == 2

    def test_n_families_property_none(self, simple_factor_model):
        """Test n_families returns 0 when families not provided."""
        result = predicted_factor_attribution(**simple_factor_model)
        assert result.n_families == 0

    def test_n_families_property_with_families(self, multi_factor_model):
        """Test n_families returns correct count when families provided."""
        result = predicted_factor_attribution(**multi_factor_model)
        assert result.n_families == 2  # Style, Industry

    def test_n_assets_property(self, simple_factor_model):
        """Test n_assets property returns correct count."""
        result = predicted_factor_attribution(**simple_factor_model)
        assert result.n_assets == 3

    def test_n_assets_zero_when_not_computed(self, simple_factor_model):
        """Test n_assets returns 0 when assets not computed."""
        result = predicted_factor_attribution(
            **simple_factor_model, compute_asset_breakdowns=False
        )
        assert result.n_assets == 0

    def test_is_rolling_false_for_single_point(self, simple_factor_model):
        """Test is_rolling is False for single-point attribution."""
        result = predicted_factor_attribution(**simple_factor_model)
        assert result.is_rolling is False
        assert result.observations is None

    def test_is_rolling_true_for_rolling(self, rolling_static_model):
        """Test is_rolling is True for rolling attribution."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=30
        )
        assert result.is_rolling is True
        assert result.observations is not None

    def test_is_realized_false_for_predicted(self, simple_factor_model):
        """Test is_realized is False for predicted attribution."""
        result = predicted_factor_attribution(**simple_factor_model)
        assert result.is_realized is False

    def test_is_realized_true_for_realized(self, static_realized_model):
        """Test is_realized is True for realized attribution."""
        result = realized_factor_attribution(**static_realized_model)
        assert result.is_realized is True


class TestComponentMuProperty:
    """Tests for Component.mu property alias."""

    def test_mu_is_alias_for_mu_contrib(self, simple_factor_model_with_perf):
        """Test that Component.mu is an alias for mu_contrib."""
        result = predicted_factor_attribution(**simple_factor_model_with_perf)

        assert result.systematic.mu == result.systematic.mu_contrib
        assert result.idio.mu == result.idio.mu_contrib
        assert result.total.mu == result.total.mu_contrib

    def test_mu_alias_for_realized(self, static_realized_model):
        """Test mu alias works for realized attribution."""
        result = realized_factor_attribution(**static_realized_model)

        assert result.systematic.mu == result.systematic.mu_contrib
        assert result.total.mu == result.total.mu_contrib


class TestAttributionPlotMethods:
    """Tests for Attribution plot methods."""

    # === Vol Bar Chart Tests ===

    def test_plot_vol_bar_returns_figure(self, simple_factor_model):
        """Test that plot_vol_bar returns a go.Figure."""
        result = predicted_factor_attribution(**simple_factor_model)
        assert isinstance(result.plot_vol_contrib(), go.Figure)

    def test_vol_bar_contains_factors(self, simple_factor_model):
        """Test that vol bar chart contains all factors."""
        result = predicted_factor_attribution(**simple_factor_model)
        bar_data = result.plot_vol_contrib().data[0]
        assert "Momentum" in bar_data.x
        assert "Value" in bar_data.x

    @pytest.mark.parametrize("include_idio,expected", [(True, True), (False, False)])
    def test_vol_bar_residual_inclusion(
        self, simple_factor_model, include_idio, expected
    ):
        """Test that vol bar chart includes/excludes idiosyncratic correctly."""
        result = predicted_factor_attribution(**simple_factor_model)
        bar_data = result.plot_vol_contrib(include_idio=include_idio).data[0]
        assert ("Idiosyncratic" in bar_data.x) == expected

    def test_vol_bar_title(self, simple_factor_model):
        """Test that vol bar has correct title."""
        result = predicted_factor_attribution(**simple_factor_model)
        assert "Vol Contribution" in result.plot_vol_contrib().layout.title.text

    def test_vol_bar_other_aggregation(self):
        """Test that plot_vol_contrib aggregates remaining factors into 'Other'."""
        # Create a model with 5 factors
        n_assets = 6
        n_factors = 5
        weights = np.ones(n_assets) / n_assets
        loading_matrix = np.eye(n_assets, n_factors)
        factor_covariance = np.diag([0.04, 0.03, 0.02, 0.01, 0.005])
        idio_covariance = np.ones(n_assets) * 0.001
        asset_names = [f"Asset{i}" for i in range(n_assets)]
        factor_names = ["F1", "F2", "F3", "F4", "F5"]

        result = predicted_factor_attribution(
            weights=weights,
            loading_matrix=loading_matrix,
            factor_covariance=factor_covariance,
            idio_covariance=idio_covariance,
            asset_names=asset_names,
            factor_names=factor_names,
        )

        # Request top_n=3, so F4 and F5 should be aggregated into "Other"
        fig = result.plot_vol_contrib(top_n=3, include_idio=True)
        bar_data = fig.data[0]

        # Should have: 3 top factors + "Other" + "Idiosyncratic" = 5 bars
        assert len(bar_data.x) == 5
        assert "Other" in bar_data.x
        assert "Idiosyncratic" in bar_data.x

        # Verify "Other" aggregates the correct variance contribution
        # The sum of pct_total_variance for all bars should equal 1.0
        total_vol_contrib = sum(bar_data.y)
        np.testing.assert_allclose(total_vol_contrib, 0.881476, rtol=1e-5)

        # "Other" should be the sum of F4 and F5 contributions
        other_idx = list(bar_data.x).index("Other")
        other_vol_contrib = bar_data.y[other_idx]

        np.testing.assert_allclose(other_vol_contrib, 0.119118, rtol=1e-5)

    # === Return Bar Chart Tests ===

    def test_plot_return_bar_returns_figure(self, simple_factor_model_with_perf):
        """Test that plot_return_bar returns a go.Figure."""
        result = predicted_factor_attribution(**simple_factor_model_with_perf)
        assert isinstance(result.plot_return_contrib(), go.Figure)

    def test_return_bar_contains_factors(self, simple_factor_model_with_perf):
        """Test that return bar chart contains all factors."""
        result = predicted_factor_attribution(**simple_factor_model_with_perf)
        bar_data = result.plot_return_contrib().data[0]
        assert "Momentum" in bar_data.x
        assert "Value" in bar_data.x

    @pytest.mark.parametrize("include_idio,expected", [(True, True), (False, False)])
    def test_return_bar_residual_inclusion(
        self, simple_factor_model_with_perf, include_idio, expected
    ):
        """Test that return bar chart includes/excludes idiosyncratic correctly."""
        result = predicted_factor_attribution(**simple_factor_model_with_perf)
        bar_data = result.plot_return_contrib(include_idio=include_idio).data[0]
        assert ("Idiosyncratic" in bar_data.x) == expected

    def test_return_bar_title(self, simple_factor_model_with_perf):
        """Test that return bar has correct title."""
        result = predicted_factor_attribution(**simple_factor_model_with_perf)
        assert "Return Contribution" in result.plot_return_contrib().layout.title.text

    def test_return_bar_other_aggregation(self):
        """Test that plot_return_contrib aggregates remaining factors into 'Other'."""
        # Create a model with 5 factors
        n_assets = 6
        n_factors = 5
        weights = np.ones(n_assets) / n_assets
        loading_matrix = np.eye(n_assets, n_factors)
        factor_covariance = np.diag([0.04, 0.03, 0.02, 0.01, 0.005])
        idio_covariance = np.ones(n_assets) * 0.001
        asset_names = [f"Asset{i}" for i in range(n_assets)]
        factor_names = ["F1", "F2", "F3", "F4", "F5"]
        factor_mu = np.array([0.08, 0.06, 0.04, 0.02, 0.01])
        idio_mu = np.ones(n_assets) * 0.005

        result = predicted_factor_attribution(
            weights=weights,
            loading_matrix=loading_matrix,
            factor_covariance=factor_covariance,
            idio_covariance=idio_covariance,
            asset_names=asset_names,
            factor_names=factor_names,
            factor_mu=factor_mu,
            idio_mu=idio_mu,
        )

        # Request top_n=3, so F4 and F5 should be aggregated into "Other"
        fig = result.plot_return_contrib(top_n=3, include_idio=True)
        bar_data = fig.data[0]

        # Should have: 3 top factors + "Other" + "Idiosyncratic" = 5 bars
        assert len(bar_data.x) == 5
        assert "Other" in bar_data.x
        assert "Idiosyncratic" in bar_data.x

        # "Other" should be the sum of contributions from remaining factors
        other_idx = list(bar_data.x).index("Other")
        other_mu = bar_data.y[other_idx]

        # Get the expected "Other" contribution from the factors
        factor_mu_contrib = result.factors.mu_contrib
        sorted_idx = np.argsort(-np.abs(factor_mu_contrib))
        expected_other = np.sum(factor_mu_contrib[sorted_idx[3:]])
        np.testing.assert_allclose(other_mu, expected_other, rtol=1e-10)

    # === Scatter Plot Tests ===

    def test_plot_return_vs_vol_scatter_returns_figure(
        self, simple_factor_model_with_perf
    ):
        """Test that plot_return_vs_vol_contrib returns a go.Figure."""
        result = predicted_factor_attribution(**simple_factor_model_with_perf)
        assert isinstance(result.plot_return_vs_vol_contrib(), go.Figure)

    def test_scatter_contains_factors(self, simple_factor_model_with_perf):
        """Test that scatter plot contains all factors (in legend)."""
        result = predicted_factor_attribution(**simple_factor_model_with_perf)
        trace_names = [t.name for t in result.plot_return_vs_vol_contrib().data]
        assert "Momentum" in trace_names
        assert "Value" in trace_names

    @pytest.mark.parametrize("include_idio,expected", [(True, True), (False, False)])
    def test_scatter_residual_inclusion(
        self, simple_factor_model_with_perf, include_idio, expected
    ):
        """Test that scatter plot includes/excludes idiosyncratic correctly."""
        result = predicted_factor_attribution(**simple_factor_model_with_perf)
        trace_names = [
            t.name
            for t in result.plot_return_vs_vol_contrib(include_idio=include_idio).data
        ]
        assert ("Idiosyncratic" in trace_names) == expected

    def test_scatter_has_reference_lines(self, simple_factor_model_with_perf):
        """Test that scatter plot has x=0 and y=0 reference lines."""
        result = predicted_factor_attribution(**simple_factor_model_with_perf)
        shapes = result.plot_return_vs_vol_contrib().layout.shapes
        assert len(shapes) >= 2

    def test_scatter_title(self, simple_factor_model_with_perf):
        """Test that scatter plot has correct title."""
        result = predicted_factor_attribution(**simple_factor_model_with_perf)
        assert "Return vs Vol" in result.plot_return_vs_vol_contrib().layout.title.text

    def test_scatter_other_aggregation(self):
        """Test that plot_return_vs_vol_contrib aggregates remaining factors."""
        # Create a model with 5 factors
        n_assets = 6
        n_factors = 5
        weights = np.ones(n_assets) / n_assets
        loading_matrix = np.eye(n_assets, n_factors)
        factor_covariance = np.diag([0.04, 0.03, 0.02, 0.01, 0.005])
        idio_covariance = np.ones(n_assets) * 0.001
        asset_names = [f"Asset{i}" for i in range(n_assets)]
        factor_names = ["F1", "F2", "F3", "F4", "F5"]
        factor_mu = np.array([0.08, 0.06, 0.04, 0.02, 0.01])
        idio_mu = np.ones(n_assets) * 0.005

        result = predicted_factor_attribution(
            weights=weights,
            loading_matrix=loading_matrix,
            factor_covariance=factor_covariance,
            idio_covariance=idio_covariance,
            asset_names=asset_names,
            factor_names=factor_names,
            factor_mu=factor_mu,
            idio_mu=idio_mu,
        )

        # Request top_n=3, so remaining factors should be aggregated into "Other"
        fig = result.plot_return_vs_vol_contrib(top_n=3, include_idio=True)

        # Should have: 3 top factors + "Other" + "Idiosyncratic" = 5 traces
        assert len(fig.data) == 5
        trace_names = [t.name for t in fig.data]
        assert "Other" in trace_names
        assert "Idiosyncratic" in trace_names

        # "Other" should be the sum of contributions from remaining factors
        other_trace = next(t for t in fig.data if t.name == "Other")
        other_vol = other_trace.x[0]
        other_mu = other_trace.y[0]

        # Get the expected "Other" contributions from the factors
        factor_vol_contrib = result.factors.vol_contrib
        factor_mu_contrib = result.factors.mu_contrib
        sorted_idx = np.argsort(-np.abs(factor_vol_contrib))
        expected_other_vol = np.sum(factor_vol_contrib[sorted_idx[3:]])
        expected_other_mu = np.sum(factor_mu_contrib[sorted_idx[3:]])
        np.testing.assert_allclose(other_vol, expected_other_vol, rtol=1e-10)
        np.testing.assert_allclose(other_mu, expected_other_mu, rtol=1e-10)

    def test_scatter_marker_sizes_proportional_to_exposure(
        self, simple_factor_model_with_perf
    ):
        """Test that single-point scatter marker sizes are proportional to |exposure|."""
        result = predicted_factor_attribution(**simple_factor_model_with_perf)
        fig = result.plot_return_vs_vol_contrib()

        # Factor traces (exclude Idiosyncratic) should have varying sizes
        sizes = [t.marker.size for t in fig.data if t.name != "Idiosyncratic"]
        assert len(sizes) >= 2
        assert len(np.unique(sizes)) > 1

    def test_scatter_size_max_parameter(self, simple_factor_model_with_perf):
        """Test that size_max parameter is respected for single-point attribution."""
        result = predicted_factor_attribution(**simple_factor_model_with_perf)
        fig = result.plot_return_vs_vol_contrib(size_max=100)

        # Max marker size should not exceed size_max (idio uses 0.4 * size_max)
        sizes = [np.asarray(t.marker.size).max() for t in fig.data]
        assert max(sizes) <= 100

    # === By Family Tests ===

    def test_by_family_aggregates(self, model_with_families):
        """Test that by_family=True aggregates to family level."""
        result = predicted_factor_attribution(**model_with_families)
        bar_data = result.plot_vol_contrib(by_family=True).data[0]
        assert "Style" in bar_data.x or "Value" in bar_data.x

    def test_by_family_requires_families(self, simple_factor_model):
        """Test that by_family=True raises error without factor_families."""
        result = predicted_factor_attribution(**simple_factor_model)
        with pytest.raises(ValueError, match="requires `factor_families`"):
            result.plot_vol_contrib(by_family=True)

    def test_plot_reserved_component_name_raises(self, simple_factor_model):
        """Reserved synthetic plot labels cannot be used as component names."""
        model = {**simple_factor_model, "factor_names": ["Other", "Value"]}
        result = predicted_factor_attribution(**model)

        with pytest.raises(ValueError, match="reserved labels"):
            result.plot_vol_contrib()

    # === Top-N Tests ===

    @pytest.fixture
    def large_factor_model(self):
        """Create a model with many factors for top_n testing."""
        n_assets, n_factors = 5, 10
        np.random.seed(42)
        return {
            "weights": np.ones(n_assets) / n_assets,
            "loading_matrix": np.random.randn(n_assets, n_factors),
            "factor_covariance": np.eye(n_factors) * 0.01,
            "idio_covariance": np.ones(n_assets) * 0.005,
            "factor_names": [f"Factor_{i}" for i in range(n_factors)],
            "asset_names": [f"Asset_{i}" for i in range(n_assets)],
        }

    def test_top_n_limits_factors(self, large_factor_model):
        """Test that top_n limits factors and aggregates remaining into Other."""
        result = predicted_factor_attribution(**large_factor_model)
        bar_data = result.plot_vol_contrib(top_n=3, include_idio=False).data[0]
        # 3 top factors + "Other" for remaining = 4 bars
        assert len(bar_data.x) == 4
        assert "Other" in bar_data.x

    def test_top_n_includes_residual_extra(self, large_factor_model):
        """Test that idiosyncratic is added beyond top_n limit and Other."""
        result = predicted_factor_attribution(**large_factor_model)
        bar_data = result.plot_vol_contrib(top_n=3, include_idio=True).data[0]
        # 3 top factors + "Other" + "Idiosyncratic" = 5 bars
        assert len(bar_data.x) == 5
        assert "Other" in bar_data.x
        assert "Idiosyncratic" in bar_data.x

    def test_top_n_none_shows_all(self, large_factor_model):
        """Test that top_n=None shows all factors."""
        result = predicted_factor_attribution(**large_factor_model)
        bar_data = result.plot_vol_contrib(top_n=None, include_idio=False).data[0]
        assert len(bar_data.x) == 10

    # === Edge Cases ===

    def test_single_factor_plot(self):
        """Test plot with a single factor."""
        result = predicted_factor_attribution(
            weights=np.array([0.5, 0.5]),
            loading_matrix=np.array([[1.0], [0.8]]),
            factor_covariance=np.array([[0.04]]),
            idio_covariance=np.array([0.01, 0.01]),
            factor_names=["SingleFactor"],
            asset_names=["A1", "A2"],
        )
        risk_fig = result.plot_vol_contrib(include_idio=True)

        assert isinstance(risk_fig, go.Figure)
        bar_data = risk_fig.data[0]
        assert "SingleFactor" in bar_data.x
        assert "Idiosyncratic" in bar_data.x

    def test_zero_factor_mu(self, simple_factor_model):
        """Test that return bar is created even with zero factor_mu."""
        result = predicted_factor_attribution(**simple_factor_model)
        return_fig = result.plot_return_contrib()

        assert isinstance(return_fig, go.Figure)
        assert len(return_fig.data[0].x) > 0

    # === Rolling Attribution Plot Tests ===

    def test_plot_vol_contrib_rolling_grouped_bar(self, rolling_static_model):
        """Test that plot_vol_contrib returns grouped bar chart for rolling attribution."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=20
        )
        fig = result.plot_vol_contrib(top_n=2, include_idio=True)

        assert isinstance(fig, go.Figure)
        # Should have: 2 top factors + "Other" (for remaining factor) + "Idiosyncratic"
        assert len(fig.data) == 4

        # Each trace should be a bar
        for trace in fig.data:
            assert trace.type == "bar"
            # x should be the observations
            assert len(trace.x) == len(result.observations)

        # Check trace names
        trace_names = [trace.name for trace in fig.data]
        assert "Other" in trace_names
        assert "Idiosyncratic" in trace_names

        # Check layout
        assert "Over Time" in fig.layout.title.text
        assert fig.layout.barmode == "group"

    # === Rolling Return-vs-Vol Scatter (Animated) Tests ===

    def test_plot_return_vs_vol_rolling_returns_figure(self, rolling_static_model):
        """Test that plot_return_vs_vol_contrib returns go.Figure for rolling."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=20
        )
        fig = result.plot_return_vs_vol_contrib()
        assert isinstance(fig, go.Figure)

    def test_plot_return_vs_vol_rolling_has_animation_frames(
        self, rolling_time_varying_model
    ):
        """Test that rolling plot_return_vs_vol_contrib has animation frames."""
        result = rolling_realized_factor_attribution(
            **rolling_time_varying_model, window_size=60, step=20
        )
        fig = result.plot_return_vs_vol_contrib()

        # px.scatter with animation_frame creates frames
        assert hasattr(fig, "frames")
        assert len(fig.frames) > 0
        # Number of frames should match number of windows
        assert len(fig.frames) == len(result.observations)

    def test_plot_return_vs_vol_rolling_title(self, rolling_static_model):
        """Test that rolling scatter has correct title."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=20
        )
        fig = result.plot_return_vs_vol_contrib()
        assert "Over Time" in fig.layout.title.text

    def test_plot_return_vs_vol_rolling_has_reference_lines(self, rolling_static_model):
        """Test that rolling scatter has x=0 and y=0 reference lines."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=20
        )
        fig = result.plot_return_vs_vol_contrib()
        # Reference lines are added as shapes
        assert len(fig.layout.shapes) >= 2

    @pytest.mark.parametrize("include_idio", [True, False])
    def test_plot_return_vs_vol_rolling_idio_inclusion(
        self, rolling_time_varying_model, include_idio
    ):
        """Test that rolling scatter includes/excludes idiosyncratic correctly."""
        result = rolling_realized_factor_attribution(
            **rolling_time_varying_model, window_size=60, step=20
        )
        fig = result.plot_return_vs_vol_contrib(include_idio=include_idio)

        # Check trace names in the legend
        trace_names = [trace.name for trace in fig.data]
        assert ("Idiosyncratic" in trace_names) == include_idio

    def test_plot_return_vs_vol_rolling_other_aggregation(self, rolling_static_model):
        """Test that rolling plot_return_vs_vol_contrib aggregates factors."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=20
        )
        # 3 factors, request top_n=2, so 1 factor goes to "Other"
        fig = result.plot_return_vs_vol_contrib(top_n=2, include_idio=True)

        trace_names = [trace.name for trace in fig.data]
        # Should have: 2 top factors + "Other" + "Idiosyncratic"
        assert len(trace_names) == 4
        assert "Other" in trace_names
        assert "Idiosyncratic" in trace_names

    def test_plot_return_vs_vol_rolling_marker_sizes_change_over_time(
        self, rolling_time_varying_model
    ):
        """Test that marker sizes change across frames with time-varying exposures."""
        result = rolling_realized_factor_attribution(
            **rolling_time_varying_model, window_size=60, step=20
        )
        fig = result.plot_return_vs_vol_contrib(include_idio=False)

        # Collect marker sizes from each frame for the first trace
        sizes_per_frame = []
        for frame in fig.frames:
            if frame.data:
                sizes_per_frame.append(frame.data[0].marker.size)

        # With time-varying exposures, sizes should differ between frames
        assert len(sizes_per_frame) >= 2
        # At least one frame should have different sizes from the first
        assert any(
            not np.allclose(sizes_per_frame[0], s) for s in sizes_per_frame[1:]
        ), "Marker sizes should change across frames with time-varying exposures"

    def test_plot_return_vs_vol_rolling_by_family(self, rolling_time_varying_model):
        """Test that rolling plot_return_vs_vol_contrib works with by_family."""
        result = rolling_realized_factor_attribution(
            **rolling_time_varying_model,
            factor_families=np.array(["Style", "Style", "Risk"]),
            window_size=60,
            step=20,
        )
        fig = result.plot_return_vs_vol_contrib(by_family=True, include_idio=True)

        trace_names = [trace.name for trace in fig.data]
        # Should have family names instead of factor names
        assert "Style" in trace_names or "Risk" in trace_names
        assert "Idiosyncratic" in trace_names

    def test_plot_return_vs_vol_rolling_size_max_parameter(
        self, rolling_time_varying_model
    ):
        """Test that size_max parameter is respected."""
        result = rolling_realized_factor_attribution(
            **rolling_time_varying_model, window_size=60, step=20
        )
        # Just verify it doesn't raise an error with custom size_max
        fig = result.plot_return_vs_vol_contrib(size_max=30)
        assert isinstance(fig, go.Figure)


class TestAssetAttribution:
    """Tests for AssetBreakdown and AssetByFactorContribution."""

    # === Predicted Attribution Asset Tests ===

    def test_assets_populated_by_default(self, simple_factor_model):
        """Test that assets breakdown is computed by default."""
        result = predicted_factor_attribution(**simple_factor_model)

        assert result.assets is not None
        assert isinstance(result.assets, AssetBreakdown)
        assert result.asset_by_factor_contrib is not None
        assert isinstance(result.asset_by_factor_contrib, AssetByFactorContribution)

    def test_compute_asset_breakdowns_false(self, simple_factor_model):
        """Test that compute_asset_breakdowns=False skips asset attribution."""
        result = predicted_factor_attribution(
            **simple_factor_model, compute_asset_breakdowns=False
        )

        assert result.assets is None
        assert result.asset_by_factor_contrib is None

    def test_assets_df_raises_when_not_computed(self, simple_factor_model):
        """Test that assets_df raises error when assets not computed."""
        result = predicted_factor_attribution(
            **simple_factor_model, compute_asset_breakdowns=False
        )

        with pytest.raises(ValueError, match="requires asset attribution"):
            result.assets_df()

    def test_asset_factor_df_raises_when_not_computed(self, simple_factor_model):
        """Test that asset_factor_df raises error when not computed."""
        result = predicted_factor_attribution(
            **simple_factor_model, compute_asset_breakdowns=False
        )

        with pytest.raises(ValueError, match="requires asset-by-factor contribution"):
            result.asset_factor_df()

    def test_assets_df_structure(self, simple_factor_model):
        """Test assets_df returns correct DataFrame structure."""
        result = predicted_factor_attribution(**simple_factor_model)
        assets_df = result.assets_df(formatted=False)

        assert isinstance(assets_df, pd.DataFrame)
        assert len(assets_df) == 3  # 3 assets

        # Check key columns are present (order may vary)
        expected_cols = {
            "Asset",
            "Weight",
            "Volatility Contribution",
            "Systematic Vol Contribution",
            "Idiosyncratic Vol Contribution",
            "% of Total Variance",
            "Expected Return Contribution",
            "Systematic Expected Return Contribution",
            "Idiosyncratic Expected Return Contribution",
            "% of Total Expected Return",
            "Standalone Volatility",
            "Standalone Expected Return",
            "Correlation with Portfolio",
        }
        assert expected_cols.issubset(set(assets_df.columns))

    def test_asset_factor_df_structure(self, simple_factor_model):
        """Test asset_factor_df returns correct pivot table structure."""
        result = predicted_factor_attribution(**simple_factor_model)
        afc_df = result.asset_factor_df(metric="vol_contrib", formatted=False)

        assert isinstance(afc_df, pd.DataFrame)
        assert afc_df.shape == (3, 2)  # 3 assets x 2 factors
        assert afc_df.index.name == "Asset"
        assert list(afc_df.columns) == ["Momentum", "Value"]

    @pytest.mark.parametrize("metric", ["vol_contrib", "mu_contrib"])
    def test_asset_factor_df_metrics(self, simple_factor_model, metric):
        """Test asset_factor_df works for both metrics."""
        result = predicted_factor_attribution(**simple_factor_model)
        afc_df = result.asset_factor_df(metric=metric, formatted=False)

        assert afc_df.shape == (3, 2)

    def test_asset_factor_df_invalid_metric(self, simple_factor_model):
        """Test asset_factor_df validates metric."""
        result = predicted_factor_attribution(**simple_factor_model)

        with pytest.raises(ValueError, match="`metric` must be"):
            result.asset_factor_df(metric="invalid")

    def test_asset_vol_contrib_sums_to_total(self, simple_factor_model):
        """Test that asset vol_contrib sums to total volatility."""
        result = predicted_factor_attribution(
            **simple_factor_model, annualized_factor=1
        )

        np.testing.assert_almost_equal(
            np.sum(result.assets.vol_contrib), result.total.vol
        )

    def test_asset_systematic_idio_sum(self, simple_factor_model):
        """Test that systematic + idio vol contribs equal total per asset."""
        result = predicted_factor_attribution(**simple_factor_model)

        np.testing.assert_array_almost_equal(
            result.assets.systematic_vol_contrib + result.assets.idio_vol_contrib,
            result.assets.vol_contrib,
        )

    def test_asset_factor_sum_matches_factor_contrib(self, simple_factor_model):
        """Test that asset-by-factor contributions sum to factor contributions."""
        result = predicted_factor_attribution(**simple_factor_model)

        # Sum over assets should give factor vol contribs
        afc_vol_sum = result.asset_by_factor_contrib.vol_contrib.sum(axis=0)
        np.testing.assert_array_almost_equal(afc_vol_sum, result.factors.vol_contrib)

    def test_asset_factor_sum_matches_asset_systematic(self, simple_factor_model):
        """Test that asset-by-factor contributions sum to asset systematic contrib."""
        result = predicted_factor_attribution(**simple_factor_model)

        # Sum over factors should give asset systematic vol contribs
        afc_vol_row_sum = result.asset_by_factor_contrib.vol_contrib.sum(axis=1)
        np.testing.assert_array_almost_equal(
            afc_vol_row_sum, result.assets.systematic_vol_contrib
        )

    # === Realized Attribution Asset Tests ===

    def test_realized_assets_populated(self, static_realized_model):
        """Test that assets breakdown is computed for realized attribution."""
        result = realized_factor_attribution(**static_realized_model)

        assert result.assets is not None
        assert isinstance(result.assets, AssetBreakdown)
        assert result.asset_by_factor_contrib is not None

    def test_realized_compute_asset_breakdowns_false(self, static_realized_model):
        """Test that compute_asset_breakdowns=False works for realized attribution."""
        result = realized_factor_attribution(
            **static_realized_model, compute_asset_breakdowns=False
        )

        assert result.assets is None
        assert result.asset_by_factor_contrib is None

    def test_realized_assets_df_structure(self, static_realized_model):
        """Test realized assets_df has correct structure."""
        result = realized_factor_attribution(**static_realized_model)
        assets_df = result.assets_df(formatted=False)

        assert isinstance(assets_df, pd.DataFrame)
        assert len(assets_df) == 5  # 5 assets

        # Check for realized-specific columns
        assert "Mean Return Contribution" in assets_df.columns
        assert "Systematic Mean Return Contribution" in assets_df.columns

    def test_realized_asset_vol_contrib_sums_to_total(self, static_realized_model):
        """Test realized asset vol_contrib sums to total."""
        result = realized_factor_attribution(
            **static_realized_model, annualized_factor=1
        )

        np.testing.assert_almost_equal(
            np.sum(result.assets.vol_contrib), result.total.vol, decimal=10
        )

    def test_realized_asset_weight_std_static(self, static_realized_model):
        """Test weight_std is zero for static weights."""
        result = realized_factor_attribution(**static_realized_model)

        np.testing.assert_array_almost_equal(result.assets.weight_std, np.zeros(5))

    def test_realized_asset_weight_std_time_varying(self, time_varying_realized_model):
        """Test weight_std is nonzero for time-varying weights."""
        result = realized_factor_attribution(**time_varying_realized_model)

        assert np.all(result.assets.weight_std > 0)

    # === Formatted Output Tests ===

    @pytest.mark.parametrize(
        "formatted,expected_dtype", [(True, object), (False, np.float64)]
    )
    def test_assets_df_formatting(self, simple_factor_model, formatted, expected_dtype):
        """Test that formatted parameter controls output type for assets_df."""
        result = predicted_factor_attribution(**simple_factor_model)
        assets_df = result.assets_df(formatted=formatted)

        assert assets_df["Standalone Volatility"].dtype == expected_dtype

    @pytest.mark.parametrize("formatted,expected_type", [(True, str), (False, float)])
    def test_asset_factor_df_formatting(
        self, simple_factor_model, formatted, expected_type
    ):
        """Test that formatted parameter controls output type for asset_factor_df."""
        result = predicted_factor_attribution(**simple_factor_model)
        afc_df = result.asset_factor_df(formatted=formatted)

        assert isinstance(afc_df.iloc[0, 0], expected_type)


class TestRollingAssetAttribution:
    """Tests for rolling asset attribution."""

    def test_rolling_assets_shape(self, rolling_static_model):
        """Test rolling asset breakdown has correct shape."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model,
            window_size=60,
            step=30,
            compute_asset_breakdowns=True,
        )

        n_windows = len(result.observations)
        n_assets = 5

        assert result.assets is not None
        assert result.assets.vol_contrib.shape == (n_windows, n_assets)
        assert result.assets.weight.shape == (n_windows, n_assets)

    def test_rolling_compute_asset_breakdowns_false(self, rolling_static_model):
        """Test compute_asset_breakdowns=False for rolling attribution."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model,
            window_size=60,
            step=30,
            compute_asset_breakdowns=False,
        )

        assert result.assets is None

    def test_rolling_asset_factor_contribs_default_off(self, rolling_static_model):
        """Test asset_factor_contribs is None by default for rolling."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=30
        )

        # By default, compute_asset_factor_contribs=False
        assert result.asset_by_factor_contrib is None

    def test_rolling_asset_factor_contribs_enabled(self, rolling_static_model):
        """Test asset_factor_contribs is computed when enabled."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model,
            window_size=60,
            step=30,
            compute_asset_factor_contribs=True,
        )

        n_windows = len(result.observations)
        n_assets = 5
        n_factors = 3

        assert result.asset_by_factor_contrib is not None
        assert result.asset_by_factor_contrib.vol_contrib.shape == (
            n_windows,
            n_assets,
            n_factors,
        )

    def test_rolling_asset_factor_df_requires_idx(self, rolling_static_model):
        """Test asset_factor_df requires observation_idx for rolling."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model,
            window_size=60,
            step=30,
            compute_asset_factor_contribs=True,
        )

        with pytest.raises(ValueError, match="must specify `observation_idx`"):
            result.asset_factor_df()

    def test_rolling_asset_factor_df_with_idx(self, rolling_static_model):
        """Test asset_factor_df works with observation_idx."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model,
            window_size=60,
            step=30,
            compute_asset_factor_contribs=True,
        )

        afc_df = result.asset_factor_df(observation_idx=0, formatted=False)

        assert afc_df.shape == (5, 3)  # 5 assets x 3 factors

    def test_rolling_asset_factor_df_invalid_idx(self, rolling_static_model):
        """Test asset_factor_df validates observation_idx."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model,
            window_size=60,
            step=30,
            compute_asset_factor_contribs=True,
        )

        with pytest.raises(IndexError, match=r"`observation_idx` .* is out of range"):
            result.asset_factor_df(observation_idx=len(result.observations))

    def test_rolling_assets_df_multiindex(self, rolling_static_model):
        """Test rolling assets_df returns MultiIndex DataFrame."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=30
        )
        assets_df = result.assets_df(formatted=False)

        assert isinstance(assets_df.index, pd.MultiIndex)
        assert assets_df.index.names == ["Observation", "Asset"]


class TestRollingFamiliesDF:
    """Tests for rolling families_df output."""

    def test_rolling_families_df_multiindex(self, rolling_static_model):
        """Test rolling families_df returns MultiIndex DataFrame."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model,
            factor_families=np.array(["Style", "Style", "Risk"]),
            window_size=60,
            step=30,
        )
        families_df = result.families_df(formatted=False)

        assert isinstance(families_df.index, pd.MultiIndex)
        assert families_df.index.names == ["Observation", "Family"]

    def test_rolling_families_breakdown_shape(self, rolling_static_model):
        """Test rolling families breakdown has correct shape."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model,
            factor_families=np.array(["Style", "Style", "Risk"]),
            window_size=60,
            step=30,
        )

        n_windows = len(result.observations)
        n_families = 2  # Style, Risk

        assert result.families.vol_contrib.shape == (n_windows, n_families)
        assert result.families.exposure.shape == (n_windows, n_families)


class TestObservationLabels:
    """Tests for observation label handling."""

    def test_string_observation_labels(self):
        """Test rolling attribution with string observation labels."""
        model = _create_realized_model(
            n_obs=100, n_assets=3, n_factors=2, include_observations=True
        )
        # Replace integer observations with strings
        model["observations"] = np.array([f"2024-01-{i + 1:02d}" for i in range(100)])

        result = rolling_realized_factor_attribution(**model, window_size=30, step=10)

        # Check that string labels are preserved (numpy uses unicode dtype for strings)
        assert result.observations.dtype.kind in ("U", "O")  # Unicode or object
        assert "2024-01-30" in result.observations  # First window end

    def test_datetime_observation_labels(self):
        """Test rolling attribution with datetime observation labels."""
        model = _create_realized_model(
            n_obs=100, n_assets=3, n_factors=2, include_observations=True
        )
        # Replace with pandas datetime
        model["observations"] = pd.date_range("2024-01-01", periods=100)

        result = rolling_realized_factor_attribution(**model, window_size=30, step=10)

        assert len(result.observations) > 0
        # Check the observations are preserved
        assert result.observations[0] == pd.Timestamp("2024-01-30")


class TestMixedStaticTimeVarying:
    """Tests for mixed static/time-varying input combinations."""

    def test_static_exposures_time_varying_weights(self):
        """Test with static exposures but time-varying weights."""
        model = _create_realized_model(
            n_obs=100,
            n_assets=5,
            n_factors=3,
            static_exposures=True,
            static_weights=False,
        )
        result = realized_factor_attribution(**model, annualized_factor=1)

        # Exposure std should still be zero (static exposures)
        # But the portfolio-level exposure varies due to changing weights
        assert result.factors.exposure_std is not None

        # Decomposition should still be additive
        sum_vol = np.sum(result.factors.vol_contrib) + result.idio.vol_contrib
        np.testing.assert_almost_equal(sum_vol, result.total.vol, decimal=10)

    def test_time_varying_exposures_static_weights(self):
        """Test with time-varying exposures but static weights."""
        model = _create_realized_model(
            n_obs=100,
            n_assets=5,
            n_factors=3,
            static_exposures=False,
            static_weights=True,
        )
        result = realized_factor_attribution(**model, annualized_factor=1)

        # Exposure std should be nonzero (time-varying exposures)
        assert np.any(result.factors.exposure_std > 0)

        # Asset weight_std should be zero (static weights)
        if result.assets is not None:
            np.testing.assert_array_almost_equal(result.assets.weight_std, np.zeros(5))


class TestAttributionGetItem:
    """Tests for Attribution.__getitem__ indexing method."""

    def test_getitem_returns_single_point_attribution(self, rolling_static_model):
        """Test that indexing rolling attribution returns a single-point Attribution."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=20
        )
        single = result[0]

        assert isinstance(single, Attribution)
        assert single.is_rolling is False
        assert single.observations is None

    def test_getitem_values_match_window(self, rolling_static_model):
        """Test that indexed values match the corresponding window."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=20
        )

        for i in [0, 1, len(result.observations) - 1]:
            single = result[i]
            # Component scalar values should match the i-th element
            np.testing.assert_almost_equal(single.total.vol, result.total.vol[i])
            np.testing.assert_almost_equal(
                single.systematic.vol_contrib, result.systematic.vol_contrib[i]
            )
            # Factor breakdown 1D values should match the i-th row
            np.testing.assert_array_almost_equal(
                single.factors.vol_contrib, result.factors.vol_contrib[i]
            )

    def test_getitem_preserves_factor_metadata_when_window_count_matches_factors(
        self, rolling_static_model
    ):
        """Test factor metadata is not sliced when n_windows equals n_factors."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model,
            factor_families=np.array(["Style", "Style", "Risk"]),
            window_size=60,
            step=70,
        )
        assert len(result.observations) == len(result.factors.names)

        single = result[0]

        np.testing.assert_array_equal(single.factors.names, result.factors.names)
        np.testing.assert_array_equal(single.factors.family, result.factors.family)
        np.testing.assert_array_equal(single.families.names, result.families.names)

    def test_getitem_preserves_asset_metadata_when_window_count_matches_assets(
        self, rolling_static_model
    ):
        """Test asset metadata is not sliced when n_windows equals n_assets."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model,
            window_size=60,
            step=35,
            compute_asset_factor_contribs=True,
        )
        assert len(result.observations) == len(result.assets.names)

        single = result[0]

        np.testing.assert_array_equal(single.assets.names, result.assets.names)
        np.testing.assert_array_equal(
            single.asset_by_factor_contrib.asset_names,
            result.asset_by_factor_contrib.asset_names,
        )
        np.testing.assert_array_equal(
            single.asset_by_factor_contrib.factor_names,
            result.asset_by_factor_contrib.factor_names,
        )

    def test_getitem_raises_typeerror_for_non_rolling(self, simple_factor_model):
        """Test that indexing non-rolling attribution raises TypeError."""
        result = predicted_factor_attribution(**simple_factor_model)

        with pytest.raises(TypeError, match="not rolling"):
            _ = result[0]

    def test_getitem_raises_indexerror_for_out_of_bounds(self, rolling_static_model):
        """Test that out-of-bounds index raises IndexError."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=20
        )
        n_windows = len(result.observations)

        with pytest.raises(IndexError, match="out of range"):
            _ = result[n_windows]

        with pytest.raises(IndexError, match="out of range"):
            _ = result[-1]


class TestAttributionPlotColors:
    """Stable name-keyed colors for bar and scatter plots."""

    def test_colors_independent_of_display_order(self):
        from skfolio.factor_model.attribution._model._attribution import (
            _IDIO_COMPONENT_COLOR,
            _OTHER_COMPONENT_COLOR,
            _colors_for_names,
            _make_component_color_map,
        )

        m = _make_component_color_map(["Momentum", "Value", "Quality"])
        c_mv = _colors_for_names(["Momentum", "Value"], m)
        c_vm = _colors_for_names(["Value", "Momentum"], m)
        assert c_mv[0] == c_vm[1]
        assert c_mv[1] == c_vm[0]
        assert m["Other"] == _OTHER_COMPONENT_COLOR
        assert m["Idiosyncratic"] == _IDIO_COMPONENT_COLOR
        assert _OTHER_COMPONENT_COLOR != _IDIO_COMPONENT_COLOR

    def test_extended_palette_cycles_by_model_index(self):
        """Color at index i matches index i+K (same hex, spaced by full palette)."""
        from skfolio.factor_model.attribution._model._attribution import (
            _EXTENDED_QUALITATIVE,
            _IDIO_COMPONENT_COLOR,
            _OTHER_COMPONENT_COLOR,
            _make_component_color_map,
        )

        k = len(_EXTENDED_QUALITATIVE)
        names = [f"F{i}" for i in range(k + 1)]
        m = _make_component_color_map(names)
        assert m["F0"] == m[f"F{k}"]
        assert m["F0"] != m["F1"]
        assert m["Other"] == _OTHER_COMPONENT_COLOR
        assert m["Idiosyncratic"] == _IDIO_COMPONENT_COLOR

    def test_consecutive_model_indices_use_distinct_plotly_colors(self):
        """Neighboring entries in ``data.names`` order get different hex values."""
        from skfolio.factor_model.attribution._model._attribution import (
            _make_component_color_map,
        )

        n = 250
        names = [f"x{i}" for i in range(n)]
        m = _make_component_color_map(names)
        for i in range(n - 1):
            assert m[f"x{i}"] != m[f"x{i + 1}"]

    def test_first_factor_uses_plotly_leader_color(self):
        """Leading color matches the first swatch of ``qualitative.Plotly``."""
        import plotly.express as px

        from skfolio.factor_model.attribution._model._attribution import (
            _make_component_color_map,
        )

        m = _make_component_color_map(["only"])
        assert m["only"] == px.colors.qualitative.Plotly[0]

    def test_risk_bar_and_scatter_same_hex_per_factor(
        self, simple_factor_model_with_perf
    ):
        from skfolio.factor_model.attribution._model._attribution import (
            _prepare_plot_data,
        )

        result = predicted_factor_attribution(**simple_factor_model_with_perf)
        attrs_bar = ["vol_contrib", "pct_total_variance", "exposure"]
        attrs_scatter = ["vol_contrib", "mu_contrib", "exposure"]
        n_bar, _, c_bar = _prepare_plot_data(
            result.factors,
            result.idio,
            attrs_bar,
            top_n=None,
            include_idio=True,
            is_rolling=False,
        )
        n_sc, _, c_sc = _prepare_plot_data(
            result.factors,
            result.idio,
            attrs_scatter,
            top_n=None,
            include_idio=True,
            is_rolling=False,
        )
        assert n_bar == n_sc
        assert c_bar == c_sc
