import numpy as np
import pandas as pd
import plotly.graph_objects as go

from skfolio.utils.figure import kde_trace, plot_kde_distributions


def generate_sample_data():
    # Simple increasing and decreasing data for two assets
    data = pd.DataFrame(
        {"asset1": np.linspace(0.01, 0.05, 10), "asset2": np.linspace(-0.02, 0.02, 10)}
    )
    weights = np.ones(10)
    weights[:5] = 2  # heavier weight for first half
    return data, weights


def test_kde_trace_unweighted():
    x = np.array([0.0, 0.5, 1.0])
    trace = kde_trace(
        x,
        fill_opacity=1.0,
        percentile_cutoff=None,
        line_color="#32a852",
        line_dash="solid",
        line_width=1,
        sample_weight=None,
        name="test",
        visible=True,
    )
    assert isinstance(trace, go.Scatter)
    assert trace.name == "test"
    assert trace.opacity == 1.0
    assert trace.visible is True
    # Ensure density values are all non-negative
    assert all(y >= 0 for y in trace.y)


def test_kde_trace_weighted():
    x = np.array([0.0, 0.5, 1.0])
    weights = np.array([1.0, 0.0, 1.0])
    trace = kde_trace(
        x,
        percentile_cutoff=0.1,
        fill_opacity=1.0,
        line_color="#32a852",
        line_dash="solid",
        line_width=1,
        name="weighted",
        visible=False,
        sample_weight=weights,
    )
    assert isinstance(trace, go.Scatter)
    assert trace.name == "weighted"
    assert trace.visible is False
    # The density at the middle should be influenced by weights
    density = np.array(trace.y)
    assert density.max() > density.min()


def test_plot_kde_distributions_default():
    data, weights = generate_sample_data()
    fig = plot_kde_distributions(data)
    assert isinstance(fig, go.Figure)
    # Only unweighted traces: two assets -> two traces
    assert len(fig.data) == 2
    names = [t.name for t in fig.data]
    assert "asset1" in names and "asset2" in names


def test_plot_kde_distributions_with_weights():
    data, weights = generate_sample_data()
    fig = plot_kde_distributions(data, sample_weight=weights, weighted_suffix="SW")
    assert isinstance(fig, go.Figure)
    # With weights: two assets unweighted + two weighted = 4 traces
    assert len(fig.data) == 4
    names = [t.name for t in fig.data]
    # Check weighted suffix applied
    assert "asset1 SW" in names and "asset2 SW" in names
