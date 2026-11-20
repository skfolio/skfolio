"""Tests for descriptors that use exponentially weighted averages."""

from itertools import pairwise

import numpy as np
import pytest

from skfolio.containers import AssetPanel, Field2D, InactivePolicy
from skfolio.descriptor import (
    DaysToCover,
    EWAmihudIlliquidity,
    EWDownsideVolatility,
    EWShareTurnover,
    EWVolatility,
)


@pytest.fixture(
    params=[
        EWShareTurnover,
        EWAmihudIlliquidity,
        DaysToCover,
        EWVolatility,
        EWDownsideVolatility,
    ]
)
def descriptor_cls(request):
    return request.param


@pytest.fixture(params=[InactivePolicy.MISSING, InactivePolicy.IGNORE])
def ew_panel(request):
    """Vary input availability independently of asset activity."""
    rng = np.random.default_rng(508)
    shape = (73, 5)
    values = {
        "returns": rng.normal(0, 0.02, shape),
        "adj_close": rng.lognormal(3, 0.4, shape),
        "adj_volume": rng.lognormal(10, 1, shape),
        "adj_shares_outstanding": rng.lognormal(16, 0.5, shape),
        "short_interest": rng.lognormal(12, 0.5, shape),
    }
    active_mask = np.ones(shape, dtype=bool)
    active_mask[:13, 1] = False
    active_mask[24:27, 2] = False
    active_mask[39:42, 3] = False
    values["adj_volume"][:4, 0] = 0
    values["adj_volume"][[18, 28, 54], 2] = 0
    values["adj_volume"][16:19, 1] = np.nan
    values["adj_volume"][:, 4] = np.nan
    values["short_interest"][10:14, 2] = np.nan
    values["short_interest"][27, 2] = np.nan
    values["returns"][31, 0] = np.nan
    values["adj_close"][32, 0] = np.nan
    values["adj_shares_outstanding"][33, 0] = np.nan
    if request.param == InactivePolicy.MISSING:
        for value in values.values():
            value[~active_mask] = np.nan
    return AssetPanel(
        fields={
            name: Field2D(value, inactive_policy=request.param)
            for name, value in values.items()
        },
        observations=np.arange(shape[0]),
        asset_names=np.array([f"asset_{asset}" for asset in range(shape[1])]),
        active_mask=active_mask,
    )


def _weighted_history(panel, descriptor_cls, half_life, min_periods):
    """Average explicit observation histories without a recursive accumulator."""
    volume = panel["adj_volume"]
    with np.errstate(divide="ignore", invalid="ignore"):
        if descriptor_cls is EWShareTurnover:
            values = volume / panel["adj_shares_outstanding"]
        elif descriptor_cls is EWAmihudIlliquidity:
            values = np.abs(panel["returns"]) / (panel["adj_close"] * volume)
        elif descriptor_cls is DaysToCover:
            values = np.where(volume > 0, volume, np.nan)
        elif descriptor_cls is EWVolatility:
            values = panel["returns"] ** 2
        else:
            values = np.minimum(panel["returns"], 0) ** 2

    result = np.full(values.shape, np.nan)
    for asset in range(panel.n_assets):
        history = []
        for t in range(panel.n_observations):
            if not panel.active_mask[t, asset]:
                history.clear()
                continue
            if np.isfinite(values[t, asset]):
                history.append(values[t, asset])
            if len(history) >= min_periods:
                ages = np.arange(len(history) - 1, -1, -1)
                weights = 2.0 ** (-ages / half_life)
                mean = np.average(history, weights=weights)
                if descriptor_cls is DaysToCover:
                    result[t, asset] = panel["short_interest"][t, asset] / mean
                elif descriptor_cls in (EWVolatility, EWDownsideVolatility):
                    result[t, asset] = np.sqrt(mean)
                else:
                    result[t, asset] = mean
    return result


@pytest.mark.parametrize("half_life", [1e-6, 0.5, 7.3, 63])
@pytest.mark.parametrize("min_periods", [1, 4])
def test_matches_weighted_history(ew_panel, descriptor_cls, half_life, min_periods):
    """Means use only each asset's observations since its last activity reset."""
    result = descriptor_cls(half_life=half_life, min_periods=min_periods).fit_transform(
        ew_panel
    )
    expected = _weighted_history(ew_panel, descriptor_cls, half_life, min_periods)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)


def test_partial_fit_matches_fit_across_resets(ew_panel, descriptor_cls):
    """Splitting calls around missing observations and resets preserves results."""
    params = dict(half_life=7.3, min_periods=4)
    expected = descriptor_cls(**params).fit_transform(ew_panel)
    estimator = descriptor_cls(**params)
    stops = [0, 1, 13, 17, 24, 26, 27, 40, 42, ew_panel.n_observations]
    result = np.concatenate(
        [
            estimator.partial_fit_transform(ew_panel[start:stop])
            for start, stop in pairwise(stops)
        ]
    )
    np.testing.assert_array_equal(result, expected)


def test_reactivation_restarts_min_periods(ew_panel, descriptor_cls):
    """Reactivated assets require new observations and discard their old estimates."""
    params = dict(half_life=1, min_periods=2)
    result = descriptor_cls(**params).fit_transform(ew_panel)
    fresh = descriptor_cls(**params).fit_transform(ew_panel[42:])
    np.testing.assert_array_equal(result[42:, 3], fresh[:, 3])
    assert np.isnan(result[42, 3])
    assert np.isfinite(result[43, 3])


@pytest.mark.parametrize("ew_panel", [InactivePolicy.IGNORE], indirect=True)
def test_inactive_values_do_not_affect_estimates(ew_panel, descriptor_cls):
    """Retained field values with active_mask=False have no effect on the estimates."""
    params = dict(half_life=7.3, min_periods=1)
    result = descriptor_cls(**params).fit_transform(ew_panel)
    masked_fields = {}
    for name in ew_panel.fields:
        values = ew_panel[name].copy()
        values[~ew_panel.active_mask] = np.nan
        masked_fields[name] = values
    masked_panel = AssetPanel(
        fields=masked_fields,
        observations=ew_panel.observations,
        asset_names=ew_panel.asset_names,
        active_mask=ew_panel.active_mask,
    )
    expected = descriptor_cls(**params).fit_transform(masked_panel)
    np.testing.assert_array_equal(result, expected)
