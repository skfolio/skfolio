"""Tests for the cross-sectional transformer helpers."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.preprocessing import CSPercentileRankScaler
from skfolio.preprocessing._transformer._cross_sectional._utils import (
    _group_key_midrank_percentile,
    _validate_and_normalize_groups,
)


def test_object_groups_with_non_integer_labels_raise():
    X = np.ones((1, 3))
    cs_groups = np.array([["a", "b", "c"]], dtype=object)
    with pytest.raises(ValueError, match=r"`cs_groups` must be an integer array\."):
        _validate_and_normalize_groups(X=X, cs_groups=cs_groups)
    with pytest.raises(ValueError, match=r"`cs_groups` must be an integer array\."):
        CSPercentileRankScaler().fit_transform(X, cs_groups=cs_groups)


def test_object_groups_with_integer_labels_are_accepted():
    X = np.ones((1, 3))
    cs_groups = np.array([[0, 1, 1]], dtype=object)
    group_ids, missing_group_mask, n_groups = _validate_and_normalize_groups(
        X=X, cs_groups=cs_groups
    )
    np.testing.assert_array_equal(group_ids, [[0, 1, 1]])
    assert not missing_group_mask.any()
    assert n_groups == 2


def test_midrank_percentile_defaults_finite_mask_and_handles_all_nan():
    X = np.full((2, 3), np.nan)
    estimation_mask = np.ones_like(X, dtype=bool)
    group_keys = np.repeat(np.arange(2), 3).reshape(2, 3)

    percentile, counts = _group_key_midrank_percentile(
        X, estimation_mask, group_keys, n_group_keys=2
    )

    assert np.isnan(percentile).all()
    np.testing.assert_array_equal(counts, [0, 0])
