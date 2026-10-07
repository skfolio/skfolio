"""Test covariance conversion and incremental learning contracts."""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.utils import get_tags
from sklearn.utils.estimator_checks import estimator_checks_generator

from skfolio.distance import CovarianceDistance, PearsonDistance
from skfolio.moments import EWCovariance, GerberCovariance


def test_input_tags():
    assert not PearsonDistance().requires_covariance_input
    assert not get_tags(PearsonDistance()).input_tags.pairwise
    model = CovarianceDistance()
    assert not model.requires_covariance_input
    model.set_params(covariance_estimator=EWCovariance())
    assert not model.requires_covariance_input
    assert get_tags(model).input_tags.allow_nan
    assert not get_tags(model).input_tags.pairwise
    model.set_params(covariance_estimator="precomputed")
    assert get_tags(model).input_tags.pairwise
    assert model.requires_covariance_input
    assert clone(model).requires_covariance_input
    assert get_tags(clone(model)).input_tags.pairwise
    model.set_params(covariance_estimator=None)
    assert not model.requires_covariance_input
    assert not get_tags(model).input_tags.pairwise


@pytest.mark.parametrize("covariance_estimator", [None, GerberCovariance()])
@pytest.mark.parametrize("fit_first", [True, False])
def test_partial_fit_requires_incremental_covariance(covariance_estimator, fit_first):
    X = np.random.default_rng(7).normal(size=(30, 4))
    model = CovarianceDistance(covariance_estimator)
    if fit_first:
        model.fit(X[:20])
    with pytest.raises(
        TypeError, match="GerberCovariance does not implement partial_fit"
    ):
        model.partial_fit(X[20:])


@pytest.mark.parametrize("fit_first", [True, False])
def test_precomputed_partial_fit_is_rejected(fit_first):
    model = CovarianceDistance("precomputed")
    if fit_first:
        model.fit(np.eye(3))
        distance = model.distance_.copy()
    with pytest.raises(TypeError, match="Call fit with the updated covariance matrix"):
        model.partial_fit(np.ones((3, 3)))
    if fit_first:
        np.testing.assert_array_equal(model.distance_, distance)
    else:
        assert not hasattr(model, "distance_")


@pytest.mark.parametrize("fit_first", [True, False])
def test_learning_matches_direct_child_and_resets(fit_first):
    X = pd.DataFrame(
        np.random.default_rng(7).normal(size=(30, 4)), columns=list("abcd")
    )
    X.iloc[:9, 2] = np.nan
    child = EWCovariance(half_life=5, min_observations=3)
    model = CovarianceDistance(child)
    direct = clone(child)
    first_method = "fit" if fit_first else "partial_fit"
    getattr(model, first_method)(X.iloc[:8])
    getattr(direct, first_method)(X.iloc[:8])
    fitted_child = model.covariance_estimator_
    for start in [8, 16, 24]:
        batch = X.iloc[start : start + 8]
        model.partial_fit(batch)
        direct.partial_fit(batch)
        assert model.covariance_estimator_ is fitted_child
        np.testing.assert_allclose(
            fitted_child.covariance_, direct.covariance_, equal_nan=True
        )
        expected = CovarianceDistance("precomputed").fit(direct.covariance_)
        np.testing.assert_allclose(model.distance_, expected.distance_, equal_nan=True)
    model.set_params(covariance_estimator__half_life=8).fit(X.iloc[-8:])
    assert model.covariance_estimator_ is not fitted_child
    assert model.covariance_estimator_.half_life == 8
    np.testing.assert_allclose(model.distance_, clone(model).fit(X.iloc[-8:]).distance_)
    assert not hasattr(clone(model), "distance_")


@pytest.mark.parametrize("absolute,power", [(False, 1), (True, 1), (False, 2)])
def test_psd_conversion_missing_and_trivial(absolute, power):
    covariance = np.array(
        [[1.0, 2.0, np.nan], [2.0, 4.0, np.nan], [np.nan, np.nan, np.nan]]
    )
    original = covariance.copy()
    model = CovarianceDistance("precomputed", absolute=absolute, power=power).fit(
        covariance
    )
    np.testing.assert_allclose(model.distance_[:2, :2], 0)
    assert np.isnan(model.distance_[2]).all()
    np.testing.assert_array_equal(covariance, original)
    assert model.covariance_estimator_ is None
    model.fit(np.full((3, 3), np.nan))
    assert np.isnan(model.distance_).all()
    covariance[:] = np.nan
    covariance[1, 1] = 4
    model.fit(covariance)
    assert model.distance_[1, 1] == 0
    assert model.codependence_[1, 1] == 1


@pytest.mark.parametrize(
    "absolute,power,codependence,distance",
    [
        (False, 2.0, 0.0625, np.sqrt(0.9375)),
        (True, 0.5, 0.5, np.sqrt(0.5)),
    ],
)
def test_float_power(absolute, power, codependence, distance):
    covariance = np.array([[1.0, -0.25], [-0.25, 1.0]])
    model = CovarianceDistance("precomputed", absolute=absolute, power=power).fit(
        covariance
    )
    np.testing.assert_allclose(
        model.codependence_, [[1, codependence], [codependence, 1]]
    )
    np.testing.assert_allclose(model.distance_, [[0, distance], [distance, 0]])


@pytest.mark.parametrize("power", [0, 0.5])
def test_invalid_power_with_no_available_assets(power):
    with pytest.raises(ValueError, match="power"):
        CovarianceDistance("precomputed", power=power).fit(np.full((2, 2), np.nan))


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
@pytest.mark.parametrize(
    "covariance,expected_corr",
    [
        (
            [[1.0, 0.5 + 1e-14], [0.5, 1.0]],
            [[1.0, 0.5], [0.5, 1.0]],
        ),
        (
            [[1.0, 1.0 + 1e-14], [1.0 + 1e-14, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ),
        (
            [
                [1.0, -0.5 - 1e-14, -0.5 - 1e-14],
                [-0.5 - 1e-14, 1.0, -0.5 - 1e-14],
                [-0.5 - 1e-14, -0.5 - 1e-14, 1.0],
            ],
            [[1.0, -0.5, -0.5], [-0.5, 1.0, -0.5], [-0.5, -0.5, 1.0]],
        ),
    ],
    ids=["asymmetry", "correlation_bounds", "negative_eigenvalue"],
)
def test_covariance_roundoff(covariance, expected_corr, scale):
    covariance = np.asarray(covariance) * scale
    original = covariance.copy()
    model = CovarianceDistance("precomputed").fit(covariance)

    expected_distance = np.sqrt((1 - np.asarray(expected_corr)) / 2)
    np.testing.assert_allclose(model.codependence_, expected_corr, rtol=0, atol=2e-14)
    np.testing.assert_allclose(model.distance_, expected_distance, rtol=0, atol=2e-14)
    np.testing.assert_array_equal(model.distance_, model.distance_.T)
    np.testing.assert_array_equal(np.diag(model.distance_), 0)
    np.testing.assert_array_equal(covariance, original)


@pytest.mark.parametrize(
    "matrix",
    [
        [[-1.0, 0], [0, 1]],
        [[0.0, 0], [0, 1]],
        [[np.inf, 0], [0, 1]],
        [[1.0, 0.1], [0.5, 1]],
        [[1.0, 1.1], [1.1, 1]],
        [[1.0, -0.9, -0.9], [-0.9, 1, -0.9], [-0.9, -0.9, 1]],
        [[1.0, np.nan], [np.nan, 1]],
        [[1.0, 0, 0], [0, 1, 0]],
    ],
)
def test_invalid_covariances(matrix):
    with pytest.raises(ValueError):
        CovarianceDistance("precomputed").fit(matrix)


def test_invalid_schema_and_metadata_are_rejected_before_learning():
    X = pd.DataFrame(np.random.default_rng(3).normal(size=(20, 3)), columns=list("abc"))
    model = CovarianceDistance(EWCovariance(half_life=4, min_observations=3)).fit(
        X[:10]
    )
    covariance = model.covariance_estimator_.covariance_.copy()
    with pytest.raises(ValueError, match="feature names"):
        model.partial_fit(X.iloc[10:, ::-1])
    np.testing.assert_array_equal(model.covariance_estimator_.covariance_, covariance)
    with pytest.raises(TypeError, match="sample_weight"):
        model.partial_fit(X[10:], sample_weight=np.ones(10))
    np.testing.assert_array_equal(model.covariance_estimator_.covariance_, covariance)


def test_named_precomputed_identity():
    matrix = pd.DataFrame(np.eye(3), index=list("abc"), columns=list("abc"))
    model = CovarianceDistance("precomputed").fit(matrix)
    np.testing.assert_array_equal(model.feature_names_in_, list("abc"))
    with pytest.raises(ValueError, match="matching"):
        model.fit(matrix.iloc[::-1])


def test_precomputed_applicable_sklearn_checks():
    # These checks manufacture indefinite matrices or scattered NaNs, which do
    # not satisfy the covariance-snapshot contract. Use valid fixtures below.
    incompatible = {
        "check_positive_only_tag_during_fit",
        "check_estimators_dtypes",
        "check_estimators_pickle",
        "check_array_api_input",
    }
    for model, check in estimator_checks_generator(CovarianceDistance("precomputed")):
        if check.func.__name__ in {
            "check_fit_score_takes_y",
            "check_n_features_in_after_fitting",
        }:
            # These checks validate fit before calling the unsupported partial_fit.
            with pytest.raises(
                TypeError, match="Call fit with the updated covariance matrix"
            ):
                check(model)
        elif check.func.__name__ not in incompatible:
            check(model)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, object])
def test_precomputed_dtype_pickle_and_readonly(dtype):
    matrix = np.array(
        [[2, 0.1, np.nan], [0.1, 3, np.nan], [np.nan, np.nan, np.nan]], dtype=dtype
    )
    matrix.setflags(write=False)
    model = CovarianceDistance("precomputed")
    assert model.fit(matrix) is model
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.distance_, model.distance_)
    restored.fit(matrix)
    np.testing.assert_array_equal(restored.distance_, model.distance_)
