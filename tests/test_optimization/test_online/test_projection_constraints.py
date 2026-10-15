"""Direct constraint checks independent of the online estimator lifecycle."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfolio.exceptions import GroupNotFoundError
from skfolio.optimization.online._mirror_maps import (
    AdaptiveMahalanobisMap,
    EntropyMirrorMap,
)
from skfolio.optimization.online._projection import (
    AutoProjector,
    ConvexProjector,
    FastProjector,
    ProjectionConfig,
    project_convex,
    project_with_turnover,
)
from skfolio.prior import FactorModel, ReturnDistribution


def project(config, candidate, geometry):
    """Use the candidate as the unconstrained minimum in each geometry."""
    projector = AutoProjector(config)
    if geometry == "euclidean":
        return projector.project(candidate)
    if geometry == "entropy":
        mirror = EntropyMirrorMap()
    else:
        mirror = AdaptiveMahalanobisMap()
        mirror.update_state(np.array([1.0, 2.0]))
    return projector.solve_update(mirror, mirror.grad_psi(candidate))


@pytest.mark.parametrize("geometry", ["euclidean", "entropy", "adaptive"])
@pytest.mark.parametrize(
    "options, expected",
    [
        pytest.param({"upper": [0.6, 1.0]}, [0.6, 0.4], id="upper-bound"),
        pytest.param({"lower": [0.0, 0.4]}, [0.6, 0.4], id="lower-bound"),
        pytest.param(
            {"groups": [["a", "b"]], "linear_constraints": ["a <= 0.6"]},
            [0.6, 0.4],
            id="named-inequality",
        ),
        pytest.param(
            {"groups": [["a", "b"]], "linear_constraints": ["a == 0.3"]},
            [0.3, 0.7],
            id="named-equality",
        ),
        pytest.param(
            {"left_inequality": [[1.0, 0.0]], "right_inequality": [0.6]},
            [0.6, 0.4],
            id="matrix-inequality",
        ),
        pytest.param(
            {"previous_weights": [0.5, 0.5], "max_turnover": 0.2},
            [0.6, 0.4],
            id="turnover",
        ),
        pytest.param(
            {"covariance": np.diag([1.0, 0.0]), "variance_bound": 0.36},
            [0.6, 0.4],
            id="variance",
        ),
        pytest.param(
            {
                "X_tracking": [[1.0, 0.0], [-1.0, 0.0]],
                "tracking_error_benchmark": [0.0, 0.0],
                "max_tracking_error": 0.6 * np.sqrt(2),
            },
            [0.6, 0.4],
            id="tracking-error",
        ),
    ],
)
def test_binding_constraints(options, expected, geometry):
    actual = project(ProjectionConfig(**options), np.array([0.9, 0.1]), geometry)
    assert_allclose(actual, expected, atol=3e-5)


@pytest.mark.parametrize("geometry", ["euclidean", "entropy", "adaptive"])
@pytest.mark.parametrize(
    "options, candidate, total",
    [
        ({"budget": 0.7}, [0.9, 0.9], 0.7),
        ({"budget": None, "min_budget": 0.8}, [0.1, 0.1], 0.8),
        ({"budget": None, "max_budget": 0.6}, [0.9, 0.9], 0.6),
        ({"budget": None, "min_budget": 0.4, "max_budget": 0.6}, [0.9, 0.9], 0.6),
    ],
)
def test_binding_budgets(options, candidate, total, geometry):
    actual = project(ProjectionConfig(**options), np.array(candidate), geometry)
    assert_allclose(actual.sum(), total, atol=3e-5)
    assert np.all(actual >= -1e-7)


@pytest.mark.parametrize("geometry", ["euclidean", "adaptive"])
@pytest.mark.parametrize("cap", [{"max_short": 0.2}, {"max_long": 1.2}])
def test_binding_signed_exposure(cap, geometry):
    config = ProjectionConfig(lower=-2.0, upper=2.0, **cap)
    actual = project(config, np.array([-0.8, 1.8]), geometry)
    assert_allclose(actual, [-0.2, 1.2], atol=3e-5)


@pytest.fixture
def factor_distribution():
    return ReturnDistribution(
        mu=np.zeros(2),
        covariance=np.eye(2),
        returns=np.zeros((2, 2)),
        factor_model=FactorModel(
            observations=np.array(["t"]),
            asset_names=np.array(["a", "b"]),
            factor_names=np.array(["Momentum"]),
            factor_families=np.array(["Style"]),
            loading_matrix=np.array([[1.0], [0.0]]),
            exposures=None,
            factor_covariance=np.eye(1),
            factor_mu=np.zeros(1),
            factor_returns=None,
            idio_covariance=np.eye(2),
            idio_mu=None,
            idio_returns=None,
            idio_variances=None,
        ),
    )


@pytest.mark.parametrize("geometry", ["euclidean", "entropy", "adaptive"])
@pytest.mark.parametrize("name", ["Momentum", "Style"])
def test_factor_constraints(factor_distribution, geometry, name):
    config = ProjectionConfig(
        groups=[["a", "b"]],
        linear_constraints=[f"{name} <= 0.6"],
        return_distribution=factor_distribution,
    )
    assert_allclose(
        project(config, np.array([0.9, 0.1]), geometry), [0.6, 0.4], atol=3e-5
    )


@pytest.mark.parametrize("geometry", ["euclidean", "entropy", "adaptive"])
def test_group_constraint_and_unknown_name(geometry):
    config = ProjectionConfig(
        groups=[["sector", "other"]], linear_constraints=["sector <= 0.6"]
    )
    assert_allclose(
        project(config, np.array([0.9, 0.1]), geometry), [0.6, 0.4], atol=3e-5
    )
    config.linear_constraints = ["missing <= 0.6"]
    with pytest.raises(GroupNotFoundError):
        project(config, np.array([0.9, 0.1]), geometry)


@pytest.mark.parametrize("geometry", ["euclidean", "entropy", "adaptive"])
@pytest.mark.parametrize(
    "options",
    [
        {"budget": None, "min_budget": 1.2, "max_budget": 0.8},
        {"groups": [["a", "b"]], "linear_constraints": ["a >= 0.8", "b >= 0.8"]},
        {"left_inequality": [[1.0, 1.0]], "right_inequality": [0.5]},
        {"covariance": np.eye(2), "variance_bound": 0.1},
        {"upper": 0.6, "previous_weights": [0.9, 0.1], "max_turnover": 0.1},
    ],
)
def test_infeasible_constraints(options, geometry):
    with pytest.raises(RuntimeError, match="projection failed"):
        project(ProjectionConfig(**options), np.array([0.5, 0.5]), geometry)


@pytest.mark.parametrize("geometry", ["euclidean", "entropy", "adaptive"])
def test_constraints_are_enforced_together(geometry):
    config = ProjectionConfig(
        lower=0.1,
        upper=0.8,
        budget=None,
        min_budget=1.0,
        max_budget=1.0,
        groups=[["a", "b"]],
        linear_constraints=["b >= 0.3"],
        left_inequality=[[1.0, 0.0]],
        right_inequality=[0.65],
        covariance=np.diag([1.0, 0.0]),
        variance_bound=0.36,
        previous_weights=[0.5, 0.5],
        max_turnover=0.1,
    )
    assert_allclose(
        project(config, np.array([0.9, 0.1]), geometry), [0.55, 0.45], atol=3e-5
    )


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"budget": None},
        {"lower": [0.1, 0.2], "upper": [0.6, 0.8]},
        {"previous_weights": [0.5, 0.5], "max_turnover": 0.2},
    ],
)
def test_fast_and_convex_projection_agree(options):
    config = ProjectionConfig(**options)
    candidate = np.array([0.9, 0.1])
    assert_allclose(
        FastProjector(config).project(candidate),
        ConvexProjector(config).project(candidate),
        atol=3e-5,
    )


@pytest.mark.parametrize("path", ["auto", "convex", "function", "entropy", "adaptive"])
@pytest.mark.parametrize(
    "options, message",
    [
        ({"max_turnover": 0.0}, "max_turnover requires previous_weights"),
        ({"variance_bound": 0.1}, "variance_bound requires covariance"),
        ({"max_tracking_error": 0.1}, "max_tracking_error requires X"),
        (
            {"max_tracking_error": 0.1, "X_tracking": [[1, 0], [0, 1]]},
            "max_tracking_error requires X",
        ),
        (
            {"max_tracking_error": 0.1, "tracking_error_benchmark": [0, 0]},
            "max_tracking_error requires X",
        ),
    ],
)
def test_missing_constraint_inputs(options, message, path):
    config = ProjectionConfig(**options)
    with pytest.raises(ValueError, match=message):
        if path == "function":
            kwargs = dict(options)
            if "X_tracking" in kwargs:
                kwargs["X"] = kwargs.pop("X_tracking")
            project_convex([0.9, 0.1], **kwargs)
        elif path == "convex":
            ConvexProjector(config).project(np.array([0.9, 0.1]))
        else:
            project(
                config, np.array([0.9, 0.1]), "euclidean" if path == "auto" else path
            )


@pytest.mark.parametrize(
    "cap", ["max_turnover", "variance_bound", "max_tracking_error"]
)
@pytest.mark.parametrize("value", [-0.1, np.nan, np.inf, [0.1, 0.2]])
@pytest.mark.parametrize("geometry", ["euclidean", "entropy", "adaptive"])
def test_invalid_caps(cap, value, geometry):
    config = ProjectionConfig(**{cap: value})
    with pytest.raises(ValueError, match=f"{cap} must be a finite nonnegative scalar"):
        project(config, np.array([0.9, 0.1]), geometry)


def test_turnover_function_rejects_missing_holdings():
    with pytest.raises(ValueError, match="max_turnover requires previous_weights"):
        project_with_turnover([0.9, 0.1], previous_weights=None, max_turnover=0.0)


def test_supporting_data_without_caps_remains_allowed():
    config = ProjectionConfig(
        previous_weights=[0.5, 0.5],
        covariance=np.eye(2),
        X_tracking=np.eye(2),
        tracking_error_benchmark=[0, 0],
    )
    assert_allclose(
        AutoProjector(config).project(np.array([0.9, 0.1])), [0.9, 0.1], atol=3e-5
    )
