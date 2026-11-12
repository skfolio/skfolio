"""Distance input contracts shared by hierarchical optimizers."""

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn import config_context

from skfolio.containers import AssetPanel, Field2D, InactivePolicy
from skfolio.distance import BaseDistance, CovarianceDistance, PearsonDistance
from skfolio.exceptions import OptimizationError
from skfolio.moments import EmpiricalCovariance, LedoitWolf
from skfolio.optimization import (
    HierarchicalEqualRiskContribution,
    HierarchicalRiskParity,
    SchurComplementary,
)
from skfolio.prior import BasePrior, ReturnDistribution


class SubsetPrior(BasePrior):
    def __init__(self, mask=(True, True, False)):
        self.mask = mask

    def fit(self, X, y=None):
        values = np.nan_to_num(np.asarray(X))
        covariance = np.cov(values, rowvar=False)
        covariance[~np.array(self.mask), :] = np.nan
        covariance[:, ~np.array(self.mask)] = np.nan
        self.return_distribution_ = ReturnDistribution(
            mu=values.mean(axis=0),
            covariance=covariance,
            returns=np.tile(values, (2, 1)),
        )
        return self

    def partial_fit(self, X, y=None):
        return self.fit(X, y)


class ExposureDistance(BaseDistance):
    """Independent distance consuming a numeric panel under a different keyword."""

    __metadata_request__fit: ClassVar[dict[str, bool]] = {"exposures": True}

    def __init__(self):
        pass

    def fit(self, X, y=None, *, exposures):
        self.columns_ = X.columns.to_numpy()
        self.n_rows_ = len(X)
        self.panel_ = exposures
        positions = [list(exposures.asset_names).index(name) for name in self.columns_]
        values = exposures["exposure"][-1, positions]
        self.distance_ = np.abs(values[:, None] - values[None, :])
        self.codependence_ = 1 - self.distance_
        return self


class CovarianceExposureDistance(ExposureDistance):
    @property
    def requires_covariance_input(self):
        return True


@pytest.mark.parametrize(
    "optimizer,params",
    [
        (HierarchicalRiskParity, {"distance_from_prior": False}),
        (HierarchicalRiskParity, {"distance_from_prior": True}),
        (SchurComplementary, {"distance_from_prior": False}),
        (SchurComplementary, {"distance_from_prior": True}),
        (HierarchicalEqualRiskContribution, {}),
    ],
)
@pytest.mark.parametrize("numeric_strings", [False, True])
@pytest.mark.parametrize("metadata_name", ["exposures", "features"])
def test_independent_metadata_distance(
    optimizer, params, numeric_strings, metadata_name
):
    names = ["20", "10", "30"] if numeric_strings else ["B", "A", "C"]
    extra = "40" if numeric_strings else "D"
    X = pd.DataFrame(np.random.default_rng(42).normal(0, 0.01, (40, 3)), columns=names)
    panel = AssetPanel(
        fields={
            "exposure": Field2D(
                np.tile([0, 0.2, 0.4, 0.8], (40, 1)),
                inactive_policy=InactivePolicy.IGNORE,
            )
        },
        observations=np.arange(40),
        asset_names=[extra, *names[::-1]],
    )
    with config_context(enable_metadata_routing=True):
        distance = ExposureDistance().set_fit_request(exposures=metadata_name)
        model = optimizer(
            prior_estimator=SubsetPrior(), distance_estimator=distance, **params
        )
        with pytest.raises(ValueError, match="same number of observations"):
            model.fit(X, **{metadata_name: panel[:20]})
        assert not hasattr(
            getattr(model, "prior_estimator_", None), "return_distribution_"
        )

        model.fit(X, **{metadata_name: panel})
        fitted = model.distance_estimator_
        from_prior = params.get("distance_from_prior", True)
        np.testing.assert_array_equal(
            fitted.columns_, names[:2] if from_prior else names
        )
        assert fitted.n_rows_ == (80 if from_prior else len(X))
        assert fitted.panel_ is panel
        np.testing.assert_allclose(fitted.distance_[:2, :2], [[0, 0.4], [0.4, 0]])
        assert model.weights_[2] == 0
        assert model.prior_estimator_.return_distribution_.returns.shape[0] == 80

        with panel.edit_masks():
            panel.active_mask[-1, -1] = False
        model.fit(X, **{metadata_name: panel})
        np.testing.assert_allclose(
            model.distance_estimator_.distance_, fitted.distance_
        )


@pytest.mark.parametrize("optimizer", [HierarchicalRiskParity, SchurComplementary])
@pytest.mark.parametrize("prefit", [False, True])
@pytest.mark.parametrize("view", [False, True])
@pytest.mark.parametrize("metadata_name", ["exposures", "features"])
def test_online_independent_metadata_distance(optimizer, prefit, view, metadata_name):
    X = pd.DataFrame(
        np.random.default_rng(42).normal(0, 0.01, (40, 3)), columns=["A", "B", "C"]
    )
    panel = AssetPanel(
        fields={"exposure": Field2D(np.tile([0, 0.2, 0.4], (40, 1)))},
        observations=np.arange(40),
        asset_names=X.columns,
    )
    with config_context(enable_metadata_routing=True):
        model = optimizer(
            prior_estimator=SubsetPrior(),
            distance_estimator=ExposureDistance().set_fit_request(
                exposures=metadata_name
            ),
        )
        first_method = "fit" if prefit else "partial_fit"
        getattr(model, first_method)(X.iloc[:20], **{metadata_name: panel[:20]})
        distribution = model.prior_estimator_.return_distribution_
        with pytest.raises(ValueError, match="same number of observations"):
            model.partial_fit(X.iloc[20:], **{metadata_name: panel[21:]})
        assert model.prior_estimator_.return_distribution_ is distribution

        batch_panel = panel[20:] if view else panel[20:].to_panel()
        model.partial_fit(X.iloc[20:], **{metadata_name: batch_panel})
        fitted = model.distance_estimator_
        assert fitted.panel_ is batch_panel
        assert fitted.n_rows_ == 2 * len(batch_panel)
        np.testing.assert_array_equal(fitted.columns_, ["A", "B"])
        np.testing.assert_allclose(fitted.distance_, [[0, 0.2], [0.2, 0]])


@pytest.mark.parametrize("optimizer", [HierarchicalRiskParity, SchurComplementary])
@pytest.mark.parametrize("method", ["fit", "partial_fit"])
def test_covariance_rejects_observation_containers_before_learning(optimizer, method):
    X = pd.DataFrame(
        np.random.default_rng(42).normal(0, 0.01, (20, 3)), columns=["A", "B", "C"]
    )
    panel = AssetPanel(
        fields={"exposure": Field2D(np.ones(X.shape))},
        observations=np.arange(len(X)),
        asset_names=X.columns,
    )
    with config_context(enable_metadata_routing=True):
        model = optimizer(
            prior_estimator=SubsetPrior(),
            distance_estimator=CovarianceExposureDistance(),
        )
        with pytest.raises(ValueError, match=r"Observation metadata.*covariance"):
            getattr(model, method)(X, exposures=panel)
        assert not hasattr(model.prior_estimator_, "return_distribution_")


def test_herc_covariance_rejects_observation_containers():
    X = pd.DataFrame(
        np.random.default_rng(42).normal(0, 0.01, (20, 3)), columns=["A", "B", "C"]
    )
    panel = AssetPanel(
        fields={"exposure": Field2D(np.ones(X.shape))},
        observations=np.arange(len(X)),
        asset_names=X.columns,
    )
    with config_context(enable_metadata_routing=True):
        model = HierarchicalEqualRiskContribution(
            prior_estimator=SubsetPrior(),
            distance_estimator=CovarianceExposureDistance(),
        )
        with pytest.raises(ValueError, match=r"Observation metadata.*covariance"):
            model.fit(X, exposures=panel)


@pytest.mark.parametrize(
    "optimizer",
    [HierarchicalEqualRiskContribution, HierarchicalRiskParity, SchurComplementary],
)
@pytest.mark.parametrize(
    "distance", [PearsonDistance(), CovarianceDistance("precomputed")]
)
@pytest.mark.parametrize("costs", [False, True])
def test_numeric_distance_uses_investable_subset(optimizer, distance, costs):
    X = pd.DataFrame(
        np.random.default_rng(42).normal(0, 0.01, (40, 3)), columns=["A", "B", "C"]
    )
    X["C"] = np.nan
    params = {
        "min_weights": [0.1, 0.2, 0],
        "max_weights": [0.6, 0.8, 1],
    }
    if costs:
        params.update(
            transaction_costs=[0.001, 0.002, 0.003],
            management_fees=[0.0001, 0.0002, 0.0003],
            previous_weights=[0.1, 0.2, 0.7],
        )
    model = optimizer(
        prior_estimator=SubsetPrior(), distance_estimator=distance, **params
    ).fit(X)
    expected = optimizer(
        prior_estimator=SubsetPrior(mask=(True, True)),
        distance_estimator=distance,
        **{name: values[:2] for name, values in params.items()},
    ).fit(X.iloc[:, :2])
    np.testing.assert_allclose(model.weights_[:2], expected.weights_)
    assert model.weights_[2] == 0
    np.testing.assert_array_equal(model.investable_mask_, [True, True, False])
    np.testing.assert_allclose(
        model.distance_estimator_.distance_, expected.distance_estimator_.distance_
    )

    assert model.distance_estimator_.n_features_in_ == 2
    np.testing.assert_array_equal(
        model.distance_estimator_.feature_names_in_, ["A", "B"]
    )


def test_herc_numeric_distance_single_investable_asset():
    X = np.random.default_rng(42).normal(0, 0.01, (40, 3))
    model = HierarchicalEqualRiskContribution(
        prior_estimator=SubsetPrior(mask=(True, False, False))
    ).fit(X)
    np.testing.assert_array_equal(model.weights_, [1, 0, 0])
    assert model.distance_estimator_ is None
    assert model.hierarchical_clustering_estimator_ is None


@pytest.mark.parametrize(
    "optimizer",
    [HierarchicalRiskParity, SchurComplementary, HierarchicalEqualRiskContribution],
)
@pytest.mark.parametrize(
    "mask,max_weights",
    [
        ((True, True, False), [0.4, 0.4, 1]),
        ((True, False, False), [0.9, 1, 1]),
        ((True, False, False), [1 - 2e-8, 1, 1]),
    ],
)
def test_infeasible_investable_weight_bounds(optimizer, mask, max_weights):
    X = np.random.default_rng(42).normal(0, 0.01, (40, 3))
    model = optimizer(prior_estimator=SubsetPrior(mask=mask), max_weights=max_weights)
    with pytest.raises(
        OptimizationError, match="infeasible for the investable assets"
    ) as error:
        model.fit(X)
    assert "lower bounds sum to 0" in str(error.value)
    upper_sum = np.asarray(max_weights)[np.asarray(mask)].sum()
    assert f"upper bounds sum to {upper_sum:.12g}" in str(error.value)


@pytest.mark.parametrize(
    "optimizer",
    [HierarchicalRiskParity, SchurComplementary, HierarchicalEqualRiskContribution],
)
@pytest.mark.parametrize("deficit", [0, 1e-10, 1e-8])
def test_single_investable_weight_bound_tolerance(optimizer, deficit):
    X = np.random.default_rng(42).normal(0, 0.01, (40, 3))
    model = optimizer(
        prior_estimator=SubsetPrior(mask=(True, False, False)),
        max_weights=[1 - deficit, 1, 1],
    ).fit(X)
    np.testing.assert_array_equal(model.weights_, [1, 0, 0])


class UnavailableDistance(BaseDistance):
    def __init__(self, pattern="row"):
        self.pattern = pattern

    def fit(self, X, y=None):
        self.distance_ = np.ones((X.shape[1], X.shape[1]))
        np.fill_diagonal(self.distance_, 0)
        if self.pattern == "row":
            self.distance_[1, :] = np.nan
            self.distance_[:, 1] = np.nan
        elif self.pattern == "pair":
            self.distance_[0, 1] = self.distance_[1, 0] = np.inf
        elif self.pattern == "shape":
            self.distance_ = self.distance_[:-1, :-1]
        else:
            self.distance_[1, 1] = np.nan
        self.codependence_ = 1 - self.distance_
        return self


@pytest.mark.parametrize(
    "optimizer",
    [HierarchicalRiskParity, SchurComplementary, HierarchicalEqualRiskContribution],
)
@pytest.mark.parametrize("pattern", ["row", "pair", "diagonal"])
@pytest.mark.parametrize("asset_names", [None, ["A", "B", "C"], [10, 20, 30]])
def test_unavailable_distances_identify_assets(optimizer, pattern, asset_names):
    X = np.random.default_rng(42).normal(0, 0.01, (40, 3))
    if asset_names is not None:
        X = pd.DataFrame(X, columns=asset_names)
    model = optimizer(
        prior_estimator=SubsetPrior(mask=(False, True, True)),
        distance_estimator=UnavailableDistance(pattern),
    )
    with pytest.raises(ValueError) as error:
        model.fit(X)
    message = "Distances are unavailable for prior-investable assets"
    if asset_names is not None:
        expected = asset_names[1:] if pattern == "pair" else asset_names[2:]
        message += f" {expected}"
    else:
        expected = [1, 2] if pattern == "pair" else [2]
        message += f" at X column positions {expected}"
    assert str(error.value) == (
        f"{message}. Check the distance estimator's data coverage and "
        "minimum-observation requirements."
    )


@pytest.mark.parametrize(
    "optimizer,params",
    [
        (HierarchicalEqualRiskContribution, {}),
        (HierarchicalRiskParity, {"distance_from_prior": True}),
        (HierarchicalRiskParity, {"distance_from_prior": False}),
        (SchurComplementary, {"distance_from_prior": True}),
        (SchurComplementary, {"distance_from_prior": False}),
    ],
)
@pytest.mark.parametrize("asset_names", [None, ["A", "B", "C"]])
def test_distance_shape_is_validated_with_or_without_labels(
    optimizer, params, asset_names
):
    X = np.random.default_rng(42).normal(0, 0.01, (40, 3))
    if asset_names is not None:
        X = pd.DataFrame(X, columns=asset_names)
    model = optimizer(
        prior_estimator=SubsetPrior(),
        distance_estimator=UnavailableDistance("shape"),
        **params,
    )
    with pytest.raises(ValueError, match="does not match its input asset schema"):
        model.fit(X)


@pytest.mark.parametrize("optimizer", [HierarchicalRiskParity, SchurComplementary])
def test_raw_distance_keeps_full_estimation_universe(optimizer):
    X = pd.DataFrame(
        np.random.default_rng(12).normal(size=(20, 3)), columns=["A", "B", "C"]
    )
    X["B"] = 0.8 * X["A"] + 0.2 * X["B"]
    X["C"] *= 4
    distance = CovarianceDistance(LedoitWolf())
    model = optimizer(
        prior_estimator=SubsetPrior(),
        distance_estimator=distance,
        distance_from_prior=False,
    ).fit(X)
    reference = distance.fit(X).distance_
    np.testing.assert_allclose(model.distance_estimator_.distance_, reference)
    assert model.weights_[2] == 0
    # Shrinkage depends on the estimation universe, even for the same A-B pair.
    subset = distance.fit(X.iloc[:, :2]).distance_
    assert not np.isclose(reference[0, 1], subset[0, 1])


class ScenarioPrior(SubsetPrior):
    def __init__(self, weighted=True):
        super().__init__()
        self.weighted = weighted

    def fit(self, X, y=None):
        super().fit(X, y)
        distribution = self.return_distribution_
        scenarios = np.asarray(X)[::-2].copy()
        weights = np.arange(1, len(scenarios) + 1, dtype=float)
        self.return_distribution_ = ReturnDistribution(
            mu=distribution.mu,
            covariance=distribution.covariance,
            returns=scenarios,
            sample_weight=weights / weights.sum() if self.weighted else None,
        )
        return self


class WeightedDistance(PearsonDistance):
    def fit(self, X, y=None, sample_weight=None):
        self.received_weights_ = sample_weight
        self.received_columns_ = X.columns
        return super().fit(X, y)


@pytest.mark.parametrize(
    "optimizer",
    [HierarchicalEqualRiskContribution, HierarchicalRiskParity, SchurComplementary],
)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("weight_request", [True, "observation_weights", False])
def test_prior_distance_scenario_weights(optimizer, nested, weighted, weight_request):
    X = pd.DataFrame(
        np.random.default_rng(18).normal(0, 0.01, (40, 3)), columns=["A", "B", "C"]
    )
    with config_context(enable_metadata_routing=True):
        consumer = EmpiricalCovariance(nearest=False) if nested else WeightedDistance()
        consumer.set_fit_request(sample_weight=weight_request)
        distance = CovarianceDistance(consumer) if nested else consumer
        name = (
            "observation_weights"
            if isinstance(weight_request, str)
            else "sample_weight"
        )
        input_weights = np.arange(1, len(X) + 1, dtype=float)
        params = {name: input_weights} if weight_request else {}
        model = optimizer(
            prior_estimator=ScenarioPrior(weighted=weighted),
            distance_estimator=distance,
        ).fit(X, **params)
    distribution = model.prior_estimator_.return_distribution_
    expected_weights = distribution.sample_weight if weight_request else None
    assert len(distribution.returns) == len(X) // 2
    if nested:
        fitted = model.distance_estimator_.covariance_estimator_
        np.testing.assert_allclose(
            fitted.covariance_,
            np.cov(
                distribution.returns[:, :2], rowvar=False, aweights=expected_weights
            ),
        )
    else:
        fitted = model.distance_estimator_
        np.testing.assert_array_equal(fitted.received_columns_, ["A", "B"])
        if expected_weights is None:
            assert fitted.received_weights_ is None
        else:
            np.testing.assert_array_equal(fitted.received_weights_, expected_weights)
    assert model.weights_[2] == 0
    np.testing.assert_array_equal(input_weights, np.arange(1, len(X) + 1))


@pytest.mark.parametrize(
    "optimizer",
    [HierarchicalEqualRiskContribution, HierarchicalRiskParity, SchurComplementary],
)
@pytest.mark.parametrize("kind", ["scalar", "array", "dict"])
def test_minimum_weights_apply_to_investable_assets(optimizer, kind):
    X = pd.DataFrame(
        np.random.default_rng(42).normal(0, 0.01, (40, 3)), columns=["A", "B", "C"]
    )
    lower = {"scalar": 0.4, "array": [0.4] * 3, "dict": dict.fromkeys(X.columns, 0.4)}
    model = optimizer(prior_estimator=SubsetPrior(), min_weights=lower[kind]).fit(X)
    assert np.all(model.weights_[:2] >= 0.4 - 1e-8)
    assert model.weights_[2] == 0
    assert model.weights_.sum() == pytest.approx(1)
    model.set_params(prior_estimator=SubsetPrior(mask=(True, True, True)))
    with pytest.raises(OptimizationError, match="infeasible"):
        model.fit(X)


@pytest.mark.parametrize(
    "optimizer",
    [HierarchicalEqualRiskContribution, HierarchicalRiskParity, SchurComplementary],
)
def test_scenario_weights_without_metadata_routing(optimizer):
    X = pd.DataFrame(
        np.random.default_rng(18).normal(0, 0.01, (40, 3)), columns=["A", "B", "C"]
    )
    with config_context(enable_metadata_routing=False):
        model = optimizer(
            prior_estimator=ScenarioPrior(), distance_estimator=WeightedDistance()
        ).fit(X)
    np.testing.assert_array_equal(
        model.distance_estimator_.received_weights_,
        model.prior_estimator_.return_distribution_.sample_weight,
    )
