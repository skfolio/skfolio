import numpy as np
import pytest
from sklearn import clone, config_context

from skfolio import ExtraRiskMeasure, RiskMeasure
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.moments import EWCovariance, ImpliedCovariance
from skfolio.optimization import HierarchicalRiskParity
from skfolio.prior import EmpiricalPrior, EntropyPooling, FactorModel


@pytest.fixture(scope="module")
def small_X(X):
    return X[["AAPL", "AMD", "BAC"]]


@pytest.fixture(
    scope="module",
    params=[
        x
        for x in [*RiskMeasure, *ExtraRiskMeasure]
        if x not in [ExtraRiskMeasure.SKEW, ExtraRiskMeasure.KURTOSIS]
    ],
)
def risk_measure(request):
    return request.param


def test_hrp_default(X):
    model = HierarchicalRiskParity(
        risk_measure=RiskMeasure.STANDARD_DEVIATION,
        hierarchical_clustering_estimator=HierarchicalClustering(
            linkage_method=LinkageMethod.SINGLE
        ),
    )
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.04948508,
                0.03483034,
                0.04967809,
                0.05488211,
                0.02595736,
                0.04482042,
                0.03332249,
                0.08091826,
                0.02840749,
                0.04412388,
                0.06297933,
                0.05717246,
                0.05285248,
                0.07430329,
                0.05499559,
                0.07914531,
                0.03391673,
                0.0396792,
                0.04812428,
                0.0504058,
            ]
        ),
    )


def test_hrp_empirical_prior(X):
    model = HierarchicalRiskParity(
        prior_estimator=EmpiricalPrior(covariance_estimator=EWCovariance())
    )
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.00923669,
                0.00427779,
                0.06213615,
                0.02367877,
                0.03596591,
                0.02150797,
                0.04545975,
                0.1480145,
                0.0800653,
                0.04264111,
                0.06024382,
                0.10169225,
                0.02671939,
                0.07198196,
                0.02399284,
                0.09213609,
                0.00500037,
                0.0520981,
                0.06382561,
                0.02932562,
            ]
        ),
    )


def test_hrp_factor_model(X, y):
    model = HierarchicalRiskParity(
        risk_measure=RiskMeasure.CVAR, prior_estimator=FactorModel()
    )
    model.fit(X, y)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.04674381,
                0.03733258,
                0.04217222,
                0.02712771,
                0.0293525,
                0.02431272,
                0.02907417,
                0.08493361,
                0.046902,
                0.07729156,
                0.0372962,
                0.04266708,
                0.04608497,
                0.07247913,
                0.07839451,
                0.08150043,
                0.02103614,
                0.06087054,
                0.05491043,
                0.05951769,
            ]
        ),
    )


def test_hrp(X, linkage_method, risk_measure):
    model = HierarchicalRiskParity(
        risk_measure=RiskMeasure.STANDARD_DEVIATION,
        hierarchical_clustering_estimator=HierarchicalClustering(
            linkage_method=linkage_method
        ),
    )
    model.fit(X)


def test_transaction_costs(X, previous_weights, transaction_costs):
    model = HierarchicalRiskParity(risk_measure=RiskMeasure.WORST_REALIZATION)
    model.fit(X)

    model_tc = HierarchicalRiskParity(
        risk_measure=RiskMeasure.WORST_REALIZATION,
        transaction_costs=transaction_costs * 1000,
        previous_weights=previous_weights,
    )
    model_tc.fit(X)
    assert np.sum(np.abs(model.weights_ - model_tc.weights_)) > 0.1


def test_hrp_small_X(small_X):
    model = HierarchicalRiskParity()
    model.fit(small_X)
    assert model.hierarchical_clustering_estimator_.n_clusters_ == 2


def test_metadata_routing(X_medium, implied_vol_medium):
    with config_context(enable_metadata_routing=True):
        model = HierarchicalRiskParity(
            prior_estimator=EmpiricalPrior(
                covariance_estimator=ImpliedCovariance().set_fit_request(
                    implied_vol=True
                )
            )
        )

        with pytest.raises(ValueError):
            model.fit(X_medium)

        model.fit(X_medium, implied_vol=implied_vol_medium)

    # noinspection PyUnresolvedReferences
    assert model.prior_estimator_.covariance_estimator_.r2_scores_.shape == (20,)


def test_hrp_weight_constraints(X):
    model = HierarchicalRiskParity(
        risk_measure=RiskMeasure.STANDARD_DEVIATION,
    )
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_[0], 0.030624328591088226)
    np.testing.assert_almost_equal(model.weights_[-1], 0.05811358822991056)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    # Min Weights
    model.set_params(min_weights={"AAPL": 0.05, "XOM": 0.08})
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_[0], 0.05)
    np.testing.assert_almost_equal(model.weights_[-1], 0.08)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    model.set_params(min_weights=0.05)
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_, np.ones(20) * 0.05)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    # Max Weights
    model.set_params(min_weights=0)
    model.set_params(max_weights={"AAPL": 0.01, "XOM": 0.03})
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_[0], 0.01)
    np.testing.assert_almost_equal(model.weights_[-1], 0.03)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    model.set_params(max_weights=0.05)
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_, np.ones(20) * 0.05)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    # Both
    model.set_params(min_weights={"AAPL": 0.05}, max_weights={"XOM": 0.03})
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_[0], 0.05)
    np.testing.assert_almost_equal(model.weights_[-1], 0.03)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    model.set_params(min_weights=0.03, max_weights=0.06)
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.03,
                0.03,
                0.03,
                0.04712633,
                0.05191067,
                0.06,
                0.06,
                0.06,
                0.03,
                0.06,
                0.06,
                0.06,
                0.04068081,
                0.06,
                0.05825588,
                0.06,
                0.03115588,
                0.05087043,
                0.06,
                0.06,
            ]
        ),
    )
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)


def test_hrp_weight_constraints_rand(X, risk_measure, linkage_method):
    model = HierarchicalRiskParity(
        risk_measure=risk_measure,
        hierarchical_clustering_estimator=HierarchicalClustering(
            linkage_method=linkage_method
        ),
    )
    np.random.seed(42)
    for _ in range(5):
        min_weights, max_weights = _random_weights_bounds(n_assets=X.shape[1])
        print(min_weights)
        model.set_params(min_weights=min_weights, max_weights=max_weights)
        model.fit(X)
        assert np.all(model.weights_ - min_weights >= -1e-8)
        assert np.all(model.weights_ - max_weights <= 1e-8)


def _random_weights_bounds(n_assets: int) -> tuple[np.ndarray, np.ndarray]:
    raw_min = np.random.rand(n_assets)
    min_total_target = np.random.rand()
    min_weights = raw_min / raw_min.sum() * min_total_target
    required_diff = max(0.0, 1.0 - min_weights.sum())
    raw_diff = np.random.rand(n_assets)
    extra_factor = 1 + np.random.rand()
    diff_total_target = required_diff * extra_factor
    diff_weights = raw_diff / raw_diff.sum() * diff_total_target
    max_weights = min_weights + diff_weights

    assert min_weights.sum() <= 1.0 <= max_weights.sum()
    assert np.all(min_weights >= 0) and np.all(max_weights >= min_weights)
    return min_weights, max_weights


def test_optim_with_equal_weighted_sample_weight(X, risk_measure):
    """No sample weight and equal-weighted sample weight should give the same result"""
    ref = HierarchicalRiskParity(risk_measure=risk_measure)
    ref.fit(X)

    model = HierarchicalRiskParity(
        risk_measure=risk_measure, prior_estimator=EntropyPooling()
    )
    model.fit(X)

    np.testing.assert_almost_equal(model.weights_, ref.weights_, 6)


@pytest.mark.parametrize(
    "risk_measure,view_params,expected_weights",
    [
        (
            RiskMeasure.CVAR,
            dict(cvar_views=["PG == 0.07"]),
            [
                0.03027,
                0.02001,
                0.02843,
                0.0395,
                0.06506,
                0.05026,
                0.05525,
                0.07618,
                0.03398,
                0.02855,
                0.06198,
                0.09286,
                0.05971,
                0.02579,
                0.03679,
                0.05522,
                0.06274,
                0.03062,
                0.06962,
                0.07718,
            ],
        ),
        (
            RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
            dict(variance_views=["PG == 0.005"]),
            [
                0.0277,
                0.02219,
                0.02003,
                0.04551,
                0.06187,
                0.05937,
                0.06236,
                0.08268,
                0.02079,
                0.04072,
                0.05868,
                0.11017,
                0.05035,
                0.02928,
                0.03423,
                0.05856,
                0.03092,
                0.03031,
                0.06729,
                0.08699,
            ],
        ),
        (
            RiskMeasure.VARIANCE,
            dict(variance_views=["PG == 0.0005"]),
            [
                0.02842,
                0.01042,
                0.01369,
                0.03213,
                0.04311,
                0.0486,
                0.06607,
                0.10018,
                0.01568,
                0.06346,
                0.05743,
                0.12451,
                0.04033,
                0.03559,
                0.0416,
                0.06218,
                0.01176,
                0.0317,
                0.08565,
                0.0875,
            ],
        ),
        (
            RiskMeasure.STANDARD_DEVIATION,
            dict(variance_views=["PG == 0.0005"]),
            [
                0.03053,
                0.01848,
                0.02181,
                0.0464,
                0.05477,
                0.05728,
                0.06654,
                0.08196,
                0.02334,
                0.04593,
                0.06176,
                0.09094,
                0.05255,
                0.0344,
                0.03727,
                0.06489,
                0.02751,
                0.03254,
                0.07604,
                0.07504,
            ],
        ),
        (
            RiskMeasure.SEMI_DEVIATION,
            dict(variance_views=["PG == 0.0005"]),
            [
                0.02992,
                0.01775,
                0.0269,
                0.04208,
                0.05807,
                0.05386,
                0.05962,
                0.08176,
                0.03087,
                0.0359,
                0.06373,
                0.08529,
                0.05585,
                0.03192,
                0.03929,
                0.06744,
                0.03824,
                0.03242,
                0.07893,
                0.07014,
            ],
        ),
        (
            RiskMeasure.SEMI_VARIANCE,
            dict(variance_views=["PG == 0.0005"]),
            [
                0.02918,
                0.01027,
                0.02065,
                0.02768,
                0.04791,
                0.04389,
                0.05557,
                0.10503,
                0.02721,
                0.0383,
                0.06376,
                0.11418,
                0.04856,
                0.03028,
                0.04887,
                0.06736,
                0.02149,
                0.03327,
                0.09425,
                0.07229,
            ],
        ),
        (
            RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
            dict(variance_views=["PG == 0.005"]),
            [
                0.0277,
                0.02219,
                0.02003,
                0.04551,
                0.06187,
                0.05937,
                0.06236,
                0.08268,
                0.02079,
                0.04072,
                0.05868,
                0.11017,
                0.05035,
                0.02928,
                0.03423,
                0.05856,
                0.03092,
                0.03031,
                0.06729,
                0.08699,
            ],
        ),
    ],
)
def test_sample_weight(X, risk_measure, view_params, expected_weights):
    ref = HierarchicalRiskParity(risk_measure=risk_measure)
    ref.fit(X)

    model = clone(ref)
    model = model.set_params(prior_estimator=EntropyPooling(**view_params))
    model.fit(X)

    assert model.weights_[15] < ref.weights_[15]

    np.testing.assert_almost_equal(model.weights_, expected_weights, 5)

    ref_ptf = ref.predict(X)
    ptf = model.predict(X)

    assert getattr(ref_ptf, risk_measure.value) < getattr(ptf, risk_measure.value)

    sample_weight = model.prior_estimator_.return_distribution_.sample_weight

    ref_ptf.sample_weight = sample_weight
    ptf.sample_weight = sample_weight

    assert getattr(ref_ptf, risk_measure.value) > getattr(ptf, risk_measure.value)
