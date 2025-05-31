import numpy as np
import pytest

from skfolio import RiskMeasure
from skfolio.distribution import Gaussian, GaussianCopula, VineCopula
from skfolio.optimization import MeanRisk
from skfolio.prior import (
    EmpiricalPrior,
    EntropyPooling,
    FactorModel,
    OpinionPooling,
    SyntheticData,
)


def test_validate_estimators_empty(X):
    model = OpinionPooling(estimators=[])
    with pytest.raises(ValueError, match="Invalid 'estimators' attribute"):
        model.fit(X)


def test_validate_opinion_probabilities_defaults(X):
    model1 = EntropyPooling()
    model2 = EntropyPooling()
    model = OpinionPooling(estimators=[("expert_1", model1), ("expert_2", model2)])
    # internal opinion_probabilities_ set after fit
    model.fit(X)
    assert np.allclose(model.opinion_probabilities_, [0.5, 0.5])


@pytest.mark.parametrize(
    "probs",
    [
        [1.0, 1.0],  # sum >1
        [-0.1, 0.1],  # negative
        [0.6],  # wrong length
    ],
)
def test_validate_opinion_probabilities_errors(X, probs):
    model1 = EntropyPooling()
    model2 = EntropyPooling()
    with pytest.raises(ValueError):
        model = OpinionPooling(
            estimators=[("expert_1", model1), ("expert_2", model2)],
            opinion_probabilities=probs,
        )
        model.fit(X)


def test_prior_error(X):
    model1 = EntropyPooling(prior_estimator=EmpiricalPrior())
    model2 = EntropyPooling()
    with pytest.raises(
        ValueError, match="Cannot set `prior_estimator` on individual estimators"
    ):
        model = OpinionPooling(
            estimators=[("expert_1", model1), ("expert_2", model2)],
        )
        model.fit(X)


@pytest.mark.parametrize("is_linear_pooling", [True, False])
@pytest.mark.parametrize("divergence_penalty", [0.0, 1.0, 10, 100])
@pytest.mark.parametrize("prior_estimator", [None, EmpiricalPrior()])
def test_no_views(X, is_linear_pooling, divergence_penalty, prior_estimator):
    model1 = EntropyPooling()
    model2 = EntropyPooling()

    model = OpinionPooling(
        estimators=[("expert_1", model1), ("expert_2", model2)],
        opinion_probabilities=[0.5, 0.2],
        is_linear_pooling=is_linear_pooling,
        divergence_penalty=divergence_penalty,
        prior_estimator=prior_estimator,
    )
    model.fit(X)
    np.testing.assert_almost_equal(
        model.return_distribution_.sample_weight, np.ones(len(X)) / len(X)
    )


@pytest.mark.parametrize("is_linear_pooling", [True, False])
@pytest.mark.parametrize(
    "divergence_penalty,opinion_probabilities,expected",
    [
        (0.0, [0.5, 0.2, 0.05], [0.5, 0.2, 0.05, 0.25]),
        (1.0, [0.5, 0.2, 0.05], [0.51924177, 0.17172076, 0.05138454, 0.25765294]),
        (10.0, [0.5, 0.2, 0.05], [0.61990552, 0.03700985, 0.05584107, 0.28724355]),
        (1000, [0.5, 0.2, 0.05], [9.9974916e-01, 0.0, 2.90e-06, 2.4793569e-04]),
        (0.0, [0.5, 0.2, 0.3], [0.5, 0.2, 0.3]),
        (1.0, [0.5, 0.2, 0.3], [0.5190896, 0.1721879, 0.3087225]),
    ],
)
def test_opinion_probabilities(
    X, is_linear_pooling, divergence_penalty, opinion_probabilities, expected
):
    model1 = EntropyPooling(mean_views=["AAPL >= 0.04"])
    model2 = EntropyPooling(mean_views=["AAPL <= -0.01"])
    model3 = EntropyPooling(mean_views=["BAC <= 0.00"])

    model = OpinionPooling(
        estimators=[("expert_1", model1), ("expert_2", model2), ("expert_3", model3)],
        opinion_probabilities=opinion_probabilities,
        is_linear_pooling=is_linear_pooling,
        divergence_penalty=divergence_penalty,
    )
    model.fit(X)
    np.testing.assert_almost_equal(model.opinion_probabilities_.sum(), 1.0)
    np.testing.assert_almost_equal(model.opinion_probabilities_, expected)


@pytest.mark.parametrize(
    "is_linear_pooling,expected",
    [
        (True, [0.00017604, 0.00020895, 0.00016989, 0.00021431, 0.00016671]),
        (False, [1.46e-06, 3.42e-06, 4.08e-06, 0.000276901, 2.40e-06]),
    ],
)
def test_is_linear_pooling(X, is_linear_pooling, expected):
    model1 = EntropyPooling(mean_views=["AAPL >= 0.04"])
    model2 = EntropyPooling(mean_views=["AAPL <= -0.01"])
    model3 = EntropyPooling(solver="CLARABEL", variance_views=["BAC <= 0.000001"])

    model = OpinionPooling(
        estimators=[("expert_1", model1), ("expert_2", model2), ("expert_3", model3)],
        opinion_probabilities=[0.5, 0.2, 0.3],
        is_linear_pooling=is_linear_pooling,
    )
    model.fit(X)
    sw = model.return_distribution_.sample_weight

    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(sw.sum(), 1.0)

    np.testing.assert_almost_equal(sw[:5], expected, 5)


def test_factor_model(X, y):
    view_1 = EntropyPooling(mean_views=["QUAL >= 0.04"])
    view_2 = EntropyPooling(mean_views=["QUAL <= -0.01"])

    opinion = OpinionPooling(
        estimators=[("expert_1", view_1), ("expert_2", view_2)],
        opinion_probabilities=[0.5, 0.2],
    )
    model = FactorModel(factor_prior_estimator=opinion)

    model.fit(X, y)

    sw = model.return_distribution_.sample_weight
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)


def test_factor_synthetic_data(X, y):
    factor_synth = SyntheticData(
        n_samples=10_000,
        distribution_estimator=VineCopula(
            log_transform=True,
            marginal_candidates=[Gaussian()],
            copula_candidates=[GaussianCopula()],
        ),
    )
    view_1 = EntropyPooling(mean_views=["QUAL >= 0.004"])
    view_2 = EntropyPooling(mean_views=["QUAL <= -0.001"])

    opinion = OpinionPooling(
        estimators=[("expert_1", view_1), ("expert_2", view_2)],
        opinion_probabilities=[0.5, 0.2],
        prior_estimator=factor_synth,
    )

    model = FactorModel(factor_prior_estimator=opinion)
    model.fit(X, y)

    sw = model.return_distribution_.sample_weight
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)


def test_optimization(X):
    views_1 = EntropyPooling(mean_views=["AAPL >= 0.004"])
    views_2 = EntropyPooling(mean_views=["AAPL <= -0.001"])

    opinion = OpinionPooling(
        estimators=[("expert_1", views_1), ("expert_2", views_2)],
        opinion_probabilities=[0.5, 0.2],
    )

    model = MeanRisk(prior_estimator=opinion, risk_measure=RiskMeasure.CVAR)
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        [
            0.0021,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0165,
            0.11468,
            0.0,
            0.15727,
            0.00624,
            0.10974,
            0.0,
            0.0,
            0.15651,
            0.20453,
            0.01889,
            0.0,
            0.19845,
            0.01509,
        ],
        5,
    )
