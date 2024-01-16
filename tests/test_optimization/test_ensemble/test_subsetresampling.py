import pytest
from skfolio import RiskMeasure
from skfolio.optimization import MeanRisk
from skfolio.optimization.ensemble._bagging import SubsetResampling
from .test_stacking import X, y, X_y


def test_subsetresampling(X, y):
    portfolio = MeanRisk(risk_measure=RiskMeasure.VARIANCE)
    model = SubsetResampling(estimator=portfolio, n_estimators=10, max_features=0.5, n_jobs=-1,
                             random_state=0)
    model.fit(X, y)

    assert hasattr(model, "weights_"), "Model must have the weights_ attribute of fitted portfolios"
    assert model.weights_.shape[0] == X.shape[1], ("Model must have as many weights as the number "
                                                   "of assets ")