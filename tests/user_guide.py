from sklearn.model_selection import train_test_split

from skfolio import (
    RiskMeasure,
)
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import MeanRisk
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()
X = prices_to_returns(X=prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

model = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    efficient_frontier_size=30,
)
model.fit(X_train)
print(model.weights_.shape)

population = model.predict(X_test)
