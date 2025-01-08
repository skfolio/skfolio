import numpy as np

from skfolio.datasets import load_sp500_dataset
from skfolio.distribution import StudentT
from skfolio.preprocessing import prices_to_returns


def test():
    X = prices_to_returns(load_sp500_dataset())
    X = np.asarray(X)

    X = np.asarray(X)[:, [0]]

    model = StudentT()
    model.fit(X)
    model.score_samples(X)
    model.score(X)
    model.bic(X)
    model.cdf(X)
    model.sample(3)
