import numpy as np
from joblib import Parallel, delayed

from skfolio.datasets import load_sp500_dataset
from skfolio.distribution import StudentTCopula, optimal_univariate_dist
from skfolio.preprocessing import prices_to_returns


def test():
    X0 = prices_to_returns(load_sp500_dataset())
    X0 = np.asarray(X0)[:, [0, 1]]
    n_assets = X0.shape[1]
    results = Parallel(n_jobs=-1)(
        delayed(optimal_univariate_dist)(X0[:, [i]]) for i in range(n_assets)
    )
    X = np.hstack([dist.cdf(X0[:, [i]]) for i, dist in enumerate(results)])

    model = StudentTCopula()
    model.fit(X)
    print(model.params_)
