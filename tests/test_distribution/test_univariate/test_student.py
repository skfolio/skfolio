import numpy as np
import scipy.stats as scs

from skfolio.datasets import load_sp500_dataset
from skfolio.distribution import StudentT
from skfolio.preprocessing import prices_to_returns


def test():
    X = prices_to_returns(load_sp500_dataset())
    returns = np.asarray(X)[:, [0]]

    model = StudentT()
    model.fit(returns)
    model.score(returns)
    model.sample(3)

    ref = scs.t
    params = ref.fit(returns)
    ref_frozen = scs.t(*params)
    ref_frozen.rvs(3)
    ref_frozen.pdf(returns)
