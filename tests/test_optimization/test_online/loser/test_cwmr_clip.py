import numpy as np

from skfolio.optimization.online._loser import FTLStrategy, FollowTheLoser


def test_cwmr_variance_clipping_bounds():
    # Construct a CWMR instance with tight variance bounds
    est = FollowTheLoser(
        strategy=FTLStrategy.CWMR,
        cwmr_sigma0=1.0,
        cwmr_min_var=0.05,
        cwmr_max_var=0.1,
        epsilon=1.0,
    )
    x = np.array([0.01, -0.01, 0.0], dtype=float)
    est.partial_fit(x)

    diag = est._cwmr_Sdiag
    assert diag is not None
    assert np.all(diag >= 0.05 - 1e-12)
    assert np.all(diag <= 0.1 + 1e-12)
