import numpy as np

from skfolio.optimization.online._loser import FollowTheLoser


def test_mean_reversion_passive_aggressive(X_small):
    m_olmar = FollowTheLoser(
        strategy="olmar",
        olmar_predictor="sma",
        olmar_window=2,
        epsilon=2.0,
        update_mode="pa",
    )
    m_pamr = FollowTheLoser(strategy="pamr", epsilon=1.0, update_mode="pa")
    m_cwmr = FollowTheLoser(
        strategy="cwmr",
        epsilon=1.0,
        cwmr_eta=0.95,
        cwmr_sigma0=0.5,
        update_mode="pa",
    )

    for m in (m_olmar, m_pamr, m_cwmr):
        m.fit(X_small)
        assert m.all_weights_.shape == X_small.shape
        assert np.allclose(m.all_weights_.sum(axis=1), 1.0, atol=1e-9)


def test_cwmr_md_runs_and_produces_portfolios():
    X = np.array([[0.0, 0.0], [0.05, -0.03]], dtype=float)
    m = FollowTheLoser(
        strategy="cwmr",
        epsilon=1.0,
        cwmr_eta=0.95,
        cwmr_sigma0=0.5,
        update_mode="md",
    )
    m.fit(X)
    assert m.all_weights_.shape == X.shape
    assert np.allclose(m.all_weights_.sum(axis=1), 1.0, atol=1e-9)
