import numpy as np


def assert_box_budget(
    w: np.ndarray,
    lower: float = 0.0,
    upper: float = 1.0,
    budget: float = 1.0,
    atol=1e-8,
):
    assert np.isfinite(w).all(), "Infinite weights!"
    assert np.all(abs(np.sum(w) - budget) <= 1e-6), "budget constraint not respected"
    assert np.all(np.max(w) - upper <= atol), "box constraint (upper) not respected"
    assert np.all(np.min(w) - lower >= -atol), "box constraint (lower) not respected"


def group_sum(
    weights: np.ndarray, groups: list[list[str]], group_name: str, axis: int
) -> float:
    labels = groups[axis]
    mask = np.array([g == group_name for g in labels], dtype=bool)
    return float(np.sum(weights[mask]))


def make_stationary_returns(T=200, gap=0.01, n=2):
    # Asset 0 dominates: r=[gap, 0, 0, ...]
    X = np.zeros((T, n), dtype=float)
    X[:, 0] = gap
    return X


def assert_simplex_trajectory(W: np.ndarray, atol=1e-9):
    s = np.sum(W, axis=1)
    assert np.all(W >= -1e-12)
    assert np.all(W <= 1 + 1e-12)
    assert np.all(np.abs(s - 1.0) <= atol)
