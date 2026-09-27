"""Entropy mirror-descent kernel and bounded-simplex KL projection.

Structure follows Carlo Nicolini's gradient / mirror-map / update separation.
This is an independent implementation of the standard EG formula, not a copy of
his general OCO engine. Constraint support here is a discussion fast path.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp


def entropy_update(weights, returns, learning_rate, lower, upper):
    """Take an EG step and apply bounds in KL, rather than Euclidean, geometry."""
    relatives = 1 + returns
    relatives /= relatives.max()
    growth = weights @ relatives
    # As in Carlo's entropy map, floor the reference before taking logarithms.
    # This permits numerical recovery from underflow; it is not exact support
    # preservation when an asset has weight zero.
    log_reference = np.log(np.maximum(weights, 1e-16))
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        log_candidate = log_reference + learning_rate * relatives / growth
    return project_entropy(log_candidate, lower, upper)


def project_entropy(log_candidate, lower, upper):
    """Minimize KL(w || candidate) over lower <= w <= upper, sum(w) = 1.

    The KKT solution is clip(exp(log_candidate - lambda), lower, upper).
    One scalar multiplier enforces the budget. Evaluate clipping in log space
    to avoid overflow. All-zero/full-budget boundary cases are explicit.
    """
    if not np.isfinite(log_candidate).all():
        raise ValueError("The entropy update must have finite log weights.")
    log_candidate = log_candidate - logsumexp(log_candidate)
    candidate = np.exp(log_candidate)
    if np.all(candidate >= lower) and np.all(candidate <= upper):
        return candidate
    if lower.sum() == 1:
        return lower.copy()
    if upper.sum() == 1:
        return upper.copy()
    with np.errstate(divide="ignore"):
        log_lower, log_upper = np.log(lower), np.log(upper)

    def allocation(multiplier):
        """Evaluate the clipped exponential allocation for a budget multiplier."""
        return np.exp(np.clip(log_candidate - multiplier, log_lower, log_upper))

    # Bracket the monotone budget residual. On one side every asset reaches its
    # upper bound; on the other every positive floor is reached (zeros underflow).
    positive = upper > 0
    left = np.min(log_candidate[positive] - log_upper[positive]) - 1
    right = np.max(log_candidate) - np.log(np.nextafter(0.0, 1.0)) + 1
    multiplier = brentq(
        lambda value: allocation(value).sum() - 1,
        left,
        right,
        xtol=1e-13,
    )
    return allocation(multiplier)
