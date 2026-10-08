"""Multiple testing module."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.utils as sku

from skfolio.typing import ArrayLike, FloatArray, IntArray

__all__ = ["MultipleTestingResult", "multiple_testing_hurdle"]


_CRITERIA = ("false_discovery_rate", "odds_ratio")


@dataclass(frozen=True)
class MultipleTestingResult:
    """Result of a multiple-testing hurdle estimation.

    Attributes
    ----------
    hurdle : float
        The selected absolute t-statistic hurdle.

    false_discovery_rate : float
        Type I error rate at `hurdle`: the expected share of the trials declared
        significant that are in fact null.

    miss_rate : float
        Type II error rate at `hurdle`, defined as the false omission rate: the
        expected share of the trials declared insignificant that are in fact
        non-null.

    odds_ratio : float
        Expected number of false discoveries per miss at `hurdle`.

    hurdles : ndarray of shape (n_hurdles,)
        The grid of candidate hurdles.

    false_discovery_rates : ndarray of shape (n_hurdles,)
        Type I error rate at each candidate hurdle.

    miss_rates : ndarray of shape (n_hurdles,)
        Type II error rate at each candidate hurdle.

    odds_ratios : ndarray of shape (n_hurdles,)
        Odds ratio at each candidate hurdle.

    t_statistics : ndarray of shape (n_trials,)
        Observed t-statistic of each trial.

    n_observations : int
        Number of observations per trial.

    n_trials : int
        Number of trials in the panel.

    non_null_ratio : float
        The assumed proportion of non-null trials used to produce the result.

    n_non_null : int
        Number of trials treated as non-null, `round(non_null_ratio * n_trials)`.

    criterion : str
        The criterion that the hurdle was selected against.

    target : float
        The targeted value of that criterion.
    """

    hurdle: float
    false_discovery_rate: float
    miss_rate: float
    odds_ratio: float
    hurdles: FloatArray = field(repr=False)
    false_discovery_rates: FloatArray = field(repr=False)
    miss_rates: FloatArray = field(repr=False)
    odds_ratios: FloatArray = field(repr=False)
    t_statistics: FloatArray = field(repr=False)
    n_observations: int
    n_trials: int
    non_null_ratio: float
    n_non_null: int
    criterion: str
    target: float

    @property
    def selected(self) -> IntArray:
        """Indices of the trials whose absolute t-statistic reaches `hurdle`."""
        return np.flatnonzero(np.abs(self.t_statistics) >= self.hurdle)

    def summary(self) -> pd.Series:
        """Summary of the result.

        Returns
        -------
        summary : series
            The result summary.
        """
        return pd.Series(
            {
                "Hurdle": self.hurdle,
                "Type I error (FDR)": self.false_discovery_rate,
                "Type II error (miss rate)": self.miss_rate,
                "Odds ratio": self.odds_ratio,
                "Selected trials": len(self.selected),
                "Trials": self.n_trials,
                "Observations": self.n_observations,
                "Assumed non-null ratio": self.non_null_ratio,
                "Criterion": self.criterion,
                "Target": self.target,
            }
        )


def _draw_indices(
    rng: np.random.RandomState,
    n_observations: int,
    n_samples: int,
    block_size: float | None,
) -> IntArray:
    """Draw `n_samples` resampled time indices of length `n_observations`.

    The same indices are applied to every trial, so the cross-sectional dependence
    between trials is carried into the resample. With `block_size`, blocks of
    consecutive observations are drawn with a geometric length, which additionally
    carries the serial dependence within each trial.
    """
    indices = rng.randint(n_observations, size=(n_samples, n_observations))
    if block_size is not None:
        # Stationary bootstrap: continue the previous block with probability
        # 1 - 1/block_size, otherwise start a new one at a fresh random position.
        keep = rng.random_sample((n_samples, n_observations)) >= 1.0 / block_size
        for j in range(1, n_observations):
            indices[:, j] = np.where(
                keep[:, j], (indices[:, j - 1] + 1) % n_observations, indices[:, j]
            )
    return indices


def _resampled_t_statistics(
    values: FloatArray, squares: FloatArray, indices: IntArray
) -> FloatArray:
    """t-statistics of every trial under each resample, from resampling multiplicities.

    Indexing the panel would materialise an array of shape
    (n_samples, n_observations, n_trials). Only the sum and the sum of squares of
    each resample are needed, and a resample is fully described by how many times
    each observation was drawn, so the counts are accumulated instead and both sums
    follow from a single matrix product. Memory is then independent of `n_samples`
    times `n_trials`.
    """
    n_samples, n_observations = indices.shape
    offsets = np.arange(n_samples) * n_observations
    counts = np.bincount(
        (indices + offsets[:, None]).ravel(), minlength=n_samples * n_observations
    ).reshape(n_samples, n_observations)
    counts = counts.astype(float)

    total = counts @ values
    total_squares = counts @ squares
    mean = total / n_observations
    variance = (total_squares - n_observations * mean**2) / (n_observations - 1)
    scale = np.sqrt(np.maximum(variance, 0.0) / n_observations)
    return np.divide(mean, scale, out=np.zeros_like(mean), where=scale > 0)


def _exceedance_counts(
    absolute_t: FloatArray, hurdles: FloatArray, n_hurdles: int
) -> IntArray:
    """Number of trials reaching each hurdle, for every resample.

    `absolute_t` has shape (n_samples, n_trials) and the result (n_samples,
    n_hurdles). Comparing every trial against every hurdle would build a
    three-dimensional array; instead each t-statistic is bucketed into the grid
    once and the buckets are cumulated from the top.
    """
    n_samples = absolute_t.shape[0]
    if absolute_t.shape[1] == 0:
        return np.zeros((n_samples, n_hurdles), dtype=int)
    # Largest hurdle not above the statistic; hurdles[0] is 0 so this is never -1.
    bucket = np.searchsorted(hurdles, absolute_t, side="right") - 1
    offsets = np.arange(n_samples) * n_hurdles
    histogram = np.bincount(
        (bucket + offsets[:, None]).ravel(), minlength=n_samples * n_hurdles
    ).reshape(n_samples, n_hurdles)
    return np.cumsum(histogram[:, ::-1], axis=1)[:, ::-1]


def multiple_testing_hurdle(
    X: ArrayLike,
    non_null_ratio: float,
    target: float = 0.05,
    criterion: str = "false_discovery_rate",
    hurdles: npt.ArrayLike | None = None,
    n_perturbations: int = 100,
    n_simulations: int = 100,
    block_size: float | None = None,
    random_state: int | np.random.RandomState | None = None,
) -> MultipleTestingResult:
    r"""Compute the t-statistic hurdle implied by a panel of trials, by double bootstrap.

    When many strategies are tried and the best is kept, the conventional hurdle of a
    single test no longer controls anything, and a fixed convention such as
    :math:`|t| > 3` does not adapt to how many trials were run or to how they are
    correlated. Harvey and Liu [1]_ estimate the hurdle from the panel of trials
    itself, and report both error rates it implies.

    The procedure resamples the panel twice. The outer resample ranks the trials and
    fixes which of them are treated as non-null, so that each inner resample is drawn
    from a population whose truth is known by construction and both false discoveries
    and misses can be counted:

    1. Resample the observations of `X` and compute the t-statistic of every trial in
       that resample.
    2. Rank the trials by those t-statistics. The top `non_null_ratio` are made
       non-null, by shifting each of them, in the original panel, to the mean it had
       in the resample. Every other trial is shifted to a zero mean.
    3. Resample the observations of that panel `n_simulations` times, and at each
       candidate hurdle count the true and false positives and negatives against the
       truth fixed in step 2.
    4. Repeat `n_perturbations` times and average the error rates.

    Taking the effect sizes in step 2 from the resample rather than from the original
    panel matters: the highest trials of the original panel are the winners of the
    search, so their in-sample means are inflated by the selection itself.

    Three error rates are averaged over the `n_perturbations` x `n_simulations`
    resamples, with :math:`FP`, :math:`TP`, :math:`FN` and :math:`TN` the counts of
    false and true positives and negatives at a given hurdle:

    .. math::

        \text{Type I} = E\left[\frac{FP}{FP + TP}\right], \quad
        \text{Type II} = E\left[\frac{FN}{FN + TN}\right], \quad
        \text{odds ratio} = E\left[\frac{FP}{FN}\right]

    Type I is the false discovery rate. Type II is the false omission rate, the share
    of the trials declared insignificant that were in fact non-null; it is the
    counterpart of the false discovery rate rather than one minus the power. The odds
    ratio prices one error against the other: a hurdle targeting an odds ratio of
    :math:`1/k` is the hurdle to use when a false discovery is judged :math:`k` times
    as costly as a miss.

    Parameters
    ----------
    X : array-like of shape (n_observations, n_trials)
        Panel of trial returns: one column per trial that the search actually
        evaluated, sharing a common time index. The multiplicity is taken from the
        number of columns, so passing only the retained trials understates it.

    non_null_ratio : float
        Assumed proportion :math:`p_0` of the trials that are genuinely non-null, in
        [0, 1). It is an assumption rather than an estimate, and the hurdle depends
        on it, so report the hurdle across the plausible range rather than at a
        single value (see the Examples section).

    target : float, default=0.05
        Targeted value of `criterion`.

    criterion : str, default="false_discovery_rate"
        The error rate that `target` applies to, either "false_discovery_rate" or
        "odds_ratio".

    hurdles : array-like of shape (n_hurdles,), optional
        Increasing grid of candidate absolute t-statistic hurdles, starting at zero.
        The default is 0 to 6 in steps of 0.05.

    n_perturbations : int, default=100
        Number of outer resamples :math:`I`.

    n_simulations : int, default=100
        Number of inner resamples :math:`J` per outer resample.

    block_size : float, optional
        Average block length of a stationary bootstrap, which preserves serial
        dependence within each trial. The default (`None`) resamples observations
        independently, as in [1]_. `skfolio.utils.bootstrap.optimal_block_size`
        estimates a block size from the data.

    random_state : int, RandomState instance or None, default=None
        Seed or random state to ensure reproducibility.

    Returns
    -------
    result : MultipleTestingResult
        The selected hurdle, the three error rates at that hurdle and their full
        curves over the candidate grid.

    Notes
    -----
    The hurdle returned is the smallest candidate such that every stricter candidate
    also meets the target, so that a target met only locally is not selected.

    With `non_null_ratio` at zero no true positive is possible, the false discovery
    rate is one as soon as any null trial clears the hurdle, and the Type I error
    rate therefore becomes the family-wise error rate. The miss rate and the odds
    ratio are degenerate in that case and are not informative.

    Cost is `n_perturbations` x `n_simulations` resamples of the panel.

    Examples
    --------
    >>> import numpy as np
    >>> from skfolio.model_selection import multiple_testing_hurdle
    >>> rng = np.random.default_rng(42)
    >>> X = rng.normal(size=(240, 40))
    >>> X[:, :2] += 3.5 / np.sqrt(240)  # two trials with a genuine edge
    >>> result = multiple_testing_hurdle(X, non_null_ratio=0.05, random_state=0)
    >>> print(f"{result.hurdle:.2f}")
    3.05

    The hurdle depends on the assumed proportion of non-null trials, and so does the
    miss rate it buys, so report them across a plausible range rather than at one
    value:

    >>> for p in [0.0, 0.05, 0.20]:
    ...     res = multiple_testing_hurdle(X, non_null_ratio=p, random_state=0)
    ...     print(f"{p:.0%} {res.hurdle:.2f} {res.miss_rate:.3f}")
    0% 3.30 0.000
    5% 3.05 0.013
    20% 2.85 0.145

    References
    ----------
    .. [1] "False (and Missed) Discoveries in Financial Economics".
        Harvey, C. R. and Liu, Y. (2020).
        The Journal of Finance, 75(5), 2503-2553.

    See Also
    --------
    skfolio.measures.deflated_sharpe_ratio
    skfolio.utils.bootstrap.optimal_block_size
    """
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {_CRITERIA}, got {criterion!r}")
    if not 0.0 <= non_null_ratio < 1.0:
        raise ValueError(
            f"non_null_ratio must be in [0, 1), got {non_null_ratio}. It is the "
            "assumed proportion of non-null trials, not a p-value."
        )
    if target <= 0:
        raise ValueError(f"target must be strictly positive, got {target}")
    if n_perturbations < 1 or n_simulations < 1:
        raise ValueError("n_perturbations and n_simulations must be strictly positive")

    X = np.asarray(sku.check_array(X, ensure_2d=True, dtype=float))
    n_observations, n_trials = X.shape
    if n_observations < 3:
        raise ValueError(
            f"X must have at least 3 observations, got {n_observations}. The "
            "t-statistic of a trial is not defined below that."
        )

    if hurdles is None:
        hurdles = np.arange(0.0, 6.0 + 1e-9, 0.05)
    else:
        hurdles = np.asarray(hurdles, dtype=float)
        if hurdles.ndim != 1 or hurdles.size == 0:
            raise ValueError("hurdles must be a non-empty one-dimensional array")
        if np.any(np.diff(hurdles) <= 0):
            raise ValueError("hurdles must be strictly increasing")
        if hurdles[0] != 0.0:
            raise ValueError("hurdles must start at zero")
    n_hurdles = hurdles.size

    rng = sku.check_random_state(random_state)

    n_non_null = round(non_null_ratio * n_trials)
    non_null_mask = np.zeros(n_trials, dtype=bool)

    observed_mean = X.mean(axis=0)
    observed_std = X.std(axis=0, ddof=1)
    observed_scale = observed_std / np.sqrt(n_observations)
    t_statistics = np.divide(
        observed_mean,
        observed_scale,
        out=np.zeros(n_trials),
        where=observed_scale > 0,
    )
    centered = X - observed_mean

    false_discovery_rates = np.zeros(n_hurdles)
    miss_rates = np.zeros(n_hurdles)
    odds_ratios = np.zeros(n_hurdles)

    for _ in range(n_perturbations):
        # Step 1: rank the trials on a resample of the panel.
        outer_indices = _draw_indices(rng, n_observations, 1, block_size)[0]
        resampled = X[outer_indices]
        resampled_mean = resampled.mean(axis=0)
        resampled_scale = resampled.std(axis=0, ddof=1) / np.sqrt(n_observations)
        resampled_t = np.divide(
            resampled_mean,
            resampled_scale,
            out=np.zeros(n_trials),
            where=resampled_scale > 0,
        )

        # Step 2: the top trials of that ranking become the non-null ones, carrying
        # the means they had in the resample; every other trial is made null.
        non_null_mask[:] = False
        if n_non_null:
            non_null_mask[np.argsort(-resampled_t, kind="stable")[:n_non_null]] = True
        pseudo = centered + np.where(non_null_mask, resampled_mean, 0.0)

        # Step 3: resample that panel and count outcomes against the known truth.
        inner_indices = _draw_indices(rng, n_observations, n_simulations, block_size)
        inner_t = _resampled_t_statistics(pseudo, pseudo**2, inner_indices)
        absolute_t = np.abs(inner_t)

        positives = _exceedance_counts(absolute_t, hurdles, n_hurdles)
        true_positives = _exceedance_counts(
            absolute_t[:, non_null_mask], hurdles, n_hurdles
        )
        false_positives = positives - true_positives
        false_negatives = n_non_null - true_positives
        true_negatives = n_trials - n_non_null - false_positives

        declared = false_positives + true_positives
        undeclared = false_negatives + true_negatives
        false_discovery_rates += np.divide(
            false_positives,
            declared,
            out=np.zeros(declared.shape),
            where=declared > 0,
        ).sum(axis=0)
        miss_rates += np.divide(
            false_negatives,
            undeclared,
            out=np.zeros(undeclared.shape),
            where=undeclared > 0,
        ).sum(axis=0)
        odds_ratios += np.divide(
            false_positives,
            false_negatives,
            out=np.zeros(false_negatives.shape),
            where=false_negatives > 0,
        ).sum(axis=0)

    n_draws = n_perturbations * n_simulations
    false_discovery_rates /= n_draws
    miss_rates /= n_draws
    odds_ratios /= n_draws

    curve = (
        false_discovery_rates if criterion == "false_discovery_rate" else odds_ratios
    )
    # The smallest hurdle such that it and every stricter hurdle meet the target.
    meets = np.maximum.accumulate(curve[::-1])[::-1] <= target
    index = int(np.argmax(meets)) if meets.any() else n_hurdles - 1

    return MultipleTestingResult(
        hurdle=float(hurdles[index]),
        false_discovery_rate=float(false_discovery_rates[index]),
        miss_rate=float(miss_rates[index]),
        odds_ratio=float(odds_ratios[index]),
        hurdles=hurdles,
        false_discovery_rates=false_discovery_rates,
        miss_rates=miss_rates,
        odds_ratios=odds_ratios,
        t_statistics=t_statistics,
        n_observations=n_observations,
        n_trials=n_trials,
        non_null_ratio=non_null_ratio,
        n_non_null=n_non_null,
        criterion=criterion,
        target=target,
    )
