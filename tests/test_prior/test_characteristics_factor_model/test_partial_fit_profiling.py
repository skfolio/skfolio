"""Profile CharacteristicsFactorModel: batch fit vs streaming partial_fit.

Uses a realistic multi-factor setup (global + industry + style factors,
500 assets, basket-neutral constraints) to identify bottlenecks that match
the real workload.

Run with::

    uv run python -m tests.test_prior.test_characteristics_factor_model.test_partial_fit_profiling
"""

from __future__ import annotations

import time
from collections import defaultdict
from functools import wraps

import sklearn.utils.validation as skv

from skfolio.datasets import make_synthetic_characteristics
from skfolio.descriptor import (
    AssetTurnover,
    BookLeverage,
    BookToPrice,
    DebtToAssets,
    EWMarketBeta,
    EWMomentum,
    LogMarketCap,
    ReturnOnAssets,
    SalesToPrice,
)
from skfolio.factor_exposure import (
    DerivedFactor,
    FixedWeightedFactor,
    GlobalFactor,
    OneHotCategoricalFactors,
)
from skfolio.prior import CharacteristicsFactorModel

N_OBS = 600
N_ASSETS = 500
SEED = 123

METHODS_TO_PROFILE = [
    "_validate_data",
    "_initialize",
    "_compute_benchmark_weights",
    "_compute_factors",
    "_validate_exposure_coverage",
    "_cross_sectional_regression",
    "_compute_factor_returns_dist",
    "_compute_idio_variances",
    "_compute_idio_covariance",
    "_compute_alpha",
    "_decompose_alpha",
]


class _Timer:
    """Accumulates wall-clock time and call counts per label."""

    def __init__(self):
        self.cumulative: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def reset(self):
        self.cumulative.clear()
        self.counts.clear()

    def wrap(self, label, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            self.cumulative[label] += time.perf_counter() - t0
            self.counts[label] += 1
            return result

        return wrapper

    def report(self, title: str, wall: float):
        total = sum(self.cumulative.get(n, 0.0) for n in METHODS_TO_PROFILE)
        w = 70
        print(f"\n{'=' * w}")
        print(f"  {title}")
        print(f"  wall: {wall:.4f}s | instrumented: {total:.4f}s")
        print(f"{'=' * w}")
        print(f"  {'Method':<40s} {'Calls':>6s} {'Time (s)':>10s} {'%':>6s}")
        print(f"  {'-' * 40} {'-' * 6} {'-' * 10} {'-' * 6}")
        for name in METHODS_TO_PROFILE:
            t = self.cumulative.get(name, 0.0)
            c = self.counts.get(name, 0)
            pct = 100 * t / total if total > 0 else 0
            print(f"  {name:<40s} {c:>6d} {t:>10.4f} {pct:>5.1f}%")

        sub_keys = sorted(k for k in self.cumulative if k not in METHODS_TO_PROFILE)
        if sub_keys:
            print()
            print("  Sub-estimator breakdown (nested in parents above)")
            print(f"  {'-' * 64}")
            for label in sub_keys:
                t = self.cumulative[label]
                c = self.counts[label]
                print(f"    {label:<42s} {c:>6d} {t:>10.4f}")
        print()


def _install_hooks(model: CharacteristicsFactorModel, timer: _Timer):
    for name in METHODS_TO_PROFILE:
        original = getattr(model, name)
        setattr(model, name, timer.wrap(name, original))


def _install_sub_estimator_hooks(model: CharacteristicsFactorModel, timer: _Timer):
    """Hook sub-estimator methods once bootstrap is complete."""
    targets = {
        "factor_prior.partial_fit": (model.factor_prior_estimator_, "partial_fit"),
        "idio_var.partial_fit": (model.idio_variance_estimator_, "partial_fit"),
    }
    fp = model.factor_prior_estimator_
    if hasattr(fp, "mu_estimator_"):
        targets["factor_prior.mu.partial_fit"] = (fp.mu_estimator_, "partial_fit")
    if hasattr(fp, "covariance_estimator_"):
        targets["factor_prior.cov.partial_fit"] = (
            fp.covariance_estimator_,
            "partial_fit",
        )

    for name, factor_est in model.named_factor_estimators_.items():
        method = "partial_fit_transform"
        if hasattr(factor_est, method):
            targets[f"factor[{name}].pft"] = (factor_est, method)
        elif hasattr(factor_est, "fit_transform"):
            targets[f"factor[{name}].ft"] = (factor_est, "fit_transform")

    targets["_get_dependency_layers"] = (model, "_get_dependency_layers")
    targets["cs_regressor.fit"] = (model.cs_regressor_, "fit")
    targets["cs_regressor.predict"] = (model.cs_regressor_, "predict")

    for label, (est, method_name) in targets.items():
        original = getattr(est, method_name)
        setattr(est, method_name, timer.wrap(label, original))


def _install_validate_hooks(timer: _Timer):
    """Patch sklearn.validate_data at import sites to measure total cost."""
    import skfolio.prior._characteristics_factor_model as cfm_mod

    original_skv = skv.validate_data
    wrapped_skv = timer.wrap("sklearn.validate_data (all sites)", original_skv)
    skv.validate_data = wrapped_skv
    cfm_mod.skv.validate_data = wrapped_skv

    def restore():
        skv.validate_data = original_skv
        cfm_mod.skv.validate_data = original_skv

    return restore


def _make_data():
    print(f"  Generating synthetic data ({N_ASSETS} assets, ~{N_OBS} dates)...")
    panel = make_synthetic_characteristics(
        n_assets=N_ASSETS,
        n_observations=N_OBS,
        random_state=SEED,
        missing_ratio=0.01,
        delisting_proba=0.10,
        late_listing_proba=0.10,
    )
    panel.ffill("market_cap").bfill("market_cap")
    X = panel.to_dataframe(fields="returns")
    n = min(N_OBS, len(X))
    return panel[:n], X.iloc[:n]


def _make_model():
    global_factor = GlobalFactor(family="market")
    industry_factors = OneHotCategoricalFactors(category="industry", family="industry")

    beta_factor = FixedWeightedFactor(descriptors=[("market_beta", EWMarketBeta())])
    size_factor = FixedWeightedFactor(descriptors=[("log_mcap", LogMarketCap())])
    momentum_factor = FixedWeightedFactor(descriptors=[("momentum", EWMomentum())])

    value_factor = FixedWeightedFactor(
        descriptors=[
            ("book_to_price", BookToPrice()),
            ("sales_to_price", SalesToPrice()),
        ]
    )

    profitability_factor = FixedWeightedFactor(
        descriptors=[
            ("asset_turnover", AssetTurnover()),
            ("return_on_assets", ReturnOnAssets()),
        ]
    )

    leverage_factor = FixedWeightedFactor(
        descriptors=[
            ("debt_to_assets", DebtToAssets()),
            ("book_leverage", BookLeverage()),
        ]
    )

    factors = [
        ("global", global_factor),
        ("industry", industry_factors),
        ("beta", beta_factor),
        ("size", size_factor),
        ("momentum", momentum_factor),
        ("value", value_factor),
        ("profitability", profitability_factor),
        ("leverage", leverage_factor),
        ("non_linear_size", DerivedFactor(source="size", func=lambda x: x**3)),
    ]

    return CharacteristicsFactorModel(
        factors=factors,
        constrained_families=[("industry", "Energy")],
        n_jobs=1,
    )


def _run_batch(panel, X):
    timer = _Timer()
    model = _make_model()
    _install_hooks(model, timer)

    t0 = time.perf_counter()
    model.fit(X, characteristics=panel)
    wall = time.perf_counter() - t0

    timer.report(f"BATCH FIT (1 call, {len(X)} obs)", wall)
    return wall


def _run_streaming(panel, X):
    timer = _Timer()
    model = _make_model()
    _install_hooks(model, timer)
    restore = _install_validate_hooks(timer)

    try:
        t0 = time.perf_counter()
        # Bootstrap: need 2 calls for exposure_lag=1
        model.partial_fit(X.iloc[:1], characteristics=panel[:1])
        model.partial_fit(X.iloc[1:2], characteristics=panel[1:2])
        _install_sub_estimator_hooks(model, timer)

        for i in range(2, len(X)):
            model.partial_fit(X.iloc[i : i + 1], characteristics=panel[i : i + 1])
        wall = time.perf_counter() - t0
    finally:
        restore()

    timer.report(f"STREAMING PARTIAL_FIT ({len(X)} calls, 1 obs each)", wall)
    return wall


def _run_scaling_test(panel, X, steps=(200, 400, 600)):
    """Run streaming partial_fit at increasing observation counts to check linearity."""
    print(f"\n{'=' * 70}")
    print("  SCALING TEST (streaming partial_fit)")
    print(f"{'=' * 70}")

    results = []
    for n in steps:
        n = min(n, len(X))
        model = _make_model()
        t0 = time.perf_counter()
        for i in range(n):
            model.partial_fit(X.iloc[i : i + 1], characteristics=panel[i : i + 1])
        wall = time.perf_counter() - t0
        results.append((n, wall))
        print(f"  {n:>5d} obs: {wall:>8.2f}s")

    if len(results) >= 2:
        n0, t0 = results[0]
        print("\n  Ratios (should be ~linear, i.e. close to N/N0):")
        for n, t in results[1:]:
            ratio = t / t0
            expected = n / n0
            print(f"    {n}/{n0} = {ratio:.2f}x  (linear would be {expected:.1f}x)")
    print()


def main():
    print("Profiling CharacteristicsFactorModel")
    print(f"  n_assets={N_ASSETS}, n_obs={N_OBS}, 9 factors (incl. industry one-hot)")
    print("=" * 70)

    panel, X = _make_data()
    actual_obs = len(X)
    print(f"  Actual observations after generation: {actual_obs}")

    n_profile = min(200, actual_obs)
    batch_wall = _run_batch(panel[:n_profile], X.iloc[:n_profile])
    stream_wall = _run_streaming(panel[:n_profile], X.iloc[:n_profile])

    print(f"{'=' * 70}")
    print(f"  Slowdown factor: {stream_wall / batch_wall:.1f}x")
    print(f"{'=' * 70}")

    _run_scaling_test(panel, X)


if __name__ == "__main__":
    main()
