"""Helpers for alpha tests on :class:`~skfolio.containers.AssetPanel`."""

from __future__ import annotations

import numpy as np

from skfolio._constants import _BENCHMARK_WEIGHTS, _IDIO_RETURNS, _IDIO_VARIANCES
from skfolio.containers import AssetPanel

_BW = _BENCHMARK_WEIGHTS
_IR = _IDIO_RETURNS
_IV = _IDIO_VARIANCES


def _renorm_benchmark_rows(bw: np.ndarray) -> np.ndarray:
    row_sums = bw.sum(axis=1, keepdims=True)
    return np.divide(bw, row_sums, out=np.zeros_like(bw), where=row_sums > 0)


def apply_idio_nan_exclusions(
    panel: AssetPanel,
    ticker: str,
    *,
    rows: slice | None = None,
    horizon: int = 3,
) -> None:
    """Inject NaN idio returns for ``ticker`` while keeping estimators well posed.

    Zeros matching `benchmark_weights` (then row-renormalizes). Sets `idio_variances`
    to `NaN` where inverse-variance WLS must skip pairs with undefined forward targets.
    Use `rows=None` to apply the pattern to the full time series for that asset.
    """
    j = int(np.flatnonzero(panel.asset_names == ticker)[0])
    bw = np.asarray(panel[_BW], dtype=float).copy()

    if rows is None:
        panel[_IR][:, j] = np.nan
        bw[:, j] = 0.0
        panel[_IV][:, j] = np.nan
    else:
        panel[_IR][rows, j] = np.nan
        bw[rows, j] = 0.0
        r0 = 0 if rows.start is None else int(rows.start)
        r1 = panel.n_observations if rows.stop is None else int(rows.stop)
        last = r1 - 1
        lo = max(0, r0 - horizon)
        hi = min(panel.n_observations - 1, last - 1)
        if hi >= lo:
            panel[_IV][lo : hi + 1, j] = np.nan

    panel[_BW] = _renorm_benchmark_rows(bw)
