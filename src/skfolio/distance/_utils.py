"""Shared distance conversion utilities."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from skfolio.typing import FloatArray
from skfolio.utils.tools import _validate_positive_real


def _corr_to_distance(
    corr: FloatArray, absolute: bool, power: float
) -> tuple[FloatArray, FloatArray]:
    r"""Transform a correlation matrix to a codependence and distance matrix.

    Some widely used distances are:

        * Standard angular distance = :math:`\sqrt{0.5 \times (1 - corr)}`
        * Absolute angular distance = :math:`\sqrt{1 - |corr|}`
        * Squared angular distance = :math:`\sqrt{1 - corr^2}`


    Parameters
    ----------
    corr : ndarray of shape (n_assets, n_assets)
        Correlation matrix. The input is not modified.

    absolute : bool
        If this is set to True, the absolute transformation is applied to the
        correlation matrix.

    power : float
        Exponent of the power transformation applied to the correlation matrix.
        Must be finite and strictly positive, with an integer value when
        `absolute=False`.

    Returns
    -------
    codependence, distance : tuple[FloatArray, FloatArray]
        Codependence and distance matrices. Codependence retains the dtype of `corr`.

    Raises
    ------
    ValueError
        If `power` is not finite and strictly positive, or has a fractional
        value when `absolute=False`.
    """
    _validate_positive_real(power, "power")
    if not absolute and power % 1 != 0:
        raise ValueError("power must have an integer value when absolute=False.")

    # Copy the correlations so the input stays unchanged.
    codependence = np.copy(corr)
    if absolute:
        np.abs(codependence, out=codependence)

    # Raising to the power of one would only add another pass over the matrix.
    if power != 1:
        np.power(codependence, power, out=codependence)

    # Scaling keeps distances in [0, 1].
    if not absolute and power % 2 == 1:
        scale = np.float64(0.5)
    else:
        scale = np.float64(1.0)
    distance = scale * (1 - codependence)

    # Clip roundoff before taking the square root. Reuse the distance array.
    np.clip(distance, 0.0, 1.0, out=distance)
    np.sqrt(distance, out=distance)
    return codependence, distance
