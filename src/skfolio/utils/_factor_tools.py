"""Shared utilities for factor-model estimators."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections import defaultdict

import numpy as np

from skfolio.linear_model import BaseCSLinearModel, CSLinearRegression
from skfolio.preprocessing import CSStandardScaler
from skfolio.typing import FloatArray, ObjArray, StrArray


def _factor_name_maps(
    factor_names: ObjArray, factor_families: ObjArray | None = None
) -> tuple[dict[str, int], dict[str, list[int]]]:
    """Build lookup maps for factor names and factor families.

    Parameters
    ----------
    factor_names : ndarray of shape (n_factors,)
        Factor names.

    factor_families : ndarray of shape (n_factors,), optional
        Family label for each factor. If `None`, the family map is empty.

    Returns
    -------
    factor_to_idx : dict of {str: int}
        Mapping from factor name to factor index.

    family_to_idx : dict of {str: list[int]}
        Mapping from family name to the factor indices in that family.
    """
    factor_to_idx = {v: i for i, v in enumerate(factor_names)}
    family_to_idx = defaultdict(list)
    if factor_families is not None:
        for i, v in enumerate(factor_families):
            family_to_idx[v].append(i)
    return factor_to_idx, dict(family_to_idx)


def _resolve_factor_name(
    name: str, factor_to_idx: dict[str, int], family_to_idx: dict[str, list[int]]
) -> set[int]:
    """Resolve one factor or family name to factor indices.

    Factor names take precedence over family names when the same label appears in both
    maps.

    Parameters
    ----------
    name : str
        Factor name or family name to resolve.

    factor_to_idx : dict of {str: int}
        Mapping from factor name to factor index.

    family_to_idx : dict of {str: list[int]}
        Mapping from family name to factor indices.

    Returns
    -------
    indices : set[int]
        Resolved factor indices.

    Raises
    ------
    ValueError
        If `name` is neither a factor name nor a family name.
    """
    if name in factor_to_idx:
        return {factor_to_idx[name]}
    if name in family_to_idx:
        return set(family_to_idx[name])
    raise ValueError(
        f"'{name}' is neither a factor name nor a family name. "
        f"Available factors: {list(factor_to_idx)}. "
        f"Available families: {list(family_to_idx)}"
    )


def _expand_factor_names(
    names: list[str], factor_to_idx: dict[str, int], family_to_idx: dict[str, list[int]]
) -> list[int]:
    """Expand factor and family names to ordered unique factor indices.

    Each input name may resolve to one factor or to all members of a family.
    Repeated factors are kept only at their first occurrence in the expanded
    sequence.

    Parameters
    ----------
    names : list of str
        Factor names or family names to expand.

    factor_to_idx : dict of {str: int}
        Mapping from factor name to factor index.

    family_to_idx : dict of {str: list[int]}
        Mapping from family name to factor indices.

    Returns
    -------
    indices : list[int]
        Ordered deduplicated factor indices.

    Raises
    ------
    ValueError
        If any name is neither a factor name nor a family name.
    """
    indices = []
    seen = set()
    for name in names:
        for idx in sorted(_resolve_factor_name(name, factor_to_idx, family_to_idx)):
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
    return indices


def _resolve_factor_subset(
    *,
    factor_names: StrArray | ObjArray,
    factor_families: StrArray | ObjArray | None,
    factor_names_to_keep: list[str] | None,
    family_names_to_keep: str | list[str] | None,
) -> tuple[slice | list[int], list[str]]:
    """Resolve a factor subset from explicit factor names or family labels.

    Parameters
    ----------
    factor_names : ndarray of shape (n_factors,)
        Available factor names.

    factor_families : ndarray of shape (n_factors,), optional
        Family label for each factor. Required when
        `family_names_to_keep` is specified.

    factor_names_to_keep : list of str, optional
        Explicit factor names to keep. Takes precedence over
        `family_names_to_keep`.

    family_names_to_keep : str, list of str, optional
        Factor families to keep. `None` includes all factors.

    Returns
    -------
    indices : slice or list of int
        Factor selector. The unfiltered case returns `slice(None)`.

    names : list of str
        Selected factor names in the same order as `indices`.

    Raises
    ------
    ValueError
        If a requested factor or family is unavailable.
    """
    factor_names = np.asarray(factor_names, dtype=str)
    all_names = factor_names.tolist()
    if factor_names_to_keep is None and family_names_to_keep is None:
        return slice(None), all_names

    if factor_names_to_keep is not None:
        missing = set(factor_names_to_keep) - set(all_names)
        if missing:
            raise ValueError(
                f"Unknown factor(s): {sorted(missing)}. Available factors: {all_names}."
            )
        return (
            [all_names.index(factor) for factor in factor_names_to_keep],
            list(factor_names_to_keep),
        )

    if factor_families is None:
        raise ValueError(
            "`family_names_to_keep` was specified but `factor_families` is None."
        )

    factor_families = np.asarray(factor_families, dtype=str)
    if factor_families.shape != factor_names.shape:
        raise ValueError(
            "`factor_families` must have the same shape as `factor_names`, "
            f"got {factor_families.shape} and {factor_names.shape}."
        )
    if isinstance(family_names_to_keep, str):
        family_names_to_keep = [family_names_to_keep]
    available_families = set(factor_families)
    unknown = set(family_names_to_keep) - available_families
    if unknown:
        raise ValueError(
            f"Unknown family/families: {sorted(unknown)}. "
            f"Available families: {sorted(available_families)}."
        )
    indices = [
        i
        for i, factor_family in enumerate(factor_families)
        if factor_family in family_names_to_keep
    ]
    return indices, [all_names[i] for i in indices]


def _cross_sectional_neutralize(
    y: FloatArray,
    x: FloatArray,
    cs_weights: FloatArray,
    cs_regressor: BaseCSLinearModel | None = None,
) -> tuple[FloatArray, FloatArray]:
    r"""Neutralize a panel against cross-sectional explanatory variables.

    For each observation :math:`t`, regress :math:`y_t` on :math:`X_t` across
    assets with cross-sectional weights and return the residuals:

    .. math::

        \tilde{y}_t = y_t - X_t \hat{\beta}_t

    Missing `y` values or rows of `x` with missing features are excluded by
    setting their effective regression weights to zero.

    Parameters
    ----------
    y : ndarray of shape (n_observations, n_assets)
        Cross-sectional response values to neutralize.

    x : ndarray of shape (n_observations, n_assets, n_features)
        Cross-sectional explanatory variables.

    cs_weights : ndarray of shape (n_observations, n_assets)
        Base cross-sectional weights. Entries whose `y` or `x` values are
        missing receive zero effective weight.

    cs_regressor : BaseCSLinearModel, optional
        Cross-sectional linear regressor used for residualization. If `None`,
        `CSLinearRegression(fit_intercept=False)` is used.

    Returns
    -------
    neutralized : ndarray of shape (n_observations, n_assets)
        Cross-sectional residuals. Missing values in `y` or `x` propagate to
        the corresponding residual entries.

    weights : ndarray of shape (n_observations, n_assets)
        Effective regression weights after missing-value exclusion.
    """
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=2)
    cs_weights = np.where(valid, cs_weights, 0.0)
    regressor = (
        CSLinearRegression(fit_intercept=False)
        if cs_regressor is None
        else cs_regressor
    )
    neutralized = y - regressor.fit(x, y, cs_weights=cs_weights).predict(x)
    return neutralized, cs_weights


def _neutralize_scores(
    neutralize_against: list[str],
    scores: FloatArray,
    exposures: FloatArray,
    cs_weights: FloatArray,
    factor_names: ObjArray,
    factor_families: ObjArray | None = None,
) -> FloatArray:
    """Neutralize descriptor scores against selected factor exposures.

    Parameters
    ----------
    neutralize_against : list of str
        Factor names or family names to neutralize each descriptor score against.

    scores : ndarray of shape (n_observations, n_assets, n_descriptors)
        Descriptor score panels. The array is modified in-place.

    exposures : ndarray of shape (n_observations, n_assets, n_factors)
        Factor exposures used as neutralization variables.

    cs_weights : ndarray of shape (n_observations, n_assets)
        Cross-sectional weights for the neutralization regressions.

    factor_names : ndarray of shape (n_factors,)
        Factor names.

    factor_families : ndarray of shape (n_factors,), optional
        Family label for each factor. If provided, `neutralize_against` may contain
        family names.

    Returns
    -------
    scores : ndarray of shape (n_observations, n_assets, n_descriptors)
        The input score array with each descriptor replaced by its neutralized residuals.

    Raises
    ------
    ValueError
        If a neutralization target is neither a factor name nor a family name.
    """
    factor_to_idx, family_to_idx = _factor_name_maps(factor_names, factor_families)
    targets_idx = _expand_factor_names(neutralize_against, factor_to_idx, family_to_idx)
    if len(targets_idx) == 0:
        return scores

    x = exposures[:, :, targets_idx]
    for i in range(scores.shape[2]):
        scores[:, :, i], _ = _cross_sectional_neutralize(
            y=scores[:, :, i], x=x, cs_weights=cs_weights
        )

    return scores


def _neutralize_exposures(
    cs_regressor: BaseCSLinearModel,
    neutralize_against: dict[str, list[str]],
    exposures: FloatArray,
    benchmark_weights: FloatArray,
    factor_names: ObjArray,
    factor_families: ObjArray,
) -> None:
    """Neutralize factor exposures against specified factors or families.

    For each entry in `neutralize_against`, the key's exposures are regressed against
    the target factors and replaced in-place by standardized residuals. Keys and targets
    accept factor names or family names. Entries are processed in insertion order, so
    later entries see exposures already modified by earlier entries.

    Parameters
    ----------
    cs_regressor : BaseCSLinearModel
        Cross-sectional linear regressor used for residualization.

    neutralize_against : dict of {str: list[str]}
        Mapping from factor name or family name to the factor names or family names it
        must be neutralized against. A family key neutralizes each factor in that family
        independently against the same targets.

    exposures : ndarray of shape (n_observations, n_assets, n_factors)
        Factor exposures. Neutralized columns are overwritten in-place.

    benchmark_weights : ndarray of shape (n_observations, n_assets)
        Cross-sectional weights for neutralization and residual scaling.

    factor_names : ndarray of shape (n_factors,)
        Factor names.

    factor_families : ndarray of shape (n_factors,)
        Family label for each factor.

    Raises
    ------
    ValueError
        If a key or target is neither a factor name nor a family name, or if a key
        resolves to factors that overlap with its neutralization targets.
    """
    factor_to_idx, family_to_idx = _factor_name_maps(factor_names, factor_families)

    for key, targets in neutralize_against.items():
        neutralize_idx = _resolve_factor_name(key, factor_to_idx, family_to_idx)
        targets_list = _expand_factor_names(targets, factor_to_idx, family_to_idx)
        targets_idx = set(targets_list)

        overlap = neutralize_idx & targets_idx
        if overlap:
            overlap_names = sorted(factor_names[i] for i in overlap)
            raise ValueError(
                f"`neutralize_against` key '{key}' resolves to factors "
                f"that overlap with its targets: {overlap_names}. "
                f"A factor cannot be neutralized against itself."
            )

        x = exposures[:, :, targets_list]
        for factor_idx in sorted(neutralize_idx):
            neutralized, cs_weights = _cross_sectional_neutralize(
                y=exposures[:, :, factor_idx],
                x=x,
                cs_weights=benchmark_weights,
                cs_regressor=cs_regressor,
            )
            exposures[:, :, factor_idx] = CSStandardScaler().fit_transform(
                neutralized, cs_weights=cs_weights
            )


def _powered_market_cap(market_caps: FloatArray, power: float = 1.0) -> FloatArray:
    r"""Compute :math:`\text{mcap}^{\text{power}}` as an unnormalized cap signal.

    Returns the raw powered market capitalizations **without** normalizing
    them to sum to one.  Downstream consumers (cross-sectional regression,
    cross-sectional scalers, basis transforms, etc.) are responsible for
    masking invalid entries and normalizing to their own validity context.

    This follows the scikit-learn `sample_weight` convention: weights
    express relative importance and need not sum to one.

    Parameters
    ----------
    market_caps : array-like of shape (n_observations, n_assets)
        Market capitalizations.  May contain NaN for delisted /
        out-of-universe assets.

    power : float, default=1.0
        Exponent applied to market caps.  Typical choices:

        * 0.0 -- equal weight (caller should handle separately)
        * 0.5 -- sqrt-cap weight
        * 1.0 -- cap weight (default)

    Returns
    -------
    cap_signal : ndarray of shape (n_observations, n_assets)
        :math:`\text{mcap}^{\text{power}}`.  NaN where `market_caps`
        is NaN.
    """
    if power != 1.0:
        market_caps = np.power(market_caps, power)
    return market_caps
