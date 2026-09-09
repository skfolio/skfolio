"""Shared utilities for factor-model estimators."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections import defaultdict

import numpy as np

from skfolio.typing import ObjArray, StrArray


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
