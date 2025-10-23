"""Datasets module."""

# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from skfolio.datasets._base import (
    load_factors_dataset,
    load_ftse100_dataset,
    load_nasdaq_dataset,
    load_sp500_dataset,
    load_sp500_implied_vol_dataset,
    load_sp500_index,
    load_usd_rates_dataset,
    load_bond_dataset,
    load_bond_metadata_dataset
)

__all__ = [
    "load_factors_dataset",
    "load_ftse100_dataset",
    "load_nasdaq_dataset",
    "load_sp500_dataset",
    "load_sp500_implied_vol_dataset",
    "load_sp500_index",
    "load_usd_rates_dataset",
    "load_bond_dataset",
    "load_bond_metadata_dataset"
]
