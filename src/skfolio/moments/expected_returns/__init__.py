"""Expected returns module."""

from skfolio.moments.expected_returns._base import (
    BaseMu,
)
from skfolio.moments.expected_returns._empirical_mu import EmpiricalMu
from skfolio.moments.expected_returns._equilibrium_mu import EquilibriumMu
from skfolio.moments.expected_returns._ew_mu import EWMu
from skfolio.moments.expected_returns._shrunk_mu import ShrunkMu, ShrunkMuMethods

__all__ = [
    "BaseMu",
    "EWMu",
    "EmpiricalMu",
    "EquilibriumMu",
    "ShrunkMu",
    "ShrunkMuMethods",
]
