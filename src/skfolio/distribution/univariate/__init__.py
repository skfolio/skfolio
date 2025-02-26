"""Univariate Distribution module."""

from skfolio.distribution.univariate._base import BaseUnivariateDist
from skfolio.distribution.univariate._gaussian import Gaussian
from skfolio.distribution.univariate._johnson_su import JohnsonSU
from skfolio.distribution.univariate._normal_inverse_gaussian import (
    NormalInverseGaussian,
)
from skfolio.distribution.univariate._selection import select_univariate_dist
from skfolio.distribution.univariate._student_t import StudentT

__all__ = [
    "BaseUnivariateDist",
    "Gaussian",
    "JohnsonSU",
    "NormalInverseGaussian",
    "StudentT",
    "select_univariate_dist",
]
