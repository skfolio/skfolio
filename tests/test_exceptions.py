"""Ensure all public custom exceptions inherit from SkfolioError."""

from __future__ import annotations

import pytest

from skfolio import exceptions
from skfolio.exceptions import SkfolioError

PUBLIC_ERRORS = [
    getattr(exceptions, name) for name in exceptions.__all__ if name != "SkfolioError"
]


def test_public_errors_are_discovered() -> None:
    """Ensure exception discovery is not empty."""
    assert PUBLIC_ERRORS


@pytest.mark.parametrize("error", PUBLIC_ERRORS, ids=lambda error: error.__name__)
def test_every_public_error_derives_from_base(error: type[Exception]) -> None:
    """Every exported error must be catchable as `SkfolioError`."""
    assert issubclass(error, SkfolioError)


def test_base_is_exported() -> None:
    """`SkfolioError` is part of the public surface, so callers can import it."""
    assert "SkfolioError" in exceptions.__all__


def test_base_still_derives_from_exception() -> None:
    """Existing `except Exception` handlers must keep working."""
    assert issubclass(SkfolioError, Exception)
