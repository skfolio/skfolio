"""Shared fixtures for online optimization tests."""

import pytest


@pytest.fixture
def X_small_single(X_small):
    """Single row of X_small for partial_fit tests."""
    return X_small.iloc[[0], :]
