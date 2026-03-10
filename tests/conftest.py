

"""Shared pytest fixtures and helpers for diversifind tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random number generator for reproducible tests."""
    return np.random.default_rng(12345)


@pytest.fixture
def small_symbols() -> list[str]:
    """Small reusable list of synthetic asset symbols."""
    return [f"A{i}" for i in range(10)]


@pytest.fixture
def medium_symbols() -> list[str]:
    """Medium reusable list of synthetic asset symbols."""
    return [f"A{i}" for i in range(20)]


def random_corr(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random positive-semidefinite correlation matrix.

    Args:
        n: Number of assets.
        rng: NumPy random generator.

    Returns:
        A valid ``n x n`` correlation matrix.
    """
    a = rng.standard_normal((n, n))
    cov = a @ a.T
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return corr


@pytest.fixture
def small_corr(rng: np.random.Generator) -> np.ndarray:
    """Reusable small correlation matrix for tests."""
    return random_corr(10, rng)


@pytest.fixture
def medium_corr(rng: np.random.Generator) -> np.ndarray:
    """Reusable medium correlation matrix for tests."""
    return random_corr(20, rng)