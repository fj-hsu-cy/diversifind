

"""Tests for the greedy search method."""

from __future__ import annotations

import numpy as np
import pytest

from diversifind import greedy
from diversifind.results import PortfolioResult


def test_greedy_returns_portfolio_result(medium_corr: np.ndarray, medium_symbols: list[str]) -> None:
    """Greedy search should return a structured PortfolioResult."""
    result = greedy(medium_corr, medium_symbols, k=5)

    assert isinstance(result, PortfolioResult)
    assert result.method == "greedy"
    assert result.k == 5
    assert result.top_k == 1
    assert len(result.top_results) == 1



def test_greedy_best_portfolio_has_k_assets(medium_corr: np.ndarray, medium_symbols: list[str]) -> None:
    """The greedy portfolio should contain exactly k distinct assets."""
    k = 5
    result = greedy(medium_corr, medium_symbols, k=k)
    best = result.best()

    assert len(best.combo_indices) == k
    assert len(best.combo_symbols) == k
    assert len(set(best.combo_indices)) == k
    assert len(set(best.combo_symbols)) == k



def test_greedy_returns_single_ranked_result(medium_corr: np.ndarray, medium_symbols: list[str]) -> None:
    """Greedy search should return exactly one ranked portfolio."""
    result = greedy(medium_corr, medium_symbols, k=4)

    assert len(result.top_results) == 1
    assert result.top_results[0].rank == 1



def test_greedy_raises_when_k_too_small(medium_corr: np.ndarray, medium_symbols: list[str]) -> None:
    """Greedy search should reject k < 2."""
    with pytest.raises(ValueError, match="k must be >= 2"):
        greedy(medium_corr, medium_symbols, k=1)



def test_greedy_raises_when_k_exceeds_n_assets(small_corr: np.ndarray, small_symbols: list[str]) -> None:
    """Greedy search should reject portfolio sizes larger than the universe."""
    with pytest.raises(ValueError, match="target portfolio size cannot exceed number of assets"):
        greedy(small_corr, small_symbols, k=len(small_symbols) + 1)



def test_greedy_raises_when_symbols_length_mismatches_corr(small_corr: np.ndarray, small_symbols: list[str]) -> None:
    """Greedy search should reject symbol lists that do not match matrix dimensions."""
    with pytest.raises(ValueError, match="symbols length must match corr dimensions"):
        greedy(small_corr, small_symbols[:-1], k=4)



def test_greedy_logdet_is_finite(medium_corr: np.ndarray, medium_symbols: list[str]) -> None:
    """Greedy search should return a finite logdet on a valid problem."""
    result = greedy(medium_corr, medium_symbols, k=5)
    best = result.best()

    assert np.isfinite(best.logdet)