"""Tests for brute-force search methods."""

from __future__ import annotations

import numpy as np
import pytest

from diversifind import bruteforce, bruteforce_mp
from diversifind.results import PortfolioResult


def test_bruteforce_returns_portfolio_result(
    small_corr: np.ndarray,
    small_symbols: list[str],
) -> None:
    """Brute force should return a structured PortfolioResult."""
    result = bruteforce(small_corr, small_symbols, k=4)

    assert isinstance(result, PortfolioResult)
    assert result.method == "bruteforce"
    assert result.k == 4
    assert result.top_k == 10
    assert len(result.top_results) > 0


def test_bruteforce_best_portfolio_has_k_assets(
    small_corr: np.ndarray,
    small_symbols: list[str],
) -> None:
    """The best brute-force portfolio should contain exactly k distinct assets."""
    k = 4
    result = bruteforce(small_corr, small_symbols, k=k)
    best = result.best()

    assert len(best.combo_indices) == k
    assert len(best.combo_symbols) == k
    assert len(set(best.combo_indices)) == k
    assert len(set(best.combo_symbols)) == k


def test_bruteforce_results_are_sorted_by_logdet(
    small_corr: np.ndarray,
    small_symbols: list[str],
) -> None:
    """Brute-force results should be ranked in descending order of logdet."""
    result = bruteforce(small_corr, small_symbols, k=4, top_k=5)
    scores = [entry.logdet for entry in result.top_results]

    assert scores == sorted(scores, reverse=True)


def test_bruteforce_respects_top_k(
    small_corr: np.ndarray,
    small_symbols: list[str],
) -> None:
    """Brute force should not return more than the requested top_k portfolios."""
    result = bruteforce(small_corr, small_symbols, k=4, top_k=3)

    assert len(result.top_results) <= 3
    assert result.top_k == 3


def test_bruteforce_raises_when_k_exceeds_n_assets(
    small_corr: np.ndarray,
    small_symbols: list[str],
) -> None:
    """Brute force should reject portfolio sizes larger than the universe."""
    with pytest.raises(ValueError, match="k cannot exceed number of assets"):
        bruteforce(small_corr, small_symbols, k=len(small_symbols) + 1)


def test_bruteforce_raises_when_symbols_length_mismatches_corr(
    small_corr: np.ndarray,
    small_symbols: list[str],
) -> None:
    """Brute force should reject symbol lists that do not match matrix dimensions."""
    with pytest.raises(ValueError, match="symbols length must match corr dimensions"):
        bruteforce(small_corr, small_symbols[:-1], k=4)


def test_bruteforce_mp_returns_portfolio_result(
    small_corr: np.ndarray,
    small_symbols: list[str],
) -> None:
    """Multiprocessing brute force should return a structured PortfolioResult."""
    result = bruteforce_mp(small_corr, small_symbols, k=4, n_jobs=2)

    assert isinstance(result, PortfolioResult)
    assert result.method == "bruteforce_mp"
    assert result.k == 4
    assert result.top_k == 10
    assert result.metadata.get("n_jobs") == 2
    assert len(result.top_results) > 0


def test_bruteforce_mp_best_portfolio_has_k_assets(
    small_corr: np.ndarray,
    small_symbols: list[str],
) -> None:
    """The multiprocessing brute-force portfolio should contain exactly k assets."""
    k = 4
    result = bruteforce_mp(small_corr, small_symbols, k=k, n_jobs=2)
    best = result.best()

    assert len(best.combo_indices) == k
    assert len(best.combo_symbols) == k
    assert len(set(best.combo_indices)) == k
    assert len(set(best.combo_symbols)) == k


def test_bruteforce_mp_matches_serial_bruteforce(
    small_corr: np.ndarray,
    small_symbols: list[str],
) -> None:
    """Multiprocessing brute force should match serial brute force on a small case."""
    k = 4
    serial = bruteforce(small_corr, small_symbols, k=k, top_k=5)
    parallel = bruteforce_mp(small_corr, small_symbols, k=k, top_k=5, n_jobs=2)

    assert parallel.best().combo_indices == serial.best().combo_indices
    assert parallel.best().combo_symbols == serial.best().combo_symbols
    assert parallel.best().logdet == pytest.approx(serial.best().logdet)


def test_bruteforce_mp_respects_top_k(
    small_corr: np.ndarray,
    small_symbols: list[str],
) -> None:
    """Multiprocessing brute force should not return more than top_k portfolios."""
    result = bruteforce_mp(small_corr, small_symbols, k=4, top_k=3, n_jobs=2)

    assert len(result.top_results) <= 3
    assert result.top_k == 3


def test_bruteforce_mp_raises_when_k_exceeds_n_assets(
    small_corr: np.ndarray,
    small_symbols: list[str],
) -> None:
    """Multiprocessing brute force should reject portfolio sizes larger than the universe."""
    with pytest.raises(ValueError, match="k cannot exceed number of assets"):
        bruteforce_mp(small_corr, small_symbols, k=len(small_symbols) + 1, n_jobs=2)


def test_bruteforce_mp_raises_when_symbols_length_mismatches_corr(
    small_corr: np.ndarray,
    small_symbols: list[str],
) -> None:
    """Multiprocessing brute force should reject symbol lists that do not match matrix dimensions."""
    with pytest.raises(ValueError, match="symbols length must match corr dimensions"):
        bruteforce_mp(small_corr, small_symbols[:-1], k=4, n_jobs=2)
