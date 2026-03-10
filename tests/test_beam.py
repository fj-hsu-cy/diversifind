"""Tests for the beam search method."""

from __future__ import annotations

import numpy as np
import pytest

from diversifind import beam, bruteforce
from diversifind.results import PortfolioResult


def test_beam_returns_portfolio_result(medium_corr: np.ndarray, medium_symbols: list[str]) -> None:
    """Beam search should return a structured PortfolioResult."""
    result = beam(medium_corr, medium_symbols, k=5)

    assert isinstance(result, PortfolioResult)
    assert result.method == "beam"
    assert result.k == 5
    assert result.top_k == 10
    assert len(result.top_results) > 0


def test_beam_best_portfolio_has_k_assets(medium_corr: np.ndarray, medium_symbols: list[str]) -> None:
    """The best returned portfolio should contain exactly k distinct assets."""
    k = 5
    result = beam(medium_corr, medium_symbols, k=k)
    best = result.best()

    assert len(best.combo_indices) == k
    assert len(best.combo_symbols) == k
    assert len(set(best.combo_indices)) == k
    assert len(set(best.combo_symbols)) == k



def test_beam_results_are_sorted_by_logdet(medium_corr: np.ndarray, medium_symbols: list[str]) -> None:
    """Beam results should be ranked in descending order of logdet."""
    result = beam(medium_corr, medium_symbols, k=5, top_k=5)
    scores = [entry.logdet for entry in result.top_results]

    assert scores == sorted(scores, reverse=True)



def test_beam_respects_top_k(medium_corr: np.ndarray, medium_symbols: list[str]) -> None:
    """Beam search should not return more than the requested top_k portfolios."""
    result = beam(medium_corr, medium_symbols, k=5, top_k=3)
    assert len(result.top_results) <= 3
    assert result.top_k == 3



def test_beam_raises_when_k_too_small(medium_corr: np.ndarray, medium_symbols: list[str]) -> None:
    """Beam search should reject k < 2."""
    with pytest.raises(ValueError, match="k must be >= 2"):
        beam(medium_corr, medium_symbols, k=1)



def test_beam_raises_when_k_exceeds_n_assets(small_corr: np.ndarray, small_symbols: list[str]) -> None:
    """Beam search should reject portfolio sizes larger than the universe."""
    with pytest.raises(ValueError, match="k=.*> n=.*"):
        beam(small_corr, small_symbols, k=len(small_symbols) + 1)



def test_beam_raises_when_symbols_length_mismatches_corr(small_corr: np.ndarray, small_symbols: list[str]) -> None:
    """Beam search should reject symbol lists that do not match matrix dimensions."""
    with pytest.raises(ValueError, match="symbols length must match corr dimensions"):
        beam(small_corr, small_symbols[:-1], k=4)



def test_beam_matches_bruteforce_on_small_problem(small_corr: np.ndarray, small_symbols: list[str]) -> None:
    """With a sufficiently wide beam, beam search should match brute force on a small case."""
    k = 4
    brute = bruteforce(small_corr, small_symbols, k=k, top_k=5)
    beam_result = beam(small_corr, small_symbols, k=k, beam_width=1000, top_k=5)

    assert beam_result.best().combo_indices == brute.best().combo_indices
    assert beam_result.best().combo_symbols == brute.best().combo_symbols
    assert beam_result.best().logdet == pytest.approx(brute.best().logdet)
