"""Tests for analytics helpers."""

from __future__ import annotations

import numpy as np
import pytest

from diversifind import beam
from diversifind.analytics import (
    analyze_portfolio,
    analyze_results,
    corr_distribution_summary,
    effective_rank,
    eigenvalue_summary,
    pairwise_abs_corr_summary,
    top_abs_corr_pairs,
)



def test_effective_rank_identity_matrix() -> None:
    """Identity correlation matrix should have full effective rank."""
    subcorr = np.eye(4)
    rank = effective_rank(subcorr)

    assert rank == pytest.approx(4.0)



def test_effective_rank_duplicate_assets_case() -> None:
    """A perfectly collinear 2x2 matrix should have effective rank 1."""
    subcorr = np.array([[1.0, 1.0], [1.0, 1.0]])
    rank = effective_rank(subcorr)

    assert rank == pytest.approx(1.0)



def test_pairwise_abs_corr_summary_returns_expected_values() -> None:
    """Pairwise correlation summary should match a simple hand-built case."""
    subcorr = np.array(
        [
            [1.0, 0.2, -0.5],
            [0.2, 1.0, 0.8],
            [-0.5, 0.8, 1.0],
        ]
    )

    summary = pairwise_abs_corr_summary(subcorr)

    assert summary["n_pairs"] == 3.0
    assert summary["min_abs_corr"] == pytest.approx(0.2)
    assert summary["mean_abs_corr"] == pytest.approx((0.2 + 0.5 + 0.8) / 3)
    assert summary["median_abs_corr"] == pytest.approx(0.5)
    assert summary["max_abs_corr"] == pytest.approx(0.8)



def test_eigenvalue_summary_identity_matrix() -> None:
    """Eigenvalue summary should be trivial for the identity matrix."""
    subcorr = np.eye(3)
    summary = eigenvalue_summary(subcorr)

    assert summary["min_eigenvalue"] == pytest.approx(1.0)
    assert summary["max_eigenvalue"] == pytest.approx(1.0)
    assert summary["condition_number"] == pytest.approx(1.0)
    assert summary["effective_rank"] == pytest.approx(3.0)



def test_analyze_portfolio_returns_expected_fields(
    small_corr: np.ndarray,
    small_symbols: list[str],
) -> None:
    """Portfolio analytics should include key diagnostics and symbols."""
    combo = [0, 1, 2, 3]
    analysis = analyze_portfolio(small_corr, combo, symbols=small_symbols)

    assert analysis["k"] == 4
    assert analysis["combo_indices"] == combo
    assert analysis["combo_symbols"] == ["A0", "A1", "A2", "A3"]
    assert "logdet" in analysis
    assert "effective_rank" in analysis
    assert "max_abs_corr" in analysis
    assert "min_eigenvalue" in analysis
    assert np.isfinite(analysis["logdet"])



def test_analyze_portfolio_raises_on_empty_combo(
    small_corr: np.ndarray,
) -> None:
    """Portfolio analytics should reject empty asset selections."""
    with pytest.raises(ValueError, match="combo must contain at least one asset index"):
        analyze_portfolio(small_corr, [])



def test_analyze_portfolio_raises_on_symbol_length_mismatch(
    small_corr: np.ndarray,
    small_symbols: list[str],
) -> None:
    """Portfolio analytics should reject symbol lists with wrong length."""
    with pytest.raises(ValueError, match="symbols length must match corr dimensions"):
        analyze_portfolio(small_corr, [0, 1, 2], symbols=small_symbols[:-1])



def test_analyze_results_returns_ranked_analytics(
    small_corr: np.ndarray,
    small_symbols: list[str],
) -> None:
    """Result analytics should preserve method metadata and per-rank analysis."""
    result = beam(small_corr, small_symbols, k=4, beam_width=100, top_k=3)
    analysis = analyze_results(small_corr, result.to_dict(), symbols=small_symbols)

    assert analysis["method"] == "beam"
    assert analysis["k"] == 4
    assert analysis["n_results"] == len(result.top_results)
    assert len(analysis["portfolio_analytics"]) == len(result.top_results)
    assert analysis["portfolio_analytics"][0]["rank"] == 1
    assert "effective_rank" in analysis["portfolio_analytics"][0]



def test_top_abs_corr_pairs_sorted_descending(
    small_corr: np.ndarray,
    small_symbols: list[str],
) -> None:
    """Strongest correlation pairs should be returned in descending order."""
    combo = [0, 1, 2, 3]
    pairs = top_abs_corr_pairs(small_corr, combo, symbols=small_symbols, top_m=3)

    assert len(pairs) <= 3
    scores = [row["abs_corr"] for row in pairs]
    assert scores == sorted(scores, reverse=True)
    assert "pair_symbols" in pairs[0]



def test_top_abs_corr_pairs_empty_for_singleton(
    small_corr: np.ndarray,
    small_symbols: list[str],
) -> None:
    """A singleton portfolio should have no pairwise-correlation pairs."""
    pairs = top_abs_corr_pairs(small_corr, [0], symbols=small_symbols)
    assert pairs == []



def test_corr_distribution_summary_returns_counts_and_quantiles(
    small_corr: np.ndarray,
) -> None:
    """Distribution summary should include pair counts and requested quantiles."""
    summary = corr_distribution_summary(small_corr, quantiles=[0.25, 0.5, 0.75])

    assert summary["n_assets"] == 10.0
    assert summary["n_pairs"] == 45.0
    assert "q25" in summary
    assert "q50" in summary
    assert "q75" in summary
    assert "max_abs_corr" in summary
    assert "mean_abs_corr" in summary



def test_corr_distribution_summary_single_asset_case() -> None:
    """A 1x1 correlation matrix should return counts without quantiles."""
    corr = np.array([[1.0]])
    summary = corr_distribution_summary(corr)

    assert summary["n_assets"] == 1.0
    assert summary["n_pairs"] == 0.0
    assert "max_abs_corr" not in summary
    assert "mean_abs_corr" not in summary
