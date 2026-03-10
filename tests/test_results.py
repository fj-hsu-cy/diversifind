

"""Tests for structured search result dataclasses."""

from __future__ import annotations

import pytest

from diversifind.results import PortfolioEntry, PortfolioResult, result_from_dict



def test_portfolio_entry_to_dict_roundtrip() -> None:
    """PortfolioEntry should round-trip cleanly through dictionary serialization."""
    entry = PortfolioEntry(
        rank=1,
        logdet=-0.123456,
        combo_indices=[1, 4, 7],
        combo_symbols=["A1", "A4", "A7"],
    )

    payload = entry.to_dict()
    rebuilt = PortfolioEntry.from_dict(payload)

    assert rebuilt == entry



def test_portfolio_result_to_dict_roundtrip() -> None:
    """PortfolioResult should round-trip cleanly through dictionary serialization."""
    result = PortfolioResult(
        method="beam",
        k=3,
        top_k=2,
        top_results=[
            PortfolioEntry(
                rank=1,
                logdet=-0.101,
                combo_indices=[0, 1, 2],
                combo_symbols=["A0", "A1", "A2"],
            ),
            PortfolioEntry(
                rank=2,
                logdet=-0.202,
                combo_indices=[0, 2, 3],
                combo_symbols=["A0", "A2", "A3"],
            ),
        ],
        metadata={"beam_width": 500},
    )

    payload = result.to_dict()
    rebuilt = PortfolioResult.from_dict(payload)

    assert rebuilt.method == result.method
    assert rebuilt.k == result.k
    assert rebuilt.top_k == result.top_k
    assert rebuilt.metadata == result.metadata
    assert rebuilt.top_results == result.top_results



def test_result_from_dict_wrapper() -> None:
    """Convenience wrapper should build a PortfolioResult from a dictionary."""
    payload = {
        "method": "greedy",
        "k": 4,
        "top_k": 1,
        "top_results": [
            {
                "rank": 1,
                "logdet": -0.55,
                "combo_indices": [1, 2, 3, 4],
                "combo_symbols": ["A1", "A2", "A3", "A4"],
            }
        ],
    }

    result = result_from_dict(payload)

    assert isinstance(result, PortfolioResult)
    assert result.method == "greedy"
    assert result.k == 4
    assert result.top_k == 1
    assert result.best().combo_symbols == ["A1", "A2", "A3", "A4"]



def test_portfolio_result_best_returns_first_entry() -> None:
    """best() should return the top-ranked portfolio entry."""
    first = PortfolioEntry(1, -0.10, [0, 1], ["A0", "A1"])
    second = PortfolioEntry(2, -0.20, [0, 2], ["A0", "A2"])

    result = PortfolioResult(
        method="beam",
        k=2,
        top_k=2,
        top_results=[first, second],
    )

    assert result.best() == first



def test_portfolio_result_best_raises_when_empty() -> None:
    """best() should fail clearly when no portfolios are present."""
    result = PortfolioResult(method="beam", k=3, top_k=5, top_results=[])

    with pytest.raises(ValueError, match="No portfolio results available"):
        result.best()



def test_portfolio_result_summary_lines_contains_metadata() -> None:
    """summary_lines() should include method, k, result count, and metadata."""
    result = PortfolioResult(
        method="bruteforce_mp",
        k=4,
        top_k=3,
        top_results=[PortfolioEntry(1, -0.1, [0, 1, 2, 3], ["A0", "A1", "A2", "A3"])],
        metadata={"n_jobs": 2},
    )

    lines = result.summary_lines()
    text = "\n".join(lines)

    assert "Method: bruteforce_mp" in text
    assert "Portfolio size: 4" in text
    assert "Returned results: 1" in text
    assert "n_jobs: 2" in text



def test_portfolio_result_pretty_contains_table_content() -> None:
    """pretty() should return a human-readable table-like summary."""
    result = PortfolioResult(
        method="beam",
        k=3,
        top_k=2,
        top_results=[
            PortfolioEntry(1, -0.111111, [0, 1, 2], ["A0", "A1", "A2"]),
            PortfolioEntry(2, -0.222222, [0, 2, 3], ["A0", "A2", "A3"]),
        ],
        metadata={"beam_width": 100},
    )

    text = result.pretty()

    assert "Method: beam" in text
    assert "Portfolio size: 3" in text
    assert "Top diversified portfolios:" in text
    assert "Rank" in text
    assert "LogDet" in text
    assert "A0, A1, A2" in text
    assert "beam_width: 100" in text