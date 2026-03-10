"""Analytics helpers for diversification search results.

This module provides post-search diagnostics for analyzing selected portfolios.
The functions here are intended to complement the search methods by helping
users understand *why* a portfolio scores well, rather than helping choose a
pairwise pruning threshold.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

import numpy as np
from .utils import combo_logdet, validate_corr


def _subset_corr(corr: np.ndarray, combo: Sequence[int]) -> np.ndarray:
    """Extract a correlation submatrix for a selected portfolio.

    Args:
        corr: Full correlation matrix.
        combo: Selected asset indices.

    Returns:
        Correlation submatrix.
    """
    idx = np.array(combo, dtype=int)
    return corr[np.ix_(idx, idx)]


def effective_rank(subcorr: np.ndarray) -> float:
    """Compute the effective rank of a correlation submatrix.

    The effective rank is defined as ``exp(H(p))``, where ``p`` is the
    normalized eigenvalue spectrum and ``H`` is Shannon entropy. It measures
    how many dimensions are meaningfully represented by the portfolio.

    Args:
        subcorr: Correlation submatrix.

    Returns:
        Effective rank.
    """
    eigvals = np.linalg.eigvalsh(subcorr)
    eigvals = np.clip(eigvals, 0.0, None)
    total = float(np.sum(eigvals))
    if total <= 0.0:
        return 0.0

    probs = eigvals / total
    probs = probs[probs > 0.0]
    entropy = -float(np.sum(probs * np.log(probs)))
    return float(np.exp(entropy))


def pairwise_abs_corr_summary(subcorr: np.ndarray) -> Dict[str, float]:
    """Summarize pairwise absolute correlations within a selected portfolio.

    Args:
        subcorr: Correlation submatrix.

    Returns:
        Dictionary with summary statistics for the unique off-diagonal absolute
        correlations.
    """
    n = subcorr.shape[0]
    if n < 2:
        return {
            "n_pairs": 0.0,
            "min_abs_corr": float("nan"),
            "mean_abs_corr": float("nan"),
            "median_abs_corr": float("nan"),
            "max_abs_corr": float("nan"),
        }

    a = np.abs(subcorr)
    i, j = np.triu_indices(n, 1)
    vals = a[i, j]

    return {
        "n_pairs": float(vals.size),
        "min_abs_corr": float(np.min(vals)),
        "mean_abs_corr": float(np.mean(vals)),
        "median_abs_corr": float(np.median(vals)),
        "max_abs_corr": float(np.max(vals)),
    }


def eigenvalue_summary(subcorr: np.ndarray) -> Dict[str, float]:
    """Summarize the eigenvalue spectrum of a selected portfolio.

    Args:
        subcorr: Correlation submatrix.

    Returns:
        Dictionary containing min/max eigenvalues, condition number, and
        effective rank.
    """
    eigvals = np.linalg.eigvalsh(subcorr)
    eigvals = np.sort(eigvals)

    min_eig = float(eigvals[0])
    max_eig = float(eigvals[-1])

    if min_eig <= 0.0:
        cond = float("inf")
    else:
        cond = float(max_eig / min_eig)

    return {
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "condition_number": cond,
        "effective_rank": float(effective_rank(subcorr)),
    }


def analyze_portfolio(
    corr: np.ndarray,
    combo: Sequence[int],
    symbols: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Compute diagnostics for a single selected portfolio.

    Args:
        corr: Full correlation matrix.
        combo: Selected asset indices.
        symbols: Optional symbols aligned with the full correlation matrix.

    Returns:
        Dictionary containing the selected assets and several diversification
        diagnostics.
    """
    validate_corr(corr)

    if len(combo) == 0:
        raise ValueError("combo must contain at least one asset index")

    subcorr = _subset_corr(corr, combo)

    out: Dict[str, Any] = {
        "k": int(len(combo)),
        "combo_indices": [int(i) for i in combo],
        "logdet": float(combo_logdet(corr, tuple(combo))),
    }

    if symbols is not None:
        if len(symbols) != corr.shape[0]:
            raise ValueError("symbols length must match corr dimensions")
        out["combo_symbols"] = [str(symbols[i]) for i in combo]

    out.update(pairwise_abs_corr_summary(subcorr))
    out.update(eigenvalue_summary(subcorr))
    return out


def analyze_results(
    corr: np.ndarray,
    result: Dict[str, Any],
    symbols: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Analyze every returned portfolio from a search result payload.

    This function expects the unified payload shape returned by the public
    search methods in ``search_methods.py``.

    Args:
        corr: Full correlation matrix.
        result: Search result dictionary returned by a search method.
        symbols: Optional symbols aligned with the full correlation matrix.
            If omitted, the function will try to infer them from the first
            result entry when available.

    Returns:
        Dictionary containing method metadata and analytics for each ranked
        portfolio.
    """
    validate_corr(corr)

    method = str(result.get("method", "unknown"))
    k = int(result.get("k", 0))
    top_results = list(result.get("top_results", []))

    if symbols is None and top_results:
        first_syms = top_results[0].get("combo_symbols")
        if first_syms is not None:
            # Presence of combo_symbols is useful per-entry, but not enough to
            # reconstruct the full symbol universe, so keep symbols=None here.
            pass

    analyzed: list[Dict[str, Any]] = []
    for entry in top_results:
        combo = entry.get("combo_indices", [])
        row = {
            "rank": int(entry.get("rank", len(analyzed) + 1)),
        }
        row.update(analyze_portfolio(corr=corr, combo=combo, symbols=symbols))
        analyzed.append(row)

    return {
        "method": method,
        "k": k,
        "n_results": int(len(analyzed)),
        "portfolio_analytics": analyzed,
    }


def top_abs_corr_pairs(
    corr: np.ndarray,
    combo: Sequence[int],
    symbols: Sequence[str] | None = None,
    top_m: int = 5,
) -> list[Dict[str, Any]]:
    """Return the strongest absolute-correlation pairs within a portfolio.

    Args:
        corr: Full correlation matrix.
        combo: Selected asset indices.
        symbols: Optional symbols aligned with the full correlation matrix.
        top_m: Number of pairs to return.

    Returns:
        Ranked list of strongest absolute-correlation pairs.
    """
    validate_corr(corr)
    subcorr = _subset_corr(corr, combo)

    n = subcorr.shape[0]
    if n < 2:
        return []

    pairs: list[Dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            row: Dict[str, Any] = {
                "abs_corr": float(abs(subcorr[i, j])),
                "pair_indices": [int(combo[i]), int(combo[j])],
            }
            if symbols is not None:
                row["pair_symbols"] = [str(symbols[combo[i]]), str(symbols[combo[j]])]
            pairs.append(row)

    pairs.sort(key=lambda x: x["abs_corr"], reverse=True)
    return pairs[:top_m]


def corr_distribution_summary(
    corr: np.ndarray,
    quantiles: Iterable[float] | None = None,
) -> Dict[str, float]:
    """Provide summary diagnostics for the distribution of |correlation| values.

    Although originally motivated by pruning thresholds, this is still useful
    as a general diagnostic for understanding the structure of a search
    universe before running the optimization.

    Args:
        corr: Correlation matrix.
        quantiles: Quantiles of the absolute-correlation distribution to
            report.

    Returns:
        Summary statistics for the upper-triangular absolute correlations.
    """
    validate_corr(corr)

    a = np.abs(corr)
    n = a.shape[0]
    i, j = np.triu_indices(n, 1)
    vals = a[i, j]

    if quantiles is None:
        quantiles = np.linspace(0.1, 0.9, 9)

    out: Dict[str, float] = {
        "n_assets": float(n),
        "n_pairs": float(vals.size),
    }

    if vals.size == 0:
        return out

    for q in quantiles:
        out[f"q{int(q * 100):02d}"] = float(np.quantile(vals, q))

    out["max_abs_corr"] = float(np.max(vals))
    out["mean_abs_corr"] = float(np.mean(vals))
    return out


__all__ = [
    "analyze_portfolio",
    "analyze_results",
    "corr_distribution_summary",
    "effective_rank",
    "eigenvalue_summary",
    "pairwise_abs_corr_summary",
    "top_abs_corr_pairs",
]