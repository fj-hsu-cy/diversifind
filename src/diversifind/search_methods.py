from __future__ import annotations

import heapq
import itertools
import multiprocessing as mp
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from .utils import combo_logdet, validate_corr
from .results import PortfolioEntry, PortfolioResult


ScoreFn = Callable[[np.ndarray, Tuple[int, ...]], float]


# =====================================================================
# Shared helpers
# =====================================================================



def _build_result(
    method: str,
    k: int,
    top_k: int,
    ranked_results: List[Tuple[float, Tuple[int, ...]]],
    symbols: list[str],
    **metadata: Any,
) -> PortfolioResult:
    """Build a structured portfolio result from scored combinations."""
    entries: list[PortfolioEntry] = []
    for rank, (score, combo) in enumerate(ranked_results[:top_k], 1):
        entries.append(
            PortfolioEntry(
                rank=int(rank),
                logdet=float(score),
                combo_indices=list(combo),
                combo_symbols=[symbols[i] for i in combo],
            )
        )

    return PortfolioResult(
        method=str(method),
        k=int(k),
        top_k=int(top_k),
        top_results=entries,
        metadata=metadata,
    )


# =====================================================================
# Beam search
# =====================================================================

def beam(
    corr: np.ndarray,
    symbols: list[str],
    k: int,
    beam_width: int = 500,
    top_k: int = 10,
) -> PortfolioResult:
    """Run beam search for logdet maximization.

    The search starts from all 2-asset portfolios. At each depth, it keeps the
    top ``beam_width`` partial portfolios ranked by their current logdet,
    expands each by adding one new asset, and keeps the best ``beam_width``
    children.

    Args:
        corr: Correlation matrix of shape ``(n, n)``.
        symbols: Symbol names aligned with the rows/columns of ``corr``.
        k: Target portfolio size.
        beam_width: Number of partial portfolios kept at each depth.
        top_k: Number of final portfolios to return.

    Returns:
        Dictionary with keys:
            - ``k``: Target portfolio size.
            - ``beam_width``: Beam width used.
            - ``top_results``: Ranked list of best portfolios found.
    """
    validate_corr(corr)

    n = corr.shape[0]
    if len(symbols) != n:
        raise ValueError("symbols length must match corr dimensions")
    if k < 2:
        raise ValueError("k must be >= 2")
    if k > n:
        raise ValueError(f"k={k} > n={n}")

    beam_width = max(1, int(beam_width))
    top_k = max(1, int(top_k))

    # Initialize beam with all 2-asset portfolios.
    beam: dict[tuple[int, ...], float] = {}
    for i in range(n - 1):
        for j in range(i + 1, n):
            combo = (i, j)
            ld = combo_logdet(corr, combo)
            if ld == float("-inf"):
                continue
            beam[combo] = ld

    if not beam:
        return PortfolioResult(
            method="beam",
            k=int(k),
            top_k=int(top_k),
            top_results=[],
            metadata={"beam_width": int(beam_width)},
        )

    beam = dict(sorted(beam.items(), key=lambda kv: kv[1], reverse=True)[:beam_width])

    # Expand one level at a time until reaching size k.
    cur_len = 2
    while cur_len < k:
        next_beam: dict[tuple[int, ...], float] = {}

        for combo, _ld in beam.items():
            combo_set = set(combo)

            for j in range(n):
                if j in combo_set:
                    continue

                new_combo = tuple(sorted(combo + (j,)))
                if len(new_combo) != cur_len + 1:
                    continue

                ld = combo_logdet(corr, new_combo)
                if ld == float("-inf"):
                    continue

                # The same child can be generated from multiple parents.
                prev = next_beam.get(new_combo)
                if prev is None or ld > prev:
                    next_beam[new_combo] = ld

        if not next_beam:
            break

        beam = dict(sorted(next_beam.items(), key=lambda kv: kv[1], reverse=True)[:beam_width])
        cur_len += 1

    final = [(c, ld) for c, ld in beam.items() if len(c) == k]
    final.sort(key=lambda x: x[1], reverse=True)
    final = final[:top_k]

    ranked_results = [(ld, combo) for combo, ld in final]
    return _build_result(
        method="beam",
        k=k,
        top_k=top_k,
        ranked_results=ranked_results,
        symbols=symbols,
        beam_width=int(beam_width),
    )



# =====================================================================
# Greedy search
# =====================================================================

def greedy(
    corr: np.ndarray,
    symbols: list[str],
    k: int,
) -> PortfolioResult:
    """Run greedy search for logdet maximization.

    The algorithm starts from the best 2-asset portfolio and then repeatedly
    adds the single asset that gives the largest improvement in the current
    portfolio log-determinant.

    This method is very fast but does not guarantee global optimality because
    each step is chosen myopically.

    Args:
        corr: Correlation matrix of shape ``(n, n)``.
        symbols: Symbol names aligned with the rows/columns of ``corr``.
        k: Target portfolio size.

    Returns:
        Dictionary with keys:
            - ``method``: Search method name.
            - ``k``: Target portfolio size.
            - ``top_k``: Number of returned portfolios.
            - ``top_results``: Ranked list of portfolios found.
    """
    validate_corr(corr)

    n = corr.shape[0]
    if len(symbols) != n:
        raise ValueError("symbols length must match corr dimensions")
    if k < 2:
        raise ValueError("k must be >= 2")
    if k > n:
        raise ValueError("target portfolio size cannot exceed number of assets")

    # --------------------------------------------------------------
    # Start from the best pair.
    # --------------------------------------------------------------
    best_pair: Optional[Tuple[int, int]] = None
    best_ld = float("-inf")

    for i in range(n - 1):
        for j in range(i + 1, n):
            ld = combo_logdet(corr, (i, j))
            if ld > best_ld:
                best_ld = ld
                best_pair = (i, j)

    if best_pair is None or best_ld == float("-inf"):
        return PortfolioResult(
            method="greedy",
            k=int(k),
            top_k=1,
            top_results=[],
            metadata={},
        )

    chosen = [best_pair[0], best_pair[1]]
    chosen_set = set(chosen)

    # --------------------------------------------------------------
    # Repeatedly add the asset giving the best immediate improvement.
    # --------------------------------------------------------------
    while len(chosen) < k:
        best_j: Optional[int] = None
        best_ld_step = float("-inf")

        for j in range(n):
            if j in chosen_set:
                continue

            combo = tuple(sorted(chosen + [j]))
            ld = combo_logdet(corr, combo)
            if ld > best_ld_step:
                best_ld_step = ld
                best_j = j

        if best_j is None or best_ld_step == float("-inf"):
            break

        chosen.append(best_j)
        chosen.sort()
        chosen_set.add(best_j)
        best_ld = best_ld_step

    ranked_results = [(float(best_ld), tuple(chosen))]
    return _build_result(
        method="greedy",
        k=k,
        top_k=1,
        ranked_results=ranked_results,
        symbols=symbols,
    )


# =====================================================================
# Full brute-force search
# =====================================================================

def bruteforce(
    corr: np.ndarray,
    symbols: list[str],
    k: int,
    top_k: int = 10,
    scoring_fn: ScoreFn = combo_logdet,
) -> PortfolioResult:
    """Naive full brute-force benchmark.

    This evaluates every combination of size ``k`` and returns the top-k scored
    portfolios under the original unconstrained objective.
    """
    validate_corr(corr)

    n_assets = corr.shape[0]
    if len(symbols) != n_assets:
        raise ValueError("symbols length must match corr dimensions")
    if k > n_assets:
        raise ValueError("k cannot exceed number of assets")

    best: List[Tuple[float, Tuple[int, ...]]] = []
    for combo in itertools.combinations(range(n_assets), k):
        score = scoring_fn(corr, combo)
        if score == float("-inf"):
            continue
        best.append((score, combo))

    best.sort(reverse=True)
    ranked_results = best[:top_k]
    return _build_result(
        method="bruteforce",
        k=k,
        top_k=top_k,
        ranked_results=ranked_results,
        symbols=symbols,
    )


# Worker globals for full brute-force multiprocessing.
_brute_worker_corr: Optional[np.ndarray] = None
_brute_worker_k: Optional[int] = None
_brute_worker_top_k: Optional[int] = None
_brute_worker_scoring_fn: Optional[ScoreFn] = None



def _brute_mp_init(
    corr: np.ndarray,
    k: int,
    top_k: int,
    scoring_fn: ScoreFn,
) -> None:
    """Initialize worker globals for multiprocessing full brute force."""
    global _brute_worker_corr
    global _brute_worker_k
    global _brute_worker_top_k
    global _brute_worker_scoring_fn

    _brute_worker_corr = corr
    _brute_worker_k = k
    _brute_worker_top_k = top_k
    _brute_worker_scoring_fn = scoring_fn



def _brute_mp_worker(prefix: int) -> List[Tuple[float, Tuple[int, ...]]]:
    """Evaluate all full brute-force combinations starting with a fixed prefix."""
    corr = _brute_worker_corr
    k = _brute_worker_k
    top_k = _brute_worker_top_k
    scoring_fn = _brute_worker_scoring_fn

    if corr is None or k is None or top_k is None or scoring_fn is None:
        raise RuntimeError("Brute worker globals were not initialized correctly")

    n = corr.shape[0]
    heap: List[Tuple[float, Tuple[int, ...]]] = []

    for rest in itertools.combinations(range(prefix + 1, n), k - 1):
        combo = (prefix,) + rest
        score = scoring_fn(corr, combo)
        if score == float("-inf"):
            continue

        item = (score, combo)
        if len(heap) < top_k:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)

    return heap



def bruteforce_mp(
    corr: np.ndarray,
    symbols: list[str],
    k: int,
    top_k: int = 10,
    scoring_fn: ScoreFn = combo_logdet,
    n_jobs: Optional[int] = None,
) -> PortfolioResult:
    """Multiprocessing full brute-force benchmark.

    The search tree is partitioned by fixing the first index, so each worker
    explores a disjoint subset of the full combination space.
    """
    validate_corr(corr)

    n = corr.shape[0]
    if len(symbols) != n:
        raise ValueError("symbols length must match corr dimensions")
    if k > n:
        raise ValueError("k cannot exceed number of assets")

    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)

    prefixes = list(range(0, n - k + 1))

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=n_jobs,
        initializer=_brute_mp_init,
        initargs=(corr, k, top_k, scoring_fn),
    ) as pool:
        worker_results = pool.map(_brute_mp_worker, prefixes)

    merged: List[Tuple[float, Tuple[int, ...]]] = []
    for worker_heap in worker_results:
        for item in worker_heap:
            if len(merged) < top_k:
                heapq.heappush(merged, item)
            elif item[0] > merged[0][0]:
                heapq.heapreplace(merged, item)

    ranked_results = sorted(merged, key=lambda x: x[0], reverse=True)
    return _build_result(
        method="bruteforce_mp",
        k=k,
        top_k=top_k,
        ranked_results=ranked_results,
        symbols=symbols,
        n_jobs=int(n_jobs),
    )


__all__ = [
    "beam",
    "greedy",
    "bruteforce",
    "bruteforce_mp",
]