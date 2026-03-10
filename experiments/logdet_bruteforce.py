import heapq
import itertools
import multiprocessing as mp
from typing import Callable, List, Optional, Tuple

import numpy as np


def _validate_corr(corr: np.ndarray) -> None:
    """Validate that the input is a square correlation matrix."""
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("corr must be a square matrix")


# ---------------------------------------------------------------------
# Multiprocessing implementation
# ---------------------------------------------------------------------

# Worker globals (set once per worker process)
_worker_corr: Optional[np.ndarray] = None
_worker_k: Optional[int] = None
_worker_top_k: Optional[int] = None
_worker_scoring_fn: Optional[Callable[[np.ndarray, Tuple[int, ...]], float]] = None


def combo_logdet(corr: np.ndarray, combo: Tuple[int, ...]) -> float:
    """
    Compute the log-determinant diversification score for a subset of assets.

    Parameters
    ----------
    corr : np.ndarray
        Full correlation matrix (N x N).
    combo : tuple[int]
        Indices of assets forming the candidate subset.

    Returns
    -------
    float
        Log determinant of the correlation submatrix.
    """

    sub = corr[np.ix_(combo, combo)]

    # numerical stability
    sign, logdet = np.linalg.slogdet(sub)

    if sign <= 0:
        return -np.inf

    return logdet


def search_bruteforce_corr(
    corr: np.ndarray,
    k: int,
    tau: float = 1,
    top_k: int = 10,
    scoring_fn: Callable[[np.ndarray, Tuple[int, ...]], float] = combo_logdet,
) -> List[Tuple[float, Tuple[int, ...]]]:
    """
    Naive brute-force baseline implementation to search for the most diversified subset of assets.

    This function operates directly on a correlation matrix and generates every combination
    of assets of size k, then filters out those with pairwise correlations above the threshold tau.

    Parameters
    ----------
    corr : np.ndarray
        Full correlation matrix (N x N).
    k : int
        Number of assets to select.
    tau : float
        Correlation pruning threshold. Combinations with any pairwise correlation above tau are discarded.
    top_k : int
        Number of best solutions to keep.
    scoring_fn : Callable
        Function to compute the diversification score for a given combination.

    Returns
    -------
    list
        List of (score, combo) sorted from best to worst.
    """

    _validate_corr(corr)
    n_assets = corr.shape[0]
    if k > n_assets:
        raise ValueError("k cannot exceed number of assets")

    best = []

    # Generate every combination and then prune invalid ones
    for combo in itertools.combinations(range(n_assets), k):

        # Prune combinations containing pairs with correlation above tau
        valid = True
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                if abs(corr[combo[i], combo[j]]) > tau:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        score = scoring_fn(corr, combo)

        best.append((score, combo))

    best.sort(reverse=True)

    return best[:top_k]


def _mp_init(
    corr: np.ndarray,
    k: int,
    top_k: int,
    scoring_fn: Callable[[np.ndarray, Tuple[int, ...]], float],
) -> None:
    """Initializer for brute-force worker processes.

    The full correlation matrix and other configuration are stored once per
    worker process to avoid repeatedly pickling them for every task.
    """
    global _worker_corr
    global _worker_k
    global _worker_top_k
    global _worker_scoring_fn

    _worker_corr = corr
    _worker_k = k
    _worker_top_k = top_k
    _worker_scoring_fn = scoring_fn



def _mp_worker(prefix: int) -> List[Tuple[float, Tuple[int, ...]]]:
    """Evaluate all brute-force combinations starting with a fixed prefix.

    Each worker is assigned a first index ``prefix`` and enumerates all
    combinations of size ``k`` that begin with that index. This partitions the
    full brute-force search tree into independent branches with no duplication.
    """
    corr = _worker_corr
    k = _worker_k
    top_k = _worker_top_k
    scoring_fn = _worker_scoring_fn

    if corr is None or k is None or top_k is None or scoring_fn is None:
        raise RuntimeError("Worker globals were not initialized correctly")

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
        else:
            if score > heap[0][0]:
                heapq.heapreplace(heap, item)

    return heap



def search_bruteforce_corr_mp(
    corr: np.ndarray,
    k: int,
    top_k: int = 10,
    scoring_fn: Callable[[np.ndarray, Tuple[int, ...]], float] = combo_logdet,
    n_jobs: Optional[int] = None,
) -> List[Tuple[float, Tuple[int, ...]]]:
    """Multiprocessing full brute-force search.

    This function evaluates the original unconstrained optimization problem by
    enumerating every combination of size ``k``. The search tree is partitioned
    by fixing the first index (prefix), allowing workers to explore independent
    branches safely and merge only their local top-k heaps.

    Args:
        corr: Full correlation matrix (N x N).
        k: Number of assets to select.
        top_k: Number of best solutions to keep.
        scoring_fn: Function used to score each combination.
        n_jobs: Number of worker processes. If ``None``, use ``cpu_count() - 1``.

    Returns:
        List of ``(score, combo)`` sorted from best to worst.
    """
    _validate_corr(corr)

    n = corr.shape[0]
    if k > n:
        raise ValueError("k cannot exceed number of assets")

    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)

    # Only prefixes that can still fit a full combination need to be explored.
    prefixes = list(range(0, n - k + 1))

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=n_jobs,
        initializer=_mp_init,
        initargs=(corr, k, top_k, scoring_fn),
    ) as pool:
        worker_results = pool.map(_mp_worker, prefixes)

    merged: List[Tuple[float, Tuple[int, ...]]] = []
    for worker_heap in worker_results:
        for item in worker_heap:
            if len(merged) < top_k:
                heapq.heappush(merged, item)
            else:
                if item[0] > merged[0][0]:
                    heapq.heapreplace(merged, item)

    return sorted(merged, key=lambda x: x[0], reverse=True)