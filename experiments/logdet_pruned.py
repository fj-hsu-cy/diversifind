import multiprocessing as mp
import numpy as np
import heapq
from typing import List, Tuple, Callable, Optional


def combo_logdet(corr: np.ndarray, combo: Tuple[int, ...]) -> float:
    """
    Compute log(det(R_sub)) for a subset of assets.

    Parameters
    ----------
    corr : np.ndarray
        Full correlation matrix (N x N)
    combo : tuple[int]
        Indices of selected assets

    Returns
    -------
    float
        log determinant of the correlation submatrix
    """

    sub = corr[np.ix_(combo, combo)]
    sign, logdet = np.linalg.slogdet(sub)

    if sign <= 0:
        return float("-inf")

    return float(logdet)


def _validate_corr(corr: np.ndarray) -> None:
    """
    Validate that corr is a 2D square symmetric matrix with finite values in [-1, 1].

    Parameters
    ----------
    corr : np.ndarray
        Correlation matrix to validate

    Raises
    ------
    ValueError
        If validation checks fail
    """
    if corr.ndim != 2:
        raise ValueError(f"Correlation matrix must be 2D, got {corr.ndim}D array")

    n, m = corr.shape
    if n != m:
        raise ValueError(f"Correlation matrix must be square, got shape {corr.shape}")

    if not np.allclose(corr, corr.T, atol=1e-8):
        raise ValueError("Correlation matrix must be symmetric")

    if not np.isfinite(corr).all():
        raise ValueError("Correlation matrix contains non-finite values")

    if np.any(corr < -1 - 1e-6) or np.any(corr > 1 + 1e-6):
        raise ValueError("Correlation matrix values must be in [-1, 1]")


def search_pruned_corr(
    corr: np.ndarray,
    k: int,
    tau: float = 0.3,
    top_k: int = 10,
    scoring_fn: Callable[[np.ndarray, Tuple[int, ...]], float] = combo_logdet,
) -> List[Tuple[float, Tuple[int, ...]]]:
    """
    Find the most diversified subset of assets using pruned brute-force search on a correlation matrix.

    The diversification score is defined by `scoring_fn`, defaulting to log det of the submatrix.

    Pruning rule:
        |corr(i, j)| <= tau
    Any pair violating the threshold is never allowed in a candidate set.

    Parameters
    ----------
    corr : np.ndarray
        Correlation matrix (N x N)
    k : int
        Number of assets to select
    tau : float
        Maximum allowed pairwise absolute correlation
    top_k : int
        Number of best combinations to keep
    scoring_fn : callable
        Function to score a subset given the correlation matrix and asset indices

    Returns
    -------
    list
        Sorted list of (score, combo) from best to worst
    """
    _validate_corr(corr)

    abs_corr = np.abs(corr)
    n = corr.shape[0]

    if k > n:
        raise ValueError("k cannot exceed number of assets")

    # Precompute forbidden indices for each asset:
    # bad_idx[i] holds indices that cannot co-exist with i because |corr| > tau
    bad_idx: List[np.ndarray] = []
    for i in range(n):
        idx = np.flatnonzero(abs_corr[i] > tau)
        idx = idx[idx != i]  # exclude self
        bad_idx.append(idx.astype(int))

    heap: List[Tuple[float, Tuple[int, ...]]] = []

    def maybe_push(combo: List[int]):
        score = scoring_fn(corr, tuple(combo))

        if score == float("-inf"):
            return

        item = (score, tuple(combo))

        if len(heap) < top_k:
            heapq.heappush(heap, item)
        else:
            if score > heap[0][0]:
                heapq.heapreplace(heap, item)

    # Use a boolean mask for forbidden assets to improve performance and avoid repeated set allocations.
    # This mask tracks assets that cannot be chosen due to pruning constraints.
    forbidden = np.zeros(n, dtype=bool)
    chosen: List[int] = []

    def backtrack(start_idx: int) -> None:
        depth = len(chosen)
        remaining = k - depth

        if remaining == 0:
            maybe_push(chosen)
            return

        if start_idx >= n:
            return

        if (n - start_idx) < remaining:
            return

        for j in range(start_idx, n):
            if forbidden[j]:
                continue

            # Apply pruning: mark j and its forbidden neighbors as forbidden,
            # record changes to undo after recursion.
            changed: List[int] = []

            if not forbidden[j]:
                forbidden[j] = True
                changed.append(j)

            for u in bad_idx[j]:
                if not forbidden[u]:
                    forbidden[u] = True
                    changed.append(u)

            chosen.append(j)
            backtrack(j + 1)
            chosen.pop()

            # Undo the forbidden markings
            for u in changed:
                forbidden[u] = False

    backtrack(0)

    results = sorted(heap, key=lambda x: x[0], reverse=True)

    return results


# ---------------------------------------------------------------------
# Multiprocessing implementation
# ---------------------------------------------------------------------

# Worker globals (set once per worker process)
_worker_corr: Optional[np.ndarray] = None
_worker_bad_idx: Optional[List[np.ndarray]] = None
_worker_k: Optional[int] = None
_worker_tau: Optional[float] = None
_worker_top_k: Optional[int] = None
_worker_scoring_fn: Optional[Callable[[np.ndarray, Tuple[int, ...]], float]] = None


def _mp_init(corr, bad_idx, k, tau, top_k, scoring_fn):
    """
    Initializer for worker processes.
    Sets read-only globals to avoid repeatedly pickling large objects.
    """
    global _worker_corr
    global _worker_bad_idx
    global _worker_k
    global _worker_tau
    global _worker_top_k
    global _worker_scoring_fn

    _worker_corr = corr
    _worker_bad_idx = bad_idx
    _worker_k = k
    _worker_tau = tau
    _worker_top_k = top_k
    _worker_scoring_fn = scoring_fn


def _mp_worker(prefix: int) -> List[Tuple[float, Tuple[int, ...]]]:
    """
    Worker that explores all combinations starting with a fixed prefix index.
    """

    corr = _worker_corr
    bad_idx = _worker_bad_idx
    k = _worker_k
    top_k = _worker_top_k
    scoring_fn = _worker_scoring_fn

    n = corr.shape[0]

    heap: List[Tuple[float, Tuple[int, ...]]] = []

    forbidden = np.zeros(n, dtype=bool)
    chosen: List[int] = []

    def maybe_push(combo: List[int]):
        score = scoring_fn(corr, tuple(combo))

        if score == float("-inf"):
            return

        item = (score, tuple(combo))

        if len(heap) < top_k:
            heapq.heappush(heap, item)
        else:
            if score > heap[0][0]:
                heapq.heapreplace(heap, item)

    def backtrack(start_idx: int):
        depth = len(chosen)
        remaining = k - depth

        if remaining == 0:
            maybe_push(chosen)
            return

        if start_idx >= n:
            return

        if (n - start_idx) < remaining:
            return

        for j in range(start_idx, n):
            if forbidden[j]:
                continue

            changed: List[int] = []

            if not forbidden[j]:
                forbidden[j] = True
                changed.append(j)

            for u in bad_idx[j]:
                if not forbidden[u]:
                    forbidden[u] = True
                    changed.append(u)

            chosen.append(j)
            backtrack(j + 1)
            chosen.pop()

            for u in changed:
                forbidden[u] = False

    # Apply the prefix choice
    j = prefix

    forbidden[j] = True
    for u in bad_idx[j]:
        forbidden[u] = True

    chosen.append(j)

    backtrack(j + 1)

    return heap


def search_pruned_corr_mp(
    corr: np.ndarray,
    k: int,
    tau: float = 0.3,
    top_k: int = 10,
    scoring_fn: Callable[[np.ndarray, Tuple[int, ...]], float] = combo_logdet,
    n_jobs: Optional[int] = None,
) -> List[Tuple[float, Tuple[int, ...]]]:
    """
    Multiprocessing version of the pruned search.

    The search tree is partitioned by fixing the first index (prefix),
    allowing each worker to explore independent branches safely.
    """

    _validate_corr(corr)

    abs_corr = np.abs(corr)
    n = corr.shape[0]

    if k > n:
        raise ValueError("k cannot exceed number of assets")

    bad_idx: List[np.ndarray] = []
    for i in range(n):
        idx = np.flatnonzero(abs_corr[i] > tau)
        idx = idx[idx != i]
        bad_idx.append(idx.astype(int))

    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)

    ctx = mp.get_context("spawn")

    prefixes = list(range(n))

    with ctx.Pool(
        processes=n_jobs,
        initializer=_mp_init,
        initargs=(corr, bad_idx, k, tau, top_k, scoring_fn),
    ) as pool:

        worker_results = pool.map(_mp_worker, prefixes)

    # Merge heaps
    merged: List[Tuple[float, Tuple[int, ...]]] = []

    for h in worker_results:
        for item in h:
            if len(merged) < top_k:
                heapq.heappush(merged, item)
            else:
                if item[0] > merged[0][0]:
                    heapq.heapreplace(merged, item)

    results = sorted(merged, key=lambda x: x[0], reverse=True)

    return results