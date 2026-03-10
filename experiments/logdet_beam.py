

"""Beam search for log-determinant portfolio diversification.

This module provides a simple beam-search baseline for the diversification
problem

    max logdet(C_S)

where C_S is the correlation submatrix of a chosen subset S.

The algorithm is heuristic: at each depth it keeps only the best
`beam_width` partial portfolios, expands them by one asset, scores the
children, and again keeps only the best `beam_width` children.

This makes beam search much cheaper than brute force, but it does not
guarantee recovery of the global optimum.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def combo_logdet(corr: np.ndarray, combo: tuple[int, ...]) -> float:
    """Compute log(det(corr[combo, combo])) safely.

    Args:
        corr: Correlation matrix.
        combo: Tuple of row/column indices defining the submatrix.

    Returns:
        The log-determinant of the selected correlation submatrix. Returns
        ``-inf`` if the submatrix is not positive definite.
    """
    idx = np.array(combo, dtype=int)
    sub = corr[np.ix_(idx, idx)]
    sign, logdet = np.linalg.slogdet(sub)
    if sign <= 0:
        return float("-inf")
    return float(logdet)



def beam_logdet_search(
    corr: np.ndarray,
    symbols: list[str],
    k: int,
    beam_width: int = 5000,
    top_k: int = 10,
) -> dict[str, Any]:
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

    Raises:
        ValueError: If ``k`` is out of range or ``corr`` / ``symbols`` are
            inconsistent.
    """
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("corr must be a square matrix")

    n = corr.shape[0]
    if len(symbols) != n:
        raise ValueError("symbols length must match corr dimensions")
    if k < 2:
        raise ValueError("k must be >= 2")
    if k > n:
        raise ValueError(f"k={k} > n={n}")

    beam_width = max(1, int(beam_width))
    top_k = max(1, int(top_k))

    # ------------------------------------------------------------------
    # Initialize beam with all 2-asset portfolios.
    # ------------------------------------------------------------------
    beam: dict[tuple[int, ...], float] = {}
    for i in range(n - 1):
        for j in range(i + 1, n):
            combo = (i, j)
            ld = combo_logdet(corr, combo)
            if ld == float("-inf"):
                continue
            beam[combo] = ld

    if not beam:
        return {"k": int(k), "beam_width": int(beam_width), "top_results": []}

    beam = dict(
        sorted(beam.items(), key=lambda kv: kv[1], reverse=True)[:beam_width]
    )

    # ------------------------------------------------------------------
    # Expand one level at a time until reaching size k.
    # ------------------------------------------------------------------
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
                # Keep only the best score for each distinct child.
                prev = next_beam.get(new_combo)
                if prev is None or ld > prev:
                    next_beam[new_combo] = ld

        if not next_beam:
            break

        beam = dict(
            sorted(next_beam.items(), key=lambda kv: kv[1], reverse=True)[:beam_width]
        )
        cur_len += 1

    # ------------------------------------------------------------------
    # Format top final portfolios.
    # ------------------------------------------------------------------
    final = [(c, ld) for c, ld in beam.items() if len(c) == k]
    final.sort(key=lambda x: x[1], reverse=True)
    final = final[:top_k]

    top_results: list[dict[str, Any]] = []
    for rank, (combo, ld) in enumerate(final, 1):
        top_results.append(
            {
                "rank": int(rank),
                "logdet": float(ld),
                "combo_indices": list(combo),
                "combo_symbols": [symbols[i] for i in combo],
            }
        )

    return {
        "k": int(k),
        "beam_width": int(beam_width),
        "top_results": top_results,
    }