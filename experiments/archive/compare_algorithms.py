

"""
compare_algorithms.py

Demonstration script comparing the naive brute‑force search and the
pruned brute‑force search for diversified portfolios.

This script is intended for the showcase repository so readers can see:

1. How to call both algorithms
2. How much faster pruning can be
3. That both methods find similar solutions

The script generates a synthetic correlation matrix and runs both
algorithms on it.
"""

import time
import numpy as np

from experiments.logdet_bruteforce import search_bruteforce_corr
from experiments.logdet_pruned import search_pruned_corr


def random_correlation_matrix(n: int, seed: int = 0) -> np.ndarray:
    """
    Generate a random positive‑definite correlation matrix.

    Parameters
    ----------
    n : int
        Number of assets

    Returns
    -------
    np.ndarray
        (n x n) correlation matrix
    """

    rng = np.random.default_rng(seed)

    # generate random returns
    X = rng.normal(size=(2000, n))

    # convert to correlation matrix
    corr = np.corrcoef(X, rowvar=False)

    return corr


def print_results(title, results, symbols=None):
    """
    Pretty print search results.
    """

    print(f"\n{title}")
    print("-" * len(title))

    for rank, (score, combo) in enumerate(results, start=1):

        if symbols is None:
            label = combo
        else:
            label = [symbols[i] for i in combo]

        print(f"{rank:2d}. score={score:.6f} | {label}")


def main():

    # problem setup
    n_assets = 50
    k = 6
    tau = 0.3

    print("Generating random correlation matrix...")

    corr = random_correlation_matrix(n_assets)

    # -----------------
    # brute force
    # -----------------

    print("\nRunning brute force search...")

    start = time.time()

    brute_results = search_bruteforce_corr(
        corr=corr,
        k=k,
        tau=tau,
        top_k=10,
    )

    brute_time = time.time() - start

    # -----------------
    # pruned search
    # -----------------

    print("Running pruned search...")

    start = time.time()

    pruned_results = search_pruned_corr(
        corr=corr,
        k=k,
        tau=tau,
        top_k=10,
    )

    pruned_time = time.time() - start

    # -----------------
    # results
    # -----------------

    print_results("Brute Force Top Results", brute_results)
    print_results("Pruned Search Top Results", pruned_results)

    print("\nRuntime comparison")
    print("-------------------")

    print(f"Brute force : {brute_time:.4f} seconds")
    print(f"Pruned      : {pruned_time:.4f} seconds")

    if pruned_time > 0:
        print(f"Speedup     : {brute_time / pruned_time:.2f}x")


if __name__ == "__main__":
    main()