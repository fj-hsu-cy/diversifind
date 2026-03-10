"""
runtime_scaling.py

Benchmark script showing how runtime scales with universe size for:

1. Brute-force search
2. Pruned search
3. Parallel pruned search

This script is intended for the showcase repository to demonstrate the
computational advantage of pruning and multiprocessing.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from experiments.logdet_bruteforce import search_bruteforce_corr
from experiments.logdet_pruned import search_pruned_corr, search_pruned_corr_mp

import json
from pathlib import Path
from typing import Optional, Tuple


def random_corr_matrix(n: int, seed: int = 0) -> np.ndarray:
    """
    Generate a random correlation matrix using simulated returns.
    """

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(3000, n))

    return np.corrcoef(X, rowvar=False)


def benchmark(
    n_assets: int,
    k: int,
    tau: float,
    repeats: int = 3,
    run_bruteforce: bool = True,
) -> Tuple[Optional[float], float, float]:
    """Benchmark the three methods on a random correlation matrix.

    Args:
        n_assets: Universe size (number of assets).
        k: Portfolio size to select.
        tau: Pruning threshold.
        repeats: Number of repeated runs per method (median is reported).
        run_bruteforce: If False, skip the brute-force method.

    Returns:
        (brute_time, pruned_time, parallel_time) where brute_time may be None.
    """

    corr = random_corr_matrix(n_assets, seed=42)

    def time_call(fn) -> float:
        times = []
        for _ in range(repeats):
            start = time.time()
            fn()
            times.append(time.time() - start)
        return float(np.median(times))

    brute_time: Optional[float] = None

    if run_bruteforce:
        brute_time = time_call(
            lambda: search_bruteforce_corr(
                corr=corr,
                k=k,
                tau=tau,
                top_k=10,
            )
        )

    pruned_time = time_call(
        lambda: search_pruned_corr(
            corr=corr,
            k=k,
            tau=tau,
            top_k=10,
        )
    )

    parallel_time = time_call(
        lambda: search_pruned_corr_mp(
            corr=corr,
            k=k,
            tau=tau,
            top_k=10,
        )
    )

    return brute_time, pruned_time, parallel_time


def main():

    tau = 0.3
    k = 6

    # Larger universe sizes to highlight scaling behavior
    sizes = list(range(20, 101, 10))  # 20, 30, 40, ..., 100

    repeats = 3
    bruteforce_cutoff = 50  # brute force becomes impractical beyond this for k=6

    brute_times: list[Optional[float]] = []
    pruned_times: list[float] = []
    parallel_times: list[float] = []

    results = {
        "tau": tau,
        "k": k,
        "repeats": repeats,
        "bruteforce_cutoff": bruteforce_cutoff,
        "sizes": sizes,
        "timings": [],
    }

    print("Running scaling benchmarks...\n")

    for n in sizes:
        run_bruteforce = n <= bruteforce_cutoff

        print(f"Testing universe size n={n} (bruteforce={'on' if run_bruteforce else 'off'})")

        brute, pruned, parallel = benchmark(
            n_assets=n,
            k=k,
            tau=tau,
            repeats=repeats,
            run_bruteforce=run_bruteforce,
        )

        brute_times.append(brute)
        pruned_times.append(pruned)
        parallel_times.append(parallel)

        if brute is not None:
            speedup_prune = brute / pruned if pruned > 0 else float("inf")
            speedup_parallel = brute / parallel if parallel > 0 else float("inf")
            print(f"  brute force : {brute:.3f}s")
            print(f"  pruned      : {pruned:.3f}s  (speedup vs brute: {speedup_prune:.1f}x)")
            print(f"  parallel    : {parallel:.3f}s  (speedup vs brute: {speedup_parallel:.1f}x)")
        else:
            print("  brute force : skipped")
            print(f"  pruned      : {pruned:.3f}s")
            print(f"  parallel    : {parallel:.3f}s")

        results["timings"].append(
            {
                "n": n,
                "brute": brute,
                "pruned": pruned,
                "parallel": parallel,
            }
        )

        print()

    # Save results to disk for reuse in README / plots
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "runtime_scaling.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to: {out_path}")

    # -----------------
    # plot results
    # -----------------

    plt.figure()

    # Plot brute force only where available
    brute_x = [n for n, t in zip(sizes, brute_times) if t is not None]
    brute_y = [t for t in brute_times if t is not None]

    if brute_x:
        plt.plot(brute_x, brute_y, marker="o", label="Brute Force")

    plt.plot(sizes, pruned_times, marker="o", label="Pruned")
    plt.plot(sizes, parallel_times, marker="o", label="Pruned + Parallel")

    plt.xlabel("Number of Assets (n)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Scaling of Portfolio Diversification Search")

    # Log scale makes scaling differences clearer as runtimes diverge
    plt.yscale("log")

    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()