"""
Basic usage example for the diversifind package.

This script demonstrates how to:
1. Construct a sample correlation matrix
2. Run the beam search diversification algorithm
3. Inspect the resulting diversified portfolio
"""

import numpy as np

from diversifind import beam, analyze_portfolio, top_abs_corr_pairs


# ---------------------------------------------------------------------
# Generate a sample correlation matrix
# ---------------------------------------------------------------------

np.random.seed(0)

n_assets = 20
symbols = [f"Asset_{i}" for i in range(n_assets)]

# Create a random symmetric matrix and normalize into a correlation matrix
A = np.random.uniform(-0.3, 0.8, size=(n_assets, n_assets))
corr = (A + A.T) / 2
np.fill_diagonal(corr, 1.0)


# ---------------------------------------------------------------------
# Run diversification search
# ---------------------------------------------------------------------

portfolio_size = 5

result = beam(
    corr=corr,
    symbols=symbols,
    k=portfolio_size,
    beam_width=500,
    top_k=5,
)


# ---------------------------------------------------------------------
# Inspect results
# ---------------------------------------------------------------------

print(result.pretty())
print()


# ---------------------------------------------------------------------
# Access the best portfolio directly
# ---------------------------------------------------------------------

best = result.best()

print("Best portfolio only:")
print("Symbols:", best.combo_symbols)
print("Log determinant:", best.logdet)
print()


# ---------------------------------------------------------------------
# Run analytics on the best portfolio
# ---------------------------------------------------------------------

analysis = analyze_portfolio(
    corr=corr,
    combo=best.combo_indices,
    symbols=symbols,
)

print("Analytics for the best portfolio:")
print(f"Effective rank: {analysis['effective_rank']:.4f}")
print(f"Max absolute correlation: {analysis['max_abs_corr']:.4f}")
print(f"Mean absolute correlation: {analysis['mean_abs_corr']:.4f}")
print(f"Min eigenvalue: {analysis['min_eigenvalue']:.4f}")
print(f"Max eigenvalue: {analysis['max_eigenvalue']:.4f}")
print()


# ---------------------------------------------------------------------
# Inspect strongest correlation pairs inside the best portfolio
# ---------------------------------------------------------------------

pairs = top_abs_corr_pairs(
    corr=corr,
    combo=best.combo_indices,
    symbols=symbols,
    top_m=3,
)

print("Strongest correlation pairs inside the best portfolio:")
for row in pairs:
    print(f"{row['pair_symbols'][0]} -- {row['pair_symbols'][1]} | |corr|={row['abs_corr']:.4f}")
