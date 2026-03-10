# Diversifind

**Diversifind** is a Python toolkit for finding the **most diversified subset of assets** from a large universe.

Given a correlation matrix and a target portfolio size $k$, 
the library searches for the portfolio that **maximizes the determinant of the correlation matrix**, 
which corresponds to selecting the most **independent return streams**.

This approach is particularly useful for **systematic trading and strategy portfolios**, 
where the goal is to combine independent signals rather than minimize variance.

---

# Quick Example

```python
import numpy as np
from diversifind import beam

# Example: random correlation matrix
np.random.seed(0)
n_assets = 20

A = np.random.randn(n_assets, n_assets)
cov = A @ A.T
std = np.sqrt(np.diag(cov))
corr = cov / np.outer(std, std)

symbols = [f"Asset_{i}" for i in range(n_assets)]

# Find the most diversified portfolio of size 5
results = beam(corr, symbols, k=5)

print(results.pretty())
```

Example output:

```
Top diversified portfolios:

Rank      LogDet  Symbols
-------------------------
1      -0.092668  Asset_8, Asset_9, Asset_12, Asset_16, Asset_19
2      -0.132573  Asset_2, Asset_4, Asset_11, Asset_14, Asset_15
3      -0.141984  Asset_1, Asset_2, Asset_4, Asset_16, Asset_17
```

---

# The Problem

Suppose you have **$M$ assets** and want to choose **$k$ assets** such that the portfolio contains the **most independent return streams**.

A natural objective is to maximize the determinant of the correlation matrix:

$$
\max_{S \subseteq \{1..M\}, |S|=k} \det(C_S)
$$

where $C_S$ is the correlation matrix of the selected assets.

A larger determinant implies:

- lower pairwise correlations
- higher effective rank
- more independent sources of return

However, this becomes a **combinatorial search problem**:

$$
\binom{M}{k}
$$

which grows extremely quickly.

For example:

| Universe | Portfolio size | Combinations |
|--------|--------|--------|
| 50 | 5 | 2,118,760 |
| 50 | 10 | 10,272,278,170 |
| 100 | 10 | 17,310,309,456,440 |

Exhaustive search quickly becomes infeasible.

---

# Algorithms Included

Diversifind implements several search strategies.

## Greedy Search

A fast heuristic:

1. Start with the best pair
2. Iteratively add the asset that maximizes the log-determinant increase

Very fast but not guaranteed optimal.

---

## Beam Search (Recommended)

Beam search maintains a pool of the **best partial portfolios** at each step.

Instead of exploring all possibilities, it expands only the most promising candidates.

Advantages:

- dramatically faster than brute force
- near-optimal results in practice
- tunable accuracy via `beam_width`

Example:

```python
results = beam(corr, symbols, k=10, beam_width=5000)
```

---

## Brute Force (Benchmark)

The library also includes a **multiprocessing brute-force search** used for benchmarking.

```python
from diversifind import bruteforce_mp

results = bruteforce_mp(corr, symbols, k=5)
```

This guarantees the optimal portfolio but becomes infeasible for large universes.

---

# Analytics Tools

Diversifind includes tools to analyze the diversification properties of portfolios.

Example:

```python
from diversifind import analyze_portfolio

best = results.best()

analysis = analyze_portfolio(
    corr=corr,
    combo=best.combo_indices,
    symbols=symbols
)

print(analysis["effective_rank"])
print(analysis["max_abs_corr"])
```

Diagnostics include:

- effective rank
- eigenvalue spectrum
- max / mean correlations
- correlation pair inspection

---

# Installation

## Install from PyPI (recommended)

Once the package is published, you will be able to install it directly with:

```
pip install diversifind
```

---

## Install from source (development)

If you want the latest development version:

```
git clone https://github.com/Varltia/diversifind.git
cd diversifind
pip install -e .
```

The `-e` flag installs the package in **editable mode**, so local code changes immediately affect the installed package.

---

# Example Scripts

The repository includes examples:

```
examples/basic_usage.py
```

Additional examples are available in the repository, including a **real data notebook** demonstrating the workflow on historical market data:

```
examples/real_data_example.ipynb
```

This notebook walks through loading price data, computing returns and correlations, running the search algorithms, 
and analyzing the resulting diversified portfolios.

---

# Research Report

This project began as an exploration of how to efficiently solve the **maximum determinant portfolio selection problem**.

The full write-up including experiments comparing greedy, beam search, and brute-force approaches is available here:

```
docs/research_report.md
```

For a reference of the public API and analytics utilities, see:

```
docs/api_guide.md
```

---

# Project Structure

```
diversifind/
    analytics.py
    results.py
    search_methods.py
    utils.py

tests/
    test_beam.py
    test_bruteforce.py
    test_greedy.py
    test_results.py
    test_utils.py
    test_analytics.py

examples/
    basic_usage.py
    real_data_example.ipynb
    data/
        sample_closes.csv

docs/
    research_report.md
    api_guide.md
```

---

# Use Cases

Diversifind can be useful for:

- systematic trading portfolios
- alpha stream diversification
- factor portfolio construction
- signal selection
- machine learning ensemble diversification

---

# License

MIT License
