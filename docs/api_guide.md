# Diversifind API Guide

This guide documents the public interface of **Diversifind**, a library for
searching highly diversified portfolios by maximizing the determinant of the
correlation submatrix (log‑det objective).

The package provides:

- Search algorithms for selecting diversified asset sets
- Structured result objects for working with search outputs
- Analytics helpers for diagnosing portfolio diversification

Diversifind focuses specifically on diversification search, rather than portfolio weighting or optimization.

---

# 1. Core Search Methods

All search methods operate on the same inputs:

- **corr** — an `NxN` correlation matrix
- **symbols** — asset identifiers aligned with the matrix
- **k** — number of assets to select

All search methods return a `PortfolioResult` object.

---

## 1.1 Beam Search (Recommended)

```python
from diversifind import beam

result = beam(
    corr,
    symbols,
    k=10,
    beam_width=500,
    top_k=10,
)
```

Beam search expands promising portfolios layer‑by‑layer while keeping only the
best `beam_width` candidates at each stage.

### Parameters

| Parameter | Description |
|---|---|
| corr | Correlation matrix (`NxN` numpy array) |
| symbols | Asset identifiers aligned with the matrix |
| k | Portfolio size |
| beam_width | Number of candidate portfolios kept per layer |
| top_k | Number of final portfolios returned |

### When to use

Beam search is the **recommended algorithm** for most real use cases.

It provides near‑optimal solutions while scaling to much larger search spaces
than brute force.

---

## 1.2 Greedy Search

```python
from diversifind import greedy

result = greedy(corr, symbols, k=10)
```

Greedy search builds the portfolio one asset at a time by selecting the asset
that maximizes the log‑det score at each step.

### Characteristics

- Extremely fast
- No tuning parameters
- May miss globally optimal portfolios

Greedy search can be useful for quick approximations.

---

## 1.3 Full Brute Force

```python
from diversifind import bruteforce

result = bruteforce(corr, symbols, k=6)
```

This method evaluates **every possible combination** of assets.

It guarantees the globally optimal portfolio but scales combinatorially.

### Complexity

```
O(n choose k)
```

Brute force is useful for:

- benchmarking
- validation
- small universes

---

## 1.4 Multiprocessing Brute Force

```python
from diversifind import bruteforce_mp

result = bruteforce_mp(
    corr,
    symbols,
    k=7,
    n_jobs=8
)
```

This version distributes the brute‑force search across multiple CPU cores.

The search space is partitioned by fixing the first asset in each
combination, allowing workers to explore disjoint branches.

### Parameters

| Parameter | Description |
|---|---|
| n_jobs | Number of worker processes |

---

# 2. Result Objects

All search methods return a **PortfolioResult** object.

## 2.1 PortfolioResult

The result object contains:

- search method used
- portfolio size
- ranked list of portfolios

Example:

```python
result = beam(corr, symbols, k=5)
```

### Access the best portfolio

```python
best = result.best()
print(best.combo_symbols)
```

### Print a formatted summary

```python
print(result.pretty())
```

Example output:

```
Method: beam
Portfolio size: 5
Returned results: 10

Top diversified portfolios:

Rank  LogDet      Assets
1     -0.093      GLDM, IGV, NVO, UNG, VZ
2     -0.103      GLDM, IGV, NVO, PM, UNG
```

### Convert to dictionary

```python
payload = result.to_dict()
```

### Restore from dictionary

```python
from diversifind.results import result_from_dict

result = result_from_dict(payload)
```

---

## 2.2 PortfolioEntry

Each portfolio inside a result is represented as a `PortfolioEntry`.

Fields:

- `rank`
- `logdet`
- `combo_indices`
- `combo_symbols`

---

# 3. Analytics

Analytics helpers help interpret why a portfolio is diversified.

```python
from diversifind.analytics import analyze_portfolio

analysis = analyze_portfolio(corr, combo=[0,1,2,3,4], symbols=symbols)
```

The output includes:

- logdet
- pairwise correlation statistics
- eigenvalue diagnostics
- effective rank

---

## 3.1 Portfolio Diagnostics

```python
analysis = analyze_portfolio(corr, combo)
```

Example output fields:

- `logdet`
- `effective_rank`
- `max_abs_corr`
- `mean_abs_corr`
- `min_eigenvalue`

---

## 3.2 Analyze Search Results

```python
from diversifind.analytics import analyze_results

analysis = analyze_results(corr, result.to_dict(), symbols)
```

### Inspecting the Results

`analyze_results` returns a dictionary containing diagnostics for each ranked
portfolio. You can inspect the returned structure directly:

```python
analysis = analyze_results(corr, result.to_dict(), symbols)

print(analysis["method"])     # search method used
print(analysis["k"])          # portfolio size
print(analysis["n_results"])  # number of portfolios analyzed
```

Each portfolio's diagnostics are stored inside `portfolio_analytics`:

```python
for row in analysis["portfolio_analytics"]:
    print("Rank:", row["rank"])
    print("Assets:", row.get("combo_symbols"))
    print("LogDet:", row["logdet"])
    print("Effective Rank:", row["effective_rank"])
    print("Max Abs Corr:", row["max_abs_corr"])
    print("Min Eigen Value:", row["min_eigenvalue"])
```

Typical diagnostics returned for each portfolio include:

| Field | Description |
|------|-------------|
| `logdet` | Diversification objective value |
| `effective_rank` | Dimensionality of the portfolio's correlation structure |
| `max_abs_corr` | Strongest pairwise correlation within the portfolio |
| `mean_abs_corr` | Average absolute pairwise correlation |
| `min_eigenvalue` | Smallest eigenvalue of the portfolio correlation matrix |

These diagnostics help explain **why a portfolio is diversified**, not just
which portfolio was selected by the search algorithm.


This computes diagnostics for every returned portfolio.

---

## 3.3 Strongest Correlation Pairs

```python
from diversifind.analytics import top_abs_corr_pairs

pairs = top_abs_corr_pairs(corr, combo, symbols)
```

Returns the most correlated asset pairs within the portfolio.

Example structure of returned results:

```python
[
    {
        "abs_corr": 0.42,
        "pair_indices": [3, 7],
        "pair_symbols": ["AAPL", "MSFT"]
    },
    {
        "abs_corr": 0.38,
        "pair_indices": [1, 4],
        "pair_symbols": ["GLDM", "VZ"]
    }
]
```

Fields:

| Field | Description |
|------|-------------|
| `abs_corr` | Absolute correlation between the two assets |
| `pair_indices` | Asset indices within the original universe |
| `pair_symbols` | Asset identifiers if `symbols` were provided |

This function is useful for identifying **hidden correlation clusters** inside
an otherwise diversified portfolio.

---

## 3.4 Correlation Distribution Diagnostics

```python
from diversifind.analytics import corr_distribution_summary

summary = corr_distribution_summary(corr)
```

This provides statistics describing the overall correlation structure of the
search universe.

Example output:

```python
{
    "n_assets": 80,
    "n_pairs": 3160,
    "q10": 0.05,
    "q20": 0.08,
    "q30": 0.11,
    "q40": 0.14,
    "q50": 0.18,
    "q60": 0.22,
    "q70": 0.27,
    "q80": 0.34,
    "q90": 0.48,
    "mean_abs_corr": 0.19,
    "max_abs_corr": 0.92
}
```

Fields:

| Field | Description |
|------|-------------|
| `n_assets` | Number of assets in the universe |
| `n_pairs` | Number of unique asset pairs |
| `qXX` | Quantiles of the absolute correlation distribution |
| `mean_abs_corr` | Average absolute correlation |
| `max_abs_corr` | Highest observed absolute correlation |

This diagnostic helps users understand the **overall correlation structure of
the search universe**, which can provide intuition about how difficult the
diversification problem will be.

---

# 4. Input Requirements

### Correlation Matrix

The correlation matrix must:

- be square
- be symmetric
- have ones on the diagonal

```
NxN numpy array
```

### Symbol Alignment

`symbols` must correspond to rows and columns of the correlation matrix.

```
len(symbols) == corr.shape[0]
```

---

# 5. Recommended Workflow

A typical workflow looks like this:

```python
from diversifind import beam
from diversifind.analytics import analyze_portfolio

result = beam(corr, symbols, k=10)

best = result.best()

analysis = analyze_portfolio(
    corr,
    combo=best.combo_indices,
    symbols=symbols
)
```

Beam search finds candidate portfolios, and analytics tools help interpret
how diversified they are.

---

# 6. Choosing a Search Method

| Method | Speed     | Optimal | Use Case |
|---|-----------|---|---|
| greedy | very fast | no | quick approximation |
| beam | fast      | usually | recommended default |
| bruteforce | slowest   | yes | benchmarking |
| bruteforce_mp | slow      | yes | parallel benchmark |

⚠️ **Warning**

Brute-force methods scale combinatorially with portfolio size:

    O(n choose k)

Even with multiprocessing, runtimes can grow extremely quickly as the
universe size or portfolio size increases.

Use brute-force search primarily for:

- benchmarking
- validation
- small universes

For most real applications, **beam search** provides a much better
speed–accuracy tradeoff.

---

# 7. Practical Advice

For most users:

- Use **beam search** with a reasonably large beam width
- Use **bruteforce** only for validation or small problems
- Use **analytics tools** to understand the resulting portfolio
