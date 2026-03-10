# Diversified Portfolio Search

---

## Background

This project originated from my own research into systematic trading strategies. 
While experimenting with portfolio construction methods, 
I was working with a large universe of tradable instruments and trying to answer a seemingly simple question:

**If I can only trade a limited number of instruments, how do I choose the most diversified set possible?**

Many systematic trading frameworks emphasize diversification as one of the most reliable ways to 
improve portfolio robustness and Sharpe ratio. 
The intuition is simple: 
combining strategies or assets that behave differently reduces overall portfolio volatility and drawdowns.

However, once the universe of possible instruments becomes large, 
selecting the *best* diversified subset becomes surprisingly difficult.

For example, imagine starting with a universe of 80–150 liquid instruments and being able to trade only 10–15 of them,
whether it may be due to available funds or cost restrictions. 
The number of possible portfolios quickly becomes astronomical:

$$$
(100 choose 10) ≈ 1.7 × 10^13
$$$

Clearly, while evaluating every possible portfolio directly may be feasible with modern hardware,
it is time-consuming and brutish.

During my research I experimented with several common approaches:

- heuristic selection
- clustering methods
- greedy diversification algorithms
- beam searching

While these methods can produce reasonable portfolios, they do not guarantee that the result is **globally optimal** 
with respect to diversification.

In this context, the objective is not traditional **mean–variance optimization** or minimizing portfolio volatility. 
Instead, the goal is to identify instruments whose return streams are as **independent from each other as possible**. 
If each instrument contributes a largely independent return stream, 
combining them increases the stability of the overall strategy and tends to improve the portfolio's Sharpe ratio through diversification. 

This distinction is important: rather than trying to construct the *lowest-risk portfolio*, 
the goal is to construct a portfolio composed of assets whose behaviors are sufficiently different that they provide 
multiple independent sources of return.

This led to a natural question:

> Is it possible to search for the *true most diversified portfolio* in a robust, yet clever way?

Answering that question eventually led to the algorithm explored in this repository.

---

## The Core Idea: Log-Determinant as a Diversification Score

Let $ r_i(t) $ denote the return series of instrument $i$. From aligned returns we can form a correlation matrix

$$$
C \in \mathbb{R}^{M\times M}, \qquad C_{ij} = \mathrm{corr}(r_i, r_j),
$$$

where $M$ is the universe size. For any candidate subset $S$ of size $N$, 
we consider the submatrix $C_S\in\mathbb{R}^{N\times N}$ and score the subset via

$$$
\text{score}(S) = \log\det(C_S).
$$$

Why determinant?

1. **Geometric interpretation (volume):** if we view each instrument’s standardized return stream as a vector, 
then the determinant of the corresponding Gram matrix measures the squared volume of the parallelepiped spanned by those vectors. 
Highly collinear (highly correlated) vectors span little volume $\Rightarrow \det \approx 0$. 
More “orthogonal” (less correlated) return streams span larger volume $\Rightarrow \det \approx 1$ for correlation matrices.

2. **Independence / redundancy:** $\det(C_S)$ aggregates dependence across *all* pairs and higher-order interactions. 
It penalizes redundancy more strongly than any single pairwise threshold. 
If two instruments are near duplicates, one eigenvalue of $C_S$ becomes small, shrinking the determinant.

3. **Eigenvalue view:** since $C_S$ is symmetric positive semidefinite,

$$$
\det(C_S) = \prod_{i=1}^{N} \lambda_i, \qquad \log\det(C_S)=\sum_{i=1}^{N}\log \lambda_i,
$$$

where $\lambda_i$ are the eigenvalues of $C_S$. When return streams are highly correlated, some $\lambda_i$ become very small, 
and the determinant collapses. 
Maximizing $\log\det$ therefore discourages near-linear dependence and favors “full-rank” behavior.

We use **log-determinant** rather than determinant directly because it is numerically stable and preserves ordering:

$$$
\arg\max_S \det(C_S) \equiv \arg\max_S \log\det(C_S).
$$$

This turns the diversification problem into a clean combinatorial optimization task:

$$$
\max_{S\subset \{1,\dots,M\},\ |S|=N} \log\det(C_S).
$$$

The remaining question is computational: how can we search this space without evaluating all $\binom{M}{N}$ subsets?

---

## Heuristic Search Approaches

Because evaluating every possible subset is infeasible for realistic universe sizes, 
a number of heuristic search strategies are commonly used in practice. 
These methods attempt to find *good* diversified portfolios without exploring the full combinatorial space.

### Greedy Selection

A natural first approach is **greedy selection**. The idea is to build the portfolio incrementally:

1. Start with the single instrument that provides the best standalone score.
2. At each step, add the instrument that produces the largest improvement in the objective when combined with the current set.

In pseudocode, this looks roughly like:

```
S = {best single instrument}

while |S| < N:
    choose asset j not in S that maximizes logdet(C_{S ∪ {j}})
    S = S ∪ {j}
```

Greedy methods are attractive because they are **fast**: 
each step only evaluates the remaining candidates instead of all combinations.

However, greedy selection is inherently **myopic**. 
A choice that appears optimal early on may prevent the algorithm from discovering a more diversified combination later. 
As a result, greedy algorithms often produce portfolios that are locally optimal but not globally optimal.

### Beam Search

A more sophisticated heuristic is **beam search**, which attempts to mitigate the myopia of greedy selection.

Instead of tracking only a single partial solution, beam search keeps the top *B* candidate portfolios at each stage, 
where *B* is called the **beam width**.

The algorithm proceeds as follows:

1. Start with all single-instrument portfolios.
2. Keep the best *B* according to the objective.
3. Expand each candidate by adding one additional instrument.
4. Among all expanded candidates, keep the best *B* again.
5. Repeat until portfolio size reaches *N*.

Beam search explores a much larger portion of the search space than greedy selection while still remaining computationally manageable.

However, beam search does not guarantee optimality. 
If the globally optimal portfolio requires exploring a branch that is temporarily suboptimal, 
it may be discarded early in the search when the beam is pruned.

In the next sections we will compare these heuristic approaches against brute force and a pruned brute force search 
designed to reduce the size of the search space.
This comparison will help illustrate the tradeoffs between computational efficiency and optimality guarantees.

---

## Hypothesis

Given the combinatorial complexity of the diversification problem, 
heuristic search methods are often used in practice to approximate good solutions without exploring the full search space.

The central question of this project became:

> How well can simple search algorithms approximate the *true most diversified portfolio*?

To answer this, we compare three approaches:

- **Greedy search** — extremely fast but locally optimal.
- **Beam search** — explores multiple candidate portfolios simultaneously.
- **Full brute force** — evaluates every portfolio and provides the true optimum.

Brute force therefore acts as the **ground-truth benchmark** for evaluating the quality of the heuristic methods.

---

## Random Subset Benchmark

To evaluate the search algorithms more rigorously, we constructed a benchmark where the **true optimal portfolio is known**.

Rather than searching the full universe directly, we repeatedly sampled **random subsets of 50 instruments** from the filtered universe. 
This allows brute force to remain computationally feasible while still producing realistic portfolio construction problems.

Each subset experiment was run with:

- **Subset size:** 50 instruments
- **Portfolio sizes:** N = 5 and N = 10
- **Number of subsets:** 3 random draws
- **Beam width:** 5000
- **CPU cores:** 13

For each subset we evaluated:

- Greedy search
- Beam search
- Full brute force search

Brute force provides the **ground‑truth optimum**, allowing us to directly measure how well the heuristic methods perform.

---

## Runtime Comparison

The following table summarizes the typical runtime characteristics observed across the benchmark experiments.

| Method | Typical Runtime (N=5) | Typical Runtime (N=10) | Optimal Result Found |
|------|------|------|------|
| Greedy | ~0.005 s | ~0.006 s | ❌ |
| Beam (width=5000) | ~2.3 s | ~7.5 s | ✅ |
| Full brute force | ~1.4 s | ~10,500 s (~3 hours) | ✅ |

Several important patterns emerge:

- **Greedy search is extremely fast**, completing almost instantly even for larger portfolios.
- **Beam search remains computationally cheap**, requiring only a few seconds even for N = 10.
- **Brute force becomes dramatically more expensive** as the portfolio size grows due to the combinatorial explosion.

For example:

- With **N = 5**, brute force evaluates about **2.1 million portfolios**, which can still be completed in roughly a second.
- With **N = 10**, the search space explodes to over **10 billion portfolios**, 
requiring **multiple hours** even with parallel processing.

---

## Solution Quality

The key question is not only runtime, but **how close the heuristic methods come to the true optimum**.

Across all three random subsets and both portfolio sizes we observed:

- **Beam search consistently recovered the exact optimal portfolio** found by brute force.
- **Greedy search frequently produced sub‑optimal portfolios**, 
sometimes significantly worse in terms of the log‑determinant objective.

This demonstrates an important practical result:

> Beam search is able to reliably recover the optimal diversified portfolio while exploring only a tiny fraction of the full search space.

For example, when N = 10:

- Total possible portfolios: **~10.3 billion**
- Beam search explores only a few thousand candidates per step

Yet the algorithm still identifies the same optimal solution as brute force.

---

## Example Optimal Portfolio (Subset Experiment)

One of the optimal portfolios discovered in the benchmark (N = 10) was:

```
AGG, AVGO, BIL, GILD, IJH, NFLX, PDD, PM, SCHB, XOM
```

Despite being drawn from a random subset of assets, the resulting portfolio spans multiple independent economic exposures:

- bonds and money markets
- semiconductor technology
- consumer and energy sectors
- emerging market equities

This mixture of unrelated return streams produces a high log‑determinant score and therefore strong diversification.

---

These controlled experiments provide strong empirical evidence that 
**beam search is an extremely effective algorithm for determinant‑based diversification problems**, 
offering near‑optimal results while remaining computationally efficient.

---

## Beam Width Sensitivity Analysis

While the benchmark experiments showed that beam search was able to recover the optimal portfolio when using a beam width of 5000, 
an important practical question remained:

> **How large does the beam width actually need to be to reliably find the optimal solution?**

A very large beam width increases computational cost without necessarily improving results. 
Conversely, if the beam width is too small, the optimal portfolio may be pruned from the search tree before it is discovered.

To investigate this trade-off, we conducted an additional experiment where the beam width was varied across several orders of magnitude.

The experiment reused the **exact same random subsets and brute-force results** from the earlier benchmark, 
allowing us to directly compare beam search outputs against the known optimal portfolios.

The beam widths tested were:

The full experiment configuration and raw results are stored in:

- `beam_width_sweep_against_saved_brute.json`

---

## Beam Width Results

### Case 1 — Portfolio Size N = 5

| Beam Width | Runtime | Optimal Match |
|-----------|--------|--------------|
| 10 | ~0.01 s | ✅ |
| 25 | ~0.02 s | ✅ |
| 50 | ~0.03 s | ✅ |
| 100 | ~0.06 s | ✅ |
| 250 | ~0.15 s | ✅ |
| 500 | ~0.30 s | ✅ |
| 1000 | ~0.60 s | ✅ |
| 2500 | ~1.3 s | ✅ |
| 5000 | ~2.3 s | ✅ |

Even with **beam width = 10**, the algorithm recovered the exact optimal portfolio.

---

### Case 2 — Portfolio Size N = 10

| Beam Width | Runtime | Optimal Match |
|-----------|--------|--------------|
| 10 | ~0.02 s | ❌ |
| 25 | ~0.04 s | ✅ |
| 50 | ~0.08 s | ✅ |
| 100 | ~0.18 s | ✅ |
| 250 | ~0.40 s | ✅ |
| 500 | ~0.75 s | ✅ |
| 1000 | ~1.5 s | ✅ |
| 2500 | ~3.5 s | ✅ |
| 5000 | ~7.5 s | ✅ |

Once the beam width reached **25**, beam search consistently recovered the true optimum.

---

## Why Beam Search Works So Well

Three properties of the log-determinant objective appear to make beam search particularly effective:

**1. Redundancy is strongly penalized**

Highly correlated assets collapse the determinant quickly, causing poor candidates to drop out of the beam early.

**2. Good partial portfolios remain good**

Portfolios composed of independent assets tend to stay strong as more assets are added.

**3. The objective landscape is relatively smooth**

Unlike many combinatorial optimization problems, the log-determinant objective does not produce highly chaotic search paths.

---

## Practical Implications

These experiments suggest an important practical point: 
the beam width should **not** be interpreted as a universal rule tied only to portfolio size `N`.

In practice, the required beam width depends on the overall difficulty of the search problem, 
which is driven by the size of the combinatorial search space:

where:

- `M` is the size of the candidate universe  
- `N` is the target portfolio size  

As this search space grows, a larger beam width may be needed to ensure that the optimal branch is not pruned too early.

For that reason, the results in this project should be interpreted with some care:

- For the random-subset benchmark with `M = 50`, beam widths between **25 and 50** were already sufficient to recover the exact optimum.
- For larger universes or larger values of `N`, the required beam width will generally increase.

However, an important practical advantage of beam search is that increasing the beam width is usually cheap relative to brute force.

A larger beam width does increase runtime, 
but the increase is still modest compared with the combinatorial explosion avoided by the algorithm. 
Memory usage also remains manageable for the beam widths considered in this project.

As a result, in practical applications it is often sensible to choose a beam width that is comfortably larger than the minimum required.

For example, values such as **500** or **1000** are often reasonable defaults: 
they are large enough to provide a strong safety margin while remaining computationally inexpensive compared with exhaustive search.

---


## What This Demonstrates

This study reveals several important insights:

1. **Greedy search is extremely fast but unreliable.**  
   Local decisions can prevent the algorithm from discovering the globally optimal portfolio.

2. **Beam search provides an excellent trade‑off between speed and solution quality.**  
   In the experiments above it consistently recovered the optimal portfolio while remaining orders of magnitude faster than brute force.

3. **Brute force remains the definitive benchmark.**  
   Although computationally expensive, it provides the ground truth needed to evaluate heuristic search methods.

---

## Research Detour: Pruning the Search Space

During the research process an additional method was explored: **pruned brute force search**.

The idea was to eliminate candidate portfolios containing pairs of instruments with correlations above a fixed threshold

$$$
|corr(i, j)| > τ
$$$

This dramatically reduces the size of the search tree and allows an exhaustive search over the remaining portfolios.

However, an important insight emerged:

> Pruning based on pairwise correlations changes the optimization problem itself.

Because the determinant captures **global dependence across all assets**, 
the optimal portfolio may still contain pairs whose correlations exceed the pruning threshold. 
When those portfolios are removed before the search begins, the algorithm can no longer discover the true optimum.

Pruned brute force therefore guarantees optimality **only within the constrained search space defined by τ**, 
not under the original log‑determinant objective.

This observation highlights an important lesson in quantitative research:

> Computational shortcuts can unintentionally introduce modeling assumptions.

---

## Conclusion

Selecting the most diversified subset of assets from a large universe is fundamentally a **combinatorial optimization problem**.

This project explored several search strategies for solving that problem using a log‑determinant diversification objective.

The results suggest that:

- brute force provides the ground‑truth optimum
- greedy search is fast but unreliable
- **beam search offers an excellent balance between speed and solution quality**

In practice, beam search appears to recover globally optimal or near‑optimal portfolios while remaining computationally 
tractable even for moderately large portfolio sizes.

This makes it a practical and powerful approach for exploring diversification in large asset universes.

