import numpy as np


def validate_corr(corr: np.ndarray) -> None:
    """Validate correlation matrix."""
    if corr.ndim != 2:
        raise ValueError("Correlation matrix must be 2D")

    n, m = corr.shape
    if n != m:
        raise ValueError("Correlation matrix must be square")

    if not np.allclose(corr, corr.T, atol=1e-8):
        raise ValueError("Correlation matrix must be symmetric")

    if not np.isfinite(corr).all():
        raise ValueError("Correlation matrix contains non-finite values")

    if not np.allclose(np.diag(corr), 1.0, atol=1e-8):
        raise ValueError("Correlation matrix must have ones on the diagonal")

    if np.any(np.abs(corr) > 1.0 + 1e-8):
        raise ValueError("Correlation values must lie within [-1, 1]")


def combo_logdet(corr: np.ndarray, combo: tuple[int, ...]) -> float:
    """Compute logdet score for a combination."""
    sub = corr[np.ix_(combo, combo)]

    sign, val = np.linalg.slogdet(sub)
    if sign <= 0:
        return float("-inf")

    return float(val)