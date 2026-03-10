import numpy as np
import pytest

from diversifind.utils import validate_corr, combo_logdet


def test_validate_corr_accepts_valid_matrix():
    """validate_corr should accept a proper correlation matrix."""
    corr = np.array([
        [1.0, 0.2, -0.1],
        [0.2, 1.0, 0.3],
        [-0.1, 0.3, 1.0],
    ])

    # Should not raise
    validate_corr(corr)


def test_validate_corr_rejects_non_square():
    """validate_corr should reject non-square matrices."""
    corr = np.ones((3, 2))

    with pytest.raises(ValueError):
        validate_corr(corr)


def test_validate_corr_rejects_non_unit_diagonal():
    """validate_corr should reject matrices whose diagonal is not all ones."""
    corr = np.array([
        [1.0, 0.1, 0.2],
        [0.1, 0.9, 0.3],  # not 1.0
        [0.2, 0.3, 1.0],
    ])

    with pytest.raises(ValueError):
        validate_corr(corr)


def test_validate_corr_rejects_out_of_bounds_values():
    """validate_corr should reject correlations outside [-1, 1]."""
    corr = np.array([
        [1.0, 1.2, 0.0],
        [1.2, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    with pytest.raises(ValueError):
        validate_corr(corr)


def test_combo_logdet_identity_matrix():
    """Log determinant of identity correlation matrix should be zero."""
    corr = np.eye(4)

    combo = (0, 1, 2, 3)

    ld = combo_logdet(corr, combo)

    assert np.isclose(ld, 0.0)


def test_combo_logdet_reduces_with_correlation():
    """Adding correlation should reduce the log determinant."""
    corr = np.array([
        [1.0, 0.5, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    combo = (0, 1, 2)

    ld = combo_logdet(corr, combo)

    # determinant < 1 so logdet < 0
    assert ld < 0


def test_combo_logdet_single_asset():
    """Single-asset portfolio should have logdet = 0."""
    corr = np.eye(3)

    combo = (1,)

    ld = combo_logdet(corr, combo)

    assert np.isclose(ld, 0.0)
