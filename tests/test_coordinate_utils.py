"""Unit tests for coordinate conversion and rotation utilities."""

import numpy as np
import pytest
from math import sqrt

from hcp_gb_generator import (
    crystal_basis_matrix,
    mb_str,
    miller_bravais_to_3axis,
    rotation_axis_angle,
    three_axis_to_miller_bravais,
)
from hcp_gb_generator._core import (
    _gcd_many,
    _hex_symmetry_equivalent_dirs,
    _loeschian_pairs,
    cart_axis_to_miller,
)


IDEAL_CA = sqrt(8.0 / 3.0)


# ---------------------------------------------------------------------------
# Miller-Bravais <-> 3-axis conversions
# ---------------------------------------------------------------------------

class TestMillerBravaisConversion:
    """Test 4-index <-> 3-index direction conversions."""

    @pytest.mark.parametrize("mb4, expected_3ax", [
        ([0, 0, 0, 1], [0, 0, 1]),
        ([1, 0, -1, 0], [2, 1, 0]),
        ([1, -1, 0, 0], [1, -1, 0]),
        ([1, 1, -2, 0], [1, 1, 0]),
        ([2, -1, -1, 0], [1, 0, 0]),
    ])
    def test_miller_bravais_to_3axis(self, mb4, expected_3ax):
        result = miller_bravais_to_3axis(mb4)
        assert result == expected_3ax

    @pytest.mark.parametrize("ax3, expected_mb4", [
        ([0, 0, 1], [0, 0, 0, 1]),
        ([1, -1, 0], [1, -1, 0, 0]),
    ])
    def test_three_axis_to_miller_bravais(self, ax3, expected_mb4):
        result = three_axis_to_miller_bravais(ax3)
        assert result == expected_mb4

    def test_roundtrip_0001(self):
        """[0001] -> 3-axis -> back to 4-index."""
        ax3 = miller_bravais_to_3axis([0, 0, 0, 1])
        mb4 = three_axis_to_miller_bravais(ax3)
        assert mb4 == [0, 0, 0, 1]

    def test_roundtrip_prism(self):
        """[10-10] -> 3-axis -> back to 4-index."""
        ax3 = miller_bravais_to_3axis([1, 0, -1, 0])
        mb4 = three_axis_to_miller_bravais(ax3)
        assert mb4 == [1, 0, -1, 0]


# ---------------------------------------------------------------------------
# Crystal basis matrix
# ---------------------------------------------------------------------------

class TestCrystalBasis:
    def test_shape(self):
        S = crystal_basis_matrix(IDEAL_CA)
        assert S.shape == (3, 3)

    def test_a1_direction(self):
        """a1 = [1,0,0] in crystal coords maps to [1, 0, 0] in Cartesian."""
        S = crystal_basis_matrix(IDEAL_CA)
        a1_cart = S @ np.array([1, 0, 0])
        np.testing.assert_allclose(a1_cart, [1.0, 0.0, 0.0])

    def test_a2_direction(self):
        """a2 = [0,1,0] maps to [-0.5, sqrt(3)/2, 0]."""
        S = crystal_basis_matrix(IDEAL_CA)
        a2_cart = S @ np.array([0, 1, 0])
        np.testing.assert_allclose(a2_cart, [-0.5, sqrt(3) / 2, 0.0], atol=1e-10)

    def test_c_direction(self):
        """c = [0,0,1] maps to [0, 0, c/a]."""
        S = crystal_basis_matrix(IDEAL_CA)
        c_cart = S @ np.array([0, 0, 1])
        np.testing.assert_allclose(c_cart, [0.0, 0.0, IDEAL_CA], atol=1e-10)

    def test_120_degree_angle(self):
        """Angle between a1 and a2 should be 120 degrees."""
        S = crystal_basis_matrix(IDEAL_CA)
        a1 = S @ np.array([1, 0, 0])
        a2 = S @ np.array([0, 1, 0])
        cos_angle = np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))
        np.testing.assert_allclose(cos_angle, -0.5, atol=1e-10)


# ---------------------------------------------------------------------------
# Rotation axis/angle extraction
# ---------------------------------------------------------------------------

class TestRotationAxisAngle:
    def test_identity(self):
        ax, angle = rotation_axis_angle(np.eye(3))
        assert angle == 0.0

    def test_90_deg_around_z(self):
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        ax, angle = rotation_axis_angle(R)
        np.testing.assert_allclose(angle, 90.0, atol=0.01)
        np.testing.assert_allclose(np.abs(ax[2]), 1.0, atol=1e-6)

    def test_180_deg(self):
        """180 deg around x-axis: [[1,0,0],[0,-1,0],[0,0,-1]]."""
        R = np.diag([1.0, -1.0, -1.0])
        ax, angle = rotation_axis_angle(R)
        np.testing.assert_allclose(angle, 180.0, atol=0.01)
        np.testing.assert_allclose(np.abs(ax[0]), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Cartesian axis to Miller index
# ---------------------------------------------------------------------------

class TestCartAxisToMiller:
    def test_c_axis(self):
        result = cart_axis_to_miller(np.array([0, 0, 1.0]), IDEAL_CA)
        assert result == [0, 0, 1]

    def test_a1_axis(self):
        result = cart_axis_to_miller(np.array([1.0, 0, 0]), IDEAL_CA)
        assert result == [1, 0, 0]


# ---------------------------------------------------------------------------
# Loeschian pairs
# ---------------------------------------------------------------------------

class TestLoeschianPairs:
    def test_sigma_7_squared(self):
        """49 = 8^2 + 3^2 - 8*3 should have coprime pair (8,3)."""
        pairs = _loeschian_pairs(49)
        assert any(u == 8 and v == 3 for u, v in pairs)

    def test_sigma_13_squared(self):
        """169 = 13^2. Should have coprime pair (15, 7) or (8, 7)."""
        pairs = _loeschian_pairs(169)
        assert len(pairs) > 0
        for u, v in pairs:
            assert u * u + v * v - u * v == 169

    def test_non_loeschian_returns_empty(self):
        """2 is not a Loeschian number."""
        pairs = _loeschian_pairs(2)
        assert pairs == []


# ---------------------------------------------------------------------------
# Hex symmetry equivalents
# ---------------------------------------------------------------------------

class TestHexSymmetry:
    def test_0001_invariant(self):
        """[001] should produce 2 unique Cartesian directions (±z)."""
        S = crystal_basis_matrix(IDEAL_CA)
        equivs = _hex_symmetry_equivalent_dirs([0, 0, 1], S)
        # All should be along z
        for v in equivs:
            np.testing.assert_allclose(v[:2], [0, 0], atol=1e-10)

    def test_prism_multiplicity(self):
        """A prism direction like [1,-1,0] should generate 12 equivalents."""
        S = crystal_basis_matrix(IDEAL_CA)
        equivs = _hex_symmetry_equivalent_dirs([1, -1, 0], S)
        assert len(equivs) == 12


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

class TestGcdMany:
    def test_basic(self):
        assert _gcd_many(6, 4) == 2

    def test_with_zeros(self):
        assert _gcd_many(0, 0, 5) == 5

    def test_coprime(self):
        assert _gcd_many(7, 3) == 1


class TestMbStr:
    def test_positive(self):
        assert mb_str([1, 0, -1, 0]) == "[1 0 \u03051 0]"

    def test_all_zero(self):
        assert mb_str([0, 0, 0, 1]) == "[0 0 0 1]"
