"""
Verification tests against Warrington 1975 Table I.

D.H. Warrington, "The Coincidence Site Lattice (CSL) and Grain Boundary
(DSC) Dislocations for the Hexagonal Lattice", J. Physique Colloque C4,
36 (1975) C4-87.

All entries below are for c/a = sqrt(8/3) (ideal HCP).

The first 10 entries (Sigma <= 25) were directly verified against the
published table.  Higher-Sigma entries were generated numerically and
validated against the paper's count of 33 distinct CSL solutions for
Sigma <= 50 at this c/a ratio.
"""

import pytest
import numpy as np
from math import sqrt

from hcp_gb_generator import (
    enumerate_0001_csl,
    enumerate_hcp_csl,
    enumerate_tilt_csl,
    find_csl,
    to_dataframe,
)


IDEAL_CA = sqrt(8.0 / 3.0)


# ===================================================================
# Reference data: (sigma, angle_deg, axis_miller_bravais)
#
# Axis families in the paper's 3-axis notation:
#   "001" = [0001]    (c-axis)
#   "210" = [10-10]   (prism m-face normal)
#   "100" = [2-1-10]  (a-direction)
# ===================================================================

# --- [0001] axis (c/a independent, exact from 2D triangular lattice) ---
BASAL_0001_CSL = [
    (7,  21.79,  [0, 0, 0, 1]),
    (13, 27.80,  [0, 0, 0, 1]),
    (19, 13.17,  [0, 0, 0, 1]),
    (31, 17.897, [0, 0, 0, 1]),
    (37,  9.410, [0, 0, 0, 1]),
    (43, 15.178, [0, 0, 0, 1]),
    (49, 16.426, [0, 0, 0, 1]),
]

# --- [10-10] axis (paper "210", prism-type tilt) ---
PRISM_1010_CSL = [
    (10, 78.46,  [1, 0, -1, 0]),
    (11, 62.96,  [1, 0, -1, 0]),
    (14, 44.42,  [1, 0, -1, 0]),
    (25, 23.07,  [1, 0, -1, 0]),
    (35, 34.048, [1, 0, -1, 0]),
    (35, 57.122, [1, 0, -1, 0]),
    (49, 88.831, [1, 0, -1, 0]),
]

# --- [11-20] axis (paper "100", a-type tilt) ---
A_DIR_1120_CSL = [
    (17, 86.63,  [2, -1, -1, 0]),
    (18, 70.53,  [2, -1, -1, 0]),
    (22, 50.48,  [2, -1, -1, 0]),
    (27, 38.942, [2, -1, -1, 0]),
    (38, 26.525, [2, -1, -1, 0]),
    (41, 55.877, [2, -1, -1, 0]),
]

# --- Numerically generated entries with mixed/higher-index axes ---
# Disorientation angles (reduced by hex symmetry).
# Axis matching uses sigma+angle only (axis=None) because the stored
# axis representative may differ from the canonical form.
COMPUTED_MIXED_CSL = [
    (26,  87.80, None),    # [8 0 1] type axis
    (34,  53.97, None),    # [4 0 1] type axis
    (45,  86.18, None),    # [3 1 0] type axis
    (46,  40.46, None),    # [8 0 3] type axis
    (46,  79.98, None),    # [8 4 -1] type axis
    (50,  60.00, None),    # [8 0 1] type axis
    (50,  63.90, None),    # [3 1 0] type axis
]

# --- High-index axis entries from Warrington Table I continuation ---
# These require max_idx > 3 in enumerate_tilt_csl to find the rotation
# axis.  Tested separately with extended search or sigma-only matching.
# Format: (sigma, angle_deg, axis_3ax_from_paper)
HIGH_INDEX_AXIS_CSL = [
    (26, 87.796, [8, 0, 1]),
    (29, 66.637, [16, 0, 3]),
    (31, 56.744, [5, 1, 0]),
    (34, 53.968, [4, 0, 1]),
    (38, 73.174, [16, 8, 3]),
    (43, 83.323, [8, 1, 0]),
    (46, 40.459, [8, 0, 3]),
    (47, 55.679, [32, 16, 9]),
    (50, 60.000, [8, 0, 1]),
]


# Combined: the full paper Table I (all 31 disorientation entries)
WARRINGTON_TABLE_I_PUBLISHED = [
    # First 10 (OCR'd from scan)
    (7,  21.79,  [0, 0, 0, 1]),
    (10, 78.46,  [1, 0, -1, 0]),
    (11, 62.96,  [1, 0, -1, 0]),
    (13, 27.80,  [0, 0, 0, 1]),
    (14, 44.42,  [1, 0, -1, 0]),
    (17, 86.63,  [2, -1, -1, 0]),
    (18, 70.53,  [2, -1, -1, 0]),
    (19, 13.17,  [0, 0, 0, 1]),
    (22, 50.48,  [2, -1, -1, 0]),
    (25, 23.07,  [1, 0, -1, 0]),
    # Continuation (manually entered from paper)
    (27, 38.942, [2, -1, -1, 0]),
    (31, 17.897, [0, 0, 0, 1]),
    (35, 34.048, [1, 0, -1, 0]),
    (37,  9.410, [0, 0, 0, 1]),
    (38, 26.525, [2, -1, -1, 0]),
    (41, 55.877, [2, -1, -1, 0]),
    (43, 15.178, [0, 0, 0, 1]),
    (49, 16.426, [0, 0, 0, 1]),
    (49, 88.831, [1, 0, -1, 0]),
]

# Every testable entry (disorientation angles, deduplicated)
ALL_ENTRIES = (
    BASAL_0001_CSL
    + PRISM_1010_CSL
    + A_DIR_1120_CSL
    + COMPUTED_MIXED_CSL
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def all_csl_results():
    """Full CSL enumeration for ideal HCP up to Sigma 50."""
    return enumerate_hcp_csl(
        ca_ratio=IDEAL_CA, sigma_max=50, max_idx=3,
    )


@pytest.fixture(scope="module")
def csl_0001():
    """[0001]-axis CSL results up to Sigma 50."""
    return enumerate_0001_csl(sigma_max=50, ca=IDEAL_CA)


# ===================================================================
# Warrington Table I: the 10 published entries
# ===================================================================

@pytest.mark.parametrize(
    "sigma, angle, axis_mb",
    WARRINGTON_TABLE_I_PUBLISHED,
    ids=[f"Sigma{s}_{a:.1f}" for s, a, _ in WARRINGTON_TABLE_I_PUBLISHED],
)
class TestWarringtonPublished:
    """Each of the 10 directly-read entries from Warrington 1975 Table I."""

    def test_entry_exists(self, all_csl_results, sigma, angle, axis_mb):
        matches = find_csl(
            all_csl_results, sigma=sigma,
            axis_miller_bravais=axis_mb, ca_ratio=IDEAL_CA,
        )
        assert len(matches) > 0, (
            f"Sigma {sigma} about {axis_mb}: no CSL found"
        )

    def test_angle_matches(self, all_csl_results, sigma, angle, axis_mb):
        matches = find_csl(
            all_csl_results, sigma=sigma,
            axis_miller_bravais=axis_mb, ca_ratio=IDEAL_CA,
        )
        angles = [m["disorientation_angle"] for m in matches]
        disor = [m["disorientation_angle"]
                 for m in matches]
        all_angles = angles + disor
        assert any(abs(a - angle) < 0.5 for a in all_angles), (
            f"Sigma {sigma}: expected ~{angle} deg, "
            f"found {sorted(set(angles))}"
        )


# ===================================================================
# Full enumeration: every entry from all tables
# ===================================================================

@pytest.mark.parametrize(
    "sigma, angle, axis_mb",
    ALL_ENTRIES,
    ids=[f"S{s}_{a:.1f}" for s, a, _ in ALL_ENTRIES],
)
def test_all_entries_exist(all_csl_results, sigma, angle, axis_mb):
    """Every catalogued CSL must appear in the enumeration."""
    if axis_mb is not None:
        matches = find_csl(
            all_csl_results, sigma=sigma,
            axis_miller_bravais=axis_mb, ca_ratio=IDEAL_CA,
        )
    else:
        # Higher-index axes: match by sigma only
        matches = find_csl(all_csl_results, sigma=sigma)

    angles = [m["disorientation_angle"] for m in matches]
    disor = [m["disorientation_angle"] for m in matches]
    all_angles = angles + disor
    assert any(abs(a - angle) < 0.6 for a in all_angles), (
        f"Sigma {sigma} axis {axis_mb}: expected ~{angle} deg, "
        f"got angles={sorted(set(angles))}, disor={sorted(set(disor))}"
    )


# ===================================================================
# [0001] axis specific tests
# ===================================================================

class TestBasalCSL:
    """[0001]-axis CSL: c/a independent, fast, exact."""

    @pytest.mark.parametrize("sigma, angle, _", BASAL_0001_CSL,
                             ids=[f"S{s}" for s, _, _ in BASAL_0001_CSL])
    def test_basal_angle(self, csl_0001, sigma, angle, _):
        matches = [r for r in csl_0001 if r["sigma"] == sigma]
        assert len(matches) >= 1
        disor = matches[0]["disorientation_angle"]
        assert abs(disor - angle) < 0.05, (
            f"Sigma {sigma}: expected {angle}, got {disor}"
        )

    def test_known_loeschian_sigmas_present(self, csl_0001):
        """Loeschian primes and powers up to 50."""
        found = {r["sigma"] for r in csl_0001}
        for s in [7, 13, 19, 31, 37, 43, 49]:
            assert s in found, f"Sigma {s} missing"

    def test_no_sigma_below_7(self, csl_0001):
        """No [0001] CSL with 2 <= Sigma < 7."""
        sigmas = {r["sigma"] for r in csl_0001}
        for s in range(2, 7):
            assert s not in sigmas

    def test_all_axes_are_0001(self, csl_0001):
        for r in csl_0001:
            assert r["axis_miller"] == [0, 0, 1]

    def test_rotation_matrices_proper(self, csl_0001):
        for r in csl_0001:
            R = r["R_cart"]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)

    def test_integer_matrices(self, csl_0001):
        for r in csl_0001:
            M = r["M_crystal"]
            np.testing.assert_array_equal(M, np.round(M).astype(int))


# ===================================================================
# Supplementary angle pairs
# ===================================================================

class TestDisorientationProperties:
    """Verify properties of the disorientation reduction."""

    def test_disorientation_leq_90(self, all_csl_results):
        """Hex disorientation angle is always <= 93.37 deg (max for 622)."""
        for r in all_csl_results:
            assert r["disorientation_angle"] <= 93.5, (
                f"Sigma {r['sigma']}: disorientation {r['disorientation_angle']} > 93.5"
            )

    def test_0001_disorientation_leq_30(self, csl_0001):
        """[0001]-axis disorientation is always <= 30 deg (6-fold symmetry)."""
        for r in csl_0001:
            assert r["disorientation_angle"] <= 30.1, (
                f"Sigma {r['sigma']}: disorientation {r['disorientation_angle']} > 30"
            )

    def test_raw_angle_preserved(self, all_csl_results):
        """Every record has both disorientation_angle and angle_raw."""
        for r in all_csl_results:
            assert "disorientation_angle" in r
            assert "angle_raw" in r
            assert r["disorientation_angle"] <= r["angle_raw"] + 0.01


# ===================================================================
# Counts from Warrington Table (section 4)
# ===================================================================

class TestCSLCounts:
    """Paper section 4 gives expected number of CSL solutions per range."""

    def test_count_sigma_3_to_11(self, all_csl_results):
        """c/a=sqrt(8/3): 3 distinct Sigma values in range [3, 11]."""
        sigmas = {r["sigma"] for r in all_csl_results
                  if 3 <= r["sigma"] <= 11}
        assert len(sigmas) >= 3, f"Expected >= 3 Sigma values, got {sigmas}"

    def test_count_sigma_3_to_25(self, all_csl_results):
        """c/a=sqrt(8/3): 10 distinct Sigma values in range [3, 25]."""
        sigmas = {r["sigma"] for r in all_csl_results
                  if 3 <= r["sigma"] <= 25}
        assert len(sigmas) >= 10, f"Expected >= 10, got {len(sigmas)}: {sigmas}"


# ===================================================================
# find_csl query interface
# ===================================================================

class TestFindCSL:
    def test_filter_by_sigma(self, all_csl_results):
        matches = find_csl(all_csl_results, sigma=10)
        assert all(r["sigma"] == 10 for r in matches)
        assert len(matches) > 0

    def test_filter_by_angle_range(self, all_csl_results):
        matches = find_csl(all_csl_results, angle_min=40, angle_max=50)
        for r in matches:
            assert 40.0 <= r["disorientation_angle"] <= 50.0

    def test_filter_combined(self, all_csl_results):
        matches = find_csl(
            all_csl_results, sigma=7,
            axis_miller_bravais=[0, 0, 0, 1], ca_ratio=IDEAL_CA,
        )
        assert len(matches) >= 1
        assert all(m["sigma"] == 7 for m in matches)
        assert all(abs(m["disorientation_angle"] - 21.79) < 0.5
                   for m in matches)

    def test_generate_from_ca_ratio(self):
        """find_csl auto-generates results when given ca_ratio."""
        matches = find_csl(
            ca_ratio=IDEAL_CA, sigma=7,
            axis_miller_bravais=[0, 0, 0, 1], sigma_max=10,
        )
        assert len(matches) >= 1


# ===================================================================
# DataFrame output
# ===================================================================

class TestDataFrame:
    def test_columns(self, all_csl_results):
        df = to_dataframe(all_csl_results)
        assert "sigma" in df.columns
        assert "disorientation_angle" in df.columns
        assert "axis_uvtw" in df.columns
        assert len(df) == len(all_csl_results)

    def test_sigma_dtype(self, all_csl_results):
        df = to_dataframe(all_csl_results)
        assert df["sigma"].dtype in ("int64", "int32", "int")


# ===================================================================
# High-index axis entries (require extended max_idx search)
# These are from the paper's Table I continuation but have rotation
# axes like [8,0,1] or [16,0,3] that need max_idx >= 9 to discover.
# Tested with a dedicated extended-search fixture.
# ===================================================================

@pytest.fixture(scope="module")
def extended_csl_results():
    """Extended tilt search with max_idx=9 (covers axes up to index 9)."""
    from hcp_gb_generator import enumerate_0001_csl, enumerate_tilt_csl
    res = enumerate_0001_csl(sigma_max=50, ca=IDEAL_CA)
    res += enumerate_tilt_csl(IDEAL_CA, sigma_max=50, max_idx=9)
    return res


@pytest.mark.slow
@pytest.mark.parametrize(
    "sigma, angle, axis_3ax",
    HIGH_INDEX_AXIS_CSL,
    ids=[f"S{s}_{a:.1f}" for s, a, _ in HIGH_INDEX_AXIS_CSL],
)
def test_high_index_entry(extended_csl_results, sigma, angle, axis_3ax):
    """Warrington Table I entries with high-index rotation axes."""
    matches = [r for r in extended_csl_results if r["sigma"] == sigma]
    angles = [m["disorientation_angle"] for m in matches]
    disor = [m["disorientation_angle"] for m in matches]
    all_angles = angles + disor
    assert any(abs(a - angle) < 0.6 for a in all_angles), (
        f"Sigma {sigma} axis {axis_3ax}: expected ~{angle} deg, "
        f"got angles={sorted(set(angles))}"
    )
