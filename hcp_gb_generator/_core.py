"""
HCP Coincidence Site Lattice (CSL) Grain Boundary Enumerator
=============================================================

Based on:
- D.H. Warrington & P. Bufalini, Scripta Met. 5 (1971) 771 (cubic method)
- D.H. Warrington, J. Physique Colloque C4 (1975) C4-87 (hexagonal extension)

For a hexagonal crystal (3-axis notation, 120 deg between a1 and a2):
A rotation R gives a CSL of index Sigma if M = Sigma * R_crystal is an
integer matrix, where R_crystal is the rotation matrix in crystal coords.

For the basal ([0001]) rotation axis the problem reduces to a 2D triangular
lattice CSL and is INDEPENDENT of c/a.

For tilt axes lying in the basal plane ([1-100], [11-20], general [uvt0]),
the c/a ratio enters through the c-axis column of M.

Usage
-----
>>> from hcp_gb_generator import enumerate_hcp_csl, find_csl
>>> results = enumerate_hcp_csl(ca_ratio=1.587, sigma_max=50)
>>> matches = find_csl(sigma=7, ca_ratio=1.587)
>>> print_csl_table(results)
"""

from __future__ import annotations

from math import gcd, sqrt, acos, degrees, radians, isqrt
from functools import reduce
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------

def miller_bravais_to_3axis(uvtw: list[int]) -> list[int]:
    """Convert 4-index Miller-Bravais [u,v,t,w] to 3-axis [U,V,W]."""
    u, v, t, w = uvtw
    U = 2 * u + v
    V = 2 * v + u
    W = w
    g = reduce(gcd, [abs(U), abs(V), abs(W)])
    if g:
        return [U // g, V // g, W // g]
    return [U, V, W]


def three_axis_to_miller_bravais(uvw: list[int]) -> list[int]:
    """Convert 3-axis [U,V,W] to canonical 4-index Miller-Bravais [u,v,t,w]."""
    U, V, W = uvw
    u3 = 2 * U - V
    v3 = 2 * V - U
    if u3 % 3 != 0 or v3 % 3 != 0:
        t = -(U + V)
        return [U, V, t, W]
    u = u3 // 3
    v = v3 // 3
    t = -(u + v)
    return [u, v, t, W]


def crystal_basis_matrix(ca: float) -> np.ndarray:
    """
    3x3 matrix S mapping crystal 3-axis coords to Cartesian (a=1).
      cart = S @ cryst
    """
    return np.array([
        [1.0, -0.5, 0.0],
        [0.0, sqrt(3) / 2, 0.0],
        [0.0, 0.0, ca],
    ])


def _gcd_many(*vals: int) -> int:
    return reduce(gcd, [abs(v) for v in vals if v != 0], 0)


def _hex_symmetry_equivalent_dirs(uvw: list[int], S: np.ndarray
                                   ) -> list[np.ndarray]:
    """
    Return Cartesian unit vectors for all hex-symmetry equivalents
    of direction [U,V,W] in 3-axis notation.

    Uses the 12 proper rotations of point group 622 plus inversion → 24.
    """
    U, V, W = uvw
    # 6-fold rotation equivalents of [U,V,W] in 3-axis hex coords:
    rot6 = [
        (U, V, W),
        (-V, U - V, W),
        (V - U, -U, W),
        (-U, -V, W),
        (V, V - U, W),
        (U - V, U, W),
    ]
    # Add 2-fold rotations about basal-plane axes (mirror + inversion equiv.)
    all_equiv = []
    for u, v, w in rot6:
        all_equiv.append((u, v, w))
        all_equiv.append((u, v, -w))   # mirror in basal plane

    # Convert to unique Cartesian unit vectors
    cart_vecs = []
    for uvw_t in all_equiv:
        c = S @ np.array(uvw_t, dtype=float)
        n = np.linalg.norm(c)
        if n > 1e-10:
            cart_vecs.append(c / n)
    return cart_vecs


# ---------------------------------------------------------------------------
# Vectorised rotation matrix utilities
# ---------------------------------------------------------------------------

def _rodrigues_batch(axes: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrices for many (axis, angle) pairs at once.

    Parameters
    ----------
    axes : (N, 3)  unit vectors
    thetas : (N,)  angles in radians

    Returns
    -------
    (N, 3, 3) rotation matrices
    """
    N = len(thetas)
    ct = np.cos(thetas)[:, None, None]          # (N,1,1)
    st = np.sin(thetas)[:, None, None]
    ux = axes[:, 0]
    uy = axes[:, 1]
    uz = axes[:, 2]

    # Outer product tensor  u_i u_j  shape (N,3,3)
    uu = axes[:, :, None] * axes[:, None, :]

    # Cross-product matrix K_ij
    K = np.zeros((N, 3, 3))
    K[:, 0, 1] = -uz
    K[:, 0, 2] = uy
    K[:, 1, 0] = uz
    K[:, 1, 2] = -ux
    K[:, 2, 0] = -uy
    K[:, 2, 1] = ux

    I3 = np.eye(3)[None, :, :]                  # (1,3,3)
    R = ct * I3 + st * K[:, :, :] + (1 - ct) * uu
    return R


def rotation_axis_angle(R: np.ndarray):
    """
    Return (axis_cart, angle_deg) from a 3x3 Cartesian rotation matrix.
    Axis is normalised and has positive leading nonzero component.
    """
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_t))

    if abs(theta) < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0

    ax = np.array([R[2, 1] - R[1, 2],
                   R[0, 2] - R[2, 0],
                   R[1, 0] - R[0, 1]])
    nrm = np.linalg.norm(ax)

    if nrm < 1e-9:
        # 180 deg
        A = R + np.eye(3)
        idx = np.argmax(np.linalg.norm(A, axis=0))
        ax = A[:, idx]
        ax = ax / np.linalg.norm(ax)
    else:
        ax = ax / nrm

    # Canonical sign
    for c in ax:
        if abs(c) > 1e-6:
            if c < 0:
                ax = -ax
            break

    return ax, degrees(theta)


def cart_axis_to_miller(ax_cart: np.ndarray, ca: float,
                        max_index: int = 8) -> list[int]:
    """
    Convert a Cartesian unit vector to the closest small-integer
    Miller index in 3-axis hex notation.  Vectorised inner search.
    """
    S = crystal_basis_matrix(ca)
    cryst = np.linalg.solve(S, ax_cart)
    cryst = cryst / (np.linalg.norm(cryst) + 1e-15)

    # Build grid of candidate [u,v,w] vectors
    rng = np.arange(-max_index, max_index + 1)
    grid = np.array(np.meshgrid(rng, rng, rng)).reshape(3, -1).T  # (K, 3)

    # Remove [0,0,0]
    nz = np.any(grid != 0, axis=1)
    grid = grid[nz]

    # Normalise each candidate
    norms = np.linalg.norm(grid.astype(float), axis=1, keepdims=True)
    grid_n = grid / norms

    # Dot product with target (cosine similarity)
    dots = grid_n @ cryst
    best_idx = np.argmax(dots)
    u, v, w = grid[best_idx]
    g = _gcd_many(int(u), int(v), int(w)) or 1
    return [int(u) // g, int(v) // g, int(w) // g]


# ---------------------------------------------------------------------------
# Core CSL construction
# ---------------------------------------------------------------------------

def integer_matrix_to_csl(M: np.ndarray, sigma: int, ca: float,
                           tol: float = 1e-5) -> Optional[dict]:
    """
    Given an integer matrix M and a candidate Sigma, check whether
    R = M / Sigma is a valid proper rotation, and return the CSL record.
    """
    R_cryst = M.astype(float) / sigma
    S = crystal_basis_matrix(ca)
    R_cart = S @ R_cryst @ np.linalg.inv(S)

    if not np.allclose(R_cart @ R_cart.T, np.eye(3), atol=tol):
        return None
    if abs(np.linalg.det(R_cart) - 1.0) > tol:
        return None

    axis, angle = rotation_axis_angle(R_cart)
    miller_ax = cart_axis_to_miller(axis, ca)

    return {
        "sigma": sigma,
        "angle_deg": round(angle, 4),
        "axis_cart": axis,
        "axis_miller": miller_ax,
        "axis_miller_bravais": three_axis_to_miller_bravais(miller_ax),
        "R_cart": R_cart,
        "M_crystal": np.asarray(M, dtype=int),
    }


# ---------------------------------------------------------------------------
# Enumerate CSL for [0001] rotation axis  (c/a-independent)
# ---------------------------------------------------------------------------

def _loeschian_pairs(target: int) -> list[tuple[int, int]]:
    """
    Find all (u, v) with u²+v²-u*v == target, gcd(u,v) == 1, u > 0.
    Vectorised: build grid once, mask.
    """
    rng = isqrt(target) + 2
    u_arr = np.arange(1, rng + 1)
    v_arr = np.arange(-rng, rng + 1)
    U, V = np.meshgrid(u_arr, v_arr, indexing="ij")
    vals = U * U + V * V - U * V
    mask = vals == target

    pairs = []
    us, vs = U[mask], V[mask]
    for u, v in zip(us, vs):
        if gcd(abs(int(u)), abs(int(v))) == 1:
            pairs.append((int(u), int(v)))
    return pairs


def enumerate_0001_csl(sigma_max: int = 50, ca: float = 1.587) -> list[dict]:
    """
    Enumerate all [0001]-axis CSL GBs up to Sigma = sigma_max.
    c/a independent (c-axis invariant under [0001] rotation).
    """
    results = []
    seen: set[tuple] = set()

    for sigma in range(2, sigma_max + 1):
        pairs = _loeschian_pairs(sigma * sigma)
        if not pairs:
            continue

        for u, v in pairs:
            M = np.array([[u, -v, 0],
                          [v, u - v, 0],
                          [0, 0, sigma]])

            cos_t = (2 * u - v) / (2.0 * sigma)
            if abs(cos_t) > 1.0:
                continue
            angle = degrees(acos(min(1.0, max(-1.0, cos_t))))

            # Disorientation in [0, 30] deg (hex 6-fold symmetry)
            ang_disor = angle % 60.0
            if ang_disor > 30.0:
                ang_disor = 60.0 - ang_disor

            key = (sigma, round(ang_disor, 3))
            if key in seen:
                continue
            seen.add(key)

            rec = integer_matrix_to_csl(M, sigma, ca)
            if rec is None:
                continue

            rec["angle_disorientation"] = round(ang_disor, 4)
            rec["axis_type"] = "[0001]"
            results.append(rec)

    results.sort(key=lambda r: (r["sigma"], r["angle_disorientation"]))
    return results


# ---------------------------------------------------------------------------
# Enumerate CSL for general (non-[0001]) rotation axes   (c/a-dependent)
# ---------------------------------------------------------------------------

def _build_candidate_axes(max_idx: int, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build unique low-index hex axis directions (excluding [001]).

    Returns
    -------
    uvw_arr : (N, 3) int array of 3-axis Miller indices
    ax_cart : (N, 3) float array of Cartesian unit vectors
    """
    rng = np.arange(-max_idx, max_idx + 1)
    grid = np.array(np.meshgrid(rng, rng, rng)).reshape(3, -1).T  # (K, 3)

    # Remove [0,0,0] and pure [0,0,w]
    nz = np.any(grid[:, :2] != 0, axis=1)
    grid = grid[nz]

    # Reduce to primitive vectors
    reduced = {}
    for row in grid:
        u, v, w = int(row[0]), int(row[1]), int(row[2])
        g = _gcd_many(u, v, w) or 1
        r = (u // g, v // g, w // g)
        # Canonical sign
        for c in r:
            if c != 0:
                if c < 0:
                    r = (-r[0], -r[1], -r[2])
                break
        reduced[r] = True

    uvw_list = list(reduced.keys())
    uvw_arr = np.array(uvw_list, dtype=int)

    # Convert to Cartesian unit vectors  (N, 3)
    cart = (S @ uvw_arr.T).T                     # (N, 3)
    norms = np.linalg.norm(cart, axis=1, keepdims=True)
    ax_cart = cart / norms

    return uvw_arr, ax_cart


def enumerate_tilt_csl(ca: float, sigma_max: int = 30,
                       max_idx: int = 10) -> list[dict]:
    """
    Enumerate CSL GBs for rotation axes NOT parallel to [0001].

    Uses vectorised batch rotation: for each candidate axis, sweep all
    angles in one numpy call, convert to crystal coords, and check
    integer conditions across all sigmas simultaneously.
    """
    results = []
    seen: set[tuple] = set()

    S = crystal_basis_matrix(ca)
    S_inv = np.linalg.inv(S)

    uvw_arr, ax_cart_arr = _build_candidate_axes(max_idx, S)
    n_axes = len(uvw_arr)

    # Angle grid (0.05 deg resolution, exclude 0)
    n_angles = 3600
    angle_indices = np.arange(1, n_angles)
    thetas = angle_indices * (np.pi / n_angles)  # (A,) in radians

    sigmas = np.arange(2, sigma_max + 1)         # (S,)

    for i_ax in range(n_axes):
        ax = ax_cart_arr[i_ax]

        # Batch rotation matrices for this axis at all angles  (A, 3, 3)
        axes_rep = np.tile(ax, (len(thetas), 1))
        R_carts = _rodrigues_batch(axes_rep, thetas)

        # Convert ALL to crystal coords:  R_cryst = S_inv @ R_cart @ S
        # Vectorised:  (A,3,3)
        R_crysts = np.einsum("ij,ajk,kl->ail", S_inv, R_carts, S)

        # For each sigma, check if sigma * R_cryst is near-integer
        for sigma in sigmas:
            M_all = sigma * R_crysts                   # (A, 3, 3)
            M_round = np.round(M_all)
            residuals = np.abs(M_all - M_round)        # (A, 3, 3)
            max_res = residuals.reshape(len(thetas), -1).max(axis=1)  # (A,)

            hits = np.where(max_res < 0.02)[0]
            if len(hits) == 0:
                continue

            for idx in hits:
                M_int = M_round[idx].astype(int)

                # Primitive check
                g = _gcd_many(*M_int.ravel())
                if g > 1:
                    continue

                rec = integer_matrix_to_csl(M_int, int(sigma), ca, tol=1e-4)
                if rec is None:
                    continue

                angle = rec["angle_deg"]
                axis_key = tuple(sorted(abs(x) for x in rec["axis_miller"]))
                key = (int(sigma), round(angle, 1), axis_key)
                if key in seen:
                    continue
                seen.add(key)

                rec["axis_type"] = "tilt"
                results.append(rec)

    results.sort(key=lambda r: (r["sigma"], r["angle_deg"]))
    return results


# ---------------------------------------------------------------------------
# Combined enumerator
# ---------------------------------------------------------------------------

def enumerate_hcp_csl(
    ca_ratio: float,
    sigma_max: int = 30,
    include_0001: bool = True,
    include_tilt: bool = True,
    max_idx: int = 10,
) -> list[dict]:
    """
    Enumerate HCP CSL grain boundaries.

    Parameters
    ----------
    ca_ratio : float
        c/a ratio (e.g. 1.587 Ti, 1.624 Mg, 1.633 ideal = sqrt(8/3)).
    sigma_max : int
        Maximum Sigma value (coincidence index).
    include_0001 : bool
        Include [0001]-axis GBs (c/a independent).
    include_tilt : bool
        Include tilt GBs with non-[0001] axes (c/a dependent).
    max_idx : int
        Maximum Miller index for candidate rotation axes.

    Returns
    -------
    list of dict with keys: sigma, angle_deg, axis_miller,
    axis_miller_bravais, axis_cart, R_cart, M_crystal
    """
    all_results = []
    if include_0001:
        all_results.extend(enumerate_0001_csl(sigma_max, ca_ratio))
    if include_tilt:
        all_results.extend(enumerate_tilt_csl(ca_ratio, sigma_max, max_idx))
    all_results.sort(key=lambda r: (r["sigma"], r["angle_deg"]))
    return all_results


# ---------------------------------------------------------------------------
# Query / filter
# ---------------------------------------------------------------------------

def find_csl(
    results: list[dict] | None = None,
    *,
    ca_ratio: float | None = None,
    sigma: int | None = None,
    axis_miller: list[int] | None = None,
    axis_miller_bravais: list[int] | None = None,
    angle_min: float = 0.0,
    angle_max: float = 180.0,
    sigma_max: int = 50,
) -> list[dict]:
    """
    Filter or generate HCP CSL records.

    If `results` is provided, filter it; otherwise generate first.

    Examples
    --------
    >>> recs = find_csl(ca_ratio=1.587, sigma=7)
    >>> recs = find_csl(results, axis_miller_bravais=[0,0,0,1])
    """
    if results is None:
        if ca_ratio is None:
            raise ValueError("Provide either `results` or `ca_ratio`.")
        results = enumerate_hcp_csl(ca_ratio, sigma_max=sigma_max)

    out = results

    if sigma is not None:
        out = [r for r in out if r["sigma"] == sigma]

    if axis_miller is not None:
        axis_miller = list(axis_miller)

    if axis_miller_bravais is not None:
        axis_miller = miller_bravais_to_3axis(axis_miller_bravais)

    if axis_miller is not None:
        _ca = ca_ratio if ca_ratio is not None else sqrt(8.0 / 3.0)
        S = crystal_basis_matrix(_ca)
        g = _gcd_many(*axis_miller) or 1
        query = [x // g for x in axis_miller]
        # Generate hex-symmetry equivalents of query axis
        q_equivs = _hex_symmetry_equivalent_dirs(query, S)
        out = [r for r in out
               if any(abs(float(np.dot(r["axis_cart"], qe))) > 0.99
                      for qe in q_equivs)]

    out = [r for r in out if angle_min <= r["angle_deg"] <= angle_max]
    return out


# ---------------------------------------------------------------------------
# Display utilities
# ---------------------------------------------------------------------------

def mb_str(uvtw: list[int]) -> str:
    """Format 4-index Miller-Bravais as a string like [0001]."""
    parts = []
    for x in uvtw:
        if x < 0:
            parts.append(f"\u0305{abs(x)}")  # overbar
        else:
            parts.append(str(x))
    return "[" + " ".join(parts) + "]"


def print_csl_table(results: list[dict], max_rows: int = 80) -> None:
    """Pretty-print a table of CSL grain boundaries."""
    if not results:
        print("No CSL grain boundaries found.")
        return

    header = f"{'Sigma':>6}  {'Angle':>10}  {'Axis [uvtw]':>16}  {'Axis [UVW]':>12}"
    print(header)
    print("-" * len(header))

    for r in results[:max_rows]:
        uvtw = r.get("axis_miller_bravais", r["axis_miller"])
        uvtw_s = mb_str(uvtw)
        uvw_s = "[" + " ".join(f"{x:>2}" for x in r["axis_miller"]) + "]"
        print(f"{r['sigma']:>6}  {r['angle_deg']:>10.2f}  {uvtw_s:>16}  {uvw_s:>12}")

    if len(results) > max_rows:
        print(f"  ... ({len(results) - max_rows} more rows)")


def to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert CSL results to a pandas DataFrame."""
    rows = []
    for r in results:
        uvtw = r.get("axis_miller_bravais", r["axis_miller"])
        rows.append({
            "sigma": r["sigma"],
            "angle_deg": r["angle_deg"],
            "axis_uvtw": mb_str(uvtw),
            "axis_UVW": "[" + " ".join(str(x) for x in r["axis_miller"]) + "]",
            "axis_type": r.get("axis_type", ""),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Command-line / demo
# ---------------------------------------------------------------------------

