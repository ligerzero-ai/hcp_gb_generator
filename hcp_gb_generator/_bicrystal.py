"""
Unified bicrystal builder for ALL grain boundary types.

Strategy (adapted from gb_code by Hossein Behtari):
  1. Define a target simulation cell (in-plane CSL vectors + stacking)
  2. Fill the cell with grain 1's lattice (original HCP)
  3. Fill the SAME cell with grain 2's lattice (rotated HCP)
  4. Stack the two half-cells along the GB normal
  5. Remove overlapping atoms at the interface

This works for twist, tilt, mixed, symmetric, and asymmetric GBs
with no special-case logic.
"""

from __future__ import annotations

from math import sqrt, gcd
from functools import reduce

import numpy as np
from ase import Atoms
from ase.build import bulk, make_supercell

from hcp_gb_generator._core import (
    crystal_basis_matrix,
    rotation_axis_angle,
)


# ---------------------------------------------------------------------------
# Core: fill a cell with a rotated lattice
# ---------------------------------------------------------------------------

def _fill_cell(element: str, a: float, c: float,
               R: np.ndarray, target_cell: np.ndarray,
               tol: float = 0.001) -> Atoms:
    """
    Fill a target cell with HCP lattice points in orientation R.

    Enumerates all lattice + basis points of the R-rotated HCP
    lattice that fall within the parallelepiped defined by target_cell.
    This works for ANY rotation — no integer supercell matrix needed.

    Adapted from the gb_code approach (Behtari et al.).

    Parameters
    ----------
    element : str
        Chemical symbol.
    a, c : float
        HCP lattice parameters.
    R : (3, 3)
        Rotation matrix (Cartesian).  np.eye(3) for unrotated.
    target_cell : (3, 3)
        Target cell vectors (rows) in Cartesian Å.
    tol : float
        Tolerance for point-in-cell check.

    Returns
    -------
    Atoms filling target_cell with the R-rotated HCP lattice.
    """
    # HCP primitive cell (Cartesian rows)
    prim_cell = np.array([
        [a, 0, 0],
        [-a / 2, a * sqrt(3) / 2, 0],
        [0, 0, c],
    ])

    # HCP basis: 2 atoms per primitive cell (fractional → Cartesian)
    basis_frac = np.array([[0.0, 0.0, 0.0],
                           [1.0 / 3, 2.0 / 3, 0.5]])
    basis_cart = basis_frac @ prim_cell   # (2, 3)

    # Rotated lattice
    rot_cell = prim_cell @ R.T
    basis_rot = basis_cart @ R.T

    # Map target cell corners to rotated crystal coordinates to find
    # the bounding box of integer lattice indices to search.
    corners = np.array([[i, j, k] for i in [0, 1]
                        for j in [0, 1] for k in [0, 1]], dtype=float)
    corner_cart = corners @ target_cell
    corner_cryst = np.linalg.solve(rot_cell.T, corner_cart.T).T

    lo = np.floor(corner_cryst.min(axis=0)).astype(int) - 1
    hi = np.ceil(corner_cryst.max(axis=0)).astype(int) + 1

    # Inverse of target cell for fractional coordinate test
    cell_inv = np.linalg.inv(target_cell.T)

    # Enumerate lattice + basis points inside the target cell
    ni = np.arange(lo[0], hi[0] + 1)
    nj = np.arange(lo[1], hi[1] + 1)
    nk = np.arange(lo[2], hi[2] + 1)
    grid = np.array(np.meshgrid(ni, nj, nk, indexing='ij')).reshape(3, -1).T
    lattice_pts = grid @ rot_cell   # (N, 3)

    all_pts = []
    for b in basis_rot:
        pts = lattice_pts + b
        frac = (cell_inv @ pts.T).T
        mask = np.all(frac >= -tol, axis=1) & np.all(frac < 1.0 - tol, axis=1)
        all_pts.append(pts[mask])

    positions = np.vstack(all_pts) if all_pts else np.zeros((0, 3))

    atoms = Atoms(
        symbols=[element] * len(positions),
        positions=positions,
        cell=target_cell,
        pbc=True,
    )
    return atoms


def _remove_overlaps(atoms: Atoms, overlap_tol: float = 0.5) -> Atoms:
    """
    Remove one atom from each pair closer than overlap_tol.

    Prefers removing grain-2 atoms (keeps grain-1 structure at interface).
    """
    if overlap_tol <= 0:
        return atoms

    pos = atoms.positions
    grain_ids = atoms.arrays.get("grain_id", np.ones(len(atoms), dtype=int))

    remove = set()
    # Only check atoms near the interface (within overlap_tol of grain boundary)
    g1_mask = grain_ids == 1
    g2_mask = grain_ids == 2
    g1_idx = np.where(g1_mask)[0]
    g2_idx = np.where(g2_mask)[0]

    if len(g1_idx) == 0 or len(g2_idx) == 0:
        return atoms

    g1_z_max = pos[g1_mask, 2].max()
    g2_z_min = pos[g2_mask, 2].min()

    # Only compare atoms near the interface
    near1 = g1_idx[np.abs(pos[g1_idx, 2] - g1_z_max) < overlap_tol * 3]
    near2 = g2_idx[np.abs(pos[g2_idx, 2] - g2_z_min) < overlap_tol * 3]

    for i2 in near2:
        for i1 in near1:
            if i1 in remove:
                continue
            d = np.linalg.norm(pos[i2] - pos[i1])
            if d < overlap_tol:
                remove.add(i2)
                break

    if remove:
        keep = [i for i in range(len(atoms)) if i not in remove]
        return atoms[keep]
    return atoms


# ---------------------------------------------------------------------------
# Target cell from CSL record
# ---------------------------------------------------------------------------

def _find_minimal_csl_cell(
    csl_record: dict,
    prim: Atoms,
) -> np.ndarray:
    """
    Find the minimal CSL supercell (volume = Σ × V_primitive).

    The CSL is the sublattice of lattice-1 vectors that are ALSO
    lattice-2 vectors.  A lattice-1 vector v (integer crystal coords)
    is a CSL vector iff  Σ M^{-1} v  is also integer.

    We find this sublattice by searching integer crystal-coord vectors
    and checking the CSL condition.

    Returns (3, 3) Cartesian cell (rows = CSL basis vectors).
    """
    M = np.asarray(csl_record["M_crystal"], dtype=float)
    sigma = int(csl_record["sigma"])
    prim_cell = np.array(prim.cell[:])

    # M^{-1} = adj(M) / det(M).  For a CSL vector v: Σ M^{-1} v ∈ Z³
    # i.e., adj(M) @ v must be divisible by Σ² (since det(M) = Σ³).
    M_int = np.round(M).astype(int)
    adjM = np.round(np.linalg.det(M_int) * np.linalg.inv(M_int)).astype(int)
    sigma_sq = sigma * sigma

    # Search for short CSL vectors
    max_r = int(np.ceil(sigma ** (1 / 3))) + 3
    csl_vecs = []

    for i in range(-max_r * 2, max_r * 2 + 1):
        for j in range(-max_r * 2, max_r * 2 + 1):
            for k in range(-max_r * 2, max_r * 2 + 1):
                if i == 0 and j == 0 and k == 0:
                    continue
                v = np.array([i, j, k])
                # CSL condition: adj(M) @ v divisible by Σ²
                w = adjM @ v
                if np.all(w % sigma_sq == 0):
                    # Convert to Cartesian and store
                    cart = v.astype(float) @ prim_cell
                    length = np.linalg.norm(cart)
                    csl_vecs.append((length, i, j, k, cart))

    # Sort by length
    csl_vecs.sort()

    # Find 3 linearly independent CSL vectors forming a cell with
    # volume = Σ × V_primitive
    target_vol = sigma * abs(np.linalg.det(prim_cell))

    for idx1 in range(len(csl_vecs)):
        v1 = csl_vecs[idx1][4]
        for idx2 in range(idx1 + 1, len(csl_vecs)):
            v2 = csl_vecs[idx2][4]
            cross12 = np.cross(v1, v2)
            if np.linalg.norm(cross12) < 1e-6:
                continue  # parallel
            for idx3 in range(idx2 + 1, len(csl_vecs)):
                v3 = csl_vecs[idx3][4]
                vol = abs(np.dot(v3, cross12))
                if vol < target_vol * 0.9:
                    continue  # too small (sub-cell)
                if abs(vol - target_vol) < target_vol * 0.01:
                    return np.array([v1, v2, v3])
                if vol > target_vol * 1.5:
                    break  # too large, no point continuing

    raise ValueError(
        f"Could not find minimal CSL cell for Sigma={sigma}. "
        f"Try increasing the search range."
    )


def _csl_slab_cell(
    csl_record: dict,
    prim: Atoms,
    n_layers: int = 6,
) -> np.ndarray:
    """
    Get a slab cell from the minimal CSL cell.

    Reorders vectors so the shortest is the stacking direction
    (replicated by n_layers for thickness).

    Returns (3, 3) Cartesian cell.
    """
    cell = _find_minimal_csl_cell(csl_record, prim)

    # Shortest vector = stacking direction
    lengths = np.linalg.norm(cell, axis=1)
    stacking_idx = np.argmin(lengths)
    order = [i for i in range(3) if i != stacking_idx] + [stacking_idx]
    cell_slab = cell[order]

    # Replicate stacking for thickness
    cell_slab[2] *= n_layers

    return cell_slab


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_bicrystal(
    csl_record: dict,
    element: str = "Ti",
    a: float = 2.95,
    c: float | None = None,
    n_layers: int = 6,
    interface_distance: float = 0.0,
    vacuum: float = 0.0,
    overlap_tol: float = 0.0,
    symmetric: bool = True,
) -> Atoms:
    """
    Build a grain boundary bicrystal from a CSL record.

    Unified builder that works for twist, tilt, mixed, symmetric,
    and asymmetric boundaries.  No special-case logic per GB type.

    Strategy:
      1. Define a target cell from the CSL supercell
      2. Fill it with grain 1 (original or -θ/2 rotated HCP)
      3. Fill it with grain 2 (R or +θ/2 rotated HCP)
      4. Stack along the stacking direction
      5. Remove overlapping atoms at the interface

    Parameters
    ----------
    csl_record : dict
        Record from ``enumerate_hcp_csl`` / ``find_csl``.
    element : str
        Chemical symbol.
    a, c : float
        Lattice parameters (Å).  c defaults to ``a * sqrt(8/3)``.
    n_layers : int
        Layers per grain along the stacking direction.
    interface_distance : float
        Spacing between grains at the GB plane (Å).
    vacuum : float
        Vacuum above bicrystal (Å).  0 = periodic (2 GBs/cell).
    overlap_tol : float
        Remove atom pairs closer than this at the interface (Å).
        0 = no removal.
    symmetric : bool
        True: grains at ±θ/2 (symmetric tilt).
        False: grain 1 at identity, grain 2 at θ (asymmetric / general).

    Returns
    -------
    ase.Atoms
        Bicrystal with ``grain_id`` array and metadata in ``info``.
    """
    if c is None:
        c = a * sqrt(8.0 / 3.0)

    prim = bulk(element, "hcp", a=a, c=c)
    R_cart = np.asarray(csl_record["R_cart"])

    # Grain rotations
    if symmetric:
        axis, angle_deg = rotation_axis_angle(R_cart)
        half_rad = np.radians(angle_deg / 2)
        ct, st = np.cos(half_rad), np.sin(half_rad)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R_half = ct * np.eye(3) + st * K + (1 - ct) * np.outer(axis, axis)
        R1 = R_half.T   # -θ/2
        R2 = R_half      # +θ/2
    else:
        R1 = np.eye(3)
        R2 = R_cart

    # Target cell from the minimal CSL (one layer thick)
    target_cell = _csl_slab_cell(csl_record, prim, n_layers=1)

    # Fill target cell with each grain's lattice
    grain1 = _fill_cell(element, a, c, R1, target_cell)
    grain2 = _fill_cell(element, a, c, R2, target_cell)

    # Replicate along stacking (axis 2) for thickness
    grain1 *= (1, 1, n_layers)
    grain2 *= (1, 1, n_layers)

    # Ensure positive z
    for slab in (grain1, grain2):
        if slab.cell[2, 2] < 0:
            frac = slab.get_scaled_positions(wrap=False)
            slab.cell[2] = -slab.cell[2]
            frac[:, 2] = -frac[:, 2]
            slab.set_scaled_positions(frac)
            slab.positions[:, 2] -= slab.positions[:, 2].min()

    # Tag grains
    grain1.arrays["grain_id"] = np.ones(len(grain1), dtype=int)
    grain2.arrays["grain_id"] = np.full(len(grain2), 2, dtype=int)

    # Stack grain2 above grain1
    offset = grain1.cell[2, 2] + interface_distance
    grain2.positions[:, 2] += offset

    bicrystal = grain1.copy()
    bicrystal.extend(grain2)

    total_height = offset + grain2.cell[2, 2]
    if vacuum > 0:
        bicrystal.cell[2, 2] = total_height + vacuum
        bicrystal.pbc = [True, True, False]
    else:
        bicrystal.cell[2, 2] = total_height
        bicrystal.pbc = [True, True, True]

    # Remove overlapping atoms at interface
    if overlap_tol > 0:
        bicrystal = _remove_overlaps(bicrystal, overlap_tol)

    # Metadata
    bicrystal.info["sigma"] = csl_record["sigma"]
    bicrystal.info["disorientation_angle"] = csl_record["disorientation_angle"]
    bicrystal.info["axis_miller"] = csl_record["axis_miller"]
    bicrystal.info["interface_distance"] = interface_distance
    bicrystal.info["vacuum"] = vacuum
    bicrystal.info["n_layers"] = n_layers
    bicrystal.info["symmetric"] = symmetric

    return bicrystal
