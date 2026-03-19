"""
Construct ASE grain boundary structures from CSL enumeration results.

Provides helper functions to go from a CSL record
(sigma, axis, angle, rotation matrix) → ASE Atoms bicrystal.

Two main construction paths:

1. **Twist GBs** (rotation axis = GB plane normal, e.g. [0001] twist):
   ``build_twist_gb()`` — builds both grains as (hkl) slabs, rotates one
   in-plane by the CSL angle, stacks them.

2. **Tilt GBs** (rotation axis lies IN the GB plane):
   ``csl_slab_directions()`` — computes the 4-index Miller-Bravais
   direction triples needed by ASE's ``HexagonalClosedPacked`` factory.
   ``build_tilt_gb()`` — end-to-end construction from a CSL record.

All functions accept the dict returned by ``enumerate_hcp_csl`` / ``find_csl``.
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
    three_axis_to_miller_bravais,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _gcd_many(*vals: int) -> int:
    return reduce(gcd, [abs(v) for v in vals if v != 0], 0)


def _reduce_vec(v: list[int]) -> list[int]:
    g = _gcd_many(*v)
    return [x // g for x in v] if g else list(v)


def _nearest_int_direction(cart_vec: np.ndarray, S: np.ndarray,
                           max_idx: int = 12) -> list[int]:
    """
    Find the smallest-integer 3-axis hex direction parallel to a
    Cartesian vector.
    """
    cryst = np.linalg.solve(S, cart_vec)
    # Try integer multiples
    for scale in range(1, max_idx * 10):
        trial = cryst * scale
        rounded = np.round(trial).astype(int)
        if np.allclose(trial, rounded, atol=0.02):
            return _reduce_vec(rounded.tolist())
    # Fallback: best small-integer approximation
    best, best_err = [1, 0, 0], 1e9
    rng = range(-max_idx, max_idx + 1)
    cn = cryst / (np.linalg.norm(cryst) + 1e-15)
    for u in rng:
        for v in rng:
            for w in rng:
                if u == v == w == 0:
                    continue
                t = np.array([u, v, w], dtype=float)
                tn = t / np.linalg.norm(t)
                err = np.linalg.norm(tn - cn)
                if err < best_err:
                    best_err = err
                    best = _reduce_vec([u, v, w])
    return best


def _to_4index(uvw: list[int]) -> list[int]:
    """3-axis [U,V,W] → 4-index [u,v,t,w] Miller-Bravais."""
    return three_axis_to_miller_bravais(uvw)


# ---------------------------------------------------------------------------
# Build the CSL supercell matrix
# ---------------------------------------------------------------------------

def csl_supercell_matrix(csl_record: dict) -> np.ndarray:
    """
    Return the 3×3 integer matrix P such that the CSL supercell
    (in crystal coordinates) is P @ primitive_cell.

    For a [0001] twist GB the in-plane part comes from the CSL
    integer matrix M; the c-axis is unchanged.
    """
    M = np.asarray(csl_record["M_crystal"])
    sigma = csl_record["sigma"]
    # M = sigma * R_crystal → the CSL supercell in grain-1 crystal coords
    # has columns [M_col1, M_col2, M_col3] (each is a lattice vector × sigma)
    # but we want the PRIMITIVE CSL vectors, not scaled by sigma.
    # For [0001] twist: P = [[u, -v, 0], [v, u-v, 0], [0, 0, 1]]
    # where M = [[u,-v,0],[v,u-v,0],[0,0,sigma]]
    # For general tilt: P = M (the full matrix IS the supercell)
    return M


# ---------------------------------------------------------------------------
# Twist GB construction
# ---------------------------------------------------------------------------

def build_twist_gb(
    csl_record: dict,
    element: str = "Ti",
    a: float = 2.95,
    c: float | None = None,
    n_layers: int = 6,
    vacuum: float = 0.0,
    gap: float = 0.5,
) -> Atoms:
    """
    Build a twist grain boundary from a [0001]-axis CSL record.

    The two grains share the same (0001) surface; the upper grain is
    rotated in-plane by the CSL angle θ.  The simulation cell uses the
    CSL supercell vectors so both grains are commensurate.

    Parameters
    ----------
    csl_record : dict
        Record from ``enumerate_0001_csl`` or ``find_csl``.
    element : str
        Chemical symbol.
    a, c : float
        Lattice parameters (Å).  If c is None, uses ``a * ca_ratio``
        where ca_ratio is inferred from the CSL record's basis matrix.
    n_layers : int
        Number of (0001) bilayers per grain.
    vacuum : float
        Vacuum above and below the bicrystal (Å).
    gap : float
        Separation between the two grains at the interface (Å).

    Returns
    -------
    ase.Atoms
        Bicrystal with ``grain_id`` array (1 = lower, 2 = upper).
    """
    if c is None:
        c = a * sqrt(8.0 / 3.0)

    M = np.asarray(csl_record["M_crystal"])
    sigma = csl_record["sigma"]
    R_cart = np.asarray(csl_record["R_cart"])

    # Build primitive HCP unit cell
    prim = bulk(element, "hcp", a=a, c=c)

    # CSL supercell matrix (crystal coords → crystal coords)
    # For [0001] twist: M = [[u,-v,0],[v,u-v,0],[0,0,sigma]]
    # The CSL supercell in Cartesian = prim.cell.T @ P.T
    # where P is the transformation matrix in crystal coords.
    # We use P = [[u, v, 0], [-v, u-v, 0], [0, 0, 1]] to get
    # a supercell whose in-plane area = sigma × primitive area.
    u, v = int(M[0, 0]), int(M[1, 0])
    P = np.array([
        [u, v, 0],
        [-v, u - v, 0],
        [0, 0, 1],
    ])

    # Build lower grain (standard orientation, CSL supercell)
    lower = make_supercell(prim, P)

    # Replicate along c for desired thickness
    lower *= (1, 1, n_layers)

    # Build upper grain: same supercell, but atoms rotated by R_cart
    upper = lower.copy()
    # Rotate atom positions around the center of the cell
    center = upper.cell.sum(axis=0) / 2
    upper.positions = (
        (upper.positions - center) @ R_cart.T + center
    )

    # Stack: shift upper grain above lower
    offset = lower.cell[2, 2] + gap
    upper.positions[:, 2] += offset

    # Combine
    bicrystal = lower.copy()
    bicrystal.arrays["grain_id"] = np.ones(len(lower), dtype=int)

    upper_tagged = upper.copy()
    upper_tagged.arrays["grain_id"] = np.full(len(upper), 2, dtype=int)

    bicrystal.extend(upper_tagged)
    bicrystal.cell[2, 2] = offset + upper.cell[2, 2] + vacuum
    bicrystal.pbc = [True, True, vacuum == 0.0]

    bicrystal.info["sigma"] = sigma
    bicrystal.info["disorientation_angle"] = csl_record["disorientation_angle"]
    bicrystal.info["axis_miller"] = csl_record["axis_miller"]
    bicrystal.info["gap"] = gap
    bicrystal.info["n_layers"] = n_layers

    return bicrystal


# ---------------------------------------------------------------------------
# Tilt GB: compute slab direction vectors
# ---------------------------------------------------------------------------

def csl_slab_directions(
    csl_record: dict,
    ca: float,
    gb_plane_3ax: list[int] | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Compute the 4-index Miller-Bravais direction triples for upper and
    lower grains of a symmetric tilt GB.

    These can be passed directly to ASE's ``HexagonalClosedPacked``
    factory or to ``HCPGBGenerator.from_basal_directions()``.

    Parameters
    ----------
    csl_record : dict
        Record from ``enumerate_tilt_csl`` or ``find_csl``.
    ca : float
        c/a ratio.
    gb_plane_3ax : list[int], optional
        GB plane normal in 3-axis notation.  If None, the symmetric
        tilt plane is computed (perpendicular to the rotation in the
        plane containing the tilt axis and the misorientation).

    Returns
    -------
    upper_dirs, lower_dirs : list of 3 lists of 4 ints
        Direction triples in 4-index Miller-Bravais [u, v, t, w].
        Index 0 = x-direction (GB normal / stacking direction)
        Index 1 = y-direction (in-plane, perpendicular to tilt axis)
        Index 2 = z-direction (tilt axis, periodic along GB)
    """
    S = crystal_basis_matrix(ca)
    R_cart = np.asarray(csl_record["R_cart"])
    axis_cart = np.asarray(csl_record["axis_cart"])
    # Use raw rotation angle for geometric construction, not disorientation
    angle = csl_record.get("angle_raw", csl_record["disorientation_angle"])

    # Tilt axis = z-direction of the slab (periodic along GB)
    z_dir_3ax = _nearest_int_direction(axis_cart, S)
    z_cart = S @ np.array(z_dir_3ax, dtype=float)
    z_cart /= np.linalg.norm(z_cart)

    if gb_plane_3ax is not None:
        # User-specified GB plane normal
        x_cart = S @ np.array(gb_plane_3ax, dtype=float)
    else:
        # Symmetric tilt: GB plane normal bisects the rotation.
        # For rotation R about axis z by angle θ, the symmetric
        # tilt plane normal is R^(1/2) applied to an arbitrary
        # in-basal-plane direction perpendicular to the tilt axis.
        # Practically: find a direction NOT parallel to z_cart.
        perp = np.array([1, 0, 0], dtype=float)
        if abs(np.dot(perp, z_cart)) > 0.9:
            perp = np.array([0, 1, 0], dtype=float)
        # Project out z-component
        perp = perp - np.dot(perp, z_cart) * z_cart
        perp /= np.linalg.norm(perp)

        # Half-rotation
        half_angle = np.radians(angle / 2)
        ct, st = np.cos(half_angle), np.sin(half_angle)
        K = np.array([
            [0, -z_cart[2], z_cart[1]],
            [z_cart[2], 0, -z_cart[0]],
            [-z_cart[1], z_cart[0], 0],
        ])
        R_half = (np.eye(3) * ct
                  + st * K
                  + (1 - ct) * np.outer(z_cart, z_cart))

        x_cart = R_half @ perp

    x_cart_unit = x_cart / np.linalg.norm(x_cart)

    # y-direction = z × x
    y_cart = np.cross(z_cart, x_cart_unit)
    y_cart /= np.linalg.norm(y_cart)

    # Convert to integer Miller indices
    x_3ax = _nearest_int_direction(x_cart_unit, S)
    y_3ax = _nearest_int_direction(y_cart, S)
    z_3ax = z_dir_3ax

    # Upper grain: standard orientation
    upper_dirs = [
        _to_4index(x_3ax),
        _to_4index(y_3ax),
        _to_4index(z_3ax),
    ]

    # Lower grain: x-direction is mirrored (symmetric tilt)
    neg_x = [-v for v in x_3ax]
    lower_dirs = [
        _to_4index(neg_x),
        _to_4index(y_3ax),
        _to_4index(z_3ax),
    ]

    return upper_dirs, lower_dirs


def build_tilt_gb(
    csl_record: dict,
    element: str = "Ti",
    a: float = 2.95,
    c: float | None = None,
    gb_plane_3ax: list[int] | None = None,
    n_layers: int = 6,
    vacuum: float = 0.0,
    gap: float = 0.5,
) -> Atoms:
    """
    Build a symmetric tilt grain boundary from a CSL record.

    Uses ASE's ``HexagonalClosedPacked`` factory to create oriented
    slabs, then stacks them.

    Parameters
    ----------
    csl_record : dict
        Record from ``enumerate_tilt_csl`` or ``find_csl``.
    element, a, c : str, float
        Element symbol and lattice parameters.
    gb_plane_3ax : list[int], optional
        GB plane normal in 3-axis hex notation.  Auto-computed if None.
    n_layers : int
        Slab thickness in unit cell layers along the stacking direction.
    vacuum, gap : float
        Vacuum padding and gap between grains (Å).

    Returns
    -------
    ase.Atoms
        Bicrystal with ``grain_id`` array.
    """
    from ase.lattice.hexagonal import HexagonalClosedPacked

    if c is None:
        c = a * sqrt(8.0 / 3.0)
    ca = c / a

    upper_dirs, lower_dirs = csl_slab_directions(
        csl_record, ca, gb_plane_3ax
    )

    # Build slabs
    size = (1, 1, n_layers)
    try:
        upper = HexagonalClosedPacked(
            symbol=element,
            latticeconstant=(a, c),
            directions=upper_dirs,
            size=size,
        )
        lower = HexagonalClosedPacked(
            symbol=element,
            latticeconstant=(a, c),
            directions=lower_dirs,
            size=size,
        )
    except Exception as exc:
        raise RuntimeError(
            f"ASE could not build slab with directions "
            f"upper={upper_dirs}, lower={lower_dirs}. "
            f"This may happen for high-index orientations. "
            f"Try specifying gb_plane_3ax explicitly.\n"
            f"Original error: {exc}"
        ) from exc

    # Stack along z (index 2)
    offset = lower.cell[2, 2] + gap
    upper.positions[:, 2] += offset

    bicrystal = lower.copy()
    bicrystal.arrays["grain_id"] = np.ones(len(lower), dtype=int)

    upper_copy = upper.copy()
    upper_copy.arrays["grain_id"] = np.full(len(upper), 2, dtype=int)

    bicrystal.extend(upper_copy)
    bicrystal.cell[2, 2] = offset + upper.cell[2, 2] + vacuum
    bicrystal.pbc = [True, True, vacuum == 0.0]

    bicrystal.info["sigma"] = csl_record["sigma"]
    bicrystal.info["disorientation_angle"] = csl_record["disorientation_angle"]
    bicrystal.info["axis_miller"] = csl_record["axis_miller"]
    bicrystal.info["upper_dirs"] = upper_dirs
    bicrystal.info["lower_dirs"] = lower_dirs
    bicrystal.info["gap"] = gap

    return bicrystal


# ---------------------------------------------------------------------------
# Convenience: auto-dispatch twist vs tilt
# ---------------------------------------------------------------------------

def build_gb(
    csl_record: dict,
    element: str = "Ti",
    a: float = 2.95,
    c: float | None = None,
    gb_plane_3ax: list[int] | None = None,
    n_layers: int = 6,
    vacuum: float = 0.0,
    gap: float = 0.5,
) -> Atoms:
    """
    Build a grain boundary bicrystal from a CSL record.

    Automatically selects twist or tilt construction based on the
    rotation axis.

    Parameters
    ----------
    csl_record : dict
        Record from ``enumerate_hcp_csl`` / ``find_csl``.
    element : str
        Chemical symbol (e.g. "Ti", "Mg", "Zr").
    a : float
        In-plane lattice parameter (Å).
    c : float, optional
        Out-of-plane lattice parameter.  If None, ``a * sqrt(8/3)``.
    gb_plane_3ax : list[int], optional
        GB plane normal (3-axis notation).  Auto-detected if None.
    n_layers : int
        Layers per grain (thickness control).
    vacuum : float
        Vacuum padding (Å).  0 for fully periodic.
    gap : float
        Gap between grains at the interface (Å).

    Returns
    -------
    ase.Atoms
        Bicrystal with ``grain_id`` array and metadata in ``atoms.info``.

    Examples
    --------
    >>> from hcp_gb_generator import find_csl
    >>> from hcp_gb_generator.builder import build_gb
    >>> recs = find_csl(ca_ratio=1.587, sigma=7,
    ...                 axis_miller_bravais=[0,0,0,1], sigma_max=10)
    >>> atoms = build_gb(recs[0], element="Ti", a=2.95, c=4.68,
    ...                  n_layers=4, gap=0.5)
    >>> atoms.info["sigma"]
    7
    """
    axis = csl_record.get("axis_miller", [0, 0, 1])

    # [0, 0, 1] in 3-axis = [0001] = c-axis → twist boundary
    is_twist = (axis[0] == 0 and axis[1] == 0 and axis[2] != 0)

    if is_twist:
        return build_twist_gb(
            csl_record, element, a, c,
            n_layers=n_layers, vacuum=vacuum, gap=gap,
        )
    else:
        return build_tilt_gb(
            csl_record, element, a, c,
            gb_plane_3ax=gb_plane_3ax,
            n_layers=n_layers, vacuum=vacuum, gap=gap,
        )
