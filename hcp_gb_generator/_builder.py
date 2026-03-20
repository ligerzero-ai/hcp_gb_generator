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

def _remove_duplicate_layer(lower: Atoms, upper: Atoms,
                            tol: float = 0.1) -> Atoms:
    """
    Remove atoms in ``upper`` that coincide with atoms in ``lower``
    at the interface (within ``tol`` Å).

    Returns a pruned copy of ``upper``.
    """
    # Interface: top of lower, bottom of upper
    z_lo_max = lower.positions[:, 2].max()
    z_up_min = upper.positions[:, 2].min()

    # Candidate atoms near the boundary in upper grain
    upper_out = upper.copy()
    keep = np.ones(len(upper_out), dtype=bool)

    for i, pos_up in enumerate(upper_out.positions):
        if abs(pos_up[2] - z_up_min) > tol:
            continue
        # Check against all atoms near the top of the lower grain
        for pos_lo in lower.positions:
            if abs(pos_lo[2] - z_lo_max) > tol:
                continue
            dist = np.linalg.norm(pos_up - pos_lo)
            if dist < tol:
                keep[i] = False
                break

    if not keep.all():
        upper_out = upper_out[keep]

    return upper_out


def _stack_grains(lower: Atoms, upper: Atoms, *,
                  interface_distance: float,
                  vacuum: float,
                  merge_boundary_layer: bool,
                  merge_tol: float,
                  csl_record: dict,
                  extra_info: dict | None = None) -> Atoms:
    """
    Stack two grain slabs into a bicrystal.

    Parameters
    ----------
    lower, upper : Atoms
        The two grain slabs (z is the stacking direction).
    interface_distance : float
        Spacing between grain surfaces at the GB plane (Å).
    vacuum : float
        Vacuum above the bicrystal (Å).  0 → fully periodic (2 GBs per cell).
        >0 → single GB with free surfaces.
    merge_boundary_layer : bool
        If True, remove duplicate atoms where the grains share a
        coincident plane (relevant for symmetric tilt GBs).
    merge_tol : float
        Distance tolerance for identifying coincident atoms (Å).
    csl_record : dict
        CSL record for metadata.
    extra_info : dict, optional
        Additional keys to store in ``atoms.info``.
    """
    if merge_boundary_layer:
        upper = _remove_duplicate_layer(lower, upper, tol=merge_tol)

    # Shift upper grain above lower
    offset = lower.cell[2, 2] + interface_distance
    upper_shifted = upper.copy()
    upper_shifted.positions[:, 2] += offset

    # Combine
    bicrystal = lower.copy()
    bicrystal.arrays["grain_id"] = np.ones(len(lower), dtype=int)

    upper_tagged = upper_shifted.copy()
    upper_tagged.arrays["grain_id"] = np.full(len(upper_shifted), 2, dtype=int)

    bicrystal.extend(upper_tagged)

    # Cell height
    total_height = offset + upper.cell[2, 2]
    if vacuum > 0.0:
        bicrystal.cell[2, 2] = total_height + vacuum
        bicrystal.pbc = [True, True, False]
    else:
        bicrystal.cell[2, 2] = total_height
        bicrystal.pbc = [True, True, True]

    # Metadata
    bicrystal.info["sigma"] = csl_record["sigma"]
    bicrystal.info["disorientation_angle"] = csl_record["disorientation_angle"]
    bicrystal.info["axis_miller"] = csl_record["axis_miller"]
    bicrystal.info["interface_distance"] = interface_distance
    bicrystal.info["vacuum"] = vacuum
    bicrystal.info["n_layers"] = lower.info.get("n_layers", None)
    if extra_info:
        bicrystal.info.update(extra_info)

    return bicrystal


def build_twist_gb(
    csl_record: dict,
    element: str = "Ti",
    a: float = 2.95,
    c: float | None = None,
    n_layers: int = 6,
    interface_distance: float = 0.5,
    vacuum: float = 0.0,
) -> Atoms:
    """
    Build a twist grain boundary from a [0001]-axis CSL record.

    The two grains share the same (0001) surface; the upper grain is
    rotated in-plane by the CSL angle.  The simulation cell uses the
    CSL supercell vectors so both grains are commensurate.

    Parameters
    ----------
    csl_record : dict
        Record from ``enumerate_0001_csl`` or ``find_csl``.
    element : str
        Chemical symbol.
    a, c : float
        Lattice parameters (Å).  If c is None, uses ``a * sqrt(8/3)``.
    n_layers : int
        Number of (0001) bilayers per grain.
    interface_distance : float
        Spacing between grain surfaces at the GB plane (Å).
        For twist GBs the two surfaces face each other without a shared
        plane, so a small positive value (~0.5 Å) is a reasonable
        starting point before relaxation.
    vacuum : float
        Vacuum above the bicrystal (Å).
        0 → fully periodic, 2 GBs per cell.
        >0 → single GB with free surfaces at the top and bottom.

    Returns
    -------
    ase.Atoms
        Bicrystal with ``grain_id`` array (1 = lower, 2 = upper).
    """
    if c is None:
        c = a * sqrt(8.0 / 3.0)

    M = np.asarray(csl_record["M_crystal"])
    R_cart = np.asarray(csl_record["R_cart"])

    prim = bulk(element, "hcp", a=a, c=c)

    u, v = int(M[0, 0]), int(M[1, 0])
    P = np.array([
        [u, v, 0],
        [-v, u - v, 0],
        [0, 0, 1],
    ])

    lower = make_supercell(prim, P)
    lower *= (1, 1, n_layers)
    lower.info["n_layers"] = n_layers

    upper = lower.copy()
    center = upper.cell.sum(axis=0) / 2
    upper.positions = (
        (upper.positions - center) @ R_cart.T + center
    )

    return _stack_grains(
        lower, upper,
        interface_distance=interface_distance,
        vacuum=vacuum,
        merge_boundary_layer=False,
        merge_tol=0.1,
        csl_record=csl_record,
    )


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
    interface_distance: float = 0.0,
    vacuum: float = 0.0,
    merge_boundary_layer: bool = True,
    merge_tol: float = 0.1,
) -> Atoms:
    """
    Build a symmetric tilt grain boundary from a CSL record.

    For a symmetric tilt GB the boundary plane is a mirror plane and
    the two grains share a coincident atomic layer.  By default this
    duplicate layer is merged (``merge_boundary_layer=True``).

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
    interface_distance : float
        Spacing between grain surfaces at the GB plane (Å).
        For symmetric tilt GBs the grains share a common plane, so the
        default is 0.0 (direct contact).
    vacuum : float
        Vacuum above the bicrystal (Å).
        0 → fully periodic, 2 GBs per cell (one at the interface,
            one at the periodic boundary).
        >0 → single GB with free surfaces at the top and bottom.
    merge_boundary_layer : bool
        If True (default), remove the duplicate atomic plane where the
        two grains meet.  This is the physically correct treatment for
        symmetric tilt GBs.
    merge_tol : float
        Distance tolerance (Å) for identifying coincident atoms at the
        boundary.

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

    # The mirrored lower grain can end up with a negative z cell vector
    # (from the mirrored GB normal direction).  Fix by flipping z.
    for slab in (lower, upper):
        if slab.cell[2, 2] < 0:
            slab.cell[2] = -slab.cell[2]
            slab.positions[:, 2] = -slab.positions[:, 2]
            # Shift so all z >= 0
            slab.positions[:, 2] -= slab.positions[:, 2].min()

    lower.info["n_layers"] = n_layers

    return _stack_grains(
        lower, upper,
        interface_distance=interface_distance,
        vacuum=vacuum,
        merge_boundary_layer=merge_boundary_layer,
        merge_tol=merge_tol,
        csl_record=csl_record,
        extra_info={"upper_dirs": upper_dirs, "lower_dirs": lower_dirs},
    )


# ---------------------------------------------------------------------------
# Rescale: enumerate at rational c/a, build at real lattice
# ---------------------------------------------------------------------------

def rescale_to_lattice(
    atoms: Atoms,
    a_real: float,
    c_real: float,
    a_csl: float | None = None,
    c_csl: float | None = None,
) -> Atoms:
    """
    Rescale a bicrystal built at rational c/a to the real lattice parameters.

    Exact tilt CSLs require rational (c/a)^2.  The standard workflow is:

    1. **Enumerate** at a nearby rational c/a  (e.g. sqrt(5/2) for Ti)
    2. **Build** the bicrystal at that rational c/a
    3. **Rescale** to the real lattice with this function

    The rescaling applies a uniform affine strain that maps the CSL
    lattice to the real lattice.  The ~0.3% misfit is physically
    accommodated by DSC dislocations at the boundary.

    Parameters
    ----------
    atoms : ase.Atoms
        Bicrystal built at rational c/a (from ``build_gb`` etc.).
    a_real, c_real : float
        Target (real) lattice parameters in Angstrom.
    a_csl, c_csl : float, optional
        The lattice parameters used during construction.
        If None, inferred from the cell (assumes the cell was built
        with ``a=a_csl`` and ``c=c_csl``).

    Returns
    -------
    ase.Atoms
        A copy with cell and positions rescaled.

    Examples
    --------
    >>> # Ti: enumerate at c/a = sqrt(5/2), build, then rescale
    >>> from math import sqrt
    >>> ca_csl = sqrt(5/2)
    >>> a_csl, c_csl = 2.95, 2.95 * ca_csl
    >>> gb = build_gb(rec, element="Ti", a=a_csl, c=c_csl)
    >>> gb_real = rescale_to_lattice(gb, a_real=2.95, c_real=4.68,
    ...                              a_csl=a_csl, c_csl=c_csl)
    """
    out = atoms.copy()

    if a_csl is None or c_csl is None:
        # Infer from cell: for HCP bicrystal, cell[0] length ~ n*a
        # and cell[2] / n_layers ~ c
        # Fallback: uniform scale using a only
        cell_lengths = atoms.cell.lengths()
        if a_csl is None:
            a_csl = a_real   # assume already at real a
        if c_csl is None:
            c_csl = c_real   # assume already at real c

    # Scale factors
    s_a = a_real / a_csl
    s_c = c_real / c_csl

    # Build the deformation gradient in Cartesian
    # HCP: x,y scale by s_a, z scales by s_c
    F = np.diag([s_a, s_a, s_c])

    out.set_cell(F @ atoms.cell[:], scale_atoms=False)
    out.positions = atoms.positions @ F.T

    # Store strain info
    out.info["rescaled"] = True
    out.info["strain_a"] = s_a - 1.0
    out.info["strain_c"] = s_c - 1.0

    return out


def build_gb_rescaled(
    csl_record: dict,
    element: str,
    a_real: float,
    c_real: float,
    ca_csl: float | None = None,
    **build_kwargs,
) -> Atoms:
    """
    Build a GB at rational c/a and rescale to real lattice parameters.

    Convenience wrapper that combines ``build_gb`` + ``rescale_to_lattice``.

    Parameters
    ----------
    csl_record : dict
        CSL record (enumerated at rational c/a).
    element : str
        Chemical symbol.
    a_real, c_real : float
        Real lattice parameters (Angstrom).
    ca_csl : float, optional
        Rational c/a used for CSL enumeration.
        If None, uses sqrt(8/3) (ideal HCP).
    **build_kwargs
        Passed to ``build_gb`` (n_layers, gap, vacuum, etc.).

    Returns
    -------
    ase.Atoms
        Bicrystal at real lattice parameters.

    Examples
    --------
    >>> from math import sqrt
    >>> from hcp_gb_generator import find_csl, build_gb_rescaled
    >>> rec = find_csl(ca_ratio=sqrt(5/2), sigma=7,
    ...               axis_miller_bravais=[1,0,-1,0], sigma_max=10)[0]
    >>> gb = build_gb_rescaled(rec, "Ti", a_real=2.95, c_real=4.68,
    ...                        ca_csl=sqrt(5/2), n_layers=4)
    """
    if ca_csl is None:
        ca_csl = sqrt(8.0 / 3.0)

    a_csl = a_real
    c_csl = a_real * ca_csl

    gb = build_gb(csl_record, element=element, a=a_csl, c=c_csl,
                  **build_kwargs)
    return rescale_to_lattice(gb, a_real, c_real, a_csl, c_csl)


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
    interface_distance: float | None = None,
    vacuum: float = 0.0,
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
    interface_distance : float, optional
        Spacing between grain surfaces at the GB plane (Å).
        If None, defaults to 0.5 for twist (no shared plane) and
        0.0 for tilt (shared coincident plane).
    vacuum : float
        Vacuum above the bicrystal (Å).
        0 → fully periodic, 2 GBs per cell.
        >0 → single GB with free surfaces at top and bottom.

    Returns
    -------
    ase.Atoms
        Bicrystal with ``grain_id`` array and metadata in ``atoms.info``.

    Examples
    --------
    >>> from hcp_gb_generator import find_csl, build_gb
    >>> recs = find_csl(ca_ratio=1.587, sigma=7,
    ...                 axis_miller_bravais=[0,0,0,1], sigma_max=10)
    >>> atoms = build_gb(recs[0], element="Ti", a=2.95, c=4.68,
    ...                  n_layers=4)
    >>> atoms.info["sigma"]
    7
    """
    axis = csl_record.get("axis_miller", [0, 0, 1])

    # [0, 0, 1] in 3-axis = [0001] = c-axis → twist boundary
    is_twist = (axis[0] == 0 and axis[1] == 0 and axis[2] != 0)

    if is_twist:
        dist = interface_distance if interface_distance is not None else 0.5
        return build_twist_gb(
            csl_record, element, a, c,
            n_layers=n_layers, interface_distance=dist, vacuum=vacuum,
        )
    else:
        dist = interface_distance if interface_distance is not None else 0.0
        return build_tilt_gb(
            csl_record, element, a, c,
            gb_plane_3ax=gb_plane_3ax,
            n_layers=n_layers, interface_distance=dist, vacuum=vacuum,
        )
