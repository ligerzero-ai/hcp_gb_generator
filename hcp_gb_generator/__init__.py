"""
HCP Coincidence Site Lattice (CSL) Grain Boundary Enumerator.

Enumerate and query CSL grain boundaries for hexagonal close-packed crystals
based on the theory of Warrington & Bufalini (1971) and Warrington (1975).
"""

__version__ = "0.1.0"

from hcp_gb_generator._core import (
    crystal_basis_matrix,
    enumerate_0001_csl,
    enumerate_hcp_csl,
    enumerate_tilt_csl,
    find_csl,
    hex_disorientation,
    mb_str,
    miller_bravais_to_3axis,
    print_csl_table,
    rotation_axis_angle,
    three_axis_to_miller_bravais,
    to_dataframe,
)
from hcp_gb_generator._builder import (
    build_gb,
    build_tilt_gb,
    build_twist_gb,
    csl_slab_directions,
    csl_supercell_matrix,
)

__all__ = [
    "crystal_basis_matrix",
    "enumerate_0001_csl",
    "enumerate_hcp_csl",
    "enumerate_tilt_csl",
    "find_csl",
    "hex_disorientation",
    "mb_str",
    "miller_bravais_to_3axis",
    "print_csl_table",
    "rotation_axis_angle",
    "three_axis_to_miller_bravais",
    "to_dataframe",
    # Builder
    "build_gb",
    "build_tilt_gb",
    "build_twist_gb",
    "csl_slab_directions",
    "csl_supercell_matrix",
]
