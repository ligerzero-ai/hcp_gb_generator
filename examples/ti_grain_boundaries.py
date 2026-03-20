"""
Example: Generate Ti grain boundaries using rational c/a approximation.

Ti has c/a ≈ 1.588, so (c/a)^2 ≈ 2.521 — irrational.
Exact tilt CSLs require rational (c/a)^2.  The standard approximation
for Ti is (c/a)^2 = 5/2, i.e. c/a = sqrt(5/2) ≈ 1.5811 (0.42% strain).

Workflow:
  1. Enumerate CSL GBs at the rational c/a = sqrt(5/2)
  2. Build bicrystal structures at that rational c/a
  3. Rescale to real Ti lattice parameters (a=2.95, c=4.68 Å)
"""

from math import sqrt

from hcp_gb_generator import (
    enumerate_hcp_csl,
    find_csl,
    print_csl_table,
    build_gb,
    build_gb_rescaled,
)

# ---------------------------------------------------------------------------
# Ti lattice parameters
# ---------------------------------------------------------------------------
a_Ti = 2.95       # Angstrom
c_Ti = 4.68       # Angstrom
ca_real = c_Ti / a_Ti               # 1.5864
ca_csl = sqrt(5 / 2)               # 1.5811 — rational approximation
strain_pct = (ca_csl - ca_real) / ca_real * 100

print(f"Real Ti: c/a = {ca_real:.4f}")
print(f"CSL approx: c/a = sqrt(5/2) = {ca_csl:.4f}")
print(f"c/a strain: {strain_pct:+.2f}%")
print()

# ---------------------------------------------------------------------------
# 1. Enumerate all CSL GBs up to Sigma 50
# ---------------------------------------------------------------------------
all_results = enumerate_hcp_csl(
    ca_ratio=ca_csl,
    sigma_max=50,
    max_idx=5,
)

print(f"Found {len(all_results)} CSL grain boundaries for Ti")
print()

# [0001] axis (twist boundaries — c/a independent)
twist_gbs = find_csl(all_results, axis_miller_bravais=[0, 0, 0, 1],
                     ca_ratio=ca_csl)
print(f"[0001] twist GBs: {len(twist_gbs)}")
print_csl_table(twist_gbs)
print()

# [10-10] axis (prism tilt)
prism_gbs = find_csl(all_results, axis_miller_bravais=[1, 0, -1, 0],
                     ca_ratio=ca_csl)
print(f"[10-10] prism tilt GBs: {len(prism_gbs)}")
print_csl_table(prism_gbs)
print()

# [11-20] axis (a-direction tilt)
a_gbs = find_csl(all_results, axis_miller_bravais=[1, 1, -2, 0],
                 ca_ratio=ca_csl)
print(f"[11-20] a-direction tilt GBs: {len(a_gbs)}")
print_csl_table(a_gbs)
print()

# ---------------------------------------------------------------------------
# 2. Build specific grain boundaries
# ---------------------------------------------------------------------------

# --- Sigma=7 [0001] twist ---
rec_twist = find_csl(all_results, sigma=7,
                     axis_miller_bravais=[0, 0, 0, 1],
                     ca_ratio=ca_csl)[0]

gb_twist = build_gb_rescaled(
    rec_twist, element="Ti",
    a_real=a_Ti, c_real=c_Ti, ca_csl=ca_csl,
    n_layers=4,
)

print(f"Sigma=7 [0001] twist GB:")
print(f"  Atoms:         {len(gb_twist)}")
print(f"  Disorientation: {gb_twist.info['disorientation_angle']:.2f} deg")
print(f"  Cell:          {gb_twist.cell.lengths().round(3)}")
print(f"  c-axis strain: {gb_twist.info['strain_c']*100:.3f}%")
print()

# --- Sigma=7 [10-10] tilt ---
rec_tilt = find_csl(all_results, sigma=7,
                    axis_miller_bravais=[1, 0, -1, 0],
                    ca_ratio=ca_csl)[0]

gb_tilt = build_gb_rescaled(
    rec_tilt, element="Ti",
    a_real=a_Ti, c_real=c_Ti, ca_csl=ca_csl,
    n_layers=4,
)

print(f"Sigma=7 [10-10] tilt GB:")
print(f"  Atoms:         {len(gb_tilt)}")
print(f"  Disorientation: {gb_tilt.info['disorientation_angle']:.2f} deg")
print(f"  Cell:          {gb_tilt.cell.lengths().round(3)}")
print(f"  c-axis strain: {gb_tilt.info['strain_c']*100:.3f}%")
print()

# --- Single-GB cell with vacuum ---
gb_single = build_gb_rescaled(
    rec_twist, element="Ti",
    a_real=a_Ti, c_real=c_Ti, ca_csl=ca_csl,
    n_layers=4, vacuum=10.0,
)

print(f"Single-GB cell (with vacuum):")
print(f"  Atoms:  {len(gb_single)}")
print(f"  PBC:    {list(gb_single.pbc)}")
print(f"  Cell z: {gb_single.cell[2,2]:.2f} Å")
print()

# ---------------------------------------------------------------------------
# 3. Export (uncomment to write files)
# ---------------------------------------------------------------------------
# from ase.io import write
# write("Ti_S7_twist.vasp", gb_twist, format="vasp")
# write("Ti_S7_tilt.vasp", gb_tilt, format="vasp")
# write("Ti_S7_twist_single_gb.vasp", gb_single, format="vasp")
# print("Exported POSCAR files.")
