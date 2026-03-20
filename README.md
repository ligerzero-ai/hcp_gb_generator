# hcp-gb-generator

Enumerate coincidence site lattice (CSL) grain boundaries for hexagonal
close-packed crystals and construct ASE bicrystal structures.

Based on the theory of:
- D.H. Warrington & P. Bufalini, *Scripta Met.* **5** (1971) 771
- D.H. Warrington, *J. Physique Colloque C4* **36** (1975) C4-87

## Installation

```bash
pip install hcp-gb-generator
```

Or for development:

```bash
git clone https://github.com/ligerzero-ai/hcp_gb_generator.git
cd hcp_gb_generator
pip install -e ".[test]"
```

## Quick start

### Enumerate available CSL grain boundaries

```python
from math import sqrt
from hcp_gb_generator import enumerate_hcp_csl, find_csl, print_csl_table

# All CSL GBs for ideal HCP up to Sigma 30
results = enumerate_hcp_csl(ca_ratio=sqrt(8/3), sigma_max=30)
print_csl_table(results)

# Filter by Sigma value
find_csl(results, sigma=7)

# Filter by rotation axis (4-index Miller-Bravais)
find_csl(results, axis_miller_bravais=[0, 0, 0, 1], ca_ratio=sqrt(8/3))

# Filter by disorientation angle range
find_csl(results, angle_min=20, angle_max=30)
```

### Build grain boundary structures

```python
from hcp_gb_generator import find_csl, build_gb

# Find Sigma=7 [0001] twist boundary
rec = find_csl(ca_ratio=1.587, sigma=7,
               axis_miller_bravais=[0, 0, 0, 1], sigma_max=10)[0]

# Build an ASE Atoms bicrystal
atoms = build_gb(rec, element="Ti", a=2.95, c=4.68, n_layers=4)

# Access grain boundary metadata
atoms.info["sigma"]                # 7
atoms.info["disorientation_angle"] # 21.79
atoms.arrays["grain_id"]           # [1, 1, ..., 2, 2, ...]
```

### Non-ideal c/a: enumerate at rational, build at real

Exact tilt CSLs require rational (c/a)^2.  For real metals, use the
nearest rational approximation for enumeration, then rescale:

```python
from math import sqrt
from hcp_gb_generator import find_csl, build_gb_rescaled

# Ti: use (c/a)^2 = 5/2 for enumeration
ca_csl = sqrt(5/2)

rec = find_csl(ca_ratio=ca_csl, sigma=7,
               axis_miller_bravais=[1, 0, -1, 0], sigma_max=10)[0]

# Build at rational c/a, then rescale to real Ti lattice
gb = build_gb_rescaled(rec, "Ti",
                       a_real=2.95, c_real=4.68,
                       ca_csl=ca_csl, n_layers=4)
# gb.info['strain_c'] → 0.34%  (tiny misfit from approximation)
```

### Single GB vs periodic cell

```python
# Fully periodic: 2 GBs per cell (one at interface, one at periodic image)
gb_periodic = build_gb(rec, "Ti", a=2.95, c=4.68, n_layers=6)
# gb_periodic.pbc → [True, True, True]

# Single GB with vacuum: free surfaces at top and bottom
gb_single = build_gb(rec, "Ti", a=2.95, c=4.68, n_layers=6, vacuum=12.0)
# gb_single.pbc → [True, True, False]
```

### Command-line interface

```bash
# List CSL GBs for ideal HCP
hcp-gb

# Specify c/a ratio and max sigma
hcp-gb 1.587 50
```

## Rational c/a approximations for common HCP metals

Exact tilt CSLs require rational (c/a)^2.  The three families from
Warrington (1975) cover all common HCP metals with < 1.2% strain:

| Family | (c/a)^2 | c/a | Metals |
|--------|---------|-----|--------|
| **5/2** | 2.500 | 1.5811 | Ti, Zr, Hf, Be, Ru, Os, Y, Sc |
| **8/3** | 2.667 | 1.6330 | Mg, Co, Re |
| **7/2** | 3.500 | 1.8708 | Zn, Cd |

Detailed strain values:

| Metal | c/a (exp) | (c/a)^2 | Best p/q | Strain (%) |
|-------|-----------|---------|----------|------------|
| **Ru** | 1.582 | 2.504 | 5/2 | -0.07 |
| **Hf** | 1.580 | 2.497 | 5/2 | +0.06 |
| **Os** | 1.579 | 2.493 | 5/2 | +0.14 |
| **Ti** | 1.588 | 2.521 | 5/2 | -0.42 |
| **Mg** | 1.624 | 2.636 | 8/3 | +0.58 |
| **Co** | 1.623 | 2.635 | 8/3 | +0.60 |
| **Y**  | 1.571 | 2.469 | 5/2 | +0.64 |
| **Zr** | 1.593 | 2.536 | 5/2 | -0.71 |
| **Zn** | 1.856 | 3.446 | 7/2 | +0.78 |
| **Cd** | 1.886 | 3.556 | 7/2 | -0.79 |
| **Sc** | 1.594 | 2.540 | 5/2 | -0.78 |
| **Be** | 1.568 | 2.459 | 5/2 | +0.83 |
| **Re** | 1.614 | 2.605 | 8/3 | +1.18 |

The [0001] twist GBs are exact for **any** c/a (the basal plane is
always a 2D triangular lattice).

## Rotation axis types

| Axis family | Paper notation | 4-index | c/a dependent? |
|-------------|---------------|---------|----------------|
| Basal       | `001`         | [0001]  | No             |
| Prism       | `210`         | [10-10] | Yes            |
| a-direction | `100`         | [11-20] | Yes            |
| Mixed       | e.g. `801`    | various | Yes            |

## Builder parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `interface_distance` | 0.0 (tilt), 0.5 (twist) | Spacing between grain surfaces at GB plane (A) |
| `vacuum` | 0.0 | Vacuum above slab (A). 0 = periodic (2 GBs/cell), >0 = single GB |
| `merge_boundary_layer` | True (tilt only) | Remove duplicate coincident plane at symmetric tilt GBs |
| `n_layers` | 6 | Layers per grain (thickness control) |

## Verification

All results are verified against Warrington 1975 Table I
(31 published CSL disorientations for c/a = sqrt(8/3)).

```bash
pytest                 # full test suite
pytest -m "not slow"   # skip high-index axis tests
```

## API reference

### Enumeration

- `enumerate_hcp_csl(ca_ratio, sigma_max)` — full enumeration
- `enumerate_0001_csl(sigma_max)` — [0001]-axis only (fast, c/a independent)
- `enumerate_tilt_csl(ca, sigma_max)` — tilt axes only (c/a dependent)
- `find_csl(results, sigma=, axis_miller_bravais=, ...)` — query/filter
- `hex_disorientation(R_cart, ca)` — compute disorientation of a rotation

### Structure construction

- `build_gb(csl_record, element, a, c, ...)` — auto-dispatch twist/tilt
- `build_gb_rescaled(csl_record, element, a_real, c_real, ca_csl)` — build + rescale to real lattice
- `build_twist_gb(csl_record, ...)` — [0001] twist boundaries
- `build_tilt_gb(csl_record, ...)` — symmetric tilt boundaries
- `rescale_to_lattice(atoms, a_real, c_real)` — rescale existing bicrystal
- `csl_slab_directions(csl_record, ca)` — compute 4-index slab directions

### Display

- `print_csl_table(results)` — formatted terminal table
- `to_dataframe(results)` — pandas DataFrame

## License

MIT
