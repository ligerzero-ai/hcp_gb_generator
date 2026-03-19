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
from hcp_gb_generator import enumerate_hcp_csl, find_csl, print_csl_table

# All CSL GBs for ideal HCP (c/a = sqrt(8/3)) up to Sigma 30
results = enumerate_hcp_csl(ca_ratio=1.633, sigma_max=30)
print_csl_table(results)

# Filter by Sigma value
find_csl(results, sigma=7)

# Filter by rotation axis (4-index Miller-Bravais)
find_csl(results, axis_miller_bravais=[0, 0, 0, 1], ca_ratio=1.633)

# Filter by angle range
find_csl(results, angle_min=20, angle_max=30)
```

### Build grain boundary structures

```python
from hcp_gb_generator import find_csl, build_gb

# Find Sigma=7 [0001] twist boundary
recs = find_csl(ca_ratio=1.587, sigma=7,
                axis_miller_bravais=[0, 0, 0, 1], sigma_max=10)

# Build an ASE Atoms bicrystal
atoms = build_gb(recs[0], element="Ti", a=2.95, c=4.68,
                 n_layers=4, gap=0.5)

# Access grain boundary metadata
atoms.info["sigma"]         # 7
atoms.info["angle_deg"]     # 38.21
atoms.arrays["grain_id"]    # [1, 1, ..., 2, 2, ...]
```

### Command-line interface

```bash
# List CSL GBs for ideal HCP
hcp-gb

# Specify c/a ratio and max sigma
hcp-gb 1.587 50
```

## Rotation axis types

The package finds CSL boundaries for three categories of rotation axes:

| Axis family | Paper notation | 4-index | c/a dependent? |
|-------------|---------------|---------|----------------|
| Basal       | `001`         | [0001]  | No             |
| Prism       | `210`         | [10-10] | Yes            |
| a-direction | `100`         | [11-20] | Yes            |
| Mixed       | e.g. `801`    | various | Yes            |

## Verification

All results are verified against Warrington 1975 Table I
(31 published CSL disorientations for c/a = sqrt(8/3)).

```bash
# Run the test suite
pytest

# Skip slow high-index axis tests
pytest -m "not slow"
```

## API reference

### Enumeration

- `enumerate_hcp_csl(ca_ratio, sigma_max)` - full enumeration
- `enumerate_0001_csl(sigma_max)` - [0001]-axis only (fast, c/a independent)
- `enumerate_tilt_csl(ca, sigma_max)` - tilt axes only (c/a dependent)
- `find_csl(results, sigma=, axis_miller_bravais=, ...)` - query/filter

### Structure construction

- `build_gb(csl_record, element, a, c, ...)` - auto-dispatch twist/tilt
- `build_twist_gb(csl_record, ...)` - [0001] twist boundaries
- `build_tilt_gb(csl_record, ...)` - symmetric tilt boundaries
- `csl_slab_directions(csl_record, ca)` - compute 4-index Miller-Bravais slab directions

### Display

- `print_csl_table(results)` - formatted terminal table
- `to_dataframe(results)` - pandas DataFrame

## License

MIT
