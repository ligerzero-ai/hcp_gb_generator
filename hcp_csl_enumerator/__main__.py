"""CLI entry point: python -m hcp_csl_enumerator [c/a] [sigma_max]"""

import sys
import time
from math import sqrt

from hcp_csl_enumerator._core import (
    enumerate_0001_csl,
    enumerate_tilt_csl,
    print_csl_table,
)


def main():
    ca = float(sys.argv[1]) if len(sys.argv) > 1 else sqrt(8 / 3)
    smax = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    print(f"Enumerating HCP CSL GBs: c/a = {ca:.4f}, Sigma <= {smax}")
    print()

    t0 = time.perf_counter()
    res0001 = enumerate_0001_csl(smax, ca)
    t1 = time.perf_counter()
    print(f"[0001]-axis GBs ({len(res0001)} found, {t1 - t0:.3f}s):")
    print_csl_table(res0001)
    print()

    t2 = time.perf_counter()
    res_tilt = enumerate_tilt_csl(ca, min(smax, 25), max_idx=3)
    t3 = time.perf_counter()
    print(f"Tilt GBs ({len(res_tilt)} found, {t3 - t2:.1f}s):")
    print_csl_table(res_tilt)


if __name__ == "__main__":
    main()
