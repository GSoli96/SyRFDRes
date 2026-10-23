#!/usr/bin/env python3
"""
plot_aggregated_syrfd.py (Root Launcher)
=============================================================================
Direct executable launcher for utility_src/plot_aggregated_syrfd.py.
Run from repository root:
    python3 plot_aggregated_syrfd.py [--metric f1] [--include-baseline]
=============================================================================
"""

import sys
from pathlib import Path

# Add utility/plotting and utility/src to sys.path
root_dir = Path(__file__).resolve().parent.parent
utility_plotting_dir = root_dir / "utility" / "plotting"
utility_src_dir = root_dir / "utility" / "src"
for _p in [utility_plotting_dir, utility_src_dir]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from plot_aggregated_syrfd import main

if __name__ == "__main__":
    sys.exit(main())
