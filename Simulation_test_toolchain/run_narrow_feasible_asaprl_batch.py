from __future__ import annotations

import sys

from Simulation_test_toolchain.run_narrow_feasible_riskidm_batch import main


if __name__ == "__main__":
    if "--policy" not in sys.argv:
        sys.argv.extend(["--policy", "asaprl"])
    main()
