# -*- coding: utf-8 -*-
# ============================================================================
#     Nonlinear-Statistical-Coupling Very Early Alpha v0.0.1.x
#     Python v3.6+
#     Created by Kevin R. Chen, Kenric P. Nelson, Daniel Svoboda,
#                John Clements, and William Thistleton
#     Licensed under Apache License v2
# ============================================================================

# Version
from .__version__ import __version__


if __name__ == "__main__":
    version_number = __version__
    print(f"Running NSC lib v{version_number}.")
else:
    version_number = __version__
    print(f"Importing NSC lib v{version_number}.")
