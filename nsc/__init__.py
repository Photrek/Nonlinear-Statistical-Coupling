# -*- coding: utf-8 -*-
# ============================================================================
#     Nonlinear-Statistical-Coupling Very Early Alpha v0.0.1.x
#     Python v3.6+
#     Created by Kevin R. Chen, Kenric P. Nelson, Daniel Svoboda,
#                John Clements, and William Thistleton
#     Licensed under Apache License v2
# ============================================================================

# Libs
from .util.function import coupled_logarithm as log, \
                           coupled_exponential as exp, \
                           coupled_entropy as entropy, \
                           coupled_cross_entropy as cross_entropy, \
                           coupled_divergence as kl_divergence
# Version
from .__version__ import __version__


if __name__ == "__main__":
    version_number = __version__
    print(f"Running NSC lib v{version_number}.")
else:
    version_number = __version__
    print(f"Importing NSC lib v{version_number}.")
