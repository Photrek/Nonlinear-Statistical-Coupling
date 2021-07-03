# -*- coding: utf-8 -*-
# ============================================================================
#     Nonlinear-Statistical-Coupling Early Alpha v0.0.3.x
#     Python v3.7+
#     Created by Kevin R. Chen, Kenric P. Nelson, Daniel Svoboda,
#                John Clements, and William Thistleton
#     Licensed under Apache License v2
# ============================================================================

# Lib
from .math.function import coupled_logarithm as log, \
                           coupled_exponential as exp
from .math.entropy import coupled_probability as prob, \
                          coupled_entropy as entropy, \
                          coupled_cross_entropy as cross_entropy, \
                          coupled_kl_divergence as kl_divergence
from .math.entropy_misc import shannon_entropy, tsallis_entropy

# Version
from .__version__ import __version__


if __name__ == "__main__":
    version_number = __version__
    print(f"Running NSC lib v{version_number}.")
else:
    version_number = __version__
    print(f"Importing NSC lib v{version_number}.")
