# Requires NumPy and
import numpy as np
from scipy.stats import rice, foldnorm


def wilson_dist_normalized_simple(E, centric=False, nargout=1):
    if centric:
        P_E = foldnorm.pdf(E, 0, 0, 1)
    else:
        P_E = rice.pdf(E, 0, 0, np.sqrt(0.5))
    return P_E


def wilson_dist_normalized(E, centric=False, nargout=1):
    """
    Wilson distribution for normalized structure factors, using either the mappings to the Rice and Folded-normal
    distributions (default) or using the equations from Rupp.

    Inputs:
    E           The normalized structure factors in a NumPy array
    centric     Boolean (scalar)
    nargout     Number of ourput arguments

    If nargout == 1, returns only the calculated value based on the Rice (acentric) or Folded-normal (centric)
    PDFs built into SciPy. Otherwise also returns the values based on Rupp's equations.
    """
    if centric:
        if nargout > 1:
            P_E = np.sqrt(2 / np.pi) * np.exp(-0.5 * E ** 2)  # Rupp eq. 7-111
        P_E2 = foldnorm.pdf(E, 0, 0, 1)
    else:
        if nargout > 1:
            P_E = 2 * E * np.exp(-(E ** 2))  # Rupp eq. 7-112
        P_E2 = rice.pdf(E, 0, 0, np.sqrt(0.5))
    if nargout == 1:
        return P_E2
    else:
        return P_E, P_E2


def dPE_dE_acentric(E):
    """
    First derivative of the acentric Wilson PDF for E wrt E
    """
    return 2 * np.exp(-(E ** 2)) * (1 - 2 * E ** 2)


def dPE_dE_centric(E):
    """
    First derivative of the centric Wilson PDF for E wrt E
    """
    return -np.sqrt(2 / np.pi) * np.exp(-0.5 * E ** 2) * E
