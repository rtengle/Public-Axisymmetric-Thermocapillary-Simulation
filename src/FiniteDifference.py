import numpy as np
from scipy.special import factorial

def getFDC(Nmax:int, center:int):
    """Automatically calculates the finite difference coefficients in the form of a matrix. The coefficients for the nth derivatives are in the nth row.

    Args:
        Nmax (int): Sample size. Must be odd.
        center (int): Distance from center of sample. (+) is to the right, (-) is to the left.

    Returns:
        _type_: _description_
    """
    # Total span away from the center
    dN = int((Nmax - 1)/2)
    # Define a matrix corresponding to the Taylor series for each point x - n around x:
    TMatrix = np.array([
        [
            (n-center)**k / factorial(k)
            for k in range(0, Nmax)
        ] for n in range(-dN, dN+1)
    ])
    # Invert the matrix to obtain the finite difference coefficients
    return np.linalg.inv(TMatrix)

pass