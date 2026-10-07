import numpy as np
from scipy.sparse.linalg import eigs
from scipy.special import j0, jn_zeros
from scipy.optimize import root
import warnings
import matplotlib.pyplot as plt

def LegendreP(l:float, g:np.ndarray):
    """Calculates the Legendre function w/ eigenvalue l across g.

    Args:
        l (float): Eigenvalue
        g (np.ndarray): Polar axis points

    Returns:
       np.ndarray: Legendre function values
    """
    t = np.linspace(0, np.pi, 200)
    (Z, G) = np.meshgrid(np.zeros(200),g)
    P = np.trapezoid(np.power(np.cos(G) + 1j*np.sin(G)*np.cos(t),l), t)/np.pi
    return np.real(P)

def getMode(nG:int, gMax:float, mode:int):
    """Generates a set Legendre mode for a pinned condition.

    Args:
        nG (int): Number of polar points to export
        gMax (float): Outer polar angle
        mode (int): Number mode to evaluate. Lowest mode is 1.

    Returns:
        _type_: _description_
    """
    # Initializes the polar angles
    g = np.linspace(0, gMax, nG)
    # At low polar spans, the Legendre function is very close to a Bessel function. Our initial guess is the nth zero of the Bessel function.
    klist = jn_zeros(0,mode)[-1]/gMax
    # Converts to the equivalent Legendre eigenvalue guess
    linit = 1/2*(-1 + np.sqrt(1 + 4 * klist**2))
    # Calculates the nth eigenvalue of the Legendre function using root finding.
    result = root(lambda l: LegendreP(l, gMax)[0], linit)
    # Saves eigenvalue and calculates the Legendre function for that value
    l = result.x[0]
    P = LegendreP(l, g)
    
    return l, P

if __name__ == '__main__':
    l, P = getMode(80, np.pi/12, [1, 10, 20, 50])
    for p in P:
        plt.plot(p)
    plt.show()