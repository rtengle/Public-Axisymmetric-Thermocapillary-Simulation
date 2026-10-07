from src.ThinFilmSystem import *
from src.ParallelProcessing import *
from src.Units import *
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.special import j0, jn_zeros
from pint import UnitRegistry
from src.LegendreModes import *
import gc
import time
import warnings
import os

# This is the script used to perform a sweep of modes 1, 2, 3, and 5. 

# Defines the unit repository
ureg = UnitRegistry()
Q_ = ureg.Quantity # type: ignore

# Number of tests to run concurrently
tests = 4

# System parameters
udict = {
    'h0': Q_(6, 'millimeters'),
    'd': Q_(6.6, 'meters'),
    'T0':  Q_(120, 'kelvin'),
    'dT': Q_(5, 'kelvin'),
    'qon': Q_(400, 'watts/meter^2'),
    'gMax': Q_(np.pi/12, 'radians'),
    'Tbg': Q_(4, 'kelvin'),
    'vareps': Q_(1, 'dimensionless'),
    'a': Q_(1, 'dimensionless'),

    'mu': Q_(100, 'centipoise'),
    'rho': Q_(1000, 'kg/m^3'),
    'beta': Q_(0.1, 'mN / (m K)'),
    's0': Q_(50, 'mN/m'),
    'k': Q_(0.16, 'W/(m K)'),
    'alpha': Q_(0.1, "mm^2/s"),
}

# Gets the dimensionless parameters from the system parameters
usys = UnitSystem(udict, ureg)
params = usys.get_params()

# Gets the polar spacing from the mode number
def getnG(settings:dict):
    return int(settings['mode']*5 + 12)

# Defines vertical spacing and number of cycles to run
params['nZ'] = 12
params['Ncycles'] = 100

# Which modes we want to test
sweepparams = {
    'mode': [1, 2, 3, 5]
}

# Defines the export string path of the mirror
def exportstring(settings:dict):
    dstring = f'results\\6.6 m Mirror\\6mm 400W 100 cP Viscosity\\'
    # dstring = f'results\\{settings['d'][0]} m Mirror\\5mm 50cP\\Multiswing\\Reverse {int(nsmall)}x{int(dT1)}K 1x{int(dT2)}K\\Mode {settings['mode']}\\'
    if not os.path.isdir(dstring):
        os.makedirs(dstring)
    return dstring + f'Mode {settings['mode']}'

params['exportstring'] = exportstring

# Gets the initial state of the mirror as a 1% deformation of the tested mode
def getHinit(settings):
    nG = settings['nG']
    gMax = settings['gMax']
    mode = settings['mode']
    l, P = getMode(nG, gMax, mode)
    settings['P'] = P
    def Hinit(g):
        return 1 - 1e-2*P
    settings['l'] = l
    return Hinit

# Returns no substrate geometry
def getD(settings):
    g = np.linspace(0, settings['gMax'], settings['nG'])
    with warnings.catch_warnings(action="ignore"):
        D = 0*settings['P']
    return D

# Defines a constant initial temperature
def getthetainit(settings):
    def thetainit(G, Z):
        return 0*G 
    return thetainit

# Defines the function that obtains the outgoing surface heat flux
def getF(settings):
    # System parameters
    xi = settings['xi']
    M = settings['M']
    ups = settings['ups']
    gMax = settings['gMax']
    a = settings['a']
    F0 = (1+np.cos(gMax))/2
    def F(t, g, theta):
        # Background Radiation
        radfactor = (1 + xi*theta)**4
        outrad = radfactor
        # Self Radiation
        inrad = 1/2*np.trapezoid(radfactor*np.sin(g), g)
        # Return total radiation (out - self - background)
        return outrad - a*inrad - a*F0*ups
    return F

# Defines the function that obtains the incoming heating flux
def getQ(settings):
    th = settings['th']
    # This is the ramp time, mostly used to help with numerical stability
    tr = th/10
    tc = settings['tc']
    tch = settings['tch']
    Q0 = settings['Q0']
    nG = settings['nG']
    zeros = np.zeros(nG)
    # Q0 spread across the surface
    Qspread = Q0 * np.ones(nG)
    def Q(t, g, theta):
        # Obtains the modulated time
        tmod = np.mod(t, tch)
        # Ramp up period
        if tmod < tr:
            return np.interp(tmod, [0,tr], [0,1]) * Qspread
        # Sustained heating
        elif tmod < th:
            return Qspread
        # Ramp down period
        elif tmod < th + tr:
            return  np.interp(tmod, [th,th+tr], [1,0]) * Qspread
        # Cooling period
        else:
            return zeros
    return Q

# Obtains the time intervals that we will be using
def getdtintervals(settings):
    th = settings['th']
    tc = settings['tc']
    tch = settings['tch']
    # Time where cooling stepping stops
    ct = tc - th/10
    # Total time for the larger time stepping
    cspan = tc - 2*th/10
    # Time where heating stepping stops
    ht = tch + tch/10
    # Total time for shorter time stepping
    hspan = th + 2*th/10

    dtintervals = []
    # Assembles all the intervals
    for i in range(settings['Ncycles']):
        dtintervals = dtintervals + [
                (tch*i + ht, hspan/3000),
                (tch*i + ht + ct, cspan/3000)
            ]

    return dtintervals

# Gets the recording intervals for 500 points per cycle.
def getrecintervals(settings):
    return [
        ( 2*settings['tch']*settings['Ncycles'], settings['tch']/500)
    ]

# Defines initial conditions
params['tinit'] = 0
params['initset'] = 0

# Defines all the derived params used. 
derivedparams = {
    'nG': getnG,
    'Hinit': getHinit,
    'thetainit': getthetainit,
    'F': getF,
    'Q': getQ,
    'D': getD,
    'dtintervals':getdtintervals,
    'recintervals':getrecintervals
}

# Starts the sweep using the TestSeries class
if __name__ == '__main__':
    series = TestSeries(params)
    series.set_sweep(sweepparams)
    series.set_derived(derivedparams)
    series.run_process(tests)