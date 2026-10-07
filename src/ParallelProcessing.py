from multiprocessing import Pool
import itertools as it
import time
from copy import deepcopy

from src.ThinFilmSystem import *
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.special import j0
import gc
from glob import glob

def sim(test_vals):
    """Helper function that sets automatically sets up the simulation based on the default settings and current test values

    Args:
        test_vals (tuple): Tuple of dictionaries containing the settings for the test.
    """
    vals, default_settings, keyslist, derived_settings = test_vals
    print(f'Running Test - {vals}')
    settings = deepcopy(default_settings)

    # Assign the sweep settings
    for key, val in zip(keyslist, vals):
        settings[key] = val

    # Create the derived settings
    for key in derived_settings:
        settings[key] = derived_settings[key](settings)

    # Set up simulation

    # Max polar span
    gMax = settings['gMax']

    # Mesh resolution
    nG = settings['nG']
    nZ = settings['nZ']

    # Space array for polar direction
    g = np.linspace(0, gMax, nG)
    z = np.linspace(-1, 0, nZ)

    # Meshgrid for system. Note that it's indexed [g, z].
    G, Z = np.meshgrid(g, z, indexing='ij')

    # Substrate
    D = settings['D']

    # Key parameters
    S = settings['S']
    M = settings['M']
    eps = settings['eps']

    # Thermal boundary conditions
    F = settings['F']
    Q = settings['Q']

    # Initial Conditions
    thetainit = settings['thetainit']
    Hinit = settings['Hinit']
    tinit = settings['tinit']

    # Time steps and recording times
    dtintervals = settings['dtintervals']
    recintervals = settings['recintervals']

    # Set up thin film system
    film = ThinFilm(G, Z, D, tinit, Hinit, thetainit, opts={'fatol':1e-7})
    film.setparams(S, M, eps, dtintervals, settings)
    film.setbc(F, Q, 1)
    film.setrec(recintervals, settings['exportstring'](settings), initset=settings['initset'])
    
    # Run the simulation
    film.solve(5)


class TestSeries:
    def __init__(self, default:dict):
        """Constructor for the TestSeries class. Use to run a series of simulations that vary a set of parameters.

        Args:
            default (dict): Default set of system parameters
        """
        self.default_settings = default
        self.derived_settings = {}
    
    def set_sweep(self, sweep_dict:dict):
        """Set parameters to sweep through. 

        Args:
            sweep_dict (dict): Dictionary containing parameters to sweep through. Format is {'param': [v1, v2, ...], ...}
        """
        # Step 1: Get the list of variables to iterate over
        self.keyslist = list(sweep_dict.keys())

        # Step 2: Get the lists to iterate over
        self.valtuple = tuple([sweep_dict[k] for k in sweep_dict])

        # Step 3: Get all possible combinations of the dicts
        self.testparams = it.product(*self.valtuple)

    def set_derived(self, derived_dict:dict):
        """Set the parameters derived by those in the sweep

        Args:
            derived_dict (dict): Dictionary of functions that calculate new parameters from the sweep. Evaluated in order put into the dict. Format is {'param': function, ...}
        """
        self.derived_settings = derived_dict
    
    def run_process(self, processes:int):
        """Run a set of simulations

        Args:
            processes (int): Number of simulations to run concurrently
        """
        pool = Pool(processes)
        testparams = [(test, self.default_settings, self.keyslist, self.derived_settings) for test in it.product(*self.valtuple)]
        pool.map(sim, testparams)
        gc.collect()
            