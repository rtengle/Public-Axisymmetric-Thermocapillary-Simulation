import numpy as np
import scipy as sp
from src.SurfaceSystem import Surface
from src.TemperatureSystem import Temperature
import pathlib
import matplotlib.pyplot as plt
import gc
from scipy.optimize import root
from scipy.interpolate import make_interp_spline
from copy import copy
from datetime import datetime

# The grid is separated into G covering polar axis and Z covering the radial axis. These are related as follows:
# Z = R/H   G = gamma
# This scheme will impact the derivatives so extra care needs to be taken.

class ThinFilm:
    def __init__(self, G: np.ndarray, Z: np.ndarray, D: np.ndarray, tinit: float, Hinit: callable, thetainit: callable, solver='krylov', opts={'fatol':1e-9}, curv=2, steadystate=False):
        """Constructor method for surface-temperature system. The physics are split between a Surface system and a Temperature system, each outputting their own residuals
        according to the governing equations. The solve method then minimizes the combined residuals over each time step until the final time is reached.

        Args:
            G (np.ndarray): Polar angle meshgrid array
            Z (np.ndarray): Relative thickness height meshgrid array
            D (np.ndarray): Underlying substrate geometry
            tinit (float): Initial time of simulation
            Hinit (function): Function returning initial height based on the polar angle
            thetainit (function): Function returning the initial temperature from G and Z
            solver (str, optional): Root finder algorithm. Defaults to 'krylov'.
            opts (dict, optional): Solver tolerance. Defaults to {'fatol':1e-9}.
            curv (int, optional): Underlying mean curvature of substrate. Defaults to 2.
            steadystate (bool, optional): If the solver is looking for a steady-state solution. Defaults to False.
        """

        self.H = Surface(G, D, Hinit, solver=solver, opts=opts, curv=curv, steadystate=steadystate)
        self.theta = Temperature(G, Z, D, thetainit, solver=solver, opts=opts, curv=curv, steadystate=steadystate)
        self.solver = solver
        self.opts = opts
        self.tinit = tinit

    def setparams(self, S: float, M: float, eps: float, dtintervals: list, settings:dict=None):
        """Sets important parameters of domain system

        Args:
            S (float): Scaled inverse capillary number
            M (float): Scaled Marangoni number
            eps (float): Transformed thin film parameter (hpin / rc)
            dtintervals (list): List of time stepping intervals. The format is [(dt, end of interval), ...].
            settings (dict, optional): Dictionary of dimensional parameters of the system. Defaults to None.
        """

        # Sets the parameters S and M for the system. These dictate the capillary pressure and mass transfer respectively
        self.H.setparams(S, M)
        self.theta.setparams(S, M, eps)
        self.dtintervals = dtintervals.copy()

        self.recdt = dtintervals.copy()
        self.settings=settings

    def setbc(self, F:callable, Q:callable, Ftype=1, Qtype=1):
        """Sets the temperature boundary conditions of the system

        Args:
            F (function): Surface boundary condition in terms of (t, gamma, surface theta).
            Q (function): Solid-liquid interface boundary condition in terms of (t, gamma, base theta).
            Ftype (int, optional): Boundary condition type of free surface. 0 is Neumann, 1 is Dirichlet or Mixed. Defaults to 1.
            Qtype (int, optional): Boundary condition type of free surface. 0 is Neumann, 1 is Dirichlet or Mixed. Defaults to 1.
        """
        # Sets the boundary conditions for the surface and temperature systems.
        self.H.setpin()
        self.theta.setbc(F, Q, Ftype, Qtype)

    def setrec(self, recintervals:list, recstring:str, recsize:float=3.12, timelimit:float=1, relpath:bool=True, initset:int=0):
        """Sets the recording intervals for the thin-film system and other information about recording.

        Args:
            recintervals (list): List of recording intervals. Format is [(dt, end interval time), ...]
            recstring (str): File path for recorded datasets.
            recsize (float, optional): Approximate max memory to allocate for recording in GB. Defaults to 3.12 GB.
            timelimit (float, optional): Time between exporting datasets in hours. Defaults to 1 hour.
            relpath (bool, optional): Sets if the path is relative. Defaults to True.
            initset (int, optional): Initial dataset number when exporting. Defaults to 0.
        """
        # Sets which points should be recorded in the simulation
        recpoints = np.zeros(0)
        self.recstring = recstring
        self.relpath = relpath
        self.initset = initset
        self.rectlimit = timelimit

        # Starts at the initial condition.
        min = 0

        # For loop over all intervals. This generates a list of all points to be recorded
        for max, spacing in recintervals:
            # This defines which points should be recorded based on the spacing provided
            pointspace = np.arange(min, max, spacing)
            # Appends the sequence to the recorded points
            recpoints = np.append(recpoints, pointspace)
            # Updates the minimum
            min = max
        
        # Defines it as a list and removes all duplicates generated by double counting
        totalrecpoints = np.unique(recpoints)
        self.recpoints = list(totalrecpoints[totalrecpoints >= self.tinit])

        # Defines the maximum size the recording arrays can be
        thetashape = self.theta.G.shape
        nG, nZ = thetashape
        bytes_per_rec = 8 * (3 * nG * nZ + 3 * nG)
        self.recmax = np.min([int(recsize/(bytes_per_rec * 1e-9)), recpoints.size])

        # Sets up the arrays used for recording the data and initializes the first recorded point as the IC

        # Free surface
        self.Hrec = np.zeros([self.H.nG, self.recmax])
        self.Hrec[:,0] = self.H.Hprev
        
        # Temperature and velocities
        thetashape = self.theta.G.shape

        self.thetarec = np.zeros(np.append(thetashape, self.recmax))
        self.Urrec = np.zeros(np.append(thetashape, self.recmax))
        self.Ugrec = np.zeros(np.append(thetashape, self.recmax))


        self.thetarec[:,:,0] = self.theta.thetaprev
        self.Urrec[:,:,0] = self.theta.Ur
        self.Ugrec[:,:,0] = self.theta.Ug

        # Heating and cooling arrays
        self.Frec = np.zeros([self.H.nG, self.recmax])
        self.Qrec = np.zeros([self.H.nG, self.recmax])
        
        self.Frec[:,0] = self.theta.bbc(0, self.theta.g, self.theta.thetaprev[:,0])
        self.Qrec[:,0] = self.theta.tbc(0, self.theta.g, self.theta.thetaprev[:,-1])

        # Recorded times. Note that the actual recorded times might vary so we're defining it as an array
        self.trec = np.zeros(self.recmax)
        self.trec[0] = self.tinit

    def solve(self, printspace=50):
        """Solver loop for thin film system. We create a loop that iterates through each time step.
        The system is solved at each step concurrently between the surface and temperature systems. This is done
        by combining the residual arrays and trying to find the zero across both. This enables significantly faster
        convergence than by solving them separately. 

        Args:
            printspace (int, optional): Number of steps before exporting current time. Defaults to 50.
        """
        
         # The overall structure is as follows:
        # - Initialize the time stepping and recording variables
        # - Enter a loop that iterates across each time step based on the interval. 
        # - Inside the loop:
        #   - Adjust the time step if necessary
        #   - Create an initial guess for the surface and temperature
        #   - Define a function that returns a combined residual for both systems
        #   - Find when the residual is equal to zero for both systems (done via Krylov solver)
        #   - Save the result to the recording arrays if applicable
        #   - Repeat

        # These are the current time, step, and dataset respectively
        t = self.tinit
        n = 0
        r = self.initset

        # Defines the final time and initial time step
        tfinal = self.dtintervals[-1][0]
        tmax, dt = self.dtintervals.pop(0)
        while tmax <= t:
            tmax, dt = self.dtintervals.pop(0)
        self.H.setdt(dt)
        self.theta.setdt(dt)

        # This just removes the zero point in the recording so the starting value isn't overwritten
        trec = self.recpoints.pop(0)
        trec = self.recpoints.pop(0)
        self.recn = 1

        datetimeprev = datetime.now()

        # Main solver loop. I might switch this to while True, but it shouldn't matter and it acts as a fail safe
        while t <= tfinal:
            # Increments the step by one
            n += 1

            # Updates time interval if the next step will go into the next range. 
            # Important for if you have events happening that could be lost if the time step is too large
            if t + dt >= tmax:
                # Checks to see if there is a next time interval. If not, exit the solver loop
                if not self.dtintervals:
                    self.export(self.recstring + f' - Dataset {r} ' + str(datetime.now()).replace(':','-'))
                    gc.collect()
                    return
                # Updates the time interval
                tmax, dt = self.dtintervals.pop(0)
                # Sets the time step in the systems
                self.H.setdt(dt)
                self.theta.setdt(dt)

            # Increments the time
            t += dt

            # Sets the initial guess for the surface and temperature
            self.H.initforward(t, self.theta.thetaprev[:,0])
            self.theta.initforward(self.H.Hnext, t)
            initguess = np.append(self.H.Hnext, self.theta.thetanext.reshape(self.theta.nG, self.theta.nZ))

            # Defines a function that returns the residual of both functions
            def reseps(guess):
                # Separates the guess into surface and temperature
                Hnext = guess[:self.theta.nG]
                thetanext = guess[self.theta.nG:].reshape(self.theta.nG, self.theta.nZ)

                # Calculates the residual from both
                resT = self.theta.residual(Hnext, thetanext, 1).reshape(-1)
                resH = self.H.residual(Hnext, thetanext[:,0], t)

                # Returns the combined residual
                return np.append(resH,resT)
            
            # Solved for the next step
            try:
                result = root(reseps, initguess, method=self.solver, options=self.opts)
            except ValueError as err:
                print('Value error detected:')
                print(err)
                print('Exporting early')
                trec = t
                self.recpoints = None

            # Prints if the solver failed and ends the simulation
            if not result.success:
                print('Time step failed to converge. Recording results and exiting sim.')
                trec = t
                self.recpoints = None
            
            if np.abs(self.H.Hnext - 1).max() >= 0.2:
                print('Sim is beginning to destabilize. Recording results and exiting sim.')
                trec = t
                self.recpoints = None

            # Converts the result to the next surface and temperature step
            self.H.Hnext = result.x[:self.theta.nG]
            self.theta.thetanext = result.x[self.theta.nG:].reshape(self.theta.nG, self.theta.nZ)

            # Prints the time every 50 iterations
            if np.mod(n, printspace) == 0:
                print(f'{t:.5f} / {tfinal:.5f}    iter:{result.nit}')
            
            # If statement for whether to record the current point
            if t >= trec:
                # Records the surface and temperature
                self.Hrec[:, self.recn] = self.H.Hnext
                self.thetarec[:,:,self.recn] = self.theta.thetanext

                # Records the velocities
                self.Urrec[:,:,self.recn] = self.theta.Ur
                self.Ugrec[:,:,self.recn] = self.theta.Ug

                # Records the fluxes
                self.Frec[:,self.recn] = self.theta.F(self.theta.thetanext[:,0])
                self.Qrec[:,self.recn] = self.theta.Q(self.theta.thetanext[:,-1])

                # Records the time
                self.trec[self.recn] = t

                self.recn += 1

                # Updates the next recording time. If another one doesn't exist, export and exit the loop
                if not self.recpoints:
                    self.export(self.recstring + f' - Dataset {r} ' + str(datetime.now()).replace(':','-'))
                    gc.collect()
                    return
                # Increments the recording index. 
                # If the array is filled or enough time has passed, export and reset the index.
                # Currently set to have a 15 minute difference
                elif self.recn == self.recmax or (datetime.now() - datetimeprev).seconds/3600 >= self.rectlimit:
                    print(f'Exporting Dataset {r}')
                    self.export(self.recstring + f' - Dataset {r} ' + str(datetime.now()).replace(':','-'))
                    datetimeprev = datetime.now()
                    r += 1
                    self.recn = 0
                    gc.collect()

                trec = self.recpoints.pop(0)
    
    def export(self, filename: str):
        """Exports data so far recorded in .npz format.

        Args:
            filename (str): File name for exported data
        """
        # Exports all the data recorded from the sim

        # Defines the filepath
        if self.relpath:
            path = pathlib.Path().resolve().__str__()+'/'+filename
        else:
            path = filename
        # Exporting early for some reason, it only exports up to the final recorded point
        ifin = self.recn

        savedict = {
            'G':self.theta.G, 
            'Z':self.theta.Z, 
            'H':self.Hrec[:,:ifin], 
            'theta':self.thetarec[:,:,:ifin], 
            'Ur':self.Urrec[:,:,:ifin], 
            'Ug':self.Ugrec[:,:,:ifin],
            't':self.trec[:ifin],
            'n':self.recpoints,
            'F':self.Frec[:,:ifin],
            'Q':self.Qrec[:,:ifin],
            'pin':self.H.bcval[-1],
            'Ftype':self.theta.btype,
            'Qtype':self.theta.ttype,
            'S':self.H.S,
            'M':self.H.M,
            'dt':self.recdt,
            'D':self.H.D
        }

        if self.settings is not None:
            paramdict = {
                'h0': self.settings['h0'],
                'd': self.settings['d'],
                'T0':  self.settings['T0'],
                'dT': self.settings['dT'],
                'qon': self.settings['qon'],
                'Tbg': self.settings['Tbg'],
                'vareps': self.settings['vareps'],
                'a': self.settings['a'],

                'mu': self.settings['mu'],
                'rho': self.settings['rho'],
                'beta': self.settings['beta'],
                's0': self.settings['s0'],
                'k': self.settings['k'],
                'alpha': self.settings['alpha'],
                'r0':self.settings['r0']
            }
            np.savez(path, **(paramdict | savedict))
        else:
            np.savez(path, **savedict)