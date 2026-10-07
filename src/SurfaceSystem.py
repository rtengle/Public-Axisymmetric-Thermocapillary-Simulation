import numpy as np
from scipy.optimize import root
import warnings
import matplotlib.pyplot as plt
from src.FiniteDifference import getFDC

class Surface:
    def __init__(self, G:np.ndarray, D:np.ndarray, Hinit:np.ndarray, solver:str='krylov', opts:dict={'fatol':1e-8}, curv:float=2, steadystate:bool=False):
        """Temperature class that describes the temperature dynamics within the liquid mirror. Called using the ThinFilm class.

        Args:
            G (np.ndarray): Polar coordinate across mesh
            Z (np.ndarray): Vertical coordinate across mesh
            D (np.ndarray): Substrate geometry
            Hinit (function): Function returning the initial surface geometry
            solver (str, optional): Solver used to solve the independent surface system. Defaults to 'krylov'.
            opts (_type_, optional): Solver function used to solve the independent surface system. Defaults to {'fatol':1e-8}.
            curv (float, optional): Underlying substrate curvature. Defaults to 2.
            steadystate (bool, optional): Determines if the system is being solved for its steady state. Defaults to False.
        """

        # Defines the polar grid
        self.g = G[:,0]
        self.nG = self.g.size
        self.dG = self.g[1] - self.g[0]
        # Defines the surface matrix.
        self.Hprev = Hinit(self.g)
        self.Hnext = Hinit(self.g)
        self.solver = solver
        self.opts = opts
        self.curv = curv
        self.steadystate = steadystate

        # This is where I define my central finite difference matrices. The left and right will be preserved for calculating other things

        # Central Difference

        # This will use N = 9 points for each step of the central difference.
        # This means that 4 on each side will be occupied apart from the edges

        NMax = 9
        CMatrix = getFDC(9, 0)
        L1Matrix = getFDC(9, -1)
        L0Matrix = getFDC(9, -2)
        R1Matrix = getFDC(9, 1)
        R0Matrix = getFDC(9, 2)

        

        self.C0 = np.zeros((self.nG, self.nG))
        for n in range(2,self.nG-2):
            self.C0[n,n] = 1

        self.C1 = np.zeros((self.nG, self.nG))
        for n in range(4,self.nG-4):
            self.C1[n,(n-4):(n+5)] = CMatrix[1,:]
        self.C1[2,:9] = L0Matrix[1,:]
        self.C1[3,:9] = L1Matrix[1,:]
        self.C1[-4,-9:] = R1Matrix[1,:]
        self.C1[-3,-9:] = R0Matrix[1,:]

        self.C1 = self.C1/self.dG
        
        self.C2 = np.zeros((self.nG, self.nG))
        for n in range(4,self.nG-4):
            self.C2[n,(n-4):(n+5)] = CMatrix[2,:]
        self.C2[2,:9] = L0Matrix[2,:]
        self.C2[3,:9] = L1Matrix[2,:]
        self.C2[-4,-9:] = R1Matrix[2,:]
        self.C2[-3,-9:] = R0Matrix[2,:]

        self.C2 = self.C2/self.dG**2

        self.C3 = np.zeros((self.nG, self.nG))
        for n in range(4,self.nG-4):
            self.C3[n,(n-4):(n+5)] = CMatrix[3,:]
        self.C3[2,:9] = L0Matrix[3,:]
        self.C3[3,:9] = L1Matrix[3,:]
        self.C3[-4,-9:] = R1Matrix[3,:]
        self.C3[-3,-9:] = R0Matrix[3,:]

        self.C3 = self.C3/self.dG**3

        self.C4 = np.zeros((self.nG, self.nG))
        for n in range(4,self.nG-4):
            self.C4[n,(n-4):(n+5)] = CMatrix[4,:]
        self.C4[2,:9] = L0Matrix[4,:]
        self.C4[3,:9] = L1Matrix[4,:]
        self.C4[-4,-9:] = R1Matrix[4,:]
        self.C4[-3,-9:] = R0Matrix[4,:]

        self.C4 = self.C4/self.dG**4

        # These are the forward/backwards difference matrices. Note that I don't use them all.

        # Left Difference Coefficients

        LMatrix = getFDC(NMax, -4)

        self.L0 = LMatrix[0,:]
        self.L1 = LMatrix[1,:]/self.dG
        self.L2 = LMatrix[2,:]/self.dG**2
        self.L3 = LMatrix[3,:]/self.dG**3
        self.L4 = LMatrix[4,:]/self.dG**4

        # Right Difference Coefficients

        self.R0 = np.flip(self.L0)
        self.R1 = -np.flip(self.L1)
        self.R2 = np.flip(self.L2)
        self.R3 = -np.flip(self.L3)
        self.R4 = np.flip(self.L4)

        # Spatial curvature functions
        with warnings.catch_warnings(action="ignore"): # Suppresses divide by zero warning since that's expected
            self.cot = 1/np.tan(self.g)
            self.csc = 1/np.sin(self.g)
            self.cot[0] = np.nan
            self.csc[0] = np.nan

        # Substrate derivatives

        self.D = D

        self.D0 = self.C0 @ D
        self.D1 = self.C1 @ D
        self.D2 = self.C2 @ D
        self.D3 = self.C3 @ D
        self.D4 = self.C4 @ D

        self.N1D, self.N2D, self.N3D, self.N4D = self.getnablas(self.D1, self.D2, self.D3, self.D4)

        self.K1 = self.curv*self.N1D + self.N3D
        self.K2 = self.curv*self.N2D + self.N4D

    def setparams(self, S:float, M:float):
        """Sets the domain parameters of the system

        Args:
            S (float): Scaled inverse capillary number
            M (float): Scaled Marangoni number
        """
        self.S = S
        self.M = M

    def setdt(self, dt:float):
        """Sets the current time step

        Args:
            dt (float): Time between steps
        """
        self.dt = dt

    def setpin(self, pin=None):
        """Defines the pinning height boudnary condition

        Args:
            pin (float, optional): Pinning height. Defaults to None where the height is set by the initial surface.
        """
        # Sets up the boundary condition matrix and value vector

        self.BC = np.zeros((self.nG, self.nG))
        
        # Conditions at g = 0. To preserve axisymmetry, the function must be even:
        self.BC[0, :9] = self.L1
        self.BC[1, :9] = self.L3 * self.dG # We multiply by dG to ensure our error doesn't take forever to converge

        # Conditions at g = gMax. First condition is pinned, second condition is no curvature:
        self.BC[-1, -9:] = self.R0
        self.BC[-2, -9:] = (self.R1*self.cot[-1] + self.R2)

        self.bcval = np.zeros(self.nG)
        if pin is None:
            self.bcval[-1] = self.Hprev[-1]
        else:
            self.bcval[-1] = pin

    def getnablas(self, F1:np.ndarray, F2:np.ndarray, F3:np.ndarray, F4:np.ndarray):
        """Obtains the powers of the nabla operator using the derivatives in the polar angle

        Args:
            F1 (np.ndarray): First derivative in gamma
            F2 (np.ndarray): Second derivative in gamma
            F3 (np.ndarray): Third derivative in gamma
            F4 (np.ndarray): Fourth derivative in gamma

        Returns:
            N1F: Gradient of F
            N2F: Laplacian of F
            N3F: Gradient of the Laplacian of F
            N4F: Biharmonic of F
        """
        # Gets harmonic powers based on the derivatives H1, H2, H3, H4
        # Notes that these are for functions where dF/dR = 0

        N1F = F1
        N2F = F2 + F1*self.cot
        N3F = F3 + F2*self.cot - F1*(self.csc**2)
        N4F = F4 + 2*F3*self.cot + (self.cot**2 - 2*self.csc**2) * F2 + self.cot * self.csc**2 * F1

        N1F[0] = 0
        N3F[0] = 0

        N2F[0] = 2 * F2[0]
        N4F[0] = 8/3 * F4[0] - 4/3 * F2[0]
    
        return N1F, N2F, N3F, N4F
            
    def residual(self, H:np.ndarray, theta:np.ndarray, t:float):
        """Calculate the residual for the current surface and temperature guess using the backwards Euler method

        Args:
            H (np.ndarray): Surface geometry of next step
            theta (np.ndarray): Temperature distribution of next step
            t (float): Current time. Not currently used as boundary conditions are static.

        Returns:
            np.ndarray: Residuals for temperature distribution.
        """
        # Calculates the residual for the free surface equation
        # Uses the finite difference matrices as the differential operators. 
        # Everything is expanded bc using the same operator multiple times causes numerical errors

        # Here's the residual structure:

        # [Left BC1, Left BC2, Domain, ..., Domain, Right BC2, Right BC1]

        # This is the domain residuals

        # Spherical curvature function for domain
        cot = np.zeros(self.nG)
        csc = np.zeros(self.nG)
        cot[2:-2] = self.cot[2:-2]
        csc[2:-2] = self.csc[2:-2]

        # Surface Spatial Derivatives
        H0 = self.C0 @ H
        H1 = self.C1 @ H
        H2 = self.C2 @ H
        H3 = self.C3 @ H
        H4 = self.C4 @ H

        # This is the nabla calculations
        N1H, N2H, N3H, N4H = self.getnablas(H1, H2, H3, H4)

        # Temperature Spatial Derivatives
        T1 = self.C1 @ theta
        T2 = self.C2 @ theta

        # Temperature Nablas
        N1T = T1
        N2T = T2 + T1*self.cot
        
        N1T[0] = 0
        N2T[0] = 2 * T2[0]

        # Capillary Pressure and Thermocapillary Fluxes
        CPF = self.S * H0**2 * N1H * (self.curv * N1H + N3H + self.K1) + 1/3 * self.S * H0**3 * (self.curv * N2H + N4H + self.K2)
        TCF = H0 * N1H * N1T + 1/2 * H0**2 * N2T

        # Domain Residual

        dHdt = self.C0 @ (H - self.Hprev)/self.dt

        dRes = self.C0 @ ( dHdt*(1-self.steadystate) + self.M*(CPF - TCF) )

        # This is the bc residuals.

        bcRes = self.BC @ H - self.bcval

        return dRes + bcRes
    
    def initforward(self, theta:np.ndarray, t:float):
        """Initializes the next time step in the surface system. Estimates the next temperature distribution using the forward Euler meethod

        Args:
            theta (np.ndarray): Solved temperature distribution
            t (float): Next time value
        """
        self.Hprev = self.Hnext

    def update(self, theta, t):
        """Depreciated function that solves the isolated surface system.

        Args:
            theta (np.ndarray): Current surface temperature
            t (np.ndarray): Current time

        Returns:
            _type_: _description_
        """
        res = lambda H: self.residual(H, theta, t)
        result = root(res, self.Hprev, method=self.solver, options=self.opts)
        if result.success is False:
            print('Surface Root Did Not Converge')
        self.Hnext = result.x
        pass