import numpy as np
from scipy.optimize import *
import warnings
import matplotlib.pyplot as plt
from src.FiniteDifference import getFDC

class Temperature:
    def __init__(self, G:np.ndarray, Z:np.ndarray, D:np.ndarray, thetainit:callable, solver:str='krylov',  opts:dict={'fatol':1e-8}, curv:float=2, steadystate:bool=False):
        """Temperature class that describes the temperature dynamics within the liquid mirror. Called using the ThinFilm class.

        Args:
            G (np.ndarray): Polar coordinate across mesh
            Z (np.ndarray): Vertical coordinate across mesh
            D (np.ndarray): Substrate geometry
            thetainit (function): Function returning the initial temperature distribution
            solver (str, optional): Solver used to solve the independent temperature system. Defaults to 'krylov'.
            opts (_type_, optional): Solver function used to solve the independent temperature system. Defaults to {'fatol':1e-8}.
            curv (float, optional): Underlying substrate curvature. Defaults to 2.
            steadystate (bool, optional): Determines if the system is being solved for its steady state. Defaults to False.
        """
        # Defines the thin film system using the following parameters:
        # G, Z: Coordinate system
        # thetainit: Initial temperature distribution across G and Z
        # stol: The solver tolerance
        # mtol: The matching tolerance
        # solver: The solver used
       
        self.G = G
        self.Z = Z
        self.D = D
        self.g = G[:,0]
        self.opts = opts
        self.solver = solver
        self.curv = curv
        self.steadystate = steadystate

        self.nG, self.nZ = np.shape(G)

        self.thetaprev = thetainit(G,Z)
        self.thetanext = thetainit(G,Z)

        self.F = np.empty(self.nG)
        self.Q = np.empty(self.nG)

        self.dZ = self.Z[0,1] - self.Z[0,0]
        self.dG = self.G[1,0] - self.G[0,0]

        # Defines finite difference coefficients for the sides. 

        Nmax = 9
        L0 = getFDC(Nmax, -4)
        L1 = getFDC(Nmax, -3)
        L2 = getFDC(Nmax, -2)
        L3 = getFDC(Nmax, -1)
        C0 = getFDC(Nmax, 0)
        R3 = getFDC(Nmax, 1)
        R2 = getFDC(Nmax, 2)
        R1 = getFDC(Nmax, 3)
        R0 = getFDC(Nmax, 4)

        # Radial Left Values
        # f @ L0Z
        self.L0Z = np.zeros((self.nZ, self.nZ))
        self.L0Z[0,0] = 1
        
        # Radial Right Values
        # f @ R0Z
        self.R0Z = np.zeros((self.nZ, self.nZ))
        self.R0Z[-1,-1] = 1

        # Polar Left Values
        # L0G @ f
        self.L0G = np.zeros((self.nG, self.nG))
        self.L0G[0,0] = 1

        # Polar Right Values
        # R0G @ f
        self.R0G = np.zeros((self.nG, self.nG))
        self.R0G[0,0] = 0
        self.R0G[-1,-1] = 0

        # 1st Radial Central Derivative
        # f @ C1Z
        self.C1Z = np.zeros((self.nZ, self.nZ)) 
        for n in range(4, self.nZ-4):
            self.C1Z[(n-4):(n+5), n] = C0[1,:]
        self.C1Z[:9,1] = L1[1,:]
        self.C1Z[:9,2] = L2[1,:]
        self.C1Z[:9,3] = L3[1,:]
        self.C1Z[-9:,-4] = R3[1,:]
        self.C1Z[-9:,-3] = R2[1,:]
        self.C1Z[-9:,-2] = R1[1,:]

        self.C1Z = self.C1Z/self.dZ

        # 1st Radial Left Derivative
        # f @ L1Z
        self.L1Z = np.zeros((self.nZ, self.nZ))
        self.L1Z[:9,0] = L0[1,:]

        self.L1Z = self.L1Z / self.dZ

        # 1st Radial Right Derivative
        # f @ R1Z
        self.R1Z = np.zeros((self.nZ, self.nZ))
        self.R1Z[-9:,-1] = R0[1,:]

        self.R1Z = self.R1Z/self.dZ

        # Total 1st Radial Derivative
        self.T1Z = self.C1Z + self.L1Z + self.R1Z

        # 2nd Radial Central Derivative
        # f @ C2Z
        self.C2Z = np.zeros((self.nZ, self.nZ)) 
        for n in range(4, self.nZ-4):
            self.C2Z[(n-4):(n+5), n] = C0[2,:]
        self.C2Z[:9,1] = L1[2,:]
        self.C2Z[:9,2] = L2[2,:]
        self.C2Z[:9,3] = L3[2,:]
        self.C2Z[-9:,-4] = R3[2,:]
        self.C2Z[-9:,-3] = R2[2,:]
        self.C2Z[-9:,-2] = R1[2,:]

        self.C2Z = self.C2Z/self.dZ**2

        # 2nd Radial Surface Derivative
        # f @ L2Z
        self.L2Z = np.zeros((self.nZ, self.nZ)) 
        self.L2Z[:9,0] = L0[2,:]

        self.L2Z = self.L2Z/self.dZ**2

        # 2nd Radial Substrate Derivative
        # f @ R2Z
        self.R2Z = np.zeros((self.nZ, self.nZ)) 
        self.R2Z[-9:,-1] = R0[2,:]

        self.R2Z = self.R2Z/self.dZ**2

        # Total 2nd Radial Derivative
        self.T2Z = self.C2Z + self.L2Z + self.R2Z

        # 1st Polar Axis Derivative
        # C1G @ f
        self.C1G = np.zeros((self.nG, self.nG)) 
        for n in range(4, self.nG-4):
            self.C1G[n, (n-4):(n+5)] = C0[1,:]
        self.C1G[1,:9] = L1[1,:]
        self.C1G[2,:9] = L2[1,:]
        self.C1G[3,:9] = L3[1,:]
        self.C1G[-4,-9:] = R3[1,:]
        self.C1G[-3,-9:] = R2[1,:]
        self.C1G[-2,-9:] = R1[1,:]

        self.C1G = self.C1G / self.dG

        # 1st Polar Edge Derivative
        # L1G @ f
        self.L1G = np.zeros((self.nG, self.nG)) 
        self.L1G[0,:9] = L0[1,:]

        self.L1G = self.L1G/self.dG

        # 1st Polar Edge Derivative
        # R1G @ f
        self.R1G = np.zeros((self.nG, self.nG)) 
        self.R1G[-1,-9:] = R0[1,:]
        self.R1G = self.R1G/self.dG

        # Total 1st Polar Derivative
        self.T1G = self.C1G + self.L1G + self.R1G

        # 2nd Polar Central Derivative
        self.C2G = np.zeros((self.nG, self.nG)) 
        for n in range(4, self.nG-4):
            self.C2G[n, (n-4):(n+5)] = C0[2,:]
        self.C2G[1,:9] = L1[2,:]
        self.C2G[2,:9] = L2[2,:]
        self.C2G[3,:9] = L3[2,:]
        self.C2G[-4,-9:] = R3[2,:]
        self.C2G[-3,-9:] = R2[2,:]
        self.C2G[-2,-9:] = R1[2,:]

        self.C2G = self.C2G / self.dG**2

        # 2nd Polar Axis Derivative
        # f @ L2G
        self.L2G = np.zeros((self.nG, self.nG)) 
        self.L2G[0,:9] = L0[2,:]
        self.L2G = self.L2G/self.dG**2

        # 2nd Radial Edge Derivative
        # f @ R2G
        self.R2G = np.zeros((self.nG, self.nG)) 
        self.R2G[-1,-9:] = R0[2,:]
        self.R2G = self.R2G/self.dG**2

        # Total 2nd Radial Derivative
        self.T2G = self.C2G + self.L2G + self.R2G

        # Surface Derivatives. Same method as before.

        self.S0 = np.identity(self.nG)

        self.S1 = np.zeros((self.nG, self.nG))
        for n in range(4, self.nG-4):
            self.S1[n, (n-4):(n+5)] = C0[1,:]
        self.S1[0,:9] = L0[1,:]
        self.S1[1,:9] = L1[1,:]
        self.S1[2,:9] = L2[1,:]
        self.S1[3,:9] = L3[1,:]
        self.S1[-4,-9:] = R3[1,:]
        self.S1[-3,-9:] = R2[1,:]
        self.S1[-2,-9:] = R1[1,:]
        self.S1[-1,-9:] = R0[1,:]

        self.S1 = self.S1 / self.dG
        
        self.S2 = np.zeros((self.nG, self.nG))
        for n in range(4, self.nG-4):
            self.S2[n, (n-4):(n+5)] = C0[2,:]
        self.S2[0,:9] = L0[2,:]
        self.S2[1,:9] = L1[2,:]
        self.S2[2,:9] = L2[2,:]
        self.S2[3,:9] = L3[2,:]
        self.S2[-4,-9:] = R3[2,:]
        self.S2[-3,-9:] = R2[2,:]
        self.S2[-2,-9:] = R1[2,:]
        self.S2[-1,-9:] = R0[2,:]

        self.S2 = self.S2 / self.dG**2
        
        self.S3 = np.zeros((self.nG, self.nG))
        for n in range(4, self.nG-4):
            self.S3[n, (n-4):(n+5)] = C0[3,:]
        self.S3[0,:9] = L0[3,:]
        self.S3[1,:9] = L1[3,:]
        self.S3[2,:9] = L2[3,:]
        self.S3[3,:9] = L3[3,:]
        self.S3[-4,-9:] = R3[3,:]
        self.S3[-3,-9:] = R2[3,:]
        self.S3[-2,-9:] = R1[3,:]
        self.S3[-1,-9:] = R0[3,:]

        self.S3 = self.S3 / self.dG**3

        self.S4 = np.zeros((self.nG, self.nG))
        for n in range(4, self.nG-4):
            self.S4[n, (n-4):(n+5)] = C0[4,:]
        self.S4[0,:9] = L0[4,:]
        self.S4[1,:9] = L1[4,:]
        self.S4[2,:9] = L2[4,:]
        self.S4[3,:9] = L3[4,:]
        self.S4[-4,-9:] = R3[4,:]
        self.S4[-3,-9:] = R2[4,:]
        self.S4[-2,-9:] = R1[4,:]
        self.S4[-1,-9:] = R0[4,:]

        self.S4 = self.S4 / self.dG**4

        # Curvature Functions
        with warnings.catch_warnings(action="ignore"): # Suppresses divide by zero warning since that's expected
            self.cot = 1/np.tan(self.G)
            self.csc = 1/np.sin(self.G)
            self.cot[0] = np.nan
            self.csc[0] = np.nan

        # Arrays for velocities and true derivaties:

        self.Ug = np.empty([self.nG, self.nZ])
        self.Ur = np.empty([self.nG, self.nZ])

        self.dTdg = np.empty([self.nG, self.nZ])
        self.dTdR = np.empty([self.nG, self.nZ])
        self.dTdR2 = np.empty([self.nG, self.nZ])

        # Substrate derivatives

        self.D0 = np.tile(self.S0 @ D, (self.nZ,1)).T
        self.D1 = np.tile(self.S1 @ D, (self.nZ,1)).T
        self.D2 = np.tile(self.S2 @ D, (self.nZ,1)).T
        self.D3 = np.tile(self.S3 @ D, (self.nZ,1)).T
        self.D4 = np.tile(self.S4 @ D, (self.nZ,1)).T

        self.N1D, self.N2D, self.N3D, self.N4D = self.getnablas(self.D1, self.D2, self.D3, self.D4)

        self.K1 = self.curv*self.N1D + self.N3D
        self.K2 = self.curv*self.N2D + self.N4D

    def setparams(self, S:float, M:float, eps:float):
        """Sets the domain parameters of the system

        Args:
            S (float): Scaled inverse capillary number
            M (float): Scaled Marangoni number
            eps (float): Transformed thin film parameter (hpin / rc)
        """
        # Sets the domain dynamics parameters
        self.S = S
        self.M = M
        self.eps = eps
        self.eps2 = eps**2
        self.eps3 = eps**3

    def setdt(self, dt:float):
        """Sets the current time step

        Args:
            dt (float): Time between steps
        """
        # Sets the time difference
        self.dt = dt

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

        N1F = F1
        N2F = F2 + F1*self.cot
        N3F = F3 + F2*self.cot - F1*(self.csc**2)
        N4F = F4 + 2*F3*self.cot + (self.cot**2 - 2*self.csc**2) * F2 + self.cot * self.csc**2 * F1

        # Limit as gamma --> 0

        N1F[0,:] = 0
        N3F[0,:] = 0

        N2F[0,:] = 2 * F2[0,:]
        N4F[0,:] = 8/3 * F4[0,:] - 4/3 * F2[0,:]
    
        return N1F, N2F, N3F, N4F

    def calcvelocities(self, stheta:np.ndarray, H:np.ndarray):
        """Calculates the velocities within the film from the surface geometry and surface temperature

        Args:
            stheta (np.ndarray): Surface temperature
            H (np.ndarray): Surface geometry
        """
        # Calculates the velocities from a given surface temperature and geometry

        # Surface Geometry Derivatives
        H0 = np.tile(self.S0 @ H, (self.nZ,1)).T
        H1 = np.tile(self.S1 @ H, (self.nZ,1)).T
        H2 = np.tile(self.S2 @ H, (self.nZ,1)).T
        H3 = np.tile(self.S3 @ H, (self.nZ,1)).T
        H4 = np.tile(self.S4 @ H, (self.nZ,1)).T

        # This is the nabla calculations
        N1H, N2H, N3H, N4H = self.getnablas(H1, H2, H3, H4)

        # First and second derivatives of the surface temperature
        T1 = np.tile(self.S1 @ stheta, (self.nZ,1)).T
        T2 = np.tile(self.S2 @ stheta, (self.nZ,1)).T

        # Nabla for temperature
        N1T = T1
        N2T = T2 + T1*self.cot
        N2T[0,:] = 2 * T2[0,:]

        # Curvature Terms

        C1 = self.curv*N1H + N3H + self.K1
        C2 = self.curv*N2H + N4H + self.K2

        # Velocity Distribution Terms

        CPD = self.S * (1/2 * self.Z**2 + self.Z) * H0**2
        CPH = self.S / 2 * self.Z**2 * H0**2
        CPK = self.S * (1/6 * self.Z**3 + 1/2 * self.Z**2) * H0**3

        TCD = self.Z * H0
        TCT = 1/2 * self.Z**2 * H0**2

        # Combining Everything

        self.Ur = CPD * self.D1 * C1 + CPH * H1 * C1 + CPK * C2 - TCD * self.D1 * N1T - TCT * N2T
        self.Ug = -CPD * C1 + TCD*N1T

        pass

    def getderivatives(self, theta:np.ndarray, H0:np.ndarray, H1:np.ndarray, H2:np.ndarray):
        """Converts the derivatives in G and Z to the derivatives in gamma and R

        Args:
            theta (np.ndarray): Temperature array
            H0 (np.ndarray): Surface geometry
            H1 (np.ndarray): Surface gradient
            H2 (np.ndarray): Surface Laplacian
        """
        # Returns the true derivatives of the system
       
        # Derivatives in our mapping
        dTdZ = theta @ self.T1Z
        dTdZ2 = theta @ self.T2Z
        dTdG = self.T1G @ theta
        dTdG2 = self.T2G @ theta
        dTdGdZ = self.T1G @ theta @ self.T1Z

        # Converts to the true derivatives
        Hinv = 1/H0
        D1iH0 = self.D1 / H0
        H1ZiH0 =  H1 * self.Z / H0
        self.dTdR = dTdZ * Hinv
        self.dTdR2 = dTdZ2 * Hinv**2
        self.dTdg = dTdG + (D1iH0 -  H1ZiH0 ) * dTdZ 
        self.dTdg2 = dTdG2 + 2 * (D1iH0 -  H1ZiH0 ) * dTdGdZ + (D1iH0 -  H1ZiH0)**2 * dTdZ2 + dTdZ / H0 * (2 * H1 * H1ZiH0 - 2 * H1 * D1iH0 + (self.D2 - self.Z * H2))
        pass
    
    def setbc(self, surf_bc:callable, base_bc:callable, surf_type:int, base_type:int):
        """Sets the free surface and solid-liquid interface thermal boundary conditions

        Args:
            surf_bc (float): Surface boundary condition function
            base_bc (float): Solid-liquid interface boundary condition function
            surf_type (int): Surface boundary condition type
            base_type (int): Solid-liquid interface boundary condition type
        """
        # Sets the boundary condition for this step
        self.bbc = surf_bc
        self.tbc = base_bc
        self.btype = surf_type
        self.ttype = base_type

    def residual(self, H:np.ndarray, theta:np.ndarray, use_transcond:int):
        """Calculate the residual for the current surface and temperature guess using the backwards Euler method

        Args:
            H (np.ndarray): Surface geometry of next step
            theta (np.ndarray): Temperature distribution of next step
            use_transcond (int): Whether to use include transverse conduction in domain residual

        Returns:
            np.ndarray: Residuals for temperature distribution.
        """

        # Calculates the residual for a given surface geometry and temperature

        # Geometry derivatives
        H0 = np.tile(self.S0 @ H, (self.nZ,1)).T
        H1 = np.tile(self.S1 @ H, (self.nZ,1)).T
        H2 = np.tile(self.S2 @ H, (self.nZ,1)).T

        # First we get the derivatives along R, gamma from the derivatives in Z, G:

        self.getderivatives(theta, H0, H1, H2)

        # Get the velocities:

        self.calcvelocities(theta[:,0], H)

        # Time differential:

        dTdt = (theta - self.thetaprev)/self.dt

        # Heat equation residuals:

        # Convection residual
        vres = self.M * (self.Ur * self.dTdR + self.Ug * self.dTdg)

        # Transverse conduction 
        transred = self.eps2 * (self.dTdg2 + self.cot * self.dTdg) * use_transcond

        # Ful residual
        res = dTdt*(1 - self.steadystate) + vres - self.dTdR2 - transred

        # Axisymmetric condition residual:

        res[0,:] = self.dTdg[0,:]

        # Edge boundary condition residual (nabla^2 theta = 0):

        res[-1,:] = (self.dTdg2[-1,:] + self.cot[-1] * self.dTdg[-1,:]) #* self.dG

        # Surface flux residual:

        Ftheta = theta[:,0]
        Qtheta = theta[:,-1]

        F = self.F(Ftheta)
        Q = self.Q(Qtheta)
        
        res[:,0] = ( (1-self.btype)*Ftheta + self.btype*self.dTdR[:,0] ) - F[:]

        # Base flux residual:

        res[:,-1] = ( (1-self.ttype)*Qtheta + self.ttype*self.dTdR[:,-1] ) - Q[:]


        return res

    def initforward(self, H:np.ndarray, t:float):
        """Initializes the next time step in the temperature system. Estimates the next temperature distribution using the forward Euler meethod

        Args:
            H (np.ndarray): Solved surface geometry
            t (float): Next time value
        """

        # Creates a guess using a forward difference bethod

        # Sets the current step to the previous step
        self.thetaprev = self.thetanext
        ""
        # Geometry derivatives
        H0 = np.tile(self.S0 @ H, (self.nZ,1)).T
        H1 = np.tile(self.S1 @ H, (self.nZ,1)).T
        H2 = np.tile(self.S2 @ H, (self.nZ,1)).T

        # Gets the previous derivatives
        self.getderivatives(self.thetaprev, H0, H1, H2)
        # Gets the previous velocities 
        self.calcvelocities(self.thetaprev[:,0], H)
        # Change in temperature
        dTdt = self.dTdR2 - self.M * (self.Ur * self.dTdR + self.Ug * self.dTdg)
        # Initial guess
        self.thetanext = self.thetaprev + dTdt * self.dt
        # New heat fluxes
        self.F = lambda theta: self.bbc(t, self.g, theta)
        self.Q = lambda theta: self.tbc(t, self.g, theta)

    def update(self, H, t):
        """Depreciated function that solves the isolated temperature system.

        Args:
            H (np.ndarray): Current surface geometry
            t (np.ndarray): Current time

        Returns:
            _type_: _description_
        """
        # No longer used
        def res(theta):
            thetashaped = theta.reshape(self.nG, self.nZ)
            return self.residual(H, thetashaped).reshape(-1)
        # res = lambda theta: self.residual(H, theta.reshape(self.nG-1, self.nZ), n).reshape(-1)
        theta0 = self.thetanext.reshape(-1)
        result = root(res, theta0, method=self.solver, options=self.opts)
        if result.success is False:
            print('Temperature Root Did Not Converge')
        self.thetanext = result.x.reshape(self.nG, self.nZ)
        pass



