import numpy as np
import matplotlib.pyplot as plt
from glob import glob, iglob
import re
from scipy.optimize import curve_fit
from scipy.special import jn_zeros
from src.LegendreModes import *
from scipy.integrate import solve_ivp, simpson, cumulative_simpson as cs
from pint import UnitRegistry
from src.Units import *
import matplotlib

# Gets unit registry
ureg = UnitRegistry()
Q_ = ureg.Quantity # type: ignore

font = {'size':16}
matplotlib.rc('font',**font)

# Creates figure
fig, axs = plt.subplots(2, 2, figsize=(12*2, 5.5*2))

# Defines which mode goes where
modes = [[1, 2], [3, 5]]

for i in range(2):
    # Location of the data files
    fstring = [
        f'results/6.6 m Mirror/6mm 400W 100 cP Viscosity',
        f'results/6.6 m Mirror/6mm 400W 100 cP Viscosity',
    ]
    for j in range(2):
        # Gets the mode npz files and sorts them in dataset order
        mode = modes[i][j]
        files = glob(fstring[i] + f'/Mode {mode} **.npz')
        files.sort(key = lambda file: int(file.split('Dataset ')[1].split()[0]))

        # Loads the first file
        file = np.load(files[0])

        # Initializes the surface, temperature, time, heating, and cooling
        H = file['H']
        T = file['theta']
        t = file['t']
        Qt = file['Q'][1,:]
        Ft = file['F'][1,:]

        for path in files[1:]:
            # Appends all other loaded files
            print(path)
            file = np.load(path)
            H = np.append(H, file['H'], axis=-1)
            T = np.append(T, file['theta'], axis=-1)
            t = np.append(t, file['t'], axis=-1)
            Qt = np.append(Qt, file['Q'][1,:], axis=-1)
            Ft = np.append(Ft, file['F'][1,:], axis=-1)

        # Function that converts the exported 
        def npz2qty(string):
            return Q_(float(file[string][0]), str(file[string][1]))

        # Creates the dimensioned number from the file
        udict = {
            'h0':npz2qty('h0'),
            'd': npz2qty('d'),
            'T0':  npz2qty('T0'),
            'dT': npz2qty('dT'),
            'qon': npz2qty('qon'),
            'gMax': Q_(np.pi/12, 'radians'),
            'Tbg': Q_(4, 'kelvins'),
            'vareps': Q_(1, 'dimensionless'),
            'a': Q_(1, 'dimensionless'),

            'mu': npz2qty('mu'),
            'rho': npz2qty('rho'),
            'beta': npz2qty('beta'),
            's0': npz2qty('s0'),
            'k': npz2qty('k'),
            'alpha': npz2qty('alpha'),
        }

        # Converts to dimensionless quantities
        usys = UnitSystem(udict, ureg)
        params = usys.get_params()

        # Relevant heating cycle parameters
        S = params['S']
        M = params['M']
        xi = params['xi']
        ups = params['ups']
        a = params['a']
        kappa = M/xi

        # Relevant heating parameters
        F = (1 + np.cos(np.pi/12))/2
        Q0 = params['Q0']
        V = 1 - a*(1 + np.cos(np.pi/12))/2 - a*ups*(1 + np.cos(np.pi/12))/2

        # Heating cycle times
        th = params['th']
        tch = params['tch']

        # Beginning of constant thickness solution

        # Radial coordinate
        R = np.linspace(-1, 0, 12)
        # Define given mode
        lval, P = getMode(21, np.pi/12, mode)
        # Create grid in space and time
        g = np.linspace(0, np.pi/12, 21)
        Gg, Rg, tg = np.meshgrid(g, R, t[:], indexing='ij')

        # Cycle time
        t0 = np.mod(tg, tch)
        # Reduced cycle time for integration purposes
        tor = np.linspace(0, tch, 2000)
        # Reduced meshgrid
        Ggr, Rgr, t0or = np.meshgrid(g, R, tor, indexing='ij')
        # Defines heating and cooling times
        t0h = t0 < th
        t0c = t0 >= th
        t0hr = t0or < th
        t0cr = t0or >= th

        # Non-sum solution

        t00o = -V*(t0 + Rg**2 / 2 - 1/6)
        t00o[t0h] += Q0 * (t0[t0h] + (Rg[t0h] + 1)**2 / 2 - 1/6)
        t00o[t0c] += Q0 * th
        t00d = 0

        t01o = V*(6*t0 + 3*Rg**2 + 1)/6
        t01o[t0h] += -Q0/6*(-2 + 3*Rg[t0h]**2 + 6*t0[t0h])
        t01o[t0c] += -Q0*th

        t01or = V*(6*t0or + 3*Rgr**2 + 1)/6
        t01or[t0hr] += -Q0/6*(-2 + 3*Rgr[t0hr]**2 + 6*t0or[t0hr])
        t01or[t0cr] += -Q0*th

        t00oR1 = -V*Rg
        t00oR1[t0h] += Q0 * (1 + Rg[t0h])

        t00oR2 = -V*np.ones(tg.shape)
        t00oR2[t0h] += Q0
        t00oR20 = (-V)*np.ones(tg[:,:,0].shape)

        # Reduced terms for integration

        t00oR1r = -V*Rgr
        t00oR1r[t0hr] += Q0 * (1 + Rgr[t0hr])

        t00oR2r = -V*np.ones(t0or.shape)
        t00oR2r[t0hr] += Q0


        # Sum solution
        for m in range(1, 3):
            beta = m * np.pi
            print(m)

            t00o += 2*Q0/(m**2 * np.pi**2) * np.cos(beta*Rg) * (np.exp(-beta**2 * (tch - th)) - np.exp(-beta**2 * tch))/(1 - np.exp(-beta**2 * tch)) * np.exp(-beta**2*t0)
            t00d -= 2*Q0/(m**2 * np.pi**2) * np.cos(beta*Rg) * (np.exp(-beta**2 * (tch - th)) - np.exp(-beta**2 * tch))/(1 - np.exp(-beta**2 * tch)) * np.exp(-beta**2*tg)

            t00d += 2*V*(-1)**m/(m**2 * np.pi**2) * np.cos(beta*Rg) * np.exp(-beta**2*tg)

            t00o[t0h] += -2*Q0/ (m**2 * np.pi**2) * np.cos(beta*Rg[t0h]) * np.exp(-beta**2 * t0[t0h])
            t00o[t0c] += 2*Q0/ (m**2 * np.pi**2) * np.cos(beta*Rg[t0c]) * (np.exp(beta**2 * (th -  t0[t0c])) - np.exp(-beta**2 * t0[t0c]))

            t01o += 2*Q0/(beta**2) * np.cos(beta*Rg) * (np.exp(-beta**2 * (tch - th)) - np.exp(-beta**2 * tch))/(1 - np.exp(-beta**2 * tch)) * np.exp(-beta**2*t0)
            t01o += 4*Q0 * np.cos(beta*Rg) * (np.exp(-beta**2 * (tch - th)) - np.exp(-beta**2 * tch))/(1 - np.exp(-beta**2 * tch)) * t0*np.exp(-beta**2*t0)
            t01o += 4*Q0 * np.cos(beta*Rg) * (np.exp(-beta**2 * (tch - th)) - np.exp(-beta**2 * tch))/((1 - np.exp(-beta**2 * tch))**2) * tch*np.exp(-beta**2*(t0+tch))
            t01o += 4*Q0 * np.cos(beta*Rg) * ((tch-th)*np.exp(-beta**2 * (tch - th)) - tch*np.exp(-beta**2 * tch))/(1 - np.exp(-beta**2 * tch)) * np.exp(-beta**2*t0)
            t01o += 2*Q0/beta * Rg*np.sin(beta*Rg) * (np.exp(-beta**2 * (tch - th)) - np.exp(-beta**2 * tch))/(1 - np.exp(-beta**2 * tch)) * np.exp(-beta**2 * t0)

            t01o[t0h] += -2*Q0/(beta**2) * ((1 + 2*beta**2*t0[t0h])*np.cos(beta*Rg[t0h]) + beta*Rg[t0h]*np.sin(beta*Rg[t0h])) * np.exp(-beta**2*t0[t0h])
            t01o[t0c] += 2*Q0/(beta**2) * ((1 + 2*beta**2*(t0[t0c] - th))*np.cos(beta*Rg[t0c]) + beta*Rg[t0c]*np.sin(beta*Rg[t0c])) * np.exp(-beta**2*(t0[t0c] - th))
            t01o[t0c] -= -2*Q0/(beta**2) * ((1 + 2*beta**2*t0[t0c])*np.cos(beta*Rg[t0c]) + beta*Rg[t0c]*np.sin(beta*Rg[t0c])) * np.exp(-beta**2*t0[t0c])

            t00oR1 += -2*Q0/(beta) * np.sin(beta*Rg) * (np.exp(-beta**2 * (tch - th)) - np.exp(-beta**2 * tch))/(1 - np.exp(-beta**2 * tch)) * np.exp(-beta**2*t0)
            t00oR1[t0h] +=  2*Q0/(beta) * np.sin(beta*Rg[t0h]) * np.exp(-beta**2 * t0[t0h])
            t00oR1[t0c] += -2*Q0/(beta) * np.sin(beta*Rg[t0c]) * (np.exp(beta**2 * (th -  t0[t0c])) - np.exp(-beta**2 * t0[t0c]))

            t00oR2 += -2*Q0 * np.cos(beta*Rg) * (np.exp(-beta**2 * (tch - th)) - np.exp(-beta**2 * tch))/(1 - np.exp(-beta**2 * tch)) * np.exp(-beta**2*t0)
            t00oR2[t0h] += 2*Q0 * np.cos(beta*Rg[t0h]) * np.exp(-beta**2 * t0[t0h])
            t00oR2[t0c] += -2*Q0 * np.cos(beta*Rg[t0c]) * (np.exp(beta**2 * (th -  t0[t0c])) - np.exp(-beta**2 * t0[t0c]))

            t00oR20 += -2*Q0 * np.cos(beta*Rg[:,:,0]) * (np.exp(-beta**2 * (tch - th)) - np.exp(-beta**2 * tch))/(1 - np.exp(-beta**2 * tch))

            # Reduced terms for integration

            t01or += 2*Q0/(beta**2) * np.cos(beta*Rgr) * (np.exp(-beta**2 * (tch - th)) - np.exp(-beta**2 * tch))/(1 - np.exp(-beta**2 * tch)) * np.exp(-beta**2*t0or)
            t01or += 4*Q0 * np.cos(beta*Rgr) * (np.exp(-beta**2 * (tch - th)) - np.exp(-beta**2 * tch))/(1 - np.exp(-beta**2 * tch)) * t0or*np.exp(-beta**2*t0or)
            t01or += 4*Q0 * np.cos(beta*Rgr) * (np.exp(-beta**2 * (tch - th)) - np.exp(-beta**2 * tch))/((1 - np.exp(-beta**2 * tch))**2) * tch*np.exp(-beta**2*(t0or+tch))
            t01or += 4*Q0 * np.cos(beta*Rgr) * ((tch-th)*np.exp(-beta**2 * (tch - th)) - tch*np.exp(-beta**2 * tch))/(1 - np.exp(-beta**2 * tch)) * np.exp(-beta**2*t0or)
            t01or += 2*Q0/beta * Rgr*np.sin(beta*Rgr) * (np.exp(-beta**2 * (tch - th)) - np.exp(-beta**2 * tch))/(1 - np.exp(-beta**2 * tch)) * np.exp(-beta**2 * t0or)

            t01or[t0hr] += -2*Q0/(beta**2) * ((1 + 2*beta**2*t0or[t0hr])*np.cos(beta*Rgr[t0hr]) + beta*Rgr[t0hr]*np.sin(beta*Rgr[t0hr])) * np.exp(-beta**2*t0or[t0hr])
            t01or[t0cr] += 2*Q0/(beta**2) * ((1 + 2*beta**2*(t0or[t0cr] - th))*np.cos(beta*Rgr[t0cr]) + beta*Rgr[t0cr]*np.sin(beta*Rgr[t0cr])) * np.exp(-beta**2*(t0or[t0cr] - th))
            t01or[t0cr] -= -2*Q0/(beta**2) * ((1 + 2*beta**2*t0or[t0cr])*np.cos(beta*Rgr[t0cr]) + beta*Rgr[t0cr]*np.sin(beta*Rgr[t0cr])) * np.exp(-beta**2*t0or[t0cr])

            t00oR1r += -2*Q0/(beta) * np.sin(beta*Rgr) * (np.exp(-beta**2 * (tch - th)) - np.exp(-beta**2 * tch))/(1 - np.exp(-beta**2 * tch)) * np.exp(-beta**2*t0or)
            t00oR1r[t0hr] += 2*Q0/(beta) * np.sin(beta*Rgr[t0hr]) * np.exp(-beta**2 * t0or[t0hr])
            t00oR1r[t0cr] += -2*Q0/(beta) * np.sin(beta*Rgr[t0cr]) * (np.exp(beta**2 * (th -  t0or[t0cr])) - np.exp(-beta**2 * t0or[t0cr]))

            t00oR2r += -2*Q0 * np.cos(beta*Rgr) * (np.exp(-beta**2 * (tch - th)) - np.exp(-beta**2 * tch))/(1 - np.exp(-beta**2 * tch)) * np.exp(-beta**2*t0or)
            t00oR2r[t0hr] += 2*Q0 * np.cos(beta*Rgr[t0hr]) * np.exp(-beta**2 * t0or[t0hr])
            t00oR2r[t0cr] += -2*Q0 * np.cos(beta*Rgr[t0cr]) * (np.exp(beta**2 * (th -  t0or[t0cr])) - np.exp(-beta**2 * t0or[t0cr]))

        # Defines t = 0 to avoid diverging sum

        t00oR2[:,:,0] = t00oR20
        t00oR2r[:,:,0] = t00oR20

        # Numerical calculation of J and H11
        IS0 = F/15
        IS = cs(y=(t01or - t00oR1r),x=t0or, axis=2, initial=IS0)
        fosc = (IS[0,0,:] - IS[0,0,-1]*t0or[0,0,:]/tch)/2
        J = simpson((IS0/2 + fosc)*t00oR2r[0,0,:], t0or[0,0,:])/tch
        H11 = -kappa*lval*(lval+1)*cs(y=(t01o[0,0,:] - t00oR1[0,0,:] - IS[0,0,-1]/tch),x=tg[0,0,:], initial=IS0)/2

        # C calculation

        t01ors = t01or.copy()
        t00oR1s = t00oR1r.copy()
        for z in range(12):
            t01ors[:,z,:] = t01or[:,0,:]
            t00oR1s[:,z,:] = t00oR1r[:,0,:]
        Acoeff = - 1/2 * Rgr**2 * (t01ors - t00oR1s)
        C2H = simpson(y=simpson(y=Acoeff[0,:,:]*t00oR1r[0,:,:],x=Rgr[0,:,0], axis=0),x=t0or[0,0,:], axis=0)/tch

        # Easy Convection Coefficients

        C4H = V/8
        C2T = V/6

        # Surface Coefficients
        K = V/2 * (tch - th - 1)
        L = -V/2 * (tch - th - 1)

        # Other values
        N2 = -lval*(lval+1)
        Ma = M / (params['eps']**2)

        # Defines the mode evolution
        def dvdt(time, v):
            dt00dt = -4*F*K*np.exp(-4*F*time)
            B = v[0]
            A = v[1]
            dBdt = kappa*(-1/3 * S * N2*(N2+2)*B + 1/2*N2*(A - K*B - F*B))
            dAdt = -4*A - (dt00dt + kappa*J*N2 - 4*K - 4*F)*B - L*dBdt - kappa*(C2H*N2*B + S*C4H*N2*(N2+2) * B - C2T*N2*A) # + kappa/Ma * N2 * (A + L * B)
            return np.array([dBdt, dAdt])

        # Radial time scale
        t1 = t*xi
        # Solve long-time evolution
        sol = solve_ivp(fun=dvdt, t_span=(t1[0], t1[-1]), y0=np.array([1,0]), t_eval=t1)

        # Extract the full solution
        H01 = sol.y[0,:]
        Hf1 = H01*(1 + xi*H11)
        # Capillary pressure only
        HS = np.exp(-1/3*S*kappa*N2*(N2-2)*t1)

        # Time in hours
        talpha = params['talpha'].to('hours').magnitude
        treal = t*talpha
        # Initial surface deformation in case dimensional units are wanted
        hinit = 1

        # Restricts time domain to 90 cycles
        tcond = (t<= tch*90)
        # Plot on desired subplot
        axs[i,j].axhline(0, color='k', linestyle='--')
        axs[i,j].plot(treal[tcond], hinit*HS[tcond], color='black', linestyle='-.', label='Capillary Pressure Only', alpha=0.7)
        axs[i,j].plot(treal[tcond], hinit*Hf1[tcond], label='Model Amplitude')
        axs[i,j].plot(treal[tcond], hinit*(H[0,tcond] - 1)/(H[0,0] - 1), color='tab:orange', label='Numercial Amplitude')
        if j == 0:
            axs[i,j].set_ylabel('Deformation Amplitude')
        if i == 0:
            axs[i,j].set_ylim([-1, 2])
        else:
            axs[i,j].set_ylim([-1, 2])
        axs[i,j].set_xlabel('Time [hours]')
        axs[i,j].legend()
        axs[i,j].set_xlim([-3*tch*talpha, 93*tch*talpha])
        axs[i,j].set_title(f'{[[6.9, 3.0],[1.9,1.1]][i][j]} m Half-Wavelength')
        
# Adjusts subplot spacing for better aesthetics
plt.subplots_adjust(hspace=0.5)
plt.show()