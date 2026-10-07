from pint import UnitRegistry
import numpy as np

class UnitSystem:
    def __init__(self, dict:dict, ureg):
        """Unit system class that automatically calculates relevant parameters from dimensional quantities via pint. 
        Parameters are given in the format of {'param': Quantity(val, units), ...}. Necessary parameters are as follows:
            
            h0: Thickness of liquid mirror
            d: Diameter of liquid mirror (1/2 the radius of curvature)
            T0: Average surface temperature
            dT: Peak-to-peak surface temperature
            qon: Magnitude of applied heating
            gMax: Polar span of mirror
            Tbg: Background temperature of space
            vareps: Emissivity of liquid surface
            a: Proportion of emitted radiation absorbed

            mu: Dynamic viscosity
            rho: Density
            beta: Thermocapillary coefficient
            s0: Average capillary pressure
            k: Thermal conductivity
            alpha: Thermal diffusivity

        Args:
            dict (dict): Dictionary of dimensioned quantities.
            ureg (UnitRegistry): Unit registry of values in dict.
        """
        self.propdict = dict
        self.ureg = ureg

    def add_prop(self, param:str, val):
        """Adds a property to the unit system

        Args:
            param (str): Property name
            val: Property value
        """
        self.propdict[param] = val

    def get_params(self):
        """Converts dimensional qualities into the relevant dimensionless parameters:

            M: Scaled Marangoni number
            S: Scaled inverse capillary number
            xi: Radiation parameter
            thmin: Dimenionless peak-to-peak temperature
            eps: Transformed thin film parameter (hpin / rc)
            Q0: Dimensionless heating magnitude
            V: Average surface radiation
            a: Proportion of emitted radiation absorbed
            tu: Mass transfer timescale
            tnu: Momentum diffusion timescale
            talpha: Thermal diffusion timescale
            tday: One day in the mass transfer timescale
            tc: Dimensionless cooling period
            th: Dimensionless heating period
            tch: Dimensionless cycle time
            gMax: Polar span
            vareps: Emissivity of fluid
            ups: Background radiation 

        Returns:
            dict: Dictionary of dimensionless parameters:
        """
        Q_ = self.ureg.Quantity
        dimless = self.ureg.Unit('dimensionless')    

        # Pulls all user-set properties
        h0 = self.propdict['h0']
        d = self.propdict['d']
        T0 = self.propdict['T0']
        dT = self.propdict['dT']
        qon = self.propdict['qon']
        gMax = self.propdict['gMax']
        Tbg = self.propdict['Tbg']
        vareps = self.propdict['vareps']
        a = self.propdict['a']

        mu = self.propdict['mu']
        rho = self.propdict['rho']
        beta = self.propdict['beta']
        r0 = d*2
        s0 = self.propdict['s0']
        k = self.propdict['k']
        alpha = self.propdict['alpha']

        # Calculation of dimensionless parameters and relevant quantities
        nu = mu / rho
        cp = k/(alpha*rho)
        eps = h0 / r0
        sb = Q_(5.670374419e-8, 'W m^-2 K^-4')
        qr = sb * T0**4
        zeta = eps * qr / k
        u = eps * beta * zeta * r0 / mu
        Re = h0 * u / nu
        Ma = ( eps * beta * zeta * r0**2 / (mu * alpha) )
        Ca = u * mu / s0
        M = eps**2  * Ma
        S = eps**3 / Ca
        xi = zeta * r0 / T0
        ups = (Tbg/(vareps*T0))**4

        # Calculating dimensionless temperature swing and heating amplitude
        thmin = - dT / (r0 * zeta)
        Q0 = qon / qr

        # Converts everything to longdoubles
        M = np.longdouble(M.to(dimless).magnitude)
        S = np.longdouble(S.to(dimless).magnitude)
        eps = np.longdouble(eps.to(dimless).magnitude)
        xi = np.longdouble(xi.to(dimless).magnitude)
        ups = np.longdouble(ups.to(dimless).magnitude)
        vareps = np.longdouble(vareps.to(dimless).magnitude)
        thmin = np.longdouble(thmin.to(dimless).magnitude)
        Q0 = np.longdouble(Q0.to(dimless).magnitude)
        gMax = np.longdouble(gMax.to('radians').magnitude)
        a = np.longdouble(a.to(dimless).magnitude)

        # Calculates the relevant timescales (mass transfer, momentum diffusion, thermal diffusion)
        tu = r0 / u
        tnu = h0**2 / nu
        talpha = h0**2 / alpha

        # Converts timescale to proper units
        tu.ito('days')
        tnu.ito('seconds')
        talpha.ito('seconds')

        # Calculates how long a day is in the mass transfer time scale
        tday = (Q_(1, 'days')/tu).to(dimless).magnitude

        # Calculates the heating cycle times
        V = 1 - a*(1 - np.cos(gMax))/2 - a*ups*(1 + np.cos(gMax))/2
        th = -thmin/(Q0-V)
        tch = Q0/V * th
        tc = tch - th

        # Combines all relevant parameters
        redparams = {
            'M': M,
            'S': S,
            'xi': xi,
            'thmin': thmin,
            'eps':eps,
            'Q0': Q0,
            'V': V,
            'a': a,
            'tu': tu,
            'tnu': tnu,
            'talpha': talpha,
            'tday': tday,
            'tc': tc,
            'th': th,
            'tch': tch,
            'gMax': gMax,
            'vareps': vareps,
            'ups': ups
        }

        # Puts all the original units into a dictionary in the format {'param': (val, units)}
        simpleunits = {key:(self.propdict[key].magnitude, str(self.propdict[key].units)) for key in self.propdict} | {'r0':(r0.magnitude, str(r0.units))}

        return simpleunits | redparams