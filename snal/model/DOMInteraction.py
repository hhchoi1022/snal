#%%
import astropy.units as u
from astropy import constants as const
import numpy as np
import os
from scipy.optimize import fsolve
import time
import pyphot
from typing import Optional
from astropy.table import Table
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from astropy.io import ascii
import warnings
from pathlib import Path
from snal.dataobjects import SNSpectrum
from snal.helper import AnalysisHelper
#matplotlib.use('Agg')
%matplotlib inline 
# suppress overflow warnings in exp
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")

# suppress fsolve iteration limit warnings
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
)

# %matplotlib inline
#%%
class Section(object): 
    """
    Simple object for collecting properties
    """

    pass

class DOMInteractionL17(AnalysisHelper):
    """
    ### Generate DOM-ejecta interaction model for Type Ia supernova (Levanon 2017)
    ##### Written by Hyeonho Choi (2023)
    =======
    Parameters
    =======

            
    =======
    Methods
    =======
    1. calc_spectrum : calculate the expected spectrum for given phase
    2. calc_magnitude : calculate the expected magnitude for given phase & filterset
    """
    
    def __init__(self,
                 E_exp : float = 1, # 10^51ergs
                 M_ej : float = 1.4, # solar mass
                 kappa : float = 0.05, # cm2/g
                 # DOM
                 M_dom : float = 0.1, # solar mass
                 V_dom : float = 5e3, # km/s
                 f_dom : float = 0.1, 
                 # Shock
                 t_delay : float = 0.5e4, # s
                 f_comp : float = 1.5
                 ):
        
        # Constants
        super().__init__()
        self.const = Section()
        self.const.c = const.c.cgs
        self.const.pi = np.pi
        self.const.sigma = const.sigma_sb.cgs
        self.const.h = const.h.cgs
        self.const.k = const.k_B.cgs
        self.const.pc = const.pc.cgs
        
        # Parameters
        self._E_exp = E_exp
        self._M_ej = M_ej
        self._kappa = kappa
        self._M_dom = M_dom
        self._V_dom = V_dom
        self._f_dom = f_dom
        self._t_delay = t_delay
        self._f_comp = f_comp
        
        self.E_exp = self._E_exp * 1e51* u.erg
        self.M_ej = self._M_ej * const.M_sun.cgs
        self.kappa = self._kappa * (u.cm**2/u.g)
        self.M_dom = self._M_dom * const.M_sun.cgs
        self.V_dom = (self._V_dom * u.km/u.s).cgs
        self.f_dom = self._f_dom 
        self.t_delay = self._t_delay * u.s
        self.f_comp = self._f_comp
        
        # Calculated parameters
        self.V_e = self._V_e()
        self.A = self._A()
        self.V_shock = self._V_shock()
        self.t_shock = self._t_shock()
        self.M_shock = self._M_shock()
        self.r_shock = self._r_shock()
        self.A_s = self._A_s()
    
    def __repr__(self) -> str:
        return (f'DOMInteractionModel(L17)(E_explosion: {self.E_exp}, '
                f't_delay:{self.t_delay :.1e}, '
                f'M_ej:{self.M_ej :.1e}, '
                f'V_e: {self.V_e :.1e}, '
                f'M_dom: {self.M_dom :.1e}, '
                f'V_dom: {self.V_dom :.1e}, '
                f'M_shock: {self.M_shock :.1e}, '
                f'V_shock: {self.V_shock :.1e}, '
                f't_shock: {self.t_shock :.1e}, '
                f'DOM mass:{self.M_dom :.1e})'
        )       
        
    def _V_e(self):
        return ((self.E_exp / (6*self.M_ej))**0.5).value * (u.cm/u.s)

    def _A(self):
        return (self.M_ej / (8*self.const.pi*self.V_e**3))

    def _V_shock(self):
        def func_velocity_shock(x):
            term1 = (self.M_dom / self.f_dom)*(x*self.V_e - self.V_dom)
            term2 = (4*self.const.pi*self.A*(self.V_e**4)*np.exp(-x))
            term3 = (x**2 + 4*x + 6)
            return term1 - term2*term3
        return (fsolve(func_velocity_shock, 5) * self.V_e)[0]
    
    def _t_shock(self):
        term1 = self.V_dom * self.t_delay
        term2 = self.V_shock - self.V_dom
        return term1 / term2
    
    def _r_shock(self):
        return self.V_dom * (self.t_delay + self.t_shock)

    def _M_shock(self):
        term1 = 4*self.const.pi*self.A*self.V_e
        term2 = np.exp(-self.V_shock/self.V_e)
        term3 = self.V_shock**2 + 2*self.V_shock*self.V_e + 2*self.V_e**2
        return (term1 * term2 * term3) + self.M_dom

    def _A_s(self):
        term1 = self.A * self.M_shock / self.M_ej 
        term2 = self.f_comp**3 * np.exp(-(self.V_shock)/(self.V_e*self.f_comp))
        term3 = (1 + self.V_shock/self.V_e/self.f_comp + (1/2)*(self.V_shock/self.V_e/self.f_comp)**2)
        return term1 * (term2 * term3)**-1

    # def _velocity_diffusion_old(self, 
    #                         t : float):
    #     """
    #     t : time since explosion [sec]
    #     """
    #     def func_velocity_diffusion(x):
    #         return 2*np.log(t) - np.log(((3*self.kappa*self.M_ej)/(4*self.const.pi*self.const.c*self.V_e)).value) + x - np.log(x+1)
    #     return self.V_e * fsolve(func_velocity_diffusion, 40)
    
    def _velocity_diffusion(self, 
                            t : float):
        """
        t : time after shock formation [sec]
        """
        
        def func_velocity_diffusion(x):
            return  np.log(t*(t+self.t_shock.value)) - np.log(((6*self.kappa*self.A_s*self.V_e**2)/(self.const.c)).value) + x / self.f_comp - np.log(x*self.f_comp+self.f_comp**2)
        return self.V_e * fsolve(func_velocity_diffusion, 40)
        
    def _mass_diffusion(self,
                        t : float):
        """
        t : time since explosion [sec]
        """
        v_diff = self._velocity_diffusion(t)
        term1 = ((4*self.const.pi*self.const.c*self.V_e)/(3*self.kappa*self.M_ej)) * t**2 * u.s**2
        term2 = v_diff / (2*(self.V_e)) * np.exp(-v_diff/self.V_e)
        return self.M_ej * (term1 + term2)
    
    ## GUARENTEED
    def _luminosity_shock(self,
                          t : float):
        """
        t : time since explosion [sec]
        """
        term1 = 2*self.const.pi*self.f_dom*self.const.c
        term2 = self.kappa*self.V_e*self.f_comp
        term3 = (self.V_shock - self.V_dom)**2
        term4 = self.t_shock
        term5 = self._velocity_diffusion(t)**2
        return (term1/term2*term3*term4*term5).value * u.erg / u.s
    
    def _radius_photosphere_old(self,
                            t: float):
        """
        t : time since explosion [sec]
        """
        term1 = self.V_e * self.f_comp
        term2 = 3*self.A_s*self.kappa
        term3 = 2*(t*u.s)**2
        term4 = self.V_e*self.f_comp
        term5 = t * u.s
        return term1 * np.log(term2/term3*term4)*term5
    
    ## GUARENTEED
    def _radius_photosphere(self,
                            t: float):
        """
        t : time after shock formation [sec]
        """
        term1 = self.V_e * self.f_comp
        term2 = 3*self.A_s*self.kappa
        term3 = 2*((t + self.t_shock.value)*u.s)**2
        term4 = self.V_e*self.f_comp
        term5 = (t + self.t_shock.value) * u.s
        return term1 * np.log(term2/term3*term4)*term5
    
    ## GUARENTEED
    def _temperature_effective(self,
                                t : float):
        """
        t : time since explosion [sec]
        """
        term1 = 4 * self.const.pi * self.f_dom
        term2 = self._radius_photosphere(t)**2 * self.const.sigma
        return ((self._luminosity_shock(t)/term1/term2)**(1/4)).value * u.K
    
    def _planck(self, 
                temp,
                wave = None,
                nu = None):
        """Calculate planck function for given temperature(K) 

        Args:
            temp : Effective temperature(K)
            wave (optional): wavelength in Angstron unit
            nu (optional): Frequency in Hz unit

        Returns:
            dict: planck functiond in f_nu, f_lamb
        """
        if wave is not None:
            w = wave/1.e8  # angstroms to cm
            nu = self.const.c.value / w
        else:
            w = (self.const.c.value / nu)
        # constants appropriate to cgs units.
        
        fnu_term1 = 2 * np.pi * self.const.h.value * nu**3 / self.const.c.value**2
        fnu_term2 = np.exp((self.const.h.value*nu)/(self.const.k.value*temp.value))
        fnu = (fnu_term1 * (1/(fnu_term2 - 1)))
        
        flamb = (fnu * 3e18 / wave**2)
        result = dict()
        result['wl'] = w
        result["nu"] = nu
        result['fnu'] = fnu
        result['flamb'] = flamb

        return result
    
    # For increasing speed, use this code
    def calc_spectrum(self,
                      td : float):
        ts = td * 86400
        ww = (10 + 20*np.arange(550))
        velocity_diffusion = self._velocity_diffusion(t = ts)
        Luminosity_shock = self._luminosity_shock(t = ts)
        Radius_phot = self._radius_photosphere(t = ts)
        Temperature_eff = self._temperature_effective(t = ts)
        bb = self._planck(temp = Temperature_eff, wave = ww)
        flux = bb['fnu'] * (4*np.pi*self.f_dom*(Radius_phot)**2) / (4*np.pi*(10*self.const.pc)**2)
        #mag = -2.5 * np.log10(flux.value) - 48.6
        #spl = interpolate.interp1d(ww, mag, kind = 'linear')
        return ww, flux, Luminosity_shock, Temperature_eff, velocity_diffusion

    def calc_magnitude(self, 
                       td : Optional[np.array], 
                       filterset : str, 
                       visualize : bool = False):
        tbl_names = ['phase'] + list(filterset)
        mag_tbl = Table(names = tbl_names)
        tmplist = []
        lumlist = []
        vellist = []
        for day in td:
            day = day.round(2)
            spec_wl, spec_flux, lum_shock, temp_eff, vel_diff = self.calc_spectrum(td = day)
            spec = SNSpectrum(spec_wl, spec_flux, 'fnu')
            filt_mag = list(spec.photometry(filterset = filterset).values())
            mag_tbl.add_row([day] + filt_mag)
            lumlist.append(float("%.3e"%lum_shock.value[0]))
            tmplist.append(float("%.1f"%temp_eff.value[0]))
            vellist.append(float("%.1e"%vel_diff.value[0]))
        mag_tbl['Luminosity_shock'] = np.array(lumlist)
        mag_tbl['Temperature_eff'] = np.array(tmplist)
        mag_tbl['Velocity_diff'] = np.array(vellist)
        lightcurve = None
        TempLumcurve = None
        if visualize:
            lightcurve = self._lightcurve(mag_tbl, filterset = filterset, dpi = 100)
            TempLumcurve = self._TempLumcurve(mag_tbl, dpi = 100)
            return mag_tbl, lightcurve, TempLumcurve
        else:
            return mag_tbl, lightcurve, TempLumcurve

    def _lightcurve(self, mag_tbl,
                    filterset : str,
                    dpi : int = 100):
        color_key, offset_key, _, _, label_key = self.load_filt_keys()
        fig = plt.figure(dpi = dpi)
        plt.text(x = 6.5, y= -7.5, s = (
                f'E_exp:{self.E_exp :.1e}\n'
                f't_delay:{self.t_delay :.1e}\n'
                f'M_ej:{self.M_ej :.1e}\n'
                f'V_e: {self.V_e :.1e}\n'
                f'M_dom: {self.M_dom :.1e}\n'
                f'V_dom: {self.V_dom :.1e}\n'
                f'M_shock: {self.M_shock :.1e}\n'
                f'V_shock: {self.V_shock :.1e}\n'
                f't_shock: {self.t_shock :.1e}\n'
                f'DOM mass:{self.M_dom :.1e}\n'))
        plt.gca().invert_yaxis()
        
        for filter_ in filterset:
            clr = color_key[filter_]
            offset = offset_key[filter_]
            label = label_key[filter_]
            plt.plot(mag_tbl['phase'], mag_tbl[filter_] + offset, color = clr, label = label)
        plt.legend(loc = 1, ncol =2)
        plt.ylim(-6, -21)
        plt.xlabel('Phase[days]')
        plt.ylabel('Magnitude[AB]')
        return fig
    
    def _TempLumcurve(self, mag_tbl,
                      dpi : int = 100):
        fig = plt.figure(dpi = dpi)

        
        fig = plt.figure(dpi = dpi)
        ax1 = plt.subplot()
        ax1.plot(mag_tbl['phase'], mag_tbl['Luminosity_shock'], c='k')
        ax1.set_yscale('log')
        ax1.set_yticks([1e41, 5e41, 1e42, 5e42, 1e43, 5e43], [1e41, 5e41, 1e42, 5e42, 1e43, 5e43])
        ax1.set_ylabel(r'$L_{shock}\ [erg/s]$', fontsize = 10)
        ax1.set_xlabel('Phase [day]')
        ax1.set_ylim(5e40, 5e43)

        ax2 = ax1.twinx()
        ax2.plot(mag_tbl['phase'], mag_tbl['Temperature_eff'], c='r')
        ax2.set_ylabel(r'$T_{eff}\ [K]$', rotation = 270)
        ax2.set_ylim(0, 120000)

        ax2.text(x = 4.5, y= 40000, s = (
        f'E_exp:{self.E_exp :.1e}\n'
        f't_delay:{self.t_delay :.1e}\n'
        f'M_ej:{self.M_ej :.1e}\n'
        f'V_e: {self.V_e :.1e}\n'
        f'M_dom: {self.M_dom :.1e}\n'
        f'V_dom: {self.V_dom :.1e}\n'
        f'M_shock: {self.M_shock :.1e}\n'
        f'V_shock: {self.V_shock :.1e}\n'
        f't_shock: {self.t_shock :.1e}\n'
        f'DOM mass:{self.M_dom :.1e}\n'))

        plt.plot(1, 1, c='k', label =r'$L_{shock}$')
        plt.plot(1, 1, c='r', label =r'$T_{eff}$')
        plt.legend()
        return fig

    def save(self, 
             td,
             filterset : str = 'UBVRIugri',
             save_directory : str = '/data1/supernova_model/DOM_model/',
             save_figures : bool = True,
             overwrite : bool = False,
             verbose : bool = False):
        subdir = os.path.join(save_directory,f'kappa%.2f'%self._kappa,f'E%.1f'%self._E_exp)   
        if not os.path.exists(subdir): 
            os.makedirs(subdir, exist_ok = True)      
        filename = '%.1f_%.1f_%.2f_%d_%.1f_%.3f_%d_%.2f'%(self._E_exp, self._M_ej, self._kappa, self._t_delay, self._f_comp, self._M_dom, self._V_dom, self._f_dom)
        filename_dat = os.path.join(subdir, f'{filename}.dat')
        if overwrite:
            mag_tbl, lightcurve, tempcurve = self.calc_magnitude(td = td, filterset = filterset, visualize = save_figures)
            mag_tbl.write(filename_dat, format='ascii.fixed_width', overwrite=True)        
            if save_figures:
                lightcurve.savefig(os.path.join(subdir, f'{filename}_LC.png'))
                tempcurve.savefig(os.path.join(subdir, f'{filename}_TL.png'))
            if verbose:
                print(f'{filename_dat} is saved. ')
        else:
            if os.path.exists(filename_dat):
                if verbose:
                    print(f'{filename_dat} is already exist. ')
                pass
            else:
                mag_tbl, lightcurve, tempcurve = self.calc_magnitude(td = td, filterset = filterset, visualize = save_figures)
                mag_tbl.write(filename_dat, format='ascii.fixed_width', overwrite=True)        
                if save_figures:
                    lightcurve.savefig(os.path.join(subdir, f'{filename}_LC.png'))
                    tempcurve.savefig(os.path.join(subdir, f'{filename}_TL.png'))
                if verbose:
                    print(f'{filename_dat} is saved. ')
    
    def get_LC(self,
               td,
               filterset : str = 'UBVRIugri',
               search_directory : str = Path.home() / 'snal/model/DOM_model/',
               force_calculate : bool = False,
               save : bool = True):
        subdir = os.path.join(search_directory,f'kappa%.2f'%self._kappa,f'E%.1f'%self._E_exp)
        if not os.path.exists(subdir): 
            os.makedirs(subdir, exist_ok = True)      
        filename = '%.1f_%.1f_%.2f_%d_%.1f_%.3f_%d_%.2f'%(self._E_exp, self._M_ej, self._kappa, self._t_delay, self._f_comp, self._M_dom, self._V_dom, self._f_dom)
        filename_dat = os.path.join(subdir, f'{filename}.dat')
        if os.path.exists(filename_dat) and not force_calculate:
            data = ascii.read(filename_dat, format = 'fixed_width')
        else:
            data, _, _ = self.calc_magnitude(td = td, filterset = filterset, visualize = False)
            if save:
                self.save(td = td, filterset = filterset, save_directory = search_directory, save_figures = True, overwrite = False)
        return data
    
    # --- add inside DOMInteractionL17 ---

    # --- inside DOMInteractionL17 ---

    def L_of_t(self, t_sec):
        """
        Luminosity at times t_sec (float or array) in seconds since explosion.
        Returns astropy Quantity [erg/s] with shape (N,).
        """
        t_sec = np.atleast_1d(t_sec)
        L_list = []
        for t in t_sec:
            # _luminosity_shock returns a Quantity with a 1-element array inside.
            val = self._luminosity_shock(t).to(u.erg/u.s).value
            # force scalar
            val = np.asarray(val).ravel()[0]
            L_list.append(val)
        return (np.asarray(L_list, dtype=float) * u.erg/u.s)

    def integrate_luminosity(self,
                            t_start_day: float = 0.0,
                            t_end_day: float = 10.0,
                            dt_sec: float = 8640.0):
        """
        Integrate L_shock from t_start_day to t_end_day using trapezoid rule.
        """
        t0 = max(t_start_day, self.t_shock.to(u.day).value)
        t1 = t_end_day
        if t1 <= t0:
            return 0.0 * u.erg, np.array([t0, t1]), np.array([0.0, 0.0]) * u.erg/u.s

        t_grid_sec = np.arange(t0*86400.0, t1*86400.0 + dt_sec, dt_sec)
        L_grid = self.L_of_t(t_grid_sec)                   # shape (N,)
        # (optional) guard against non-finite values
        m = np.isfinite(L_grid.value)
        E_rad = np.trapz(L_grid.value[m], t_grid_sec[m]) * u.erg
        return E_rad, (t_grid_sec/86400.0), L_grid

#%%
self = DOMInteractionL17(E_exp = 1.0, M_ej = 1.4, M_dom = 0.05, t_delay= 2e4, f_comp = 1.5, f_dom = 0.3)
self.V_shock
#%%
self.M_shock/self.M_dom
#%%
self._r_shock()/const.R_sun.cgs
#%%
self._A_s
#%%
DOM_1 = DOMInteractionL17(E_exp=1.5, M_ej=1.4, kappa=0.05, t_delay=5000, f_comp=1.5, M_dom=0.1, V_dom=5000, f_dom=0.10)
DOM_2 = DOMInteractionL17(E_exp=1.0, M_ej=1.0, kappa=0.05, t_delay=5000, f_comp=1.5, M_dom=0.1, V_dom=5000, f_dom=0.10)
DOM_3 = DOMInteractionL17(E_exp=1.5, M_ej=1.0, kappa=0.05, t_delay=5000, f_comp=1.5, M_dom=0.1, V_dom=5000, f_dom=0.10)
DOM_4 = DOMInteractionL17(E_exp=1.0, M_ej=1.4, kappa=0.05, t_delay=5000, f_comp=1.5, M_dom=0.1, V_dom=5000, f_dom=0.10)
#%%
td = np.arange(0.1, 10, 0.3)
mag_tbl_1 = DOM_1.calc_magnitude(td = td, filterset = 'UBVRIugri', visualize = False)[0]
mag_tbl_2 = DOM_2.calc_magnitude(td = td, filterset = 'UBVRIugri', visualize = False)[0]
mag_tbl_3 = DOM_3.calc_magnitude(td = td, filterset = 'UBVRIugri', visualize = False)[0]
mag_tbl_4 = DOM_4.calc_magnitude(td = td, filterset = 'UBVRIugri', visualize = False)[0]
#%%
plt.figure(figsize = (8, 6))
plt.plot(mag_tbl_1['phase'], mag_tbl_1['U'] + 2, c='purple', linestyle = ':', label = 't_delay = 20000')
plt.plot(mag_tbl_1['phase'], mag_tbl_1['B'] , c='blue', linestyle = ':')
plt.plot(mag_tbl_1['phase'], mag_tbl_1['V'] - 2, c='green', linestyle = ':')
plt.plot(mag_tbl_2['phase'], mag_tbl_2['U'] + 2, c='purple', linestyle = '-.', label = 't_delay = 1000')
plt.plot(mag_tbl_2['phase'], mag_tbl_2['B'] , c='blue', linestyle = '-.')
plt.plot(mag_tbl_2['phase'], mag_tbl_2['V'] - 2, c='green', linestyle = '-.')
plt.plot(mag_tbl_3['phase'], mag_tbl_3['U'] + 2, c='purple', linestyle = '--', label = 't_delay = 5000')
plt.plot(mag_tbl_3['phase'], mag_tbl_3['B'] , c='blue', linestyle = '--')
plt.plot(mag_tbl_3['phase'], mag_tbl_3['V'] - 2, c='green', linestyle = '--')
plt.plot(mag_tbl_4['phase'], mag_tbl_4['U'] + 2, c='purple', linestyle = '-', label = 't_delay = 5000')
plt.plot(mag_tbl_4['phase'], mag_tbl_4['B'] , c='blue', linestyle = '-')
plt.plot(mag_tbl_4['phase'], mag_tbl_4['V'] - 2, c='green', linestyle = '-')
plt.yticks([-10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21])
plt.ylim(-10, -21)
plt.xlim(0, 10)
plt.legend()
plt.grid()
#%%
DOM_lowmass = DOMInteractionL17(E_exp=1.4, M_ej=1.4, kappa=0.05, t_delay=5000, f_comp=1.5, M_dom=0.0001, V_dom=5000, f_dom=0.15)
DOM_highmass = DOMInteractionL17(E_exp=1.4, M_ej=1.4, kappa=0.05, t_delay=5000, f_comp=1.5, M_dom=0.001, V_dom=5000, f_dom=0.15)
DOM_highermass = DOMInteractionL17(E_exp=1.4, M_ej=1.4, kappa=0.05, t_delay=5000, f_comp=1.5, M_dom=0.01, V_dom=5000, f_dom=0.15)
DOM_highestmass = DOMInteractionL17(E_exp=1.4, M_ej=1.4, kappa=0.05, t_delay=5000, f_comp=1.5, M_dom=0.1, V_dom=5000, f_dom=0.15)
#%%
lum_1 = DOM_lowmass.L_of_t(ts)
lum_2 = DOM_highmass.L_of_t(ts)
lum_3 = DOM_highermass.L_of_t(ts)
lum_4 = DOM_highestmass.L_of_t(ts)
#%%
plt.plot(td, lum_1.value, c='purple', linestyle = ':', label = 'M_dom = 0.0001')
plt.plot(td, lum_2.value, c='blue', linestyle = ':', label = 'M_dom = 0.001')
plt.plot(td, lum_3.value, c='green', linestyle = ':', label = 'M_dom = 0.01')
plt.plot(td, lum_4.value, c='purple', linestyle = '-', label = 'M_dom = 0.1')
plt.legend()
#%%
# %%
td = np.arange(0.1, 10, 0.1)
mag_tbl_1 = DOM_lowmass.calc_magnitude(td = td, filterset = 'UBVRIugri', visualize = False)[0]
mag_tbl_2 = DOM_highmass.calc_magnitude(td = td, filterset = 'UBVRIugri', visualize = False)[0]
mag_tbl_3 = DOM_highermass.calc_magnitude(td = td, filterset = 'UBVRIugri', visualize = False)[0]
mag_tbl_4 = DOM_highestmass.calc_magnitude(td = td, filterset = 'UBVRIugri', visualize = False)[0]
# %%
plt.plot(mag_tbl_1['phase'], mag_tbl_1['U'] + 2, c='purple', linestyle = ':')
plt.plot(mag_tbl_1['phase'], mag_tbl_1['B'] , c='blue', linestyle = ':')
plt.plot(mag_tbl_1['phase'], mag_tbl_1['V'] - 2, c='green', linestyle = ':')
plt.plot(mag_tbl_2['phase'], mag_tbl_2['U'] + 2, c='purple', linestyle = '--')
plt.plot(mag_tbl_2['phase'], mag_tbl_2['B'] , c='blue', linestyle = '--')
plt.plot(mag_tbl_2['phase'], mag_tbl_2['V'] - 2, c='green', linestyle = '--')
plt.plot(mag_tbl_3['phase'], mag_tbl_3['U'] + 2, c='purple', linestyle = '-.')
plt.plot(mag_tbl_3['phase'], mag_tbl_3['B'] , c='blue', linestyle = '-.')
plt.plot(mag_tbl_3['phase'], mag_tbl_3['V'] - 2, c='green', linestyle = '-.')
plt.plot(mag_tbl_4['phase'], mag_tbl_4['U'] + 2, c='purple', linestyle = '-')
plt.plot(mag_tbl_4['phase'], mag_tbl_4['B'] , c='blue', linestyle = '-')
plt.plot(mag_tbl_4['phase'], mag_tbl_4['V'] - 2, c='green', linestyle = '-')
plt.ylim(-11, -21)
plt.xlim(0, 10)
plt.grid()

# %%
print('Original Model')
print(DOM_lowmass._temperature_effective(2))
print(DOM_highmass._temperature_effective(2))
print(DOM_highermass._temperature_effective(2))

# %%
print(DOM_lowmass._luminosity_shock(2))
print(DOM_highmass._luminosity_shock(2))
print(DOM_highermass._luminosity_shock(2))

# %%
print(DOM_lowmass._velocity_diffusion(2))
print(DOM_highmass._velocity_diffusion(2))
print(DOM_highermass._velocity_diffusion(2))

# %%
print(DOM_lowmass._velocity_diffusion_rough(2))
print(DOM_highmass._velocity_diffusion_rough(2))
print(DOM_highermass._velocity_diffusion_rough(2))
# %%
print(DOM_lowmass._radius_photosphere(2))
print(DOM_highmass._radius_photosphere(2))
print(DOM_highermass._radius_photosphere(2))
# %%
print(DOM_lowmass._radius_photosphere_rough(2))
print(DOM_highmass._radius_photosphere_rough(2))
print(DOM_highermass._radius_photosphere_rough(2))



# %%
if __name__ == '__main__':
    def process_params(E_exp, M_ej, kappa, t_delay, f_comp, M_dom, v_dom, f_dom, home_dir, td):
        #print(f'Start: E_exp = {E_exp}, M_ej = {M_ej}, kappa = {kappa}, t_delay = {t_delay}, f_comp = {f_comp}, M_dom = {M_dom}, v_dom = {v_dom}, f_dom = {f_dom}') 
        dirname = home_dir / f'kappa{kappa}/E{E_exp}'
        filename = '%.1f_%.1f_%.2f_%d_%.1f_%.3f_%d_%.2f'%(E_exp, M_ej, kappa, t_delay, f_comp, M_dom, v_dom, f_dom)
        if not os.path.exists(dirname): 
            os.makedirs(dirname, exist_ok = True) 
        filename_dat = os.path.join(dirname, f'{filename}.dat')
        if not os.path.exists(filename_dat):            
            #print(f'Start calculation: E_exp = {E_exp}, M_ej = {M_ej}, kappa = {kappa}, t_delay = {t_delay}, f_comp = {f_comp}, M_dom = {M_dom}, v_dom = {v_dom}, f_dom = {f_dom}') 
            DOM = DOMInteractionL17(E_exp=E_exp, M_ej=M_ej, kappa=kappa, t_delay=t_delay, f_comp=f_comp, M_dom=M_dom, V_dom=v_dom, f_dom=f_dom)
            result, lightcurve, tempcurve = DOM.calc_magnitude(td=td, filterset='UBVRIugri', visualize = True)
            result.write(filename_dat, format='ascii.fixed_width', overwrite=True)
            lightcurve.savefig(os.path.join(dirname, f'{filename}_LC.png'))
            tempcurve.savefig(os.path.join(dirname, f'{filename}_TL.png'))
            plt.close('all')
            #print(f'{home_dir}kappa{kappa}/E{E_exp}/{filename} is saved. ')
#%%

#%%

from tqdm import tqdm
from pathlib import Path
import time
if __name__ == '__main__':
    home_dir = Path.home() / 'snal/model/DOM_model/' 
    #home_dir = '/data7/yunyi/temp_supernova/Gitrepo/Research/model/DOM_model/' 
    range_E_exp = np.round(np.arange(1.0, 1.6, 0.1), 1)  # 10^51 ergs, rounded to 2 decimal places
    range_M_ej = np.round([1.5], 1)   # solar mass, rounded to 2 decimal places
    range_kappa = np.round([0.05], 2)              # cm^2/g, rounded to 2 decimal places
    range_t_delay = np.round(np.logspace(1, 3.5, 30), 0)  # s, rounded to 0 decimal places (integer-like)
    range_t_delay = np.round(np.arange(1e1, 1e3, 10), 0)
    range_f_comp = np.round([1.5], 1)                    # compress fraction, rounded to 2 decimal places
    range_M_dom = np.round(np.logspace(-4, -1, 100), 5)  # solar mass, rounded to 2 decimal places
    range_M_dom = np.round(np.arange(0.01, 0.7, 0.02), 2)
    range_v_dom = np.int_(np.arange(1e3, 5e3, 200))  # km/s, rounded and converted to integer
    range_f_dom = np.round([0.1], 1)  # fraction of DOM mass, rounded to 2 decimal places
    td = np.arange(0.1, 10, 0.1)
    expected_time = len(range_E_exp) * len(range_M_ej) * len(range_kappa) * len(range_t_delay) * len(range_f_comp) * len(range_M_dom) * len(range_v_dom) * len(range_f_dom) * 3 / 3600
    n_workers = 50
    print('expected time: ', expected_time/n_workers, 'Hour')
#%%
if __name__ == '__main__':
    param_combinations = [(E_exp, M_ej, kappa, t_delay, f_comp, M_dom, v_dom, f_dom)
                          for E_exp in range_E_exp
                          for M_ej in range_M_ej
                          for kappa in range_kappa
                          for t_delay in range_t_delay
                          for f_comp in range_f_comp
                          for M_dom in range_M_dom
                          for v_dom in range_v_dom
                          for f_dom in range_f_dom]
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_params, *params, home_dir, td) for params in param_combinations]
        # Use tqdm to wrap the as_completed iterator for progress tracking
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Parameters"):
            try:
                future.result()  # This will raise an exception if the function call raised one.
            except Exception as e:
                print(f"Generated an exception: {e}")

#%%
import numpy as np
import matplotlib.pyplot as plt

def parameter_sweep(
    target_parameter: str = 'f_dom',
    result_parameter: str = 'V_shock',
    param_min: float = 0.01,
    param_max: float = 0.3,
    log_scale: bool = False,
    param_step: int = 30,
    visualize: bool = True
    ):
    """
    Sweep a parameter in DOMInteractionL17 and measure how a result quantity changes.
    
    Parameters
    ----------
    target_parameter : str
        Name of the parameter to vary (e.g., 'f_dom', 'M_dom', 'V_dom', etc.)
    result_parameter : str
        Attribute name of the model to record (e.g., 'V_shock', 't_shock', 'M_shock', etc.)
    param_min, param_max, param_step : float
        Range of values for the parameter sweep.
    fixed_params : dict
        Dictionary of all fixed parameters passed to DOMInteractionL17.
    visualize : bool
        If True, plot the result.
    
    Returns
    -------
    params : np.ndarray
        Array of varied parameter values.
    results : np.ndarray
        Array of result values corresponding to each parameter.
    """
    fixed_params = dict(
            E_exp=1.0, M_ej=1.4, kappa=0.05, 
            t_delay=5000, f_comp=1.5,
            M_dom=0.05, V_dom=5000, f_dom=0.1
        )

    if log_scale:
        params = np.logspace(np.log10(param_min), np.log10(param_max), param_step)
    else:
        params = np.linspace(param_min, param_max, param_step)
    results = []

    for value in params:
        # set varied parameter
        fixed_params[target_parameter] = value

        # create model
        model = DOMInteractionL17(**fixed_params)

        # read result (could be Quantity or float)
        result_value = getattr(model, result_parameter)
        # convert to scalar if needed
        if hasattr(result_value, 'value'):
            result_value = result_value.value
        results.append(result_value)

    results = np.array(results)

    # visualization
    if visualize:
        plt.figure(figsize=(6,4))
        plt.plot(params, results, marker='o', lw=2, c='k')
        plt.xlabel(target_parameter)
        plt.ylabel(result_parameter)
        plt.title(f'{result_parameter} vs {target_parameter}')
        plt.grid(True, ls='--', alpha=0.5)
        plt.tight_layout()
        if log_scale:
            plt.xscale('log')
        plt.show()

    return params, results

#%%
parameter_range = dict()
parameter_range['V_dom'] = [1000, 5000, False]
parameter_range['f_dom'] = [0.1, 0.8, False]
parameter_range['M_dom'] = [0.0001, 0.1, True]
parameter_range['E_exp'] = [1, 1.4, False]
parameter_range['M_ej'] = [1, 1.4, False]
parameter_range['kappa'] = [0.01, 0.2, False]
parameter_range['t_delay'] = [1e1, 1e4, True]
parameter_range['f_comp'] = [1.0, 2.0, False]

result_to_see = 't_shock'

for key in parameter_range.keys():
    parameter_sweep(
        target_parameter=key,
        result_parameter=result_to_see,
        param_min=parameter_range[key][0],
        param_max=parameter_range[key][1],
        param_step=30,
        log_scale=parameter_range[key][2],
        visualize = True
    )
#%%
mass_range = np.logspace(np.log10(0.0001), np.log10(0.2), 30)
E_rad_list = []
for mass in mass_range:
    DOM = DOMInteractionL17(E_exp=1.4, M_ej=1.4, kappa=0.05, t_delay=3000, f_comp=1.5, M_dom=mass, V_dom=5000, f_dom=0.1)
    E_rad, t_grid, L_grid = DOM.integrate_luminosity(t_start_day=0.0, t_end_day=30.0, dt_sec=86400.0)
    E_rad_list.append(E_rad.value)
#%%
plt.plot(mass_range, E_rad_list)
plt.xscale('log')
plt.xlabel('Mass [M_sun]')
plt.ylabel('Integrated Luminosity [erg]')
plt.title('Integrated Luminosity vs Mass')
plt.grid()
plt.yscale('log')
plt.show()

# %%
