#%%
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import numpy as np
from typing import Optional
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy import constants as const
from astropy.io import ascii
from pathlib import Path
from snal.helper import AnalysisHelper
from snal.dataobjects import SNSpectrum
matplotlib.use('Agg')
#%%
WAVE_GRID = 100 + 20 * np.arange(550)   # 100 Å → 11200 Å, step 20 Å

class CompanionInteractionK10(AnalysisHelper):
    """
    ### Generate Companion interaction model for Type Ia supernova (Kasen 2010)
    ##### Written by prof.Myungshin Im &&& modified by Gu Lim (2021?) &&& modified by Hyeonho Choi (2023)
    =======
    Parameters
    =======
    1. rstar : float = radius of the companion star in solar radius unit
    2. m_wd : float = mass of the white dwarf in solar mass unit 
    3. ke : float = opacity scaled by 0.2cm^2/g
    4. v9 : float = ejecta velocity scaled by 10^9cm/s
    5. commonangle : bool = commonangle or optimal angle
            If false, calculate the optimal luminosity of the model (Optimal angle)
            If true, calculate the averaged angle that can be detected 

            
    =======
    Methods
    =======
    1. calc_spectrum : calculate the expected spectrum for given phase
    2. calc_magnitude : calculate the expected magnitude for given phase & filterset
    """
    
    def __init__(self,
                 rstar : float,
                 m_wd : float,
                 kappa : float = 0.2,
                 v9 : float = 1.0,
                 commonangle : bool = False
                 ):
        
        self.rstar = rstar
        self.wdmass = m_wd
        self.ke = kappa/0.2
        self.v9 = v9
        self.commonangle = commonangle
        self.sigma = const.sigma_sb.cgs.value
        self.a13 = 2*self.rstar*6.955e-3
        self.Mc = self.wdmass/1.40
        self.d10pc = 10 * const.pc.cgs.value
        self.c = const.c.cgs.value
        self.h = const.h.cgs.value
        self.k = const.k_B.cgs.value

    def __repr__(self) -> str:
        rstar = self.rstar
        wdmass = self.wdmass
        commonangle = self.commonangle
        
        return (f'CompanionInteractionModel(Companion radius:{rstar}, '
                f'WD mass:{wdmass}, '
                f'CommonAngle:{commonangle})'
        )

    def planck(self, 
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
            nu = self.c / w
        else:
            w = (self.c / nu)
        # constants appropriate to cgs units.
        
        fnu_term1 = 2 * np.pi * self.h * nu**3 / self.c**2
        fnu_term2 = np.exp((self.h*nu)/(self.k*temp))
        fnu = (fnu_term1 * (1/(fnu_term2 - 1)))
        
        flamb = (fnu * 3e18 / wave**2)
        result = dict()
        result['wl'] = w
        result["nu"] = nu
        result['fnu'] = fnu
        result['flamb'] = flamb
        return result
    
    def _luminosity_shock(self, td):
        logL_c = 43 + np.log10(self.a13) + (1/4)*np.log10(self.Mc) + (7/4)*np.log10(self.v9) + (-3/4)*np.log10(self.ke) + (-1/2)*np.log10(td) 
        if self.commonangle:
            return logL_c -1
        else:
            return logL_c
        
    def _temperature_effective(self, td):
        logT_eff = np.log10(2.5) + 4 + (1/4)*np.log10(self.a13) + (-35/36)*np.log10(self.ke) + (-37/72)*np.log10(td) 
        return logT_eff
    
    def calc_spectrum(self, td):
        # Collision luminosity (eq 22) 
        logL_c = self._luminosity_shock(td = td)
        # Effective temperature from T_eff = (L_collision/(4*pi*r_ph**2))^(-1/4)
        logT_eff = self._temperature_effective(td = td)
        
        # photosphere radius where optical depth is 1
        logR_phot = (1/2) * (logL_c - 4*logT_eff - np.log10(4*np.pi*self.sigma))
        
        # Blackbody radiation for given effective temperature 
        ww = (100 + 20*np.arange(550))
        bb = self.planck(temp = 10**logT_eff, wave = ww)
        
        # Absolute flux_nu corresponding collisional luminosity
        flux_lamb = bb['flamb'] * (4*np.pi*(10**logR_phot)**2) / (4*np.pi*self.d10pc**2)
        flux_nu = bb['fnu'] * (4*np.pi*(10**logR_phot)**2) / (4*np.pi*self.d10pc**2)
        return ww, flux_nu
        
    def calc_magnitude(self, 
                       td : Optional[np.array], 
                       filterset : str = 'UBVRIugri', 
                       visualize : bool = False):
        tbl_names = ['phase'] + list(filterset)
        mag_tbl = Table(names = tbl_names)
        tmplist = []
        lumlist = []
        for day in td:
            lumlist.append(self._luminosity_shock(td = day))
            tmplist.append(self._temperature_effective(td = day))
            spec_wl, spec_flux = self.calc_spectrum(td = day)
            spec = SNSpectrum(spec_wl, spec_flux, 'fnu')
            filt_mag = list(spec.photometry(filterset = filterset).values())
            mag_tbl.add_row([day] + filt_mag)
        mag_tbl['Luminosity_shock'] = 10**np.array(lumlist)
        mag_tbl['Temperature_eff'] = 10**np.array(tmplist)
        lightcurve = None
        TempLumcurve = None
        if visualize:
            lightcurve = self._lightcurve(mag_tbl, filterset = filterset, dpi = 100)
            TempLumcurve = self._TempLumcurve(mag_tbl, dpi = 100)
            return mag_tbl, lightcurve, TempLumcurve
        return mag_tbl, lightcurve, TempLumcurve
    
    # def calc_spectrum_fast(self, L, T, wave=None):
    #     """
    #     Faster version of calc_spectrum:
    #     directly use luminosity L and temperature T without recomputing log terms.
        
    #     Args:
    #         L : float
    #             Luminosity [erg/s]
    #         T : float
    #             Effective temperature [K]
    #         wave : ndarray, optional
    #             Wavelength grid in Angstrom (default = 100–11200 Å step 20)
    #     """
    #     if wave is None:
    #         wave = WAVE_GRID   # pre-defined global constant

    #     # Planck function once
    #     bb = self.planck(temp=T, wave=wave)

    #     # photosphere radius
    #     R_phot = np.sqrt(L / (4*np.pi*self.sigma * T**4))

    #     # flux at 10 pc
    #     scale = (R_phot / self.d10pc)**2
    #     flux_nu = bb['fnu'] * scale
    #     return wave, flux_nu
    
    # def calc_magnitude(self, td, filterset='UBVRIugri', visualize=False):
    #     td = np.asarray(td)
        
    #     # vectorized calculations
    #     logL_c = (43 + np.log10(self.a13) + 0.25*np.log10(self.Mc)
    #             + 1.75*np.log10(self.v9) - 0.75*np.log10(self.ke)
    #             - 0.5*np.log10(td))
    #     if self.commonangle:
    #         logL_c -= 1

    #     logT_eff = (np.log10(2.5) + 4 + 0.25*np.log10(self.a13)
    #                 - 35/36*np.log10(self.ke) - 37/72*np.log10(td))

    #     # now loop once over td for photometry only
    #     mag_tbl = Table(names=['phase'] + list(filterset))
    #     lumlist = 10**logL_c
    #     tmplist = 10**logT_eff

    #     for day, L, T in zip(td, lumlist, tmplist):
    #         spec_wl, spec_flux = self.calc_spectrum_fast(L, T)   # rewrite using direct L,T
    #         spec = SNSpectrum(spec_wl, spec_flux, 'fnu')
    #         filt_mag = list(spec.photometry(filterset=filterset).values())
    #         mag_tbl.add_row([day] + filt_mag)

    #     mag_tbl['Luminosity_shock'] = lumlist
    #     mag_tbl['Temperature_eff'] = tmplist
    #     return mag_tbl, None, None
    
    def _lightcurve(self, mag_tbl,
                    filterset : str,
                    dpi : int = 100):
        color_key, offset_key, _, _, label_key = self.load_filt_keys()
        fig = plt.figure(dpi = dpi)
        plt.text(x = 8.5, y= -8, s = (
                f'R_comp:{self.rstar :.1e}\n'
                f'M_wd:{self.wdmass :.1e}\n'
                f'kappa:{self.ke*0.2 :.1e}\n'
                f'v9: {self.v9 :.1e}\n'
                f'Commonangle: {self.commonangle}\n'))
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
        #ax1.set_ylim(5e40, 5e43)

        ax2 = ax1.twinx()
        ax2.plot(mag_tbl['phase'], mag_tbl['Temperature_eff'], c='r')
        ax2.set_ylabel(r'$T_{eff}\ [K]$', rotation = 270)
        ax2.set_ylim(0, 80000)

        ax2.text(x = 4.5, y= 40000, s = (
                f'R_comp:{self.rstar :.1e}\n'
                f'M_wd:{self.wdmass :.1e}\n'
                f'kappa:{self.ke*0.2 :.1e}\n'
                f'v9: {self.v9 :.1e}\n'
                f'Commonangle: {self.commonangle}\n'))

        plt.plot(1, 1, c='k', label =r'$L_{shock}$')
        plt.plot(1, 1, c='r', label =r'$T_{eff}$')
        plt.legend()
        return fig
    
    def save(self, 
             td,
             filterset : str = 'UBVRIugri',
             save_directory : str = '/data1/supernova_model/Comp_model/',
             save_figures : bool = True,
             overwrite : bool = False,
             verbose : bool = False):
        subdir = os.path.join(save_directory,f'M%.2f'%self.wdmass)   
        if not os.path.exists(subdir): 
            os.makedirs(subdir, exist_ok = True)      
        filename = '%.2f_%.2f_%.2f'%(self.rstar, self.wdmass, self.v9)
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
               search_directory : str = Path.home() / 'snal/model/comp_model/',
               force_calculate : bool = False,
               save : bool = True):
        subdir = os.path.join(search_directory,f'M%.2f'%self.wdmass)   
        if not os.path.exists(subdir): 
            os.makedirs(subdir, exist_ok = True)      
        filename = '%.2f_%.2f_%.2f'%(self.rstar, self.wdmass, self.v9)
        filename_dat = os.path.join(subdir, f'{filename}.dat')
        if os.path.exists(filename_dat) and not force_calculate:
            data = ascii.read(filename_dat, format = 'fixed_width')
        else:
            data, _, _ = self.calc_magnitude(td = td, filterset = filterset, visualize = False)
            if save:
                self.save(td = td, filterset = filterset, save_directory = search_directory, save_figures = True, overwrite = False)
        return data
# %%
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path
#%%
if __name__ == "__main__":
    import time
    home_dir = Path.home() / "snal/model/comp_model/" 
    range_rstar = np.arange(0.5, 30, 0.05)
    range_m_wd = np.arange(1.0, 1.4, 0.05)
    range_v9 = np.arange(0.7, 1.4, 0.05)
    td = np.arange(0.1, 10, 0.1)

    def process_params(rstar, m_wd, v9, home_dir, td):

        directory = os.path.join(home_dir, f"M{m_wd}")
        os.makedirs(directory, exist_ok=True)
        file_path = f"{directory}/{rstar}_{m_wd}_{v9}.dat"
        if not os.path.exists(file_path):
            Comp = CompanionInteractionK10(rstar=rstar, m_wd=m_wd, v9=v9,
                                        commonangle=False, kappa=0.2)
            result, lightcurve, tempcurve = Comp.calc_magnitude(td=td, filterset='UBVRIugri', visualize=True)

            result.write(file_path,
                        format="ascii.fixed_width", overwrite=True)
            lightcurve.savefig(f"{directory}/{rstar}_{m_wd}_{v9}_LC.png")
            tempcurve.savefig(f"{directory}/{rstar}_{m_wd}_{v9}_TL.png")
        else:
            pass

        return f"Completed: rstar={rstar}, m_wd={m_wd}, v9={v9}"

    param_combinations = [(round(r, 2), round(m, 2), round(v, 2))
                          for r in range_rstar
                          for m in range_m_wd
                          for v in range_v9]

    # tqdm progress bar for total number of parameter sets
    with ProcessPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(process_params, r, m, v, home_dir, td): (r, m, v)
                   for r, m, v in param_combinations}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing models"):
            try:
                result = future.result()
                tqdm.write(result)  # safe print with tqdm
            except Exception as e:
                rstar, m_wd, v9 = futures[future]
                tqdm.write(f"Error with rstar={rstar}, m_wd={m_wd}, v9={v9}: {e}")
# %%
