
#%%
from astropy.io import ascii
path_imsng = './data/SN2021aefx_formatted_Host_dereddening_MW_dereddening.ascii_fixed_width'
path_h22 = './data/Hosseinzadeh2022_formatted_Host_dereddening_MW_dereddening.ascii_fixed_width'

tbl_imsng = ascii.read(path_imsng, format = 'fixed_width')
tbl_h22 = ascii.read(path_h22, format = 'fixed_width')
tbl_h22 = tbl_h22[tbl_h22['observatory'] == 'LasCumbres1m']

#%%
from ezphot.dataobjects import LightCurve
lc_imsng = LightCurve()
lc_imsng.data = tbl_imsng
lc_imsng.plt_params.xlim = [59500, 59730]
lc_imsng.plt_params.ylim = [20, 8]
lc_imsng.plt_params.figure_figsize = (12, 8)
lc_imsng.plot(ra = 64.9725, dec= -54.948081, flux_key = 'mag', fluxerr_key = 'e_mag')

#%%
lc_h22 = LightCurve()
lc_h22.data = tbl_h22
lc_h22.plt_params.xlim = [59500, 59730]
lc_h22.plt_params.ylim = [20, 8]
lc_h22.plt_params.figure_figsize = (12, 8)
lc_h22.plot(ra = 64.9725, dec= -54.948081, flux_key = 'mag', fluxerr_key = 'e_mag')

#%%
from astropy.table import vstack
lc_all = LightCurve()
lc_all.data = vstack([tbl_imsng, tbl_h22])
lc_all.plt_params.xlim = [59500, 59730]
lc_all.plt_params.ylim = [20, 8]
lc_all.plt_params.figure_figsize = (12, 8)
lc_all.plot(ra = 64.9725, dec= -54.948081, flux_key = 'mag', fluxerr_key = 'e_mag')
#%%
from astropy.io import ascii
from astropy.table import vstack
from astropy.table import Table
import astropy.units as u
import sncosmo
import numpy as np
import os, glob
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import seaborn as sns
import numpy as np

#%%
from snal.helper import AnalysisHelper
helper = AnalysisHelper()
color_key, offset_key, filter_key_sncosmo, _, name_key = helper.load_filt_keys()
#%% Filter registration
def registerfilter(responsefile,
                   name,
                   force = True):
    tbl = ascii.read(responsefile, format = 'fixed_width')
    band = sncosmo.Bandpass(tbl['wavelength'], tbl['response'], wave_unit = u.AA, name = name)
    sncosmo.register(band, force = force)
#%%
# KCT 
list_responsefile = glob.glob('../helper/transmission/KCT_STX16803/STX16803_sdss2_?')
KCT_filterkey = dict()
for responsefile in list_responsefile:
    filter_ = responsefile[-1]
    filename = os.path.basename(responsefile)
    registerfilter(responsefile, filename, force = True)
    KCT_filterkey[filter_] = filename
    
# RASA36
RASA36_responsefile = glob.glob("../helper/transmission/RASA36_KL4040/KL4040*")[0]
RASA_filterkey = dict()
filter_ = RASA36_responsefile[-1]
filename = os.path.basename(RASA36_responsefile)
registerfilter(RASA36_responsefile, filename, force = True)
RASA_filterkey[filter_] = filename

# Lascumbres1m
list_responsefile = glob.glob('../helper/transmission/LasCumbres/Las*')
Las_filterkey = dict()
for responsefile in list_responsefile:
    filter_ = responsefile[-1]
    filename = os.path.basename(responsefile)
    registerfilter(responsefile, filename, force = True)
    Las_filterkey[filter_] = filename
#%%
fit_tbl = vstack([tbl_imsng, tbl_h22])
phase_min = 59538
phase_max = 59595
filter_key = 'BVgri'
fit_tbl_mjdcut = fit_tbl[(fit_tbl['mjd'] > phase_min) & (fit_tbl['mjd'] < phase_max)]
fit_tbl_filtercut = fit_tbl_mjdcut[[filter in filter_key for filter in fit_tbl_mjdcut['filter']]]
fit_tbl = fit_tbl_filtercut
#%%
filterkeylist = [] 
magsyslist = []
for observatory, filter_ in zip(fit_tbl['observatory'], fit_tbl['filter']):
    filter_key = filter_key_sncosmo[filter_]
    # if observatory == 'KCT':
    #     filter_key = KCT_filterkey[filter_]
    # if observatory == 'RASA36':
    #     filter_key = RASA_filterkey[filter_]
    # if (observatory == 'LasCumbres1m') & (filter_ in 'UBVRI'):
    #     filter_key = Las_filterkey[filter_]
    if filter_ in 'UBVRI':
        magsys = 'vega'
    else:
        magsys = 'ab'
    filterkeylist.append(filter_key)
    magsyslist.append(magsys)
    

fit_tbl['filter_sncosmo'] = filterkeylist
fit_tbl['magsys'] = magsyslist
show_tbl = fit_tbl
formatted_fit_tbl = helper.SNcosmo_format(fit_tbl['mjd'], fit_tbl['mag'], fit_tbl['e_mag'], fit_tbl['filter_sncosmo'], magsys = fit_tbl['magsys'], zp = 25)

# %%
# %%
def salt3_to_salt2(x1_salt3, c_salt3):
    M = np.array([[1.028, 0.138],
                [0.002, 0.985]])
    y = np.array([x1_salt3 - 0.005,
                c_salt3 - 0.013])
    x1_salt2, c_salt2 = np.linalg.solve(M, y)
    return x1_salt2, c_salt2

def draw_mu_samples(result, model, band='bessellb', magsys='ab',
                    alpha=0.145, sigma_alpha=0.007,  # Betoule et al. 2015, Table 10
                    beta=3.059,  sigma_beta=0.093,
                    M=-19.02,   sigma_M=0.03,
                    t_window=(-20, 45), nsamp=20000):
    """
    Sample the distance modulus ? = mB_max - M + alpha*x1 - beta*c with full error propagation.
    result: sncosmo fit result
    model:  sncosmo.Model used in the fit
    """
    # parameter ordering from sncosmo fit
    names = result.param_names
    cov = result.covariance  # 5x5 typically for ['z','t0','x0','x1','c']
    parameters = result.parameters
    if not 'z' in result.vparam_names:
        names = names[1:]
        parameters = parameters[1:]
    p_mean = np.array([parameters[names.index(n)] for n in names[:len(cov)]])

    rng = np.random.default_rng()
    params_samples = rng.multivariate_normal(p_mean, cov, size=nsamp)

    # priors/uncertainties for calibration constants (assumed independent)
    alpha_samp = rng.normal(alpha, sigma_alpha, size=nsamp)
    beta_samp  = rng.normal(beta,  sigma_beta,  size=nsamp)
    M_samp     = rng.normal(M,     sigma_M,     size=nsamp)

    mu = np.empty(nsamp)
    for i, ps in enumerate(params_samples):
        # set model parameters for this draw
        for n, v in zip(names, ps):
            model.set(**{n: float(v)})

        # recompute B-band light curve around maximum and take the minimum magnitude
        t0 = ps[names.index('t0')]
        t_grid = np.arange(t0 + t_window[0], t0 + t_window[1], 0.2)
        mb_grid = model.bandmag(band, magsys, t_grid)
        mb_max = np.nanmin(mb_grid)  # minimum AB mag = peak brightness

        x1_draw = ps[names.index('x1')]
        c_draw  = ps[names.index('c')]

        mu[i] = mb_max - M_samp[i] + alpha_samp[i]*x1_draw - beta_samp[i]*c_draw

    mu_mean = np.mean(mu)
    mu_std  = np.std(mu, ddof=1)
    return mu_mean, mu_std


def fit_one_table(target_tbl, group_id=[1]):
    try:
        model = sncosmo.Model(source='salt2', version = '2.4')
        import sfdmap        
        model.set(z = 0.005017)

        # --- Light curve fit ---
        result , fitted_model= sncosmo.fit_lc(
            target_tbl, model,
            ['t0', 'x0', 'x1', 'c'], #salt3
            bounds = {}
        )
        sncosmo.plot_lc(target_tbl, model=fitted_model, errors=result.errors, figtext = '', ncol = 3,  xfigsize = 10, tighten_ylim=False, color = 'black')

        # --- SALT3?SALT2 ?? ---
        x1 = result.parameters[result.param_names.index('x1')]
        c  = result.parameters[result.param_names.index('c')]
        e_x1  = result.errors['x1']
        e_c   = result.errors['c']
        
        # param_stretch = 0.98+ 0.091*x1+ 0.003*x1**2- 0.00075*x1**3
        # e_param_stretch = np.sqrt((0.091*e_x1)**2+(0.003*2*e_x1)**2+(0.00075*3*e_x1)**2)
        # delmag = 1.09- 0.161*x1+ 0.013*x1**2- 0.00130*x1**3
        # e_delmag = np.sqrt((0.161*e_x1)**2+(0.013*2*e_x1)**2+(0.00130*3*e_x1)**2)
        t_max = result.parameters[1]

        # --- Distance modulus ??? ---
        #mu_mean, mu_std = draw_mu_samples(result, fitted_model, band='bessellb')
        
        t_range = np.arange(t_max - 10, t_max + 15, 0.01)
        mag_B = fitted_model.bandmag('bessellb', 'ab', t_range)
        magB_max = np.nanmin(mag_B)
        absmag = (-19.05) - 0.141 * x1 + 3.101 * c
        mu_mean = magB_max - absmag
        mu_std = np.sqrt(0.03**2 + (np.sqrt((0.007/0.145)**2+(e_x1/x1)**2))**2 + (np.sqrt((0.093/3.059)**2+(np.abs(e_c/c))**2))**2)

        distance   = 10**((mu_mean +5)/5)
        e_distance = 10**((mu_mean+mu_std +5)/5) - distance

        return {
            "result_fit": result,
            "fitted_model": fitted_model,
            "len_group": len(group_id),
            "group_id": group_id,
            "z": result.parameters[result.param_names.index('z')],
            "x1": x1, "c": c,
            "e_x1": e_x1, "e_c": e_c,
            "mu": mu_mean, "e_mu": mu_std,
            "distance": distance, "e_distance": e_distance,
            # "s": param_stretch, "e_s": e_param_stretch,
            # 'm15': delmag, 'e_m15': e_delmag,
            't_max': t_max
        }
    except Exception as e:
        return {"group_id": group_id, "error": str(e)}
# %%
result = fit_one_table(formatted_fit_tbl)
# %%
print("Global chi^2:", result['result_fit'].chisq)
print("Global dof:", result['result_fit'].ndof)
print("Global reduced chi^2:", result['result_fit'].chisq / result['result_fit'].ndof)
#%%
result
#%%
plt.show()
# %%
