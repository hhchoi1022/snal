
#%%
from astropy.io import ascii
from snal.helper import AnalysisHelper
import numpy as np
from snal.model import DOMInteractionL17
from lmfit import minimize, Parameters
from astropy.table import vstack
import matplotlib.pyplot as plt
from snal.helper import ABVegaMagnitude
#%matplotlib inline 


helper = AnalysisHelper()
path_imsng = './data/SN2021aefx_formatted_Host_dereddening_MW_dereddening.ascii_fixed_width'
path_h22 = './data/Hosseinzadeh2022_formatted_Host_dereddening_MW_dereddening.ascii_fixed_width'

tbl_imsng = ascii.read(path_imsng, format = 'fixed_width')
tbl_h22 = ascii.read(path_h22, format = 'fixed_width')
tbl_h22 = tbl_h22[tbl_h22['observatory'] == 'LasCumbres1m']
tbl_all = vstack([tbl_imsng, tbl_h22])
DM = 31.133854 # From 3.SNcosmo.py
#%%
# # CONVERT VEGA TO AB MAGNITUDE
# from snal.helper import ABVegaMagnitude
mag = ABVegaMagnitude(tbl_all['mag'], magsys = tbl_all['magsys'], filter_ = tbl_all['filter'])
tbl_all['mag'] = mag.AB
tbl_all['magsys'] = 'AB'
#%%
from ezphot.dataobjects import LightCurve
lc_imsng = LightCurve()
lc_imsng.data = tbl_imsng
lc_imsng.plt_params.xlim = [59500, 59730]
lc_imsng.plt_params.ylim = [20, 8]
lc_imsng.plt_params.figure_figsize = (12, 8)
fig, ax, _= lc_imsng.plot(ra = 64.9725, dec= -54.948081, flux_key = 'mag', fluxerr_key = 'e_mag')
#%%
from astropy.table import vstack
lc_all = LightCurve()
lc_all.data = tbl_all
lc_all.plt_params.xlim = [59500, 59730]
lc_all.plt_params.ylim = [20, 8]
lc_all.plt_params.figure_figsize = (12, 8)
lc_all.plot(ra = 64.9725, dec= -54.948081, flux_key = 'mag', fluxerr_key = 'e_mag')

#%%# ---- model ----
def fireball_model(time, amplitude, exptime, alpha):
    dt = np.asarray(time) - exptime
    dt = np.clip(dt, 1e-6, None)  # avoid <=0
    flux = amplitude * (dt**alpha)
    return np.nan_to_num(flux, nan=1e-6, posinf=1e6, neginf=1e-6)

def band_model(params, time, b):
    exptime = params['exptime'].value
    amp     = params[f'amp_{b}'].value
    alpha   = params[f'alpha_{b}'].value
    return fireball_model(time, amp, exptime, alpha)

def residuals(params, x_list, y_list, e_list, band_list):
    res = []
    for i, b in enumerate(band_list):
        mod = band_model(params, x_list[i], b)
        res.append((y_list[i] - mod) / e_list[i])  # unsquared residuals
    return np.concatenate(res)

def get_DEI_spline(model_DEI,
                   exptime_DEI,
                   filterset : str = 'UBVRIugri',
                   smooth : float = 0.05):
    spl_dict = dict()
    for filter_ in filterset:
        model_mag = model_DEI[filter_]
        inf_idx = np.isinf(model_mag)
        mag_DEI = model_mag[~inf_idx]
        phase_DEI = model_DEI['phase'][~inf_idx]
        spl, _ = helper.interpolate_spline(phase_DEI + exptime_DEI, mag_DEI, show = False, smooth = smooth)
        spl_dict[filter_] = spl
    return spl_dict

def fit_both(fit_tbl,
             E_exp,
             M_ej,
             kappa,
             M_dom,
             V_dom,
             f_dom,
             t_delay,
             f_comp,
             
             fit_method = 'leastsq'
             ):

    def chisq_both(params, x_fit, y_fit, e_y_fit, filter_key, model_DEI):
        vals = params.valuesdict()
        exptime_DEI = vals['exptime_DEI']
        exptime_FB = vals['exptime_FB']
        chisq_allfilter = []
        spl_allfilt_DEI = get_DEI_spline(model_DEI = model_DEI, exptime_DEI = exptime_DEI, filterset = filter_key)
        for mjd, obs_flux, obs_fluxerr, filter_ in zip(x_fit, y_fit, e_y_fit, filter_key):
            spl_DEI = spl_allfilt_DEI[filter_]
            fireball_alpha = vals[f'alpha_{filter_}']
            fireball_amplitude = vals[f'amplitude_{filter_}']
            fireball_flux = fireball_model(time = mjd, amplitude = fireball_amplitude, alpha = fireball_alpha, exptime = exptime_FB)
            DEI_flux = helper.mag_to_flux(spl_DEI(mjd)+DM)
            both_flux = fireball_flux + DEI_flux
            chisq_singlefilter = ((obs_flux - both_flux)/obs_fluxerr)
            chisq_allfilter.append(chisq_singlefilter)
        return np.concatenate(chisq_allfilter)

    # Input
    filter_tbls = fit_tbl.group_by('filter').groups
    filter_key = filter_tbls.keys['filter']
    fit_table = {filter_:filter_tbl for filter_, filter_tbl in zip(filter_key, filter_tbls)}
    x_fit = [np.array((fit_table[filter_]['mjd'].tolist())) for filter_ in filter_key]
    y_fit = [np.array((fit_table[filter_]['flux'].tolist())) for filter_ in filter_key]
    e_y_fit = [np.array((fit_table[filter_]['e_flux'].tolist())) for filter_ in filter_key]

    # Parameters
    fit_params_DEI_FB = Parameters()
    fit_params_DEI_FB.add('exptime_DEI', value = 59528.5, min = 59525, max = 59535)
    fit_params_DEI_FB.add('exptime_FB', value = 59528.5, min = 59525, max = 59535)
    for filter_ in filter_key:
        fit_params_DEI_FB.add(f'alpha_{filter_}', value = 2, min = 0, max = 4)
        fit_params_DEI_FB.add(f'amplitude_{filter_}', value = 3000, min = 1, max = 500000)
    
    # Fitting
    t_range = np.arange(0.1, 10, 0.1)
    DOM_model = DOMInteractionL17(E_exp = E_exp, M_ej = M_ej, kappa = kappa, M_dom = M_dom, V_dom = V_dom, f_dom = f_dom, t_delay = t_delay, f_comp = f_comp)
    model_DEI = DOM_model.get_LC(td = t_range, filterset = ''.join(filter_key), search_directory = model_directory, save = False)
    out = minimize(chisq_both, fit_params_DEI_FB, args = (x_fit, y_fit, e_y_fit, filter_key, model_DEI), method = fit_method)
    return out

#%% FITTING

model_directory = '/home/hhchoi1022/snal/model/DOM_model'
result_directory = '/home/hhchoi1022/snal/result/SN2021aefx/DOM_fit_result'
fit_filterset : str = 'BVgri'
fit_start_mjd : int = 59529
fit_end_mjd : int = 59538.27207782408 # Half maximum mag MJD

# Construct table for fitting
fit_idx = [filter_ in fit_filterset for filter_ in tbl_all['filter']]
fit_tbl = tbl_all[fit_idx]
fit_tbl = fit_tbl[(fit_tbl['mjd'] > fit_start_mjd)&(fit_tbl['mjd'] < fit_end_mjd)]
fit_tbl.sort('mjd')
fit_tbl['flux'] = helper.mag_to_flux(fit_tbl['mag'])
fit_tbl['e_flux'] = fit_tbl['e_mag']*helper.mag_to_flux(fit_tbl['mag'])*2.303/2.5
fit_tbl['absmag'] = (fit_tbl['mag'] - DM).round(3)

lc_fit = LightCurve()
lc_fit.data = fit_tbl
lc_fit.plt_params.xlim = [fit_start_mjd, fit_end_mjd]
lc_fit.plt_params.ylim = [20, 8]
lc_fit.plt_params.figure_figsize = (6, 8)
lc_fit.plot(ra = 64.9725, dec= -54.948081, flux_key = 'mag', fluxerr_key = 'e_mag')
#%%

import emcee
import numpy as np
import multiprocessing

# --- define log-prior ---
def log_prior(theta):
    E_exp, M_ej, log_M_dom, V_dom, log_t_delay = theta
    
    # Convert back
    M_dom = 10 ** log_M_dom
    t_delay = 10 ** log_t_delay
    
    if not (0.5 < E_exp < 2.0): return -np.inf
    if not (1.0 < M_ej < 2.0): return -np.inf
    if not (1e-4 < M_dom < 0.2): return -np.inf
    if not (1000 < V_dom < 8000): return -np.inf
    if not (10 < t_delay < 1e4): return -np.inf
    return 0.0
#%%
def log_likelihood(theta):
    # theta order: [E_exp, M_ej, log_kappa, log_M_dom, V_dom, f_dom, log_t_delay, f_comp]
    E_exp, M_ej, log_M_dom, V_dom, log_t_delay = theta
    
    # Convert logs back to linear space before passing to model
    M_dom = 10 ** log_M_dom
    t_delay = 10 ** log_t_delay

    try:
        result = fit_both(
            fit_tbl=fit_tbl,
            E_exp=E_exp, M_ej=M_ej, kappa=0.05,
            M_dom=M_dom, V_dom=V_dom, f_dom=0.1,
            t_delay=t_delay, f_comp=1.5,
            fit_method='leastsq'
        )
        chi2 = result.redchi
        if not np.isfinite(chi2):
            return -np.inf
        return -0.5 * chi2
    except Exception:
        return -np.inf
    
#%%
def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

# Initial guess (from your brute-force intuition)
initial = np.array([
    1.4,            # E_exp
    1.4,            # M_ej
    np.log10(0.01), # log_M_dom
    4000,           # V_dom
    np.log10(3000), # log_t_delay
])
ndim = len(initial)
nwalkers = 24
pos = initial + 1e-2 * np.random.randn(nwalkers, ndim)


#%%
import time
import matplotlib.pyplot as plt

# number of steps (you can increase later)
nsteps = 5000   # try 200–500 for a test run; later ~2000–5000

# --- Run with multiprocessing ---
# On Linux/macOS, "fork" is best (it shares memory cleanly)
# On Windows, use "spawn" instead of "fork"
with multiprocessing.get_context("fork").Pool(processes=64) as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior, pool=pool
    )
    
    print("Running MCMC...")
    start_time = time.time()
    
    sampler.run_mcmc(pos, nsteps, progress=True)
    
    end_time = time.time()
    print(f"MCMC finished in {(end_time - start_time)/60:.2f} minutes")
# %%
#%%
# Access the full chain
chains = sampler.get_chain()      # shape: (nsteps, nwalkers, ndim)
logprob = sampler.get_log_prob()  # log-posterior values

# quick diagnostic plot
fig, axes = plt.subplots(ndim, figsize=(10, 10), sharex=True)
for i in range(ndim):
    axes[i].plot(chains[:, :, i], "k", alpha=0.3)
    axes[i].set_ylabel(f"param {i}")
axes[-1].set_xlabel("step number")
plt.tight_layout()
plt.show()

#%%
flat_samples = sampler.get_chain(discard=50, thin=10, flat=True)
print("Number of samples:", len(flat_samples))

import corner
labels = ["E_exp","M_ej","log_M_dom","V_dom","log_t_delay"]
corner.corner(flat_samples, labels=labels, show_titles=True)
plt.show()

# %%
# %%
#%%
best_params = np.median(flat_samples, axis=0)
for name, val in zip(labels, best_params):
    print(f"{name:>12s}: {val:.4f}")

# %%
