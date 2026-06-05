
#%%
from typing import Any
from astropy.io import ascii
from snal.helper import AnalysisHelper
import numpy as np
from snal.model import DOMInteractionL17
from lmfit import minimize, Parameters
from astropy.table import vstack
import matplotlib.pyplot as plt
from snal.helper import ABVegaMagnitude
import matplotlib
#%matplotlib inline 
matplotlib.use('Agg')

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
# mag = ABVegaMagnitude(tbl_all['mag'], magsys = tbl_all['magsys'], filter_ = tbl_all['filter'])
# tbl_all['mag'] = mag.AB
# tbl_all['magsys'] = 'AB'
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
import numpy as np
import os
import time
import multiprocessing as mp
from astropy.table import Table
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
range_f_dom = np.round([0.1], 1)  # fraction of DOM mass, rounded to 2 decimal places  # fraction of DOM mass, rounded to 2 decimal placestd = np.arange(0.1, 10, 0.1)
len(range_E_exp) * len(range_M_ej) * len(range_kappa) * len(range_t_delay) * len(range_f_comp) * len(range_M_dom) * len(range_v_dom) * len(range_f_dom)
#%%
def process_combination(args):
    E_exp, M_ej, kappa, t_delay, f_comp, M_dom, V_dom, f_dom, fit_tbl = args
    header_parameters = ['E_exp','M_ej','kappa','t_delay','f_comp','M_dom','V_dom','f_dom']
    header_fitvalues = ['exptime_DEI', 'exptime_FB']
    header_fitconfig = ['success','nfev', 'ndata', 'nvar', 'chisq', 'redchisqr', 'aic', 'bic']
    for filter_ in fit_filterset:
        header_fitvalues.append(f'alpha_{filter_}')
        header_fitvalues.append(f'amplitude_{filter_}')
    tot_header = header_parameters + header_fitvalues + header_fitconfig
    result_tbl = Table(names = tot_header)
    result_tbl.add_row(vals = np.zeros(len(result_tbl.colnames)))
    if os.path.exists(f'{result_directory}/kappa{kappa}/E{E_exp}/{E_exp}_{M_ej}_{kappa}_{t_delay}_{f_comp}_{M_dom}_{V_dom}_{f_dom}.fit'):
        return
    try:
        result = fit_both(
                          fit_tbl=fit_tbl,
                          E_exp=E_exp,
                          M_ej=M_ej,
                          kappa=kappa,
                          M_dom=M_dom,
                          V_dom=V_dom,
                          f_dom=f_dom,
                          t_delay=t_delay,
                          f_comp=f_comp,
                          fit_method='leastsq'
                          )
        data_parameters = dict(E_exp=E_exp, M_ej=M_ej, kappa=kappa, t_delay=t_delay, f_comp=f_comp, M_dom=M_dom, V_dom=V_dom, f_dom=f_dom)
        data_fitvalues = result.params.valuesdict()
        data_fitconfig = dict(success=result.success, nfev=result.nfev, ndata=result.ndata, nvar=result.nvarys, chisq=result.chisqr, redchisqr=result.redchi, aic=result.aic, bic=result.bic)
        all_data = {**data_parameters, **data_fitvalues, **data_fitconfig}
        all_values = [all_data[colname] for colname in result_tbl.columns]
        result_tbl.add_row(vals=all_values)
    except:
        data_parameters = dict(E_exp=E_exp, M_ej=M_ej, kappa=kappa, t_delay=t_delay, f_comp=f_comp, M_dom=M_dom, V_dom=V_dom, f_dom=f_dom)
        data_fitvalues = {value: 99999 for value in header_fitvalues}
        data_fitconfig = dict(success=False, nfev=99999, ndata=99999, nvar=99999, chisq=99999, redchisqr=99999, aic=99999, bic=99999)
        all_data = {**data_parameters, **data_fitvalues, **data_fitconfig}
        all_values = [all_data[colname] for colname in result_tbl.columns]
        result_tbl.add_row(vals=all_values)
    os.makedirs(f'{result_directory}/kappa{kappa}/E{E_exp}', exist_ok = True)
    result_tbl.remove_row(index = 0)
    result_tbl.write(f'{result_directory}/kappa{kappa}/E{E_exp}/{E_exp}_{M_ej}_{kappa}_{t_delay}_{f_comp}_{M_dom}_{V_dom}_{f_dom}.fit', format='ascii.fixed_width', overwrite=True)
#%%
# Prepare the list of all combinations of parameters
all_combinations = [(E_exp, M_ej, kappa, t_delay, f_comp, M_dom, V_dom, f_dom, fit_tbl)
                    for E_exp in range_E_exp
                    for M_ej in range_M_ej
                    for kappa in range_kappa
                    for t_delay in range_t_delay
                    for f_comp in range_f_comp
                    for M_dom in range_M_dom
                    for V_dom in range_v_dom
                    for f_dom in range_f_dom]
#%%
# Use multiprocessing to process the combinations in parallel with tqdm
from tqdm import tqdm
with mp.Pool(processes=50) as pool:
    pool.map(process_combination, tqdm(all_combinations, desc="Processing combinations"))

# %% Read result with multiprocessing
from concurrent.futures import ProcessPoolExecutor
from astropy.table import vstack, Table
from astropy.io import ascii
from tqdm import tqdm
import glob

def construct_file_format(E_exp, M_ej, kappa, t_delay, f_comp, M_dom, V_dom, f_dom):
    return f'{result_directory}/kappa{kappa}/E{E_exp}/{E_exp}_{M_ej}_{kappa}_{t_delay}_{f_comp}_{M_dom}_{V_dom}_{f_dom}.fit'

def read_one_file(file_):
    """Function executed in each process."""
    try:
        tbl = ascii.read(file_, format='fixed_width')
        return tbl
    except:
        return None

# Collect file list
files = [construct_file_format(E_exp, M_ej, kappa, t_delay, f_comp, M_dom, V_dom, f_dom)
         for E_exp in range_E_exp
         for M_ej in range_M_ej
         for kappa in range_kappa
         for t_delay in range_t_delay
         for f_comp in range_f_comp
         for M_dom in range_M_dom
         for V_dom in range_v_dom
         for f_dom in range_f_dom]

#files = glob.glob(f'{result_directory}/*/*/*.fit')
#%%
# Read in parallel
tables = []
with ProcessPoolExecutor(60) as executor:
    for tbl in tqdm(executor.map(read_one_file, files), total=len(files), desc="Reading files"):
        if tbl is not None:
            tables.append(tbl)
#%%
# Stack all tables
result_tbl = vstack(tables)

# Sort and save
result_tbl.sort('redchisqr')
result_tbl.write(f'{result_directory}/DOM_fit_result.fit',
                 format='ascii.fixed_width',
                 overwrite=True)
# %%

result_tbl = Table().read(f'{result_directory}/DOM_fit_result.fit', format = 'ascii.fixed_width')
#%%
result_tbl_selected = result_tbl[(result_tbl['M_ej'] == 1.5) & (result_tbl['f_dom'] < 0.15)]# & (result_tbl['V_dom'] == 4700) ]
#result_tbl_selected = result_tbl
#result_tbl_selected = result_tbl[(result_tbl['E_exp'] == 1.4) & (result_tbl['M_dom'] < 0.1) & (result_tbl['t_delay'] == 100) & (result_tbl['V_dom'] > 4500) ]
#%%
lc_fig = LightCurve()

lc_fig.data = tbl_all
lc_fig.FILTER_OFFSET['U'] = -3
lc_fig.FILTER_OFFSET['B'] = -2
lc_fig.FILTER_OFFSET['V'] = -1
lc_fig.plt_params.xlim = [59526, 59540]
lc_fig.plt_params.ylim = [22.5, 8]
lc_fig.plt_params.figure_figsize = (6, 8)
fig, ax, _ = lc_fig.plot(ra = 64.9725, dec= -54.948081, flux_key = 'mag', fluxerr_key = 'e_mag')

#%%
import matplotlib.pyplot as plt
i = 0
result_values = result_tbl_selected[i]
exptime_DEI = result_values['exptime_DEI']
exptime_FB = result_values['exptime_FB']
filter_key = fit_tbl.group_by('filter').groups.keys['filter']
#%%
phase_min_FB = np.max([59526, result_values['exptime_FB']])
phase_min_DEI = np.max([59526, result_values['exptime_DEI']])
phase_range_FB = np.arange(phase_min_FB, 59540, 0.1)
phase_range_DEI = np.arange(phase_min_DEI, 59540, 0.1)

DOM_model = DOMInteractionL17(E_exp = result_values['E_exp'], M_ej = result_values['M_ej'], kappa = result_values['kappa'], t_delay = result_values['t_delay'], f_comp = result_values['f_comp'], M_dom = result_values['M_dom'], V_dom = result_values['V_dom'], f_dom = result_values['f_dom'])
DOM_LC = DOM_model.get_LC(td = np.arange(0.01, 10, 0.01), filterset  = 'UBVRIugri', search_directory = model_directory, save = True, force_calculate= True)
spl_allfilt_DEI = get_DEI_spline(DOM_LC, exptime_DEI = result_values['exptime_DEI'], filterset = 'UBVRIugri')

for filter_ in filter_key:
    amp = result_values[f'amplitude_{filter_}']
    alpha= result_values[f'alpha_{filter_}']
    flux_FB = fireball_model(time = phase_range_DEI, amplitude = amp, alpha = alpha, exptime = result_values['exptime_FB'])
    spl_DEI = spl_allfilt_DEI[filter_]
    flux_DEI = helper.mag_to_flux(spl_DEI(phase_range_DEI)+DM)
    flux_both = flux_FB+ flux_DEI
    mag_model = helper.flux_to_mag(flux_FB, zp = 25)
    mag_DOM = helper.flux_to_mag(flux_DEI, zp = 25)
    mag_both = helper.flux_to_mag(flux_both, zp = 25)
    ax.plot(phase_range_DEI, mag_model + lc_fig.FILTER_OFFSET[filter_], c = lc_fig.FILTER_COLOR[filter_], label = rf'[{filter_} + {lc_fig.FILTER_OFFSET[filter_]}] $\alpha = {round(alpha,2)}$', linestyle= ':', linewidth = 1)
    ax.plot(phase_range_DEI, mag_DOM + lc_fig.FILTER_OFFSET[filter_], c = lc_fig.FILTER_COLOR[filter_], linestyle= '--', linewidth = 1)
    ax.plot(phase_range_DEI, mag_both + lc_fig.FILTER_OFFSET[filter_], c = lc_fig.FILTER_COLOR[filter_], linestyle= '-', linewidth = 1)
    if filter_ == 'U':
        mag_U_model = mag_model
        mag_U_CEI = mag_DOM
        mag_U_both = mag_both
    if filter_ == 'B':
        mag_B_model = mag_model
        mag_B_CEI = mag_DOM
        mag_B_both = mag_both
    if filter_ == 'V':
        mag_V_model = mag_model
        mag_V_CEI = mag_DOM
        mag_V_both = mag_both
    if filter_ == 'g':
        mag_g_model = mag_model
        mag_g_CEI = mag_DOM
        mag_g_both = mag_both 
    if filter_ == 'r':
        mag_r_model = mag_model
        mag_r_CEI = mag_DOM
        mag_r_both = mag_both
spl_DEI = spl_allfilt_DEI['U']
flux_DEI = helper.mag_to_flux(spl_DEI(phase_range_DEI)+DM)
flux_both = flux_DEI
mag_both = helper.flux_to_mag(flux_both, zp = 25)
ax.plot(phase_range_DEI, mag_both + lc_fig.FILTER_OFFSET['U'], c = lc_fig.FILTER_COLOR['U'], linestyle= '--', linewidth = 1, alpha = 1)
#
#plt.legend(loc = 4)
#observed_data.show_lightcurve( day_binsize = 5, color_BV = False, color_gr = False, color_UB = False, UL = True, label = False, label_location=2, scatter_size= 120)
#observed_data.show_lightcurve( day_binsize = 5, color_BV = True, color_gr = True, color_UB = True, UL = True, label = False, label_location=2, scatter_size= 120)

# #plt.plot(phase_range_DEI, mag_U_model - mag_B_model -0.5, c = 'cyan', label = 'U-B', linestyle= '--', linewidth = 1)
# plt.plot(phase_range_DEI, mag_B_model - mag_V_model + 0.5, c = 'b', label = 'B-V', linestyle= '--', linewidth = 1)
# plt.plot(phase_range_DEI, mag_g_model - mag_r_model, c = 'g', label = 'g-r', linestyle= '--', linewidth = 1)
# #plt.plot(phase_range_DEI, mag_U_CEI - mag_B_CEI -0.5, c = 'cyan', label = 'U-B', linestyle= ':', linewidth = 1)
# plt.plot(phase_range_DEI, mag_B_CEI - mag_V_CEI +0.5, c = 'b', label = 'B-V', linestyle= ':', linewidth = 1)
# plt.plot(phase_range_DEI, mag_g_CEI - mag_r_CEI, c = 'g', label = 'g-r', linestyle= ':', linewidth = 1)
# #plt.plot(phase_range_DEI, mag_U_both - mag_B_both-0.5, c = 'cyan', label = 'U-B', linestyle= '-', linewidth = 1)
# plt.plot(phase_range_DEI, mag_B_both - mag_V_both+0.5, c = 'b', label = 'B-V', linestyle= '-', linewidth = 1)
# plt.plot(phase_range_DEI, mag_g_both - mag_r_both, c = 'g', label = 'g-r', linestyle= '-', linewidth = 1)
# #plt.ylim(-1, 1.7)
# ax1.set_xlim(phase_range_FB[0]-1, 59537)
# ax2.set_xlim(phase_range_FB[0]-1, 59537)
# ax1.set_ylim(22.5, 8)
# # %%

# %%
fig
#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
param = 't_delay'
values = np.unique(result_tbl[param])

medians = []
q1s = []
q3s = []
bests = []
for v in values:
    subset = result_tbl[result_tbl[param] == v]['redchisqr']
    medians.append(np.median(subset))
    bests.append(np.min(subset))
    q1s.append(np.percentile(subset, 25))
    q3s.append(np.percentile(subset, 75))

plt.figure(figsize=(6,4))
plt.plot(values, bests, '-o', label = 'best', c = 'red')
plt.plot(values, medians, '-o', label=f'median = {np.median(medians):.2f}')
plt.fill_between(values, q1s, q3s, alpha=0.3, label='Q1?Q3')
plt.xlabel(param)
plt.ylabel('redchisqr')
plt.ylim(7, 15)
plt.legend()
plt.show()
#%%

import numpy as np
import pandas as pd
from astropy.table import Table
result_tbl_selected = result_tbl[(result_tbl['M_ej'] == 1.5)]# & (result_tbl['M_dom']== 0.05) ]# & (result_tbl['M_dom'] < 0.1) ]#& (result_tbl['f_dom'] == 0.14)]# & (result_tbl['V_dom'] == 4700) ]

# astropy.Table -> pandas.DataFrame
df = result_tbl_selected.to_pandas()

# ?? ???? ??
phys_params = ['E_exp','M_ej','kappa','t_delay','f_comp','M_dom','V_dom','f_dom']

# "?? ?" ??: ?? 10% (??) + ???? min*1.2 ??? ???
min_chi = df['redchisqr'].min()
df_topN = df.nsmallest(int(len(df)*0.05), 'redchisqr').copy()

# ?? ??? ?? (??? ?? 10%)
df_good = df_topN.reset_index(drop=True)
df_good = df[df['redchisqr'] < 20]
print("All / good:", len(df), len(df_good))

# %%
# ???(??), ????(??·??)
corr_pearson  = df_good[phys_params].corr(method='pearson')
corr_spearman = df_good[phys_params].corr(method='spearman')

print("== Pearson ==");  print(corr_pearson.round(2))
print("\n== Spearman =="); print(corr_spearman.round(2))

# ??? ??? (matplotlib? ??)
import numpy as np
import matplotlib.pyplot as plt

def plot_corr(cmat, title):
    plt.figure(figsize=(6,5))
    im = plt.imshow(cmat, vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(phys_params)), phys_params, rotation=45, ha='right')
    plt.yticks(range(len(phys_params)), phys_params)
    plt.title(title)
    # ? ?? ??
    for i in range(len(phys_params)):
        for j in range(len(phys_params)):
            plt.text(j, i, f"{cmat.iloc[i, j]:.2f}",
                     ha="center", va="center", fontsize=8, color="w" if abs(cmat.iloc[i, j])>0.5 else "k")
    plt.tight_layout(); plt.show()

plot_corr(corr_pearson,  "Correlation (Pearson, good fits)")
plot_corr(corr_spearman, "Correlation (Spearman, good fits)")


# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt

def plot_chisq_contour(df, xpar, ypar, zpar='redchisqr', bins=30):
    # 2D grid ??
    x, y, z = df[xpar], df[ypar], df[zpar]
    xi = np.linspace(x.min(), x.max(), bins)
    yi = np.linspace(y.min(), y.max(), bins)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method='linear')

    plt.figure(figsize=(5,4))
    c = plt.contourf(Xi, Yi, Zi, levels=np.linspace(8, 30, 101),
                     cmap='viridis_r')
    plt.colorbar(c, label='Reduced $\chi^2$ (bright = good fit)')
    plt.xlabel(xpar)
    plt.ylabel(ypar)
    plt.title(f'{xpar} vs {ypar} (?² map, range 10?20)')
    plt.tight_layout()
    plt.show()
plot_chisq_contour(df, 't_delay', 'f_dom')
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm
from scipy.stats import binned_statistic_2d


xpar = 't_delay'
ypar = 'M_dom'
zpar = 'redchisqr'
bins = 100

x, y, z = df_good[xpar], df_good[ypar], df_good[zpar]
xi = np.linspace(x.min(), x.max(), bins)
yi = np.linspace(y.min(), y.max(), bins)
Xi, Yi = np.meshgrid(xi, yi)
# Zi = griddata((x, y), z, (Xi, Yi), method='linear')
Zi_linear = griddata((x, y), z, (Xi, Yi), method='linear')
Zi_linear = np.clip(Zi_linear, a_min=8, a_max=16)
# set colormap for masked values
cmap = plt.cm.viridis_r.copy()
cmap.set_bad(color='black')   # choose any color (e.g. gray, black, or extend colormap)

plt.figure(figsize=(5,4))
# mask NaNs to use cmap.set_bad
masked_Zi = np.ma.masked_invalid(Zi_linear)
c = plt.contourf(
    Xi, Yi, Zi_linear,
    levels=np.linspace(8, 30, 200),
    cmap=cmap, vmin = 9, vmax = 15
)
plt.colorbar(c, label='Reduced $\chi^2$ (bright = good fit)')
plt.xlabel(xpar)
plt.ylabel(ypar)
#plt.xlim(80, 800)
plt.title(f'{xpar} vs {ypar} (?² map, range 10?20)')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.show()



# %%
