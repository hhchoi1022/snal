
#%%
from astropy.io import ascii
from snal.helper import AnalysisHelper
import numpy as np
from snal.model import CompanionInteractionK10
from lmfit import minimize, Parameters
from astropy.table import vstack
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
lc_imsng.plot(ra = 64.9725, dec= -54.948081, flux_key = 'mag', fluxerr_key = 'e_mag')
lc_h22 = LightCurve()
lc_h22.data = tbl_h22
lc_h22.plt_params.xlim = [59500, 59730]
lc_h22.plt_params.ylim = [20, 8]
lc_h22.plt_params.figure_figsize = (12, 8)
lc_h22.plot(ra = 64.9725, dec= -54.948081, flux_key = 'mag', fluxerr_key = 'e_mag')
#%%
from astropy.table import vstack
lc_all = LightCurve()
lc_all.data = tbl_all
lc_all.plt_params.xlim = [59500, 59730]
lc_all.plt_params.ylim = [20, 8]
lc_all.plt_params.figure_figsize = (12, 8)
lc_all.plot(ra = 64.9725, dec= -54.948081, flux_key = 'mag', fluxerr_key = 'e_mag')
#%%
lc_all.plt_params.figure_figsize = (6, 8)
lc_all.plt_params.xlim = [59526, 59538]
fig, ax, _ = lc_all.plot(ra = 64.9725, dec= -54.948081, flux_key = 'mag', fluxerr_key = 'e_mag')
#%%

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

def get_CEI_spline(model_CEI,
                   exptime_CEI,
                   filterset : str = 'BVgri',
                   smooth : float = 0.05):
    spl_dict = dict()
    for filter_ in filterset:
        model_mag = model_CEI[filter_]
        inf_idx = np.isinf(model_mag)
        mag_CEI = model_mag[~inf_idx]
        phase_CEI = model_CEI['phase'][~inf_idx]
        spl, _ = helper.interpolate_spline(phase_CEI + exptime_CEI, mag_CEI, show = False, smooth = smooth)
        spl_dict[filter_] = spl
    return spl_dict
def fit_both(fit_tbl,
             rstar,
             m_wd,
             v9 : float = 1.0,
             fit_method : str = 'leastsq'
             ):

    def chisq_both(params, x_fit, y_fit, e_y_fit, filter_key, model_CEI):
        vals = params.valuesdict()
        exptime_CEI = vals['exptime_CEI']
        exptime_FB = vals['exptime_FB']
        chisq_allfilter = []
        spl_allfilt_CEI = get_CEI_spline(model_CEI = model_CEI, exptime_CEI = exptime_CEI, filterset = filter_key)
        for mjd, obs_flux, obs_fluxerr, filter_ in zip(x_fit, y_fit, e_y_fit, filter_key):
            spl_CEI = spl_allfilt_CEI[filter_]
            fireball_alpha = vals[f'alpha_{filter_}']
            fireball_amplitude = vals[f'amplitude_{filter_}']
            fireball_flux = fireball_model(time = mjd, amplitude = fireball_amplitude, alpha = fireball_alpha, exptime = exptime_FB)
            CEI_flux = helper.mag_to_flux(spl_CEI(mjd)+DM)
            both_flux = fireball_flux + CEI_flux
            chisq_singlefilter = (obs_flux - both_flux) / obs_fluxerr
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
    fit_params_CEI_FB = Parameters()
    fit_params_CEI_FB.add('exptime_CEI', value = 59528.5, min = 59525, max = 59535)
    fit_params_CEI_FB.add('exptime_FB', value = 59528.5, min = 59525, max = 59535)
    for filter_ in filter_key:
        fit_params_CEI_FB.add(f'alpha_{filter_}', value = 2, min = 0, max = 4)
        fit_params_CEI_FB.add(f'amplitude_{filter_}', value = 3000, min = 1, max = 500000)
    
    # Fitting
    t_range = np.arange(0.1, 10, 0.1)
    Comp_model = CompanionInteractionK10(rstar = rstar, m_wd = m_wd, v9 = v9)
    model_CEI = Comp_model.get_LC(td = t_range, filterset = ''.join(filter_key), search_directory = model_directory, save = True)
    out = minimize(chisq_both, fit_params_CEI_FB, args = (x_fit, y_fit, e_y_fit, filter_key, model_CEI), method = fit_method)
    return out

#%% FITTING

model_directory = '/home/hhchoi1022/snal/model/comp_model'
result_directory = '/home/hhchoi1022/snal/result/SN2021aefx/comp_fit_result'
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
range_rstar = np.round(np.arange(0.5, 30, 0.05),2)
range_m_wd =  np.round(np.arange(1.0, 1.45, 0.05), 2)
range_v9 = np.round(np.arange(0.7, 1.45, 0.05), 2)
#%%
def process_combination(args):
    rstar, m_wd, v9, fit_tbl = args
    header_parameters =['rstar','m_wd','v9']
    header_fitvalues = ['exptime_CEI', 'exptime_FB']
    header_fitconfig = ['success','nfev', 'ndata', 'nvar', 'chisq', 'redchisqr', 'aic', 'bic']
    for filter_ in fit_filterset:
        header_fitvalues.append(f'alpha_{filter_}')
        header_fitvalues.append(f'amplitude_{filter_}')
    tot_header = header_parameters + header_fitvalues + header_fitconfig
    result_tbl = Table(names = tot_header)
    result_tbl.add_row(vals = np.zeros(len(result_tbl.colnames)))
    if os.path.exists(f'{result_directory}/M%.2f/%.2f_%.2f_%.2f.fit'%(m_wd, rstar, m_wd, v9)):
        return
    try:
        result = fit_both(fit_tbl=fit_tbl,
                          rstar = rstar,
                          m_wd = m_wd,
                          v9 = v9,
                          fit_method='leastsq'
                          )
        data_parameters = dict(rstar = rstar, m_wd = m_wd, v9 = v9)
        data_fitvalues = result.params.valuesdict()
        data_fitconfig = dict(success = result.success, nfev = result.nfev, ndata = result.ndata, nvar = result.nvarys, chisq = result.chisqr, redchisqr = result.redchi, aic = result.aic, bic = result.bic)
        all_data = {**data_parameters, **data_fitvalues, **data_fitconfig}
        all_values = [all_data[name] for name in result_tbl.colnames]
        result_tbl.add_row(vals=all_values)
    except:
        data_parameters = dict(rstar = rstar, m_wd = m_wd, v9 = v9)
        data_fitvalues = {value : 99999 for value in header_fitvalues}
        data_fitconfig = dict(success = False, nfev = 99999, ndata = 99999, nvar = 99999, chisq = 99999, redchisqr = 99999, aic = 99999, bic = 99999)
        all_data = {**data_parameters, **data_fitvalues, **data_fitconfig}
        all_values = [all_data[name] for name in result_tbl.colnames]
        result_tbl.add_row(vals=all_values)
    #os.makedirs(f'/data7/yunyi/temp_supernova/result/Comp_fit_result/M{m_wd}', exist_ok = True)
    os.makedirs(f'{result_directory}/M%.2f'%m_wd, exist_ok = True)
    result_tbl.remove_row(index = 0)
    #result_tbl.write(f'/data7/yunyi/temp_supernova/result/Comp_fit_result/M{m_wd}/Rstar_{rstar}_V9_{v9}.txt', format = 'ascii.fixed_width', overwrite = True)
    result_tbl.write(f'{result_directory}/M%.2f/%.2f_%.2f_%.2f.fit'%(m_wd, rstar, m_wd, v9), format = 'ascii.fixed_width', overwrite = True)
# %%
os.makedirs(f'{result_directory}', exist_ok = True)
#os.makedirs(f'/data7/yunyi/temp_supernova/result/Comp_fit_result', exist_ok=True)
#%%
# Prepare the list of all combinations of parameters
all_combinations = [(rstar, m_wd, v9, fit_tbl)
                    for rstar in range_rstar
                    for m_wd in range_m_wd
                    for v9 in range_v9
                    ]
#%%
# Use multiprocessing to process the combinations in parallel with tqdm
from tqdm import tqdm
with mp.Pool(processes=50) as pool:
    pool.map(process_combination, tqdm(all_combinations, desc="Processing combinations"))

# %%
import glob
from astropy.table import vstack
from concurrent.futures import ProcessPoolExecutor
from astropy.table import vstack, Table
from astropy.io import ascii
from tqdm import tqdm
import glob

def read_one_file(file_):
    """Function executed in each process."""
    try:
        tbl = ascii.read(file_, format='fixed_width')
        return tbl
    except:
        return None

result_key = f'{result_directory}/*/*.fit'
files = glob.glob(result_key)
tables = []
with ProcessPoolExecutor() as executor:
    for tbl in tqdm(executor.map(read_one_file, files), total=len(files), desc="Reading files"):
        if tbl is not None:
            tables.append(tbl)

# Stack all tables
result_tbl = vstack(tables)
#%%
result_tbl.sort('redchisqr')
result_tbl.write(f'{result_directory}/comp_fit_result.fit', format = 'ascii.fixed_width', overwrite = True)
# %%
result_tbl = Table().read(f'{result_directory}/comp_fit_result.fit', format = 'ascii.fixed_width')

lc_fig = LightCurve()
lc_fig.FILTER_OFFSET['U'] = -3
lc_fig.FILTER_OFFSET['B'] = -2
lc_fig.FILTER_OFFSET['V'] = -1
lc_fig.data = tbl_all
lc_fig.plt_params.xlim = [59526, 59540]
lc_fig.plt_params.ylim = [22.5, 8]
lc_fig.plt_params.figure_figsize = (6, 8)
fig, ax, _ = lc_fig.plot(ra = 64.9725, dec= -54.948081, flux_key = 'mag', fluxerr_key = 'e_mag')

#%%
i = 0
result_values = result_tbl[i]
exptime_CEI = result_values['exptime_FB']
exptime_FB = result_values['exptime_CEI']
filter_key = fit_tbl.group_by('filter').groups.keys['filter']

phase_min_FB = np.max([59526, result_values['exptime_FB']])
phase_min_CEI = np.max([59526, result_values['exptime_CEI']])
phase_min_CEI = 59528
phase_range_FB = np.arange(phase_min_FB, 59540, 0.1)
phase_range_CEI = np.arange(phase_min_CEI, 59540, 0.1)
phase_range_CEI = np.arange(np.min([phase_min_FB, phase_min_CEI]), 59540, 0.1)


CEI_model = CompanionInteractionK10(rstar = result_values['rstar'], m_wd = result_values['m_wd'], v9 = result_values['v9'])
CEI_LC = CEI_model.get_LC(td = np.arange(0.1, 10, 0.1), filterset = 'UBVRIugri', search_directory = model_directory, save = False, force_calculate =True)
spl_allfilt_CEI = get_CEI_spline(CEI_LC, exptime_CEI = result_values['exptime_CEI'], filterset = 'UBVRIugri')

for filter_ in filter_key:
    amp = result_values[f'amplitude_{filter_}']
    alpha= result_values[f'alpha_{filter_}']
    flux_FB = fireball_model(time = phase_range_CEI, amplitude = amp, alpha = alpha, exptime = result_values['exptime_FB'])
    spl_CEI = spl_allfilt_CEI[filter_]
    flux_CEI = helper.mag_to_flux(spl_CEI(phase_range_CEI)+DM)
    flux_both = flux_FB+ flux_CEI
    mag_model = helper.flux_to_mag(flux_FB, zp = 25)
    mag_DOM = helper.flux_to_mag(flux_CEI, zp = 25)
    mag_both = helper.flux_to_mag(flux_both, zp = 25)
    ax.plot(phase_range_CEI, mag_model + lc_fig.FILTER_OFFSET[filter_], c =lc_fig.FILTER_COLOR[filter_], label = rf'[{filter_} + {lc_fig.FILTER_OFFSET[filter_]}] $\alpha = {round(alpha,2)}$', linestyle= ':', linewidth = 1, alpha = 0.4)
    ax.plot(phase_range_CEI, mag_DOM + lc_fig.FILTER_OFFSET[filter_], c = lc_fig.FILTER_COLOR[filter_], linestyle= '--', linewidth = 1, alpha = 0.4)
    ax.plot(phase_range_CEI, mag_both + lc_fig.FILTER_OFFSET[filter_], c = lc_fig.FILTER_COLOR[filter_], linestyle= '-', linewidth = 1, alpha = 1)
    # For color plot 
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
spl_CEI = spl_allfilt_CEI['U']
flux_CEI = helper.mag_to_flux(spl_CEI(phase_range_CEI)+DM)
flux_both = flux_CEI
mag_both = helper.flux_to_mag(flux_both, zp = 25)
ax.plot(phase_range_CEI, mag_both + lc_fig.FILTER_OFFSET['U'], c = lc_fig.FILTER_COLOR['U'], linestyle= '--', linewidth = 1, alpha = 1)
#%%
fig
# %%

# %%
