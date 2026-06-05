#%%
from ezphot.dataobjects import LightCurve
from ezphot.utils import DataBrowser
from astropy.io import ascii
from astropy.table import Table
from astropy.table import vstack
#%%
databrowser = DataBrowser('scidata')
databrowser.objname = 'NGC1566'
catalogset = databrowser.search(pattern='sub*.cat', return_type='catalog')
catalogset.select_sources(ra = 64.9725, dec= -54.948081)
#%%

#%%
lc = LightCurve(catalogset)
lc.extract_source_info(ra = 64.9725, dec= -54.948081)
#%%
lc.plt_params.figure_figsize = (6, 10)
lc.plt_params.xlim = [59525, 59535]
lc.plt_params.ylim = [20, 8]
lc.plot(ra = 64.9725, dec= -54.948081, flux_key = 'MAGSKY_APER_2', fluxerr_key = 'MAGERR_APER_2')
#%%
imsng_tbl = ascii.read('./data/SN2021aefx.ascii_fixed_width', format = 'fixed_width')
imsng_tbl = imsng_tbl[(imsng_tbl['mjd'] > 59500) & (imsng_tbl['mjd'] < 59730)]
imsng_tbl.sort('mjd')
imsng_tbl_g = imsng_tbl[imsng_tbl['filter'] == 'g']
imsng_tbl_r = imsng_tbl[imsng_tbl['filter'] == 'r']
imsng_tbl_i = imsng_tbl[imsng_tbl['filter'] == 'i']
#%%
from ezphot.helper import Helper
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
helper_ezphot = Helper()
merged_tbl = helper_ezphot.match_table(imsng_tbl_g, imsng_tbl_r, key1 = 'mjd', key2 = 'mjd')
#%%
# Data
x = merged_tbl['mjd_1']
y = merged_tbl['MAGSKY_APER_1_1'] - merged_tbl['MAGSKY_APER_1_2']

# Sort by x (important for spline fitting)
sorted_idx = np.argsort(x)
x_sorted, y_sorted = x[sorted_idx], y[sorted_idx]
x_postpeak = x[x > 59600]
y_postpeak = y[x > 59600]

# Fit spline (s is smoothing factor; smaller s = tighter fit)
spl = UnivariateSpline(x_sorted, y_sorted, s=0.07)
linear = np.polyfit(x_postpeak, y_postpeak, 1)
linear_fn = np.poly1d(linear)

# Evaluate
x_fit = np.linspace(x_sorted.min(), x_sorted.max(), 10000)
y_fit = spl(x_fit)
x_fit_postpeak = np.linspace(59600, 59650, 500)
y_fit_postpeak = linear_fn(x_fit_postpeak)
# --- Plot ---
plt.figure(figsize=(6,4))
plt.scatter(x, y, color='gray', label='Observed (g?r)')
plt.plot(x_fit, y_fit, color='red', label='Spline fit')
plt.plot(x_fit_postpeak, y_fit_postpeak, color='blue', label='Post-peak linear fit')
plt.axvline(59600, ls='--', color='black', alpha=0.3, label='Post-peak boundary')
plt.xlabel('MJD')
plt.ylabel('g ? r')
plt.legend()
plt.tight_layout()
plt.show()
#%%
from ezphot.imageobjects import ScienceImage
mag_keys = ['MAGSKY_AUTO', 'MAGSKY_APER', 'MAGSKY_APER_1', 'MAGSKY_APER_2', 'MAGSKY_APER_3', 'MAGSKY_APER_4']
corr_mag_dict = {mag: [] for mag in mag_keys}
for data in imsng_tbl_g:
    target_img = ScienceImage(data['target_img'])
    k = target_img.header['K_COLOR_APER_2_G-R']
    c = target_img.header['C_COLOR_APER_2_G-R']
    
    if target_img.mjd > 59650:
        gr = linear_fn(target_img.mjd)
    else:
        gr = spl(target_img.mjd)
    corrected_mag_offset = k * gr + c
    for mag in mag_keys:
        corr_mag_dict[mag].append(data[mag] + corrected_mag_offset)
for mag in mag_keys:
    mag_key_new = mag + '_CORR'
    imsng_tbl_g[mag_key_new] = corr_mag_dict[mag]
#%%
corr_mag_dict = {mag: [] for mag in mag_keys}
for data in imsng_tbl_r:
    target_img = ScienceImage(data['target_img'])
    k = target_img.header['K_COLOR_APER_2_G-R']
    c = target_img.header['C_COLOR_APER_2_G-R']
    
    if target_img.mjd > 59650:
        gr = linear_fn(target_img.mjd)
    else:
        gr = spl(target_img.mjd)
    corrected_mag_offset = k * gr + c
    for mag in mag_keys:
        corr_mag_dict[mag].append(data[mag] + corrected_mag_offset)
for mag in mag_keys:
    mag_key_new = mag + '_CORR'
    imsng_tbl_r[mag_key_new] = corr_mag_dict[mag]
#%%
corr_mag_dict = {mag: [] for mag in mag_keys}
for data in imsng_tbl_i:
    target_img = ScienceImage(data['target_img'])
    k = target_img.header['K_COLOR_APER_2_G-R']
    c = target_img.header['C_COLOR_APER_2_G-R']
    
    if target_img.mjd > 59650:
        gr = linear_fn(target_img.mjd)
    else:
        gr = spl(target_img.mjd)
    corrected_mag_offset = k * gr + c
    for mag in mag_keys:
        corr_mag_dict[mag].append(data[mag] + corrected_mag_offset)
for mag in mag_keys:
    mag_key_new = mag + '_CORR'
    imsng_tbl_i[mag_key_new] = corr_mag_dict[mag]
#%%
imsng_tbl = vstack([imsng_tbl_g, imsng_tbl_r, imsng_tbl_i])
imsng_tbl.sort(['observatory','filter', 'mjd'])
imsng_tbl.write('./data/SN2021aefx_corrected.ascii_fixed_width', format= 'ascii.fixed_width', overwrite=  True)
#%%
tbl = ascii.read('./data/SN2021aefx.ascii_fixed_width', format = 'fixed_width')
tbl_corrected = ascii.read('./data/SN2021aefx_corrected.ascii_fixed_width', format = 'fixed_width')
#%%

# %%
lc = LightCurve()
lc.data = tbl
lc.plt_params.figure_figsize = (14, 10)
lc.plt_params.xlim= [59540, 59560]
lc.plt_params.xlim= [59500, 59730]
lc.plt_params.ylim = [12.3, 11.7]
lc.plt_params.ylim = [21, 11]
lc.plot(ra = 64.9725, dec= -54.948081, flux_key = 'MAGSKY_APER_2', fluxerr_key = 'MAGERR_APER_2')
#%%
lc_corr = LightCurve()
lc_corr.data = tbl_corrected
lc_corr.plt_params.figure_figsize = (14, 10)
lc_corr.plt_params.xlim= [59540, 59560]
lc_corr.plt_params.xlim= [59500, 59730]
lc_corr.plt_params.ylim = [12.3, 11.7]
lc_corr.plt_params.ylim = [21, 11]
lc_corr.plot(ra = 64.9725, dec= -54.948081, flux_key = 'MAGSKY_APER_2_CORR', fluxerr_key = 'MAGERR_APER_2')
# %%
lc.data.sort(['observatory','filter', 'mjd'])
lc.data.write('./data/SN2021aefx.ascii_fixed_width', format= 'ascii.fixed_width', overwrite=  True)
# %%
from ezphot.utils import DataBrowser
from ezphot.dataobjects import LightCurve
from astropy.io import ascii
tbl = ascii.read('./data/SN2021aefx_corrected.ascii_fixed_width', format = 'fixed_width')
tbl_h22 = ascii.read('./data/Hosseinzadeh2022.dat', format = 'fixed_width')
lc = LightCurve()
lc.data = tbl
# %%
lc.plt_params.figure_figsize = (12,8)
lc.plt_params.xlim= [59500, 59720]
lc.plt_params.ylim = [20, 11]
lc.plot(ra = 64.9725, dec= -54.948081, flux_key = 'MAGSKY_APER_2', fluxerr_key = 'MAGERR_APER_2')
#%%
from astropy.table import Table
import numpy as np
tbl_formatted = Table()
tbl_formatted['mjd'] = tbl['mjd']
tbl_formatted['mag'] = tbl['MAGSKY_APER_2_CORR']
tbl_formatted['e_mag'] = np.sqrt(tbl['MAGERR_APER_2']**2 + tbl['ZPERR_APER_2']**2)
tbl_formatted['magsys'] = 'AB'
tbl_formatted['filter'] = tbl['filter']
tbl_formatted['depth'] = tbl['depth']
tbl_formatted['observatory'] = tbl['observatory']
tbl_formatted['detected'] = [True if mag > 0 else False for mag in tbl_formatted['mag']]
tbl_formatted.write('./data/SN2021aefx_formatted.ascii_fixed_width', format = 'ascii.fixed_width', overwrite = True)
# %%
tbl_h22_formatted = Table()
tbl_h22_formatted['mjd'] = tbl_h22['obsdate']
tbl_h22_formatted['mag'] = tbl_h22['mag']
tbl_h22_formatted['e_mag'] = tbl_h22['e_mag']
tbl_h22_formatted['magsys'] = tbl_h22['magsys']
tbl_h22_formatted['filter'] = tbl_h22['filter']
tbl_h22_formatted['depth'] = tbl_h22['depth_5sig']
tbl_h22_formatted['zp'] = tbl_h22['zp']
tbl_h22_formatted['observatory'] = tbl_h22['observatory']
tbl_h22_formatted['detected'] = tbl_h22['detected']
tbl_h22_formatted.write('./data/Hosseinzadeh2022_formatted.ascii_fixed_width', format = 'ascii.fixed_width', overwrite = True)
# %%
tbl_formatted_imsng = ascii.read('./data/SN2021aefx_formatted.ascii_fixed_width', format = 'fixed_width')
tbl_formatted_h22 = ascii.read('./data/Hosseinzadeh2022_formatted.ascii_fixed_width', format = 'fixed_width')
tbl_formatted_h22 = tbl_formatted_h22[tbl_formatted_h22['observatory'] == 'LasCumbres1m']
tbl_formatted_all = vstack([tbl_formatted_imsng, tbl_formatted_h22])
lc = LightCurve()
lc.data = tbl_formatted_all
lc.plt_params.figure_figsize = (6, 8)
lc.plt_params.xlim = [59525, 59650]
lc.plt_params.ylim = [20, 8]
lc.plot(ra = 64.9725, dec= -54.948081, flux_key  = 'mag', fluxerr_key = 'e_mag')
# %%
