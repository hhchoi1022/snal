

#%%
from snal.helper import ExtinctionCorrector
# %%
corrector_imsng = ExtinctionCorrector(filename = './data/SN2021aefx_formatted.ascii_fixed_width')

# %%
corrector_imsng.correct_host_extinction(ebv = 0.097, mag_key = 'mag')
corrector_imsng.correct_mw_extinction(ra = 64.9708333, dec = -54.9480556, mwRv = 3.10, mag_key= 'mag')
# %%
from ezphot.dataobjects import LightCurve
# %%
lc = LightCurve()
# %%
lc.data = corrector_imsng.corrected_data
# %%
lc.plt_params.figure_figsize = (12, 8)
lc.plt_params.xlim = [59500, 59700]
lc.plt_params.ylim = [20, 11]
figure = lc.plot(ra = 64.9708333, dec = -54.9480556, flux_key = 'mag', fluxerr_key = 'e_mag')
# %%
figure[0]
# %%

corrector_h22 = ExtinctionCorrector(filename = './data/Hosseinzadeh2022_formatted.ascii_fixed_width')
#%%
corrector_h22.correct_host_extinction(ebv = 0.097, mag_key = 'mag')
corrector_h22.correct_mw_extinction(ra = 64.9708333, dec = -54.9480556, mwRv = 3.10, mag_key= 'mag')
# %%
lc = LightCurve()
lc.data = corrector_h22.corrected_data[corrector_h22.corrected_data['observatory'] == 'LasCumbres1m']
# %%
lc.plt_params.xlim = [59500, 59720]
lc.plt_params.ylim = [20, 8]
lc.plt_params.figure_figsize = (12, 8)
figure = lc.plot(ra = 64.9708333, dec = -54.9480556, flux_key = 'mag', fluxerr_key = 'e_mag')
# %%
corrector_imsng.save()
corrector_h22.save()
# %%
from astropy.io import ascii
from astropy.table import vstack
tbl_h22_corrected = ascii.read('./data/Hosseinzadeh2022_formatted_Host_dereddening_MW_dereddening.ascii_fixed_width', format = 'fixed_width')
tbl_h22_corrected = tbl_h22_corrected[tbl_h22_corrected['observatory'] == 'LasCumbres1m']
tbl_imsng_corrected = ascii.read('./data/SN2021aefx_formatted_Host_dereddening_MW_dereddening.ascii_fixed_width', format = 'fixed_width')
tbl_corrected = vstack([tbl_imsng_corrected, tbl_h22_corrected])
# %%
lc = LightCurve()
lc.data = tbl_corrected
lc.plt_params.xlim = [59500, 59720]
lc.plt_params.ylim = [20, 8]
lc.plt_params.figure_figsize = (12, 8)
figure = lc.plot(ra = 64.9708333, dec = -54.9480556, flux_key = 'mag', fluxerr_key = 'e_mag')
# %%
