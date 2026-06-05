
#%%
from ezphot.dataobjects import Spectrum
from snal.utils import TNSQuerier
#%%
tnsquerier = TNSQuerier()
meta, photometry_tbl, spectroscopy_tbl = tnsquerier.get_object(objname = 'SN2025ajmj')
#%%
from astropy.io import ascii
tbl = ascii.read(spectroscopy_tbl['ascii_file'][0])
# %%
spec = Spectrum(wavelength = tbl['col1'], flux = tbl['col2'])
# %%
result_synphot = spec.synphot(filterset = 'medium', visualize = True, visualize_transmission = True)
synphot_dict, fig_synphot, ax_synphot, ax_transmission = result_synphot
# %%
result_from_7dt_path = '/home/hhchoi1022/bridge/alert/AT2025ajmj/AT2025ajmj_20260126_091243_ezphot_PS.dat'
result_from_7dt = ascii.read(result_from_7dt_path, format = 'fixed_width')
# %%
from ezphot.dataobjects import PhotometricSpectrum
photspectrum = PhotometricSpectrum()
# %%
photspectrum.data = result_from_7dt
# %%
result = photspectrum.plot()
# %%
fig, _, ax, tbl = result
#%%
ax = ax['2025-12-30 07:19']
# %%
for filter_, value in synphot_dict.items():
    if filter_ in set(tbl['filter']):
        ax.scatter(value['wl_pivot'], value['mag'], s = 50, marker = 'D')

# %%
fig['2025-12-30 07:19']

# %%
