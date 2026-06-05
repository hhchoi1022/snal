


#%%
from ezphot.dataobjects import PhotometricSpectrum
from snal.utils import TNSQuerier
from ezphot.utils import DataBrowser
# %%
tnsquerier = TNSQuerier()
# %%
result = tnsquerier.get_object(objname = 'AT2026jqi',  with_spectra= True, save= True)
# %%
spec_file = result[2]['ascii_file'][0]
# %%

from ezphot.dataobjects import Spectrum
# %%
spec = Spectrum(spec_file)
# %%
spec.synphot(filterset = 'medium', visualize = True, visualize_transmission = False, visualize_spectrum = False)
# %%
dbrowser = DataBrowser('scidata')
dbrowser.objname = 'T02080'
catalog_set = dbrowser.search(pattern = '*trac*', return_type = 'catalog')
# %%
photspec = PhotometricSpectrum(catalog_set)
photspec.plt_params.ylim = [14, 20]
result = photspec.plot(ra = 323.94508745, dec = -63.90354722222222,
flux_key = 'MAG_TRACT7DT', fluxerr_key = 'MAGERR_TRACT7DT', depth_key = 'UL_TRACT7DT', zperr_key = 'ZPERR_TRACT7DT')
# %%
obsdate_key = '2026-04-16 07:13'
fig = result[0][obsdate_key]
ax = result[2][obsdate_key]
#%%
result_spec = spec.synphot(filterset = 'medium', visualize = False, visualize_transmission = False, ax = None, visualize_spectrum = False)
# %%
result_spec[0]
# %%
for filter_, value in result_spec[0].items():
    ax.scatter(value['wl_pivot'], value['mag'], s = 50, marker = 'D')
# %%
fig
# %%
