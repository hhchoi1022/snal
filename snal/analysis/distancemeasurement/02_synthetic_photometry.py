
#%%
from snal.utils import OSCQuerier
from pathlib import Path
from astropy.table import Table
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from ezphot.dataobjects import LightCurve
# %%

oscquerier = OSCQuerier()
SNAL_DIR = oscquerier.config.save_dir
COLOR_MAP_PATH = Path(SNAL_DIR) / 'color_map.json'
COLOR_MAP = json.load(open(COLOR_MAP_PATH, 'r'))
# COLOR_MAP = LightCurve.FILTER_COLOR
AUTO_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
AUTO_i = 0
# %% Load effective table
from astropy.time import Time
mjd_start = Time('2001-01-01').mjd
num_photometry_threshold = 5
num_spectra_threshold = 0
effective_tbl = oscquerier.summary_tbl[
    (oscquerier.summary_tbl['num_photometry'] > num_photometry_threshold) & 
    (oscquerier.summary_tbl['num_spectra'] > num_spectra_threshold) &
    (oscquerier.summary_tbl['discovery_date'] > mjd_start)
    ]
print('Number of objects: ', len(effective_tbl))
#%%
# for i in range(len(effective_tbl)):
def get_meta_idx(path, objname):
    name = Path(path).name
    return int(name.replace(f"{objname}_spectra_", "").split("_")[0])
def get_filter_color(filter_name, color_map):
    global AUTO_i
    if filter_name not in color_map or color_map[filter_name] is None:
        color_map[filter_name] = AUTO_COLORS[AUTO_i % len(AUTO_COLORS)]
        AUTO_i += 1
    return color_map[filter_name]
#%%
from ezphot.dataobjects import Spectrum
# Define pyphot filters first. 
pyphot_filters = ['U', 'B', 'V', 'R', 'I', 'u', 'g', 'r', 'i', 'z', 'medium']
_, pyphot_filters, _, _, _ = Spectrum(wavelength = [1000, 1001], flux = [0,0], wavelength_unit = 'AA', flux_unit = 'flamb').synphot(filterset = pyphot_filters, visualize = False, visualize_transmission = False)
#%%

from tqdm import tqdm
for i in tqdm(range(0, len(effective_tbl)), desc = 'Processing objects'):
    objname = effective_tbl['objname'][i]

    # Query metadata from TNS
    # Get metadata, photometry_tbl, spectroscopy_tbl from TNS queried
    tns_metadata_path = Path(SNAL_DIR) / objname / f"{objname}_meta_TNS.json"
    tns_photometry_path = Path(SNAL_DIR) / objname / f"{objname}_photometry_TNS.csv"
    tns_spectroscopy_path = Path(SNAL_DIR) / objname / f"{objname}_spectroscopy_TNS.csv"

    tns_metadata = dict()
    tns_photometry_tbl = Table()
    tns_spectroscopy_tbl = Table()
    tns_exists = False
    tns_photometry_exists = False
    tns_spectroscopy_exists = False
    if tns_metadata_path.exists():
        tns_metadata = json.load(open(tns_metadata_path, 'r'))
        tns_exists = True
    if tns_photometry_path.exists():
        tns_photometry_tbl = Table.read(tns_photometry_path, format='ascii.fixed_width')
        tns_photometry_exists = True
    if tns_spectroscopy_path.exists():
        tns_spectroscopy_tbl = Table.read(tns_spectroscopy_path, format='ascii.fixed_width')
        tns_spectroscopy_exists = True
    objtype_dict =  tns_metadata.get('object_type', {})
    objtype = objtype_dict.get('name', 'Unknown')
    hostname = tns_metadata.get('hostname', 'Unknown')
    if hostname is None:
        hostname = 'Unknown'

    # Photometry
    photometry_path = Path(SNAL_DIR) / objname / f"{objname}_photometry_OSC.csv"
    photometry_tbl = Table.read(photometry_path, format='ascii.fixed_width')
    mjd_min = np.min(photometry_tbl['mjd'].astype(float))
    mjd_max = np.max(photometry_tbl['mjd'].astype(float))
    # Remove masked filters
    f = photometry_tbl['filter']
    if hasattr(f, "mask"):
        photometry_tbl = photometry_tbl[~f.mask]
    filters = list(set(photometry_tbl['filter']))
    photometry_tbl_filter = photometry_tbl.group_by('filter').groups


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"{objname} ({objtype}, {hostname.replace(' ', '')})")
    for phot_tbl_filter in photometry_tbl_filter:
        mjd = phot_tbl_filter['mjd'].astype(float)
        mag = phot_tbl_filter['mag'].astype(float)
        magerr = phot_tbl_filter['magerr'].astype(float)
        filter = phot_tbl_filter['filter'][0]
        if np.ma.is_masked(filter):
            continue
        color = get_filter_color(filter, COLOR_MAP)
        # color = 'k'
        ax.errorbar(mjd, mag, yerr=magerr, fmt='o', label=f'{filter} ({len(mjd)} points)', color=color)
    #ax.set_xlim(57000, 57100)
    ax.invert_yaxis()
    # Get ylim from ax 
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    xticks = ax.get_xticks()

    # Spectra
    spectra_data_paths = list((Path(SNAL_DIR) / objname).glob(f"{objname}_spectra_*_data_OSC.csv"))
    spectra_data_paths.sort()
    spectra_meta_paths = list((Path(SNAL_DIR) / objname).glob(f"{objname}_spectra_*_meta_OSC.json"))
    spectra_meta_paths.sort()
    spectra_dict = dict()
    for spectra_data_path, spectra_meta_path in zip(spectra_data_paths, spectra_meta_paths):
        idx = get_meta_idx(spectra_data_path, objname)
        spectra_dict[idx] = dict()
        spectra_dict[idx]['data'] = Table.read(spectra_data_path, format='csv')
        spectra_dict[idx]['meta'] = json.load(open(spectra_meta_path, 'r'))
    spectra_dict_within_photometry_range = dict()
    for idx, spectrum_data in spectra_dict.items():
        spec_meta = spectrum_data['meta']
        mjd = float(spec_meta['mjd'])
        if mjd >= mjd_min and mjd <= mjd_max:
            spectra_dict_within_photometry_range[idx] = spectrum_data
            
    import numpy as np
    from ezphot.dataobjects import Spectrum
    synphot_dict = {}
    synphot_filters = ['U', 'B', 'V', 'R', 'I', 'u', 'g', 'r', 'i', 'z', 'medium']
    for idx, spectrum_data in spectra_dict_within_photometry_range.items():
        header = spectrum_data['meta']
        mjd = header['mjd']
        unit_flux = header['u_fluxes']
        if unit_flux == 'Uncalibrated':
            continue
        data = spectrum_data['data']
        data.sort('wavelength')
        
        if 'e_flux' in data.colnames:
            spectrum = Spectrum(wavelength = data['wavelength'], flux = data['flux'], fluxerr = data['e_flux'], wavelength_unit = 'AA', flux_unit = 'flamb')
        else:
            spectrum = Spectrum(wavelength = data['wavelength'], flux = data['flux'], wavelength_unit = 'AA', flux_unit = 'flamb')
        
        synphot_dict = spectrum.synphot(filterset = synphot_filters, pyphot_filters = pyphot_filters, visualize = False, visualize_transmission = False)[0]
        synphot_path = Path(SNAL_DIR) / objname / f"{objname}_synphot_{idx}_data_OSC.csv"
        rows = []
        for filter, dict_synphot in synphot_dict.items():
            mag = dict_synphot['mag']
            if np.isfinite(mag):
                row = dict()
                for dict_key, dict_value in dict_synphot.items():
                    row[dict_key] = dict_value
                row['filter'] = filter
                for key, value in header.items():
                    row[key] = value
                rows.append(row)
                
                if filter in filters:
                    color = get_filter_color(filter, COLOR_MAP)
                    ax.scatter(float(mjd), float(mag), facecolor = color, edgecolor = 'r', s = 40, marker = 'o', zorder=100)
            
        synphot_tbl = Table(rows=rows)
        synphot_tbl.write(synphot_path, format='ascii.fixed_width', overwrite=True)

    for spectrum_data in spectra_dict.values():
        spec_meta = spectrum_data['meta']
        ax.axvline(float(spec_meta['mjd']), color='k', linestyle='--', alpha = 0.1)
    ax.axvline(0, color='k', linestyle='--', alpha = 0.1, label = f'Spectrum ({len(spectra_dict)} points)')
    ax.scatter(0, 0, facecolor = 'none', edgecolor = 'r', s = 40, marker = 'o', label = 'Synthetic Photometry')

    ax.legend()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(xticks, xticks.astype(int))
    fig.savefig(Path(SNAL_DIR) / objname / f"{objname}_synthetic_photometry.png", dpi = 300, bbox_inches = 'tight')
    # Cloase figure
    plt.close(fig)
    # %%
