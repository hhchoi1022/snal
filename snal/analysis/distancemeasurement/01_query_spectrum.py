#%%   
from snal.utils import OSCQuerier
from pathlib import Path
from astropy.table import Table
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
# %%
oscquerier = OSCQuerier()
# %%
from astropy.time import Time
mjd_start = Time('2001-01-01').mjd
num_photometry_threshold = -1
num_spectra_threshold = -1
effective_tbl = oscquerier.summary_tbl[
    (oscquerier.summary_tbl['num_photometry'] > num_photometry_threshold) & 
    (oscquerier.summary_tbl['num_spectra'] > num_spectra_threshold) &
    (oscquerier.summary_tbl['discovery_date'] > mjd_start)
    ]
print('Number of objects: ', len(effective_tbl))
# %%
from tqdm import tqdm
for i in tqdm(range(len(effective_tbl))):
# for i in np.arange(2000, len(effective_tbl)):
    # print('Object name: ', effective_tbl['objname'][i])
    photometry_tbl = oscquerier.get_photometry(objname = effective_tbl['objname'][i], save = True, verbose = False)
    spec_header, spec_data = oscquerier.get_spectra(objname = effective_tbl['objname'][i], save = True, verbose = False)

# %%
from pathlib import Path
from snal.utils import TNSQuerier
from ezphot.dataobjects import LightCurve
def get_meta_idx(path, objname):
    name = Path(path).name
    return int(name.replace(f"{objname}_spectra_", "").split("_")[0])
def get_filter_color(filter_name, color_map):
    global auto_i
    if filter_name not in color_map or color_map[filter_name] is None:
        color_map[filter_name] = auto_colors[auto_i % len(auto_colors)]
        auto_i += 1
    return color_map[filter_name]

tnsquerier = TNSQuerier()
snal_dir = oscquerier.config.save_dir
color_map = LightCurve.FILTER_COLOR
auto_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
auto_i = 0
#%%
for i in range(len(effective_tbl)):
    objname = effective_tbl['objname'][i]

    # Query metadata from TNS
    try:
        metadata, photometry_tbl, spectroscopy_tbl = tnsquerier.get_object(objname = objname, save = True, verbose = False)
    except Exception as e:
        metadata = dict()
    objtype_dict =  metadata.get('object_type', {})
    objtype = objtype_dict.get('name', 'Unknown')
    hostname = metadata.get('hostname', 'Unknown')
    if hostname is None:
        hostname = 'Unknown'

    # Query photometry and spectra from OSC
    photometry_path = Path(snal_dir) / objname / f"{objname}_photometry_OSC.csv"
    spectra_data_paths = list((Path(snal_dir) / objname).glob(f"{objname}_spectra_*_data_OSC.csv"))
    spectra_data_paths.sort()
    spectra_meta_paths = list((Path(snal_dir) / objname).glob(f"{objname}_spectra_*_meta_OSC.json"))
    spectra_meta_paths.sort()
    photometry_tbl = Table.read(photometry_path, format='ascii.fixed_width')
    spectra_dict = dict()
    for spectra_data_path, spectra_meta_path in zip(spectra_data_paths, spectra_meta_paths):
        idx = get_meta_idx(spectra_data_path, objname)
        spectra_dict[idx] = dict()
        spectra_dict[idx]['data'] = Table.read(spectra_data_path, format='csv')
        spectra_dict[idx]['meta'] = json.load(open(spectra_meta_path, 'r'))

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
        color = get_filter_color(filter, color_map)
        ax.errorbar(mjd, mag, yerr=magerr, fmt='o', label=f'{filter} ({len(mjd)} points)', color=color)
    #ax.set_xlim(57000, 57100)
    ax.invert_yaxis()
    # Get ylim from ax 
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    xticks = ax.get_xticks()

    for spectrum_data in spectra_dict.values():
        spec_meta = spectrum_data['meta']
        ax.axvline(float(spec_meta['mjd']), color='k', linestyle='--', alpha = 0.1)
    ax.axvline(0, color='k', linestyle='--', alpha = 0.1, label = f'Spectrum ({len(spectra_dict)} points)')

    ax.legend()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(xticks, xticks.astype(int))
    fig.savefig(f"{snal_dir}/{objname}/{objname}_lightcurve.png", dpi=300)
    plt.close(fig)


#%%
COLOR_MAP_PATH = Path(snal_dir) / 'color_map.json'
with open(COLOR_MAP_PATH, 'w') as f:
    json.dump(color_map, f)
    print(f"Color map saved to: {COLOR_MAP_PATH}")
#%%