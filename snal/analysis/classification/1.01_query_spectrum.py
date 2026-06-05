
#%%

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
#%%
from snal.utils import TNSQuerier
from snal.utils import OSCQuerier
from pathlib import Path
from astropy.table import Table
import json
import numpy as np
import matplotlib.pyplot as plt
from ezphot.helper import Helper
import sncosmo
from astropy.io import ascii
# %% Load the table
oscquerier = OSCQuerier()
SNAL_DIR = oscquerier.config.save_dir
COLOR_MAP_PATH = Path(SNAL_DIR) / 'color_map.json'
COLOR_MAP = json.load(open(COLOR_MAP_PATH, 'r'))
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
good_objnames = []
for i in range(len(effective_tbl)):
    objname = effective_tbl['objname'][i]
    spectra_result_path = Path(SNAL_DIR) / objname / f"{objname}_spectra_quality.csv"
    if not spectra_result_path.exists():
        continue
    spectra_result = Table.read(spectra_result_path, format='ascii.fixed_width')
    if len(spectra_result) == 0:
        continue
    spectra_result_good_and_normal = spectra_result[(spectra_result['good_sign'] == 'Good')]# | (spectra_result['good_sign'] == 'Normal') | (spectra_result['good_sign'] == 'Bad') | (spectra_result['good_sign'] == 'Worst')]
    if len(spectra_result_good_and_normal) > 0:
        good_objnames.append(objname)
#%%
from tqdm import tqdm
tnsquerier = TNSQuerier()
all_meta_rows = []
for objname in good_objnames:
    tns_meta_path = Path(SNAL_DIR) / objname / f"{objname}_meta_TNS.json"
    if not tns_meta_path.exists():
        try:
            meta_json_tns = tnsquerier.get_object(objname = objname)[0]
        except:
            continue
    else:
        meta_json_tns = json.load(open(tns_meta_path, 'r'))
    meta_json_tns['transient_type'] = meta_json_tns['object_type']['name']
    meta_json_tns.pop('object_type')
    meta_json_tns.pop('discmagfilter')
    meta_json_tns.pop('reporting_group')
    meta_json_tns.pop('discovery_data_source')
    meta_tbl_osc = oscquerier.get_object(objname = objname)
    meta_spectra_quality = Table.read(Path(SNAL_DIR) / objname / f"{objname}_spectra_quality.csv", format='ascii.fixed_width')
    meta_spectra_quality_good = meta_spectra_quality[meta_spectra_quality['good_sign'] == 'Good']
    print('Type: ', meta_json_tns['transient_type'], 'Num_photometry:', meta_tbl_osc['num_photometry'][0], 'Num_spectra:', len(meta_spectra_quality_good),'/',meta_tbl_osc['num_spectra'][0])
    all_meta_tbl_row = dict()
    all_meta_tbl_row.update(meta_json_tns)
    all_meta_tbl_row.update(meta_tbl_osc[0])
    all_meta_tbl_row.update(dict(num_good_spectra = len(meta_spectra_quality_good)))
    all_meta_rows.append(all_meta_tbl_row)
#%%
good_meta_tbl = Table(all_meta_rows)
good_meta_tbl.write(Path(SNAL_DIR) / 'good_meta_tbl.ecsv', format='ascii.ecsv', overwrite=True)
# %%
