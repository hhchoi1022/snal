
#%%
from pathlib import Path
import numpy as np
import json
import h5py
import csv
import hashlib
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any
from astropy.table import Table
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
# %%
# 1. Load the spreadsheet
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1wwbBlAvOUamYM-l_F-N6xd4UYPEcKhntzLETNtP-m1o/edit?gid=0#gid=0"
import gspread
from astropy.table import Table

gc = gspread.service_account(
    filename="/home/hhchoi1022/code/SNID-SAGE/googlesheet.json"
)

sh = gc.open_by_url(SPREADSHEET_URL)
ws = sh.get_worksheet(0)

rows = ws.get_all_records(numericise_ignore=["all"])
SNID_tbl = Table(rows)
SNID_tbl_clean = SNID_tbl[SNID_tbl['Quality note'] == 'P']
all_types_SNID_SAGE = list(set(SNID_tbl['Type']))
for type in all_types_SNID_SAGE:
    print(f"'{type}'")
#%%
# 2. Load the NGSF-7DT templates
sample_dir = '/home/hhchoi1022/code/NGSF_7DT/NGSF_7DT/bank/original_resolution/sne/'
sample_dir = Path(sample_dir)
all_types_NGSF_7DT = list(sample_dir.glob('*'))
all_types_NGSF_7DT = [type.name for type in all_types_NGSF_7DT]
for type in all_types_NGSF_7DT:
    print(f"'{type}'")

#%%
# 3. Load the metadata of SNID-SAGE templates
failed_history = {}
for idx in range(60, len(SNID_tbl_clean)):
    try:
        SNID_row = SNID_tbl_clean[idx]
        objname = SNID_row['Objname']
        NGSF_7DT_TEMPLATE_DIR = Path("/home/hhchoi1022/code/NGSF_7DT/NGSF_7DT/bank/")
        TEMPLATE_DIR = Path('/home/hhchoi1022/code/SNAL/snal/analysis/classification/template_result')
        all_templates_meta = list(Path(TEMPLATE_DIR).glob(f'*{objname}*meta.json'))
        from tqdm import tqdm
        wiserep_rows = []
        snid_rows = []
        other_rows = []
        for template_meta in tqdm(all_templates_meta, desc = 'Processing templates...'):
            with open(template_meta, 'r') as f:
                template_meta_dict = json.load(f)
            wiserep_meta = template_meta_dict.pop('wiserep_meta')
            wiserep_rows.append(wiserep_meta)
            snid_meta = template_meta_dict.pop('snid_sage_meta')
            snid_rows.append(snid_meta)
            other_rows.append(template_meta_dict)
        wiserep_meta_tbl = Table(wiserep_rows)
        snid_meta_tbl = Table(snid_rows)
        other_meta_tbl = Table(other_rows)
        # for colname in wiserep_meta_tbl.colnames:
        #     print(colname)

        # from astropy.io import ascii
        # # 4. Load the example wiserep_spectra.csv file
        # sample_path = '/home/hhchoi1022/code/NGSF_7DT/NGSF_7DT/bank/original_resolution/sne/II/2014cn/wiserep_spectra.csv'
        # example_tbl = ascii.read(sample_path)
        # for colname in example_tbl.colnames:
        #     print(colname)

        # MAP COLUMN NAMES FROM WISEREP to NGSF-7DT
        column_mapping = {
            "Spec. ID": "Spec. ID",
            "Obj. ID": "Obj. ID",
            "Obj. IAU Name": "IAU name",
            "Obj. Alt. Name/s": "Internal name/s",
            "Obj Type": "Obj. Type",
            "Redshift": "Redshift",
            "Obs-date (UT)": "Obs-date",
            "Exp-time": "Exp-time",
            "Observer/s": "Observer/s",
            "Reducer/s": "Reducer/s",
            "Group": "Source group",
            "Spectrum ascii File": "Ascii file",
            "Spectrum fits File": "Fits file",
            "Spec. Type": "Spec. type",
            "Quality": "Spec. quality",
            "Extinction-Corrected": "Extinction-Corrected",
            "Flux Calibrated By": "Flux Calibrated By",
            "WL Medium": "WL Medium",
            "Assoc. Groups": "Associated groups",
            "Public": "Public",
            "End Prop. Period": "End prop. period",
            "Publish": "Publish",
            "Contrib": "Contrib",
            "Remarks": "Remarks",
            "Created by": "Created by",
            "Creation date (UT)": "Creation date",
            "wl_min": "Lambda-min",
            "wl_max": "Lambda-max",
        }
        missing_columns = ["Obj. RA", 
            "Obj. DEC", 
            "JD", 
            "Telescope",
            "Instrument", 
            "WL Units", 
            "Flux Unit Coefficient", 
            "Spec. units",
            "Aperture (slit)", 
            "HA", 
            "Airmass", 
            "Dichroic", 
            "Grism", 
            "Grating", 
            "Blaze",
            "Del-Lambda",
            "Created by",
            "Creation date",
        ]

        from wiserep_api import get_target_property, get_target_class
        props = ["type", "redshift", "host", "coords", "coords_deg"]
        info = get_target_property(objname, props)

        print(f'================={objname}==============')
        # print(dict(zip(props, info)))

        sn_class = get_target_class(objname)
        print("classification:", sn_class)

        #6. Add Telescope and Instrument information to the example table
        from astropy.time import Time
        telinfo_all = wiserep_meta_tbl['Tel / Inst']
        all_telescopes = []
        all_instruments = []
        for telinfo in telinfo_all:
            telescope = telinfo.split(' / ')[0]
            instrument = telinfo.split(' / ')[1]
            all_telescopes.append(telescope)
            all_instruments.append(instrument)
        wiserep_meta_tbl['Telescope'] = all_telescopes
        wiserep_meta_tbl['Instrument'] = all_instruments
        wiserep_meta_tbl['JD'] = Time(wiserep_meta_tbl['Obs-date (UT)'], format = 'iso').jd
        ra, dec = info[4].split(' ')
        wiserep_meta_tbl['Obj. RA'] = ra
        wiserep_meta_tbl['Obj. DEC'] = dec
        wiserep_meta_tbl.rename_columns(list(column_mapping.keys()), list(column_mapping.values()))

        # MAP TRANSIENT TYPE FROM SNID to NGSF-7DT
        map_snid_to_ngsf_7dt = {
            "SN": 'SN', #
            "SN I": 'I', #
            "SN Ia": "Ia-norm",

            "SN Ia-rapid": "Ia-rapid", #
            "SN Ia-02cx-like": "Ia 02cx like",
            "SN Ia-02es-like": "Ia 02es like",
            "SN Ia-99aa-like": "Ia 99aa-like", #
            "SN Ia-91bg-like": "Ia 91bg-like",
            "SN Ia-91T-like": "Ia 91T-like",
            "SN Ib/c-Ca-rich": "Ca-Ib/c", #
            "SN Ia-pec": "Ia-pec",
            "SN Ia-Ca-rich": "Ca-Ia",
            "SN Ia-SC": "super_chandra",
            "SN Ia-CSM": "Ia-CSM",
            "SN Ia-CSM-ambigious": "Ia-CSM-(ambigious)",

            "SN Ib": "Ib",

            "SN Ibn": "Ibn", #
            "SN Ib-Ca-rich": "Ca-Ib",

            "SN Ib/c": "Ib/c", #
            "SN Ib-pec": "Ib-pec", #

            "SN Ic": "Ic",
            "SN Ic-Ca-rich": "Ca-Ic", #
            "SN Icn": "Icn", #
            "SN Ic-BL": "Ic-BL",
            "SN Ic-pec": "Ic-pec", #

            
            "SN II": "II",
            "SN II-pec": "II-pec", #    

            "SN IIb": "IIb",
            "SN IIP": "IIP", #
            
            "SN IIn": "IIn",
            "SN IIn-pec": "IIn-pec", #

            
            "ILRT": "ILRT",
            "LRN": "LRN", #    
            "SLSN-I": "SLSN-I",
            "SLSN-R": "SLSN-R", #
            "SLSN-Ib": "SLSN-Ib",
            "SLSN-II": "SLSN-II",
            "SLSN-IIn": "SLSN-IIn",
            "TDE": "TDE", #
            "TDE-H": "TDE H",
            "TDE-H+He": "TDE H+He",
            "TDE-He": "TDE He",
            "FBOT": 'FBOT'
        }

        # 8. Save the template spectrum to NGSF_7DT
        '''
        Folder tree of the template is
        Original data
            NGSF_7DT_TEMPLATE_DIR/original_resolution/sne/<transient_type>/<objname>/<ascii_file>
        7DT data
            NGSF_7DT_TEMPLATE_DIR/binnings/7DT/sne/<transient_type>/<objname>/<ascii_file>
        '''

        transient_type = map_snid_to_ngsf_7dt[SNID_row['Type']]
        objname = objname
        template_folder_original = NGSF_7DT_TEMPLATE_DIR / 'original_resolution' / 'sne' / transient_type / objname
        template_folder_7DT = NGSF_7DT_TEMPLATE_DIR / 'binnings' / '7DT' / 'sne' / transient_type / objname
        redshift_applied = SNID_row['Redshift']

        import shutil
        for i in range(len(wiserep_meta_tbl)):
            wiserep_meta = wiserep_meta_tbl[i]
            snid_meta = snid_meta_tbl[i]
            other_meta = other_meta_tbl[i]
            
            epoch = snid_meta['epoch']
            original_path = list(TEMPLATE_DIR.glob(f'*{objname}_{epoch}.json'))[0]
            with open(original_path, 'r') as f:
                original_dict = json.load(f)
            w_rest_original = np.array(original_dict['w_wiserep_original'])
            w_rest_rebinned = np.array(original_dict['w_wiserep_rebinned'])
            w_observed_original = w_rest_original * (1 + float(redshift_applied))
            w_observed_rebinned = w_rest_rebinned * (1 + float(redshift_applied))

            f_original = np.array(original_dict['f_wiserep_original'])
            f_rebinned = np.array(original_dict['f_wiserep_rebinned_corrected'])
            filepath =  wiserep_meta['Ascii file']
            
            # Save the original template
            (template_folder_original / 'raw').mkdir(parents=True, exist_ok=True)
            if (template_folder_original / filepath).exists():
                # Move to raw folder
                shutil.move(template_folder_original / filepath, template_folder_original / 'raw' / filepath)
            with open(template_folder_original / 'raw' /  filepath, 'w') as f:
                for wave, flux in zip(w_observed_original, f_original):
                    f.write(f'{wave} {flux}\n')
            # Save the rebinned template
            with open(template_folder_original /  filepath, 'w') as f:
                for wave, flux in zip(w_observed_rebinned, f_rebinned):
                    f.write(f'{wave} {flux}\n')
            # Save the 7DT resolution
            synphot_path = list(TEMPLATE_DIR.glob(f'*{objname}_{epoch}_synphot.ascii'))[0]
            synphot_tbl = ascii.read(synphot_path)
            # Normalize the synphot flux 
            synphot_tbl['col2'] = synphot_tbl['col2'] / np.nanmean(synphot_tbl['col2'])
            if (template_folder_7DT / filepath).exists():
                (template_folder_7DT / 'raw').mkdir(parents=True, exist_ok=True)
                shutil.move(template_folder_7DT / filepath, template_folder_7DT / 'raw' / filepath)
            
            (template_folder_7DT).mkdir(parents=True, exist_ok=True)
            with open(template_folder_7DT / filepath, 'w') as f:
                for wave, flux in zip(synphot_tbl['col1'], synphot_tbl['col2']):
                    f.write(f'{wave} {flux}\n')

        # 10. Merge wiserep_meta_tbl and ngsf_meta_tbl
        ngsf_meta_path = template_folder_original/'wiserep_spectra.csv'
        if Path(ngsf_meta_path).exists():
            ngsf_meta_tbl = ascii.read(ngsf_meta_path)
            from astropy.table import vstack
            import numpy as np

            t1 = wiserep_meta_tbl.copy()
            t2 = ngsf_meta_tbl.copy()

            shared_cols = set(t1.colnames) & set(t2.colnames)

            for col in shared_cols:
                dtype1 = t1[col].dtype
                dtype2 = t2[col].dtype

                if dtype1 != dtype2:
                    # print(f"Converting {col}: {dtype1} + {dtype2} -> str")
                    t1[col] = t1[col].astype(str)
                    t2[col] = t2[col].astype(str)

            merged_tbl = vstack([t1, t2], join_type="outer", metadata_conflicts="silent")
            from astropy.table import unique
        else:
            merged_tbl = wiserep_meta_tbl.copy()

        merged_tbl = unique(merged_tbl, keys="Spec. ID")
        # # Save 
        merged_tbl.write(ngsf_meta_path, format='csv', overwrite=True)
    except Exception as e:
        failed_history[objname] = e
# %%
# Update mjd_of_maximum_brightness.csv
path_mjd_of_maximum_brightness = Path('/home/hhchoi1022/code/NGSF_7DT/NGSF_7DT/mjd_of_maximum_brightness.csv')
mjd_of_maximum_brightness_tbl = ascii.read(path_mjd_of_maximum_brightness)
#%%
# Save it to table
from astropy.table import Table

rows = []

matched = np.isin(mjd_of_maximum_brightness_tbl['Name'], SNID_tbl_clean['Objname'])
unmatched = ~matched
matched_mjd_of_maximum_brightness_tbl = mjd_of_maximum_brightness_tbl[matched]
unmatched_mjd_of_maximum_brightness_tbl = mjd_of_maximum_brightness_tbl[unmatched]
#%%
from astropy.io import ascii
from astropy.table import Table, vstack
from pathlib import Path

path_mjd_of_maximum_brightness = Path('/home/hhchoi1022/code/NGSF_7DT/NGSF_7DT/mjd_of_maximum_brightness.csv')
tbl = ascii.read(path_mjd_of_maximum_brightness)

new_rows = []

for SNID_row in SNID_tbl_clean:
    objname = SNID_row['Objname']
    mjd_peak_new = SNID_row['mjd_max_phase']

    matched_idx = tbl['Name'] == objname

    if any(matched_idx):
        tbl['mjd_peak'][matched_idx] = mjd_peak_new
    else:
        new_rows.append([objname, mjd_peak_new, '-', 0])

if len(new_rows) > 0:
    new_tbl = Table(
        rows=new_rows,
        names=['Name', 'mjd_peak', 'band_peak', 'isupperlimit']
    )
    new_tbl['mjd_peak'] = new_tbl['mjd_peak'].astype(float)
    tbl = vstack([tbl, new_tbl])

tbl.write(path_mjd_of_maximum_brightness, format='csv', overwrite=True)

# tbl.write(path_mjd_of_maximum_brightness, format='csv', overwrite=True)
#%%

for SNID_row in SNID_tbl_clean:
    objname = SNID_row['Objname']
    mjd_peak = SNID_row['mjd_max_phase']
    band_peak = '-'
    is_upperlimit = 0

    matched = mjd_of_maximum_brightness_tbl[mjd_of_maximum_brightness_tbl['Name'] == objname]

    if len(matched) > 0:
        band_peak = matched['band_peak'][0]
        is_upperlimit = matched['isupperlimit'][0]

    rows.append([objname, mjd_peak, band_peak, is_upperlimit])

new_tbl = Table(
    rows=rows,
    names=['Objname', 'mjd_peak', 'band_peak', 'is_upperlimit']
)
new_tbl.write(path_mjd_of_maximum_brightness, format='csv', overwrite=True)
#%%




#%%
all_filepaths = [path.name for path in list((template_folder_7DT).glob('*'))]
from NGSF_7DT.Header_Binnings import bin_spectrum
filepath = all_filepaths[0]
original_filepath = template_folder_original / 'raw' / filepath
binned_filepath = template_folder_original / filepath
original_tbl = ascii.read(original_filepath)
binned_tbl = ascii.read(binned_filepath)
spec_original = np.array((np.array(list(original_tbl['col1'])), np.array(list(original_tbl['col2'])))).T
spec_binned = np.array((np.array(list(binned_tbl['col1'])), np.array(list(binned_tbl['col2'])))).T
spec_original_binned = bin_spectrum(spec_original, 10)
spec_binned_binned = bin_spectrum(spec_binned, 10)

plt.plot(spec_original_binned['lam_bin'] / (1 + float(redshift_applied)), spec_original_binned['bin_flux'], label = 'Original')
plt.plot(spec_binned_binned['lam_bin'] / (1 + float(redshift_applied)), spec_binned_binned['bin_flux'], label = 'Binned')
plt.legend()
#%%
all_filepaths = [path.name for path in list((template_folder_7DT / 'raw').glob('*'))]
tbl2 = ascii.read(template_folder_7DT / filepath)
fig, ax = plt.subplots(figsize = (10, 5))

if (template_folder_7DT/ 'raw').exists():
    tbl1 = ascii.read(template_folder_7DT / 'raw' /  filepath)
    ax.scatter(tbl1['col1'], tbl1['col2'], facecolor = 'none', edgecolor = 'r')
ax1 = ax.twinx()
ax1.plot(spec_original_binned['lam_bin'] / (1 + float(redshift_applied)), spec_original_binned['bin_flux'], label = 'Original')
ax1.scatter(tbl2['col1'], tbl2['col2'], facecolor = 'none', edgecolor = 'b')
ax1.legend()

# %%

