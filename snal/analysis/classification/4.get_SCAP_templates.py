


#%%
import os
import json

SCAT_SPECTRA_SUMMARY_PATH = '/home/hhchoi1022/code/SCAT/templates/scat_dr1_v1.1_spectra.csv'
SCAT_SOURCE_SUMMARY_PATH  = '/home/hhchoi1022/code/SCAT/templates/scat_dr1_v1.1_sources.csv'
SCAT_SCI_FOLDER = '/home/hhchoi1022/code/SCAT/templates/scat_dr1_v1.1_sci'
SCAT_SPECTRA_FOLDER = '/home/hhchoi1022/code/SCAT/templates/scat_dr1_v1.1_spectra_lite'
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy.table import Table
# matplotlib.use("Agg")
%matplotlib inline 

tbl_spectra = Table.read(SCAT_SPECTRA_SUMMARY_PATH, format='ascii.csv', comment='#')
tbl_sources = Table.read(SCAT_SOURCE_SUMMARY_PATH, format='ascii.csv', comment='#')

#%% 1. Phase distribution by transient type and quality

pairs = sorted(set(zip(tbl_spectra['sptype'], tbl_spectra['subtype'])))
pair_to_y = {pair: i for i, pair in enumerate(pairs)}
pair_counts = {pair: int(np.sum((tbl_spectra['sptype'] == sp) & (tbl_spectra['subtype'] == sub))) for pair in pairs for sp, sub in [pair]}
pair_labels = [f'{sp} - {sub} ({pair_counts[(sp, sub)]})' for sp, sub in pairs]

sptype_boundaries = []
prev_sp = None
for i, (sp, _) in enumerate(pairs):
    if sp != prev_sp and prev_sp is not None:
        sptype_boundaries.append(i - 0.5)
    prev_sp = sp

quality_colors = {'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'}
quality_order = ['Bronze', 'Silver', 'Gold']

fig, ax = plt.subplots(figsize=(14, 12))

for quality in quality_order:
    mask = tbl_spectra['quality'] == quality
    subset = tbl_spectra[mask]
    y_vals = np.array([pair_to_y[(sp, sub)] for sp, sub in zip(subset['sptype'], subset['subtype'])])
    jitter = np.random.default_rng(42).uniform(-0.2, 0.2, size=len(y_vals))
    ax.scatter(
        subset['phase'], y_vals + jitter,
        c=quality_colors[quality],
        label=f'{quality} ({np.sum(mask)})',
        s=12, alpha=0.6, edgecolors='k', linewidths=0.3,
    )

for boundary in sptype_boundaries:
    ax.axhline(boundary, color='gray', ls='-', lw=0.5, alpha=0.5)

ax.set_yticks(range(len(pairs)))
ax.set_yticklabels(pair_labels, fontsize=9)
ax.set_xlabel('Phase [days from max]')
ax.set_ylabel('Transient Type')
ax.set_title(f'SCAT DR1 — Phase Distribution by Transient Subtype\n({len(tbl_spectra)} spectra from {len(set(tbl_spectra["tns_name"]))} transients)')
ax.legend(title='Quality', loc='upper right')
ax.axvline(0, color='gray', ls='--', lw=0.8, zorder=0)
ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
# fig.savefig('scat_dr1_phase_by_subtype.png', dpi=150)
plt.show()
# %%
def read_header(filename):
    header = {}
    with open(filename, "r") as f:
        for line in f:
            # Stop when the numeric spectrum data begins
            if not line.startswith("#"):
                break

            # Remove leading "#"
            line = line[1:].strip()

            # Skip column-name/unit lines like:
            # # lbda flux error
            # # [A] [erg/s/cm2/A]
            if "=" not in line:
                continue

            # Split key/value from comment
            key, value = line.split("=", 1)

            key = key.strip()

            # Remove explanatory comment after //
            value = value.split("//")[0].strip()

            header[key] = value
    return header

#%% 2. Filtering the spectra by quality
from tqdm import tqdm
# 2.1. Filtering tentative classification
non_tentative_mask = np.isin(tbl_sources['tentative_classification'], 'False')
non_tentative_sources = tbl_sources[non_tentative_mask]['tns_name']
tbl_spectra = tbl_spectra[(np.isin(tbl_spectra['tns_name'], non_tentative_sources))]
# 2.2. Filtering the spectra by SNR
tbl_spectra = tbl_spectra[(tbl_spectra['median_snr_blue'] > 5) & (tbl_spectra['median_snr_red'] > 10)]
# 2.3. Filtering the spectra by subtype (only use the ones in the list)
subtype_to_use = np.array([
    #'nova',
    #'unknown',
    '91bg',
    #'XRB',
    'Ia',
    #'YSO',
    'II',
    '03fg',
    #'lensed',
    'SN',
    #'flash',
    #'ANT',
    'FBOT',
    'Ibn',
    #'BLLac',
    '02es',
    'TDE',
    'Ib',
    'IIn',
    'SLSN-I',
    'Ic-BL',
    '91T',
    'CV',
    #'AGN',
    'Ic',
    'IIb',
])
tbl_spectra = tbl_spectra[(np.isin(tbl_spectra['subtype'].astype(str), subtype_to_use))]
# 2.4. Filtering the spectra by nan mask. If masked wavelength range is larger than 300A, filter out
wl_min = 3700
wl_max = 7300
wl_mask_range_threshold = 300  # Angstrom

keep_mask = []

for row in tqdm(tbl_spectra):
    objname = row["tns_name"]

    sourceinfo_filename = "source_info.json"
    sourceinfo_path = os.path.join(SCAT_SCI_FOLDER, objname, sourceinfo_filename)

    with open(sourceinfo_path, "r") as f:
        sourceinfo_dict = json.load(f)

    spec_filename = row["filename"]
    spec_path = os.path.join(SCAT_SCI_FOLDER, objname, "ascii", spec_filename)

    spec_tbl = Table.read(spec_path, format="ascii", comment="#")

    header = read_header(spec_path)
    flux_scale = float(header["FLUXNORM"])

    wl = np.asarray(spec_tbl["col1"], dtype=float)
    flux = np.asarray(spec_tbl["col2"], dtype=float) * flux_scale
    fluxerr = np.asarray(spec_tbl["col3"], dtype=float) * flux_scale

    # Region of interest
    in_range = (wl >= wl_min) & (wl <= wl_max)

    # NaN pixels inside wavelength range
    nan_in_range = in_range & (np.isnan(flux) | np.isnan(fluxerr))

    if np.any(nan_in_range):
        nan_wl = wl[nan_in_range]
        nan_range = nan_wl.max() - nan_wl.min()
    else:
        nan_range = 0.0

    # Keep only if NaN-covered wavelength range is smaller than threshold
    keep = nan_range < wl_mask_range_threshold
    keep_mask.append(keep)

tbl_spectra = tbl_spectra[keep_mask]

tbl_spectra_good = tbl_spectra[(tbl_spectra['quality'] == 'Silver') | (tbl_spectra['quality'] == 'Gold')]
tbl_spectra_bad = tbl_spectra[(tbl_spectra['quality'] == 'Bronze')]
#%%
from ezphot.dataobjects import Spectrum
# for i,row in enumerate(tbl_spectra):

i = 14
row = tbl_spectra_good[i]
objname = row['tns_name']
objtype = row['sptype']
subtype = row['subtype']
sourceinfo_filenmae = 'source_info.json'
sourceinfo_path = os.path.join(SCAT_SCI_FOLDER, objname, sourceinfo_filenmae)
sourceinfo_dict = json.load(open(sourceinfo_path, 'r'))
spec_filename = row['filename']
spec_path = os.path.join(SCAT_SCI_FOLDER, objname, 'ascii', spec_filename)
spec_tbl = Table.read(spec_path, format='ascii', comment='#')
header = read_header(spec_path)
flux_scale = float(header['FLUXNORM'])

wl = spec_tbl['col1']
flux = spec_tbl['col2'] * flux_scale
fluxerr = spec_tbl['col3'] * flux_scale
# Linear interpolation to fill the missing values
# flux = np.interp(wl, wl[~np.isnan(flux)], flux[~np.isnan(flux)])
# fluxerr = np.interp(wl, wl[~np.isnan(fluxerr)], fluxerr[~np.isnan(fluxerr)])
isnan_mask = np.isnan(flux) | np.isnan(fluxerr)

spec = Spectrum(wavelength = wl, flux = flux, fluxerr = fluxerr, wavelength_unit = 'AA', flux_unit = 'flamb')
fig, ax = spec.show('AB')
ax.set_title(f'{objname} ({objtype} {subtype}, Phase = {row["phase"]:.2f} days)')
ax.vlines(
    wl[isnan_mask],
    ymin=ax.get_ylim()[0],
    ymax=ax.get_ylim()[1],
    color="red",
    alpha=0.2,
    linewidth=0.5)
text_str = ''
if 'lc_redchi2' in sourceinfo_dict:
    text_str += f'LC Chisq: {sourceinfo_dict["lc_redchi2"]}\n'
if 'tmax' in sourceinfo_dict:
    text_str += f'T_max = {sourceinfo_dict["tmax"]:.2f}+-{sourceinfo_dict["tmax_error_stat"]:.2f} days\n'
if 'median_snr_blue' in row.colnames:
    text_str += f'SNR_blue: {row["median_snr_blue"]:.2f}\n'
if 'median_snr_red' in row.colnames:
    text_str += f'SNR_red: {row["median_snr_red"]:.2f}'
ax.text(0.5, 0.2, text_str, transform=ax.transAxes, fontsize=12, ha='left', va='top')
# %%
synphot_result = spec.synphot(filterset = 'medium', visualize = True, visualize_transmission = False, visualize_spectrum = True)
#%%
