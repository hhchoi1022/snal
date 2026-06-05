
#%%
from bridge.alertmonitor import AlertClassifier
from snal.utils import OSCQuerier
from astropy.table import Table, join
from astropy.stats import sigma_clip
from pathlib import Path
import json
import re
import matplotlib
import numpy as np
from ezphot.dataobjects import Spectrum
from ezphot.dataobjects import PhotometricSpectrum
from astropy.time import Time
import matplotlib.pyplot as plt
#%%
SNAL_DIR = Path('/home/hhchoi1022/snal/data/')

medium_filterset_0 = [
    # 'm375w',
    'm386',
    'm400',
    'm425',
    # 'm425w',
    'm438',
    'm450',
    # 'm466w',
    'm475',
    'm483',
    'm500',
    'm512',
    'm525',
    'm534',
    'm550',
    'm561',
    'm575',
    'm586',
    'm600',
    'm615',
    'm625',
    'm640',
    'm650',
    'm661',
    'm675',
    # 'm692w',
    'm700',
    # 'm710w',
    'm725',
    'm750',
    'm769w',
    'm775',
    'm800',
    'm825',
    'm832w',
    'm850',
    'm875',
]
medium_filterset_1 = [
    'm400',
    'm425',
    'm450',
    'm475',
    'm500',
    'm525',
    'm550',
    'm575',
    'm600',
    'm625',
    'm650',
    'm675',
    'm700',
    'm725',
    'm750',
    'm775',
    'm800',
    'm825',
    'm850',
    'm875',
]
medium_filterset_2 = [
    'm400',
    'm450',
    'm500',
    'm550',
    'm600',
    'm650',
    'm700',
    'm750',
    'm800',
    'm850',
]
medium_filterset_3 = [
    'm400',
    'm500',
    'm600',
    'm700',
    'm800',
]
medium_filterset_4 = [
    'm400',
    'm600',
    'm800',
]
medium_filterset_5 = [
    # 'm375w',
    'm400',
    'm425',
    # 'm425w',
    'm438',
    'm450',
    # 'm466w',
    'm475',
    'm483',
    'm500',
    'm512',
    'm525',
    'm534',
    'm550',
    'm561',
    'm575',
    'm586',
    'm600',
    'm615',
    'm625',
    'm640',
    'm650',
    'm661',
    'm675',
    # 'm692w',
    'm700',
    # 'm710w',
]
medium_filterset_dict = {
    0: medium_filterset_0,
    1: medium_filterset_1,
    2: medium_filterset_2,
    3: medium_filterset_3,
    4: medium_filterset_4,
    5: medium_filterset_5}

#%%
RESULT_DIR = Path('/home/hhchoi1022/snal/data/')
all_wiserep_meta = list(RESULT_DIR.glob('*/wiserep_spectra.csv'))
#%%

# all_results_dict_medium0 = []
# all_results_dict_medium1 = []
# all_results_dict_medium2 = []
# all_results_dict_medium3 = []
# all_results_dict_medium4 = []
# all_results_dict_medium5 = []
# #%%

# maxdate_info_tbl = Table.read(Path(SNAL_DIR) / 'mjd_of_maximum_brightness.csv', format='csv')
# from tqdm import tqdm
# failed_filelist = []
# for i in tqdm(range(len(all_wiserep_meta))):
#     try:

#         wiserep_meta = all_wiserep_meta[i]
#         wiserep_meta_tbl = Table.read(wiserep_meta, format='csv')
#         objname = wiserep_meta.parent.name
#         maxdate = maxdate_info_tbl['mjd_peak'][maxdate_info_tbl['Name'] == objname][0]
#         for row in wiserep_meta_tbl:
#             spec_filename = row['Ascii file']
#             spec_filepath = wiserep_meta.parent / spec_filename
#             spec_tbl = ascii.read(spec_filepath)

#             classification_result_medium0 = ascii.read(wiserep_meta.parent / (Path(spec_filename).stem + '_medium0.fit'))
#             if objname in classification_result_medium0['SN_SUBCLASS']:
#                 idx = classification_result_medium0['SN_SUBCLASS'] == objname
#                 classification_result_medium0 = classification_result_medium0[~idx]

#             classification_result_medium1 = ascii.read(wiserep_meta.parent / (Path(spec_filename).stem + '_medium1.fit'))
#             if objname in classification_result_medium1['SN_SUBCLASS']:
#                 idx = classification_result_medium1['SN_SUBCLASS'] == objname
#                 classification_result_medium1 = classification_result_medium1[~idx]

#             classification_result_medium2 = ascii.read(wiserep_meta.parent / (Path(spec_filename).stem + '_medium2.fit'))
#             if objname in classification_result_medium2['SN_SUBCLASS']:
#                 idx = classification_result_medium2['SN_SUBCLASS'] == objname
#                 classification_result_medium2 = classification_result_medium2[~idx]

#             classification_result_medium3 = ascii.read(wiserep_meta.parent / (Path(spec_filename).stem + '_medium3.fit'))
#             if objname in classification_result_medium3['SN_SUBCLASS']:
#                 idx = classification_result_medium3['SN_SUBCLASS'] == objname
#                 classification_result_medium3 = classification_result_medium3[~idx]

#             classification_result_medium4 = ascii.read(wiserep_meta.parent / (Path(spec_filename).stem + '_medium4.fit'))
#             if objname in classification_result_medium4['SN_SUBCLASS']:
#                 idx = classification_result_medium4['SN_SUBCLASS'] == objname
#                 classification_result_medium4 = classification_result_medium4[~idx]

#             classification_result_medium5 = ascii.read(wiserep_meta.parent / (Path(spec_filename).stem + '_medium5.fit'))
#             if objname in classification_result_medium5['SN_SUBCLASS']:
#                 idx = classification_result_medium5['SN_SUBCLASS'] == objname
#                 classification_result_medium5 = classification_result_medium5[~idx]

#             result_dict_medium0 = {}
#             result_dict_medium0['objname'] = objname
#             result_dict_medium0['maxdate'] = maxdate
#             result_dict_medium0['wl_min'] = np.min(row['Lambda-min'])
#             result_dict_medium0['wl_max'] = np.max(row['Lambda-max'])
#             result_dict_medium0['num_synphot'] = len(spec_tbl)
#             result_dict_medium0['num_filterset'] = len(medium_filterset_0)

#             result_dict_medium1 = {}
#             result_dict_medium1['objname'] = objname
#             result_dict_medium1['maxdate'] = maxdate
#             result_dict_medium1['wl_min'] = np.min(row['Lambda-min'])
#             result_dict_medium1['wl_max'] = np.max(row['Lambda-max'])
#             result_dict_medium1['num_synphot'] = len(spec_tbl)
#             result_dict_medium1['num_filterset'] = len(medium_filterset_1)

#             result_dict_medium2 = {}
#             result_dict_medium2['objname'] = objname
#             result_dict_medium2['maxdate'] = maxdate
#             result_dict_medium2['wl_min'] = np.min(row['Lambda-min'])
#             result_dict_medium2['wl_max'] = np.max(row['Lambda-max'])
#             result_dict_medium2['num_synphot'] = len(spec_tbl)
#             result_dict_medium2['num_filterset'] = len(medium_filterset_2)

#             result_dict_medium3 = {}
#             result_dict_medium3['objname'] = objname
#             result_dict_medium3['maxdate'] = maxdate
#             result_dict_medium3['wl_min'] = np.min(row['Lambda-min'])
#             result_dict_medium3['wl_max'] = np.max(row['Lambda-max'])
#             result_dict_medium3['num_synphot'] = len(spec_tbl)
#             result_dict_medium3['num_filterset'] = len(medium_filterset_3)

#             result_dict_medium4 = {}
#             result_dict_medium4['objname'] = objname
#             result_dict_medium4['maxdate'] = maxdate
#             result_dict_medium4['wl_min'] = np.min(row['Lambda-min'])
#             result_dict_medium4['wl_max'] = np.max(row['Lambda-max'])
#             result_dict_medium4['num_synphot'] = len(spec_tbl)
#             result_dict_medium4['num_filterset'] = len(medium_filterset_4)

#             result_dict_medium5 = {}
#             result_dict_medium5['objname'] = objname
#             result_dict_medium5['maxdate'] = maxdate
#             result_dict_medium5['wl_min'] = np.min(row['Lambda-min'])
#             result_dict_medium5['wl_max'] = np.max(row['Lambda-max'])
#             result_dict_medium5['num_synphot'] = len(spec_tbl)
#             result_dict_medium5['num_filterset'] = len(medium_filterset_5)

            
#             result_dict_medium0.update(dict(row))
#             result_dict_medium1.update(dict(row))
#             result_dict_medium2.update(dict(row))
#             result_dict_medium3.update(dict(row))
#             result_dict_medium4.update(dict(row))
#             result_dict_medium5.update(dict(row))
#             result_dict_medium0.update(dict(classification_result_medium0[0]))
#             result_dict_medium1.update(dict(classification_result_medium1[0]))
#             result_dict_medium2.update(dict(classification_result_medium2[0]))
#             result_dict_medium3.update(dict(classification_result_medium3[0]))
#             result_dict_medium4.update(dict(classification_result_medium4[0]))
#             result_dict_medium5.update(dict(classification_result_medium5[0]))
#             all_results_dict_medium0.append(result_dict_medium0)
#             all_results_dict_medium1.append(result_dict_medium1)
#             all_results_dict_medium2.append(result_dict_medium2)
#             all_results_dict_medium3.append(result_dict_medium3)
#             all_results_dict_medium4.append(result_dict_medium4)
#             all_results_dict_medium5.append(result_dict_medium5)
#             # print(f"Real: {row['Obj. Type']}")
#             # print(classification_result_medium0['SN_TYPE'][0])
#             # print(classification_result_medium1['SN_TYPE'][0])
#             # print(classification_result_medium2['SN_TYPE'][0])
#             # print(classification_result_medium3['SN_TYPE'][0])
#             # print(classification_result_medium4['SN_TYPE'][0])
#             # print(classification_result_medium5['SN_TYPE'][0])
#     except Exception as e:
#         print(f"Error: {e}")
#         failed_filelist.append(wiserep_meta)
# #%%
# #%%
# #%%
# # all_tables_tbl_raw = Table(all_results_dict_raw)
# all_results_tbl_medium0 = Table(all_results_dict_medium0)
# all_results_tbl_medium1 = Table(all_results_dict_medium1)
# all_results_tbl_medium2 = Table(all_results_dict_medium2)
# all_results_tbl_medium3 = Table(all_results_dict_medium3)
# all_results_tbl_medium4 = Table(all_results_dict_medium4)
# all_results_tbl_medium5 = Table(all_results_dict_medium5)
# all_results_tbl_medium0['transient_type'] = all_results_tbl_medium0['Obj. Type']
# all_results_tbl_medium1['transient_type'] = all_results_tbl_medium1['Obj. Type']
# all_results_tbl_medium2['transient_type'] = all_results_tbl_medium2['Obj. Type']
# all_results_tbl_medium3['transient_type'] = all_results_tbl_medium3['Obj. Type']
# all_results_tbl_medium4['transient_type'] = all_results_tbl_medium4['Obj. Type']
# all_results_tbl_medium5['transient_type'] = all_results_tbl_medium5['Obj. Type']
# all_tables_tbl_raw.write(Path(SNAL_DIR) / 'NGSF_all_results_tbl_raw.csv', format='ascii.fixed_width', overwrite=True)
#%%
# all_results_tbl_medium0.write(Path(SNAL_DIR) / 'NGSF_all_results_tbl_medium0.csv', format='ascii.fixed_width', overwrite=True)
# all_results_tbl_medium1.write(Path(SNAL_DIR) / 'NGSF_all_results_tbl_medium1.csv', format='ascii.fixed_width', overwrite=True)
# all_results_tbl_medium2.write(Path(SNAL_DIR) / 'NGSF_all_results_tbl_medium2.csv', format='ascii.fixed_width', overwrite=True)
# all_results_tbl_medium3.write(Path(SNAL_DIR) / 'NGSF_all_results_tbl_medium3.csv', format='ascii.fixed_width', overwrite=True)
# all_results_tbl_medium4.write(Path(SNAL_DIR) / 'NGSF_all_results_tbl_medium4.csv', format='ascii.fixed_width', overwrite=True)
# all_results_tbl_medium5.write(Path(SNAL_DIR) / 'NGSF_all_results_tbl_medium5.csv', format='ascii.fixed_width', overwrite=True)
#%%
# all_tables_tbl_raw = Table.read(Path(SNAL_DIR) / 'NGSF_all_results_tbl_raw.csv', format='ascii.fixed_width')
all_results_tbl_medium0 = Table.read(Path(SNAL_DIR) / 'NGSF_all_results_tbl_medium0.csv', format='ascii.fixed_width')
all_results_tbl_medium1 = Table.read(Path(SNAL_DIR) / 'NGSF_all_results_tbl_medium1.csv', format='ascii.fixed_width')
all_results_tbl_medium2 = Table.read(Path(SNAL_DIR) / 'NGSF_all_results_tbl_medium2.csv', format='ascii.fixed_width')
all_results_tbl_medium3 = Table.read(Path(SNAL_DIR) / 'NGSF_all_results_tbl_medium3.csv', format='ascii.fixed_width')
all_results_tbl_medium4 = Table.read(Path(SNAL_DIR) / 'NGSF_all_results_tbl_medium4.csv', format='ascii.fixed_width')
all_results_tbl_medium5 = Table.read(Path(SNAL_DIR) / 'NGSF_all_results_tbl_medium5.csv', format='ascii.fixed_width')
#%%

def _col_to_float_array(col, nrows):
    """Table column -> 1d float array; invalid / masked -> nan."""
    out = np.full(nrows, np.nan, dtype=float)
    for i in range(nrows):
        v = col[i]
        if hasattr(col, 'mask') and col.mask is not None and col.mask[i]:
            continue
        if v is None:
            continue
        if isinstance(v, str) and v.strip() in ('', 'None', 'nan', 'NaN'):
            continue
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _coarse_supernova_family_from_metadata(transient_type):
    """Map TNS-style transient_type to a coarse family for comparison with NGSF SN_TYPE."""
    t = str(transient_type).strip().upper().replace('SN', '').strip()
    if not t or t == 'NONE':
        return 'unknown'
    if t.startswith('IA') or t.startswith('IA-'):
        return 'Ia'
    if 'IIP' in t or 'IIN' in t:
        return 'II'
    if t.startswith('II'):
        return 'II'
    if t.startswith('IB') or t.startswith('IC'):
        return 'Ibc'
    return 'unknown'


def _coarse_supernova_family_from_fit(sn_type):
    """Map NGSF SN_TYPE string to the same coarse family."""
    t = str(sn_type).strip().upper()
    if not t or t == 'NONE':
        return 'unknown'
    if t.startswith('IA') or t.startswith('IA-') or t.startswith('SUPER_'):
        return 'Ia'
    if t.startswith('II') or 'IIB' in t or 'IIN' in t or t == 'II':
        return 'II'
    if t.startswith('IB') or t.startswith('IC'):
        return 'Ibc'
    return 'unknown'


def add_ngsf_metadata_comparison(tbl):
    """
    Add observed phase (mjd - maxdate) and deltas vs fit (Z, Phase, coarse type).

    Fit columns: Z, Phase, SN_TYPE. Metadata: redshift, transient_type, mjd, maxdate.
    """
    n = len(tbl)
    mjd = Time(_col_to_float_array(tbl['JD'], n), format = 'jd').mjd
    maxdate = _col_to_float_array(tbl['maxdate'], n)
    phase_obs_mjd = mjd - maxdate

    z_fit = _col_to_float_array(tbl['Z'], n)
    z_meta = _col_to_float_array(tbl['Redshift'], n)
    phase_fit = _col_to_float_array(tbl['Phase'], n)

    out = tbl.copy()
    out['phase_obs_mjd'] = phase_obs_mjd
    out['delta_z'] = z_fit - z_meta
    out['delta_phase'] = phase_fit - phase_obs_mjd

    fam_meta = [_coarse_supernova_family_from_metadata(x) for x in tbl['transient_type']]
    fam_fit = [_coarse_supernova_family_from_fit(x) for x in tbl['SN_TYPE']]
    out['type_family_meta'] = fam_meta
    out['type_family_fit'] = fam_fit
    out['type_family_match'] = np.array(
        [m == f and m != 'unknown' for m, f in zip(fam_meta, fam_fit)],
        dtype=bool,
    )
    return out


_JOIN_KEY_OBJ = 'objname'
_JOIN_KEY_MJD_ROUND = '__mjd_join__'
_MEDIUM_TABLE_SUFFIX = '_med'


def merge_ngsf_raw_and_medium(tbl_raw_comp, tbl_medium_comp):
    """
    Inner-join raw vs medium-band NGSF result tables on ``objname`` and rounded ``mjd`` (same spectrum).

    ``mjd`` is rounded to 6 decimals before joining to avoid float mismatch. Columns present in
    both tables get ``_med`` on the medium side (e.g. ``Z`` vs ``Z_med``).
    """
    tr = tbl_raw_comp.copy()
    tm = tbl_medium_comp.copy()
    if _JOIN_KEY_OBJ not in tr.colnames or _JOIN_KEY_OBJ not in tm.colnames:
        raise ValueError(f'missing join key {_JOIN_KEY_OBJ!r}')
    if 'mjd' not in tr.colnames or 'mjd' not in tm.colnames:
        raise ValueError("missing 'mjd' column")
    n_r, n_m = len(tr), len(tm)
    tr[_JOIN_KEY_MJD_ROUND] = np.round(_col_to_float_array(tr['mjd'], n_r), 6)
    tm[_JOIN_KEY_MJD_ROUND] = np.round(_col_to_float_array(tm['mjd'], n_m), 6)
    join_keys = [_JOIN_KEY_OBJ, _JOIN_KEY_MJD_ROUND]
    overlap = (set(tr.colnames) & set(tm.colnames)) - set(join_keys)
    for c in sorted(overlap):
        tm.rename_column(c, f'{c}{_MEDIUM_TABLE_SUFFIX}')
    out = join(tr, tm, keys=join_keys, join_type='inner')
    out.remove_column(_JOIN_KEY_MJD_ROUND)
    return out


def add_medium_vs_raw_metrics(merged_tbl):
    """Add deltas and SN_TYPE agreement columns (expects ``Z`` / ``Phase`` / ``SN_TYPE`` raw and ``*_med``)."""
    t = merged_tbl.copy()
    n = len(t)
    z_r = _col_to_float_array(t['Z'], n)
    z_m = _col_to_float_array(t[f'Z{_MEDIUM_TABLE_SUFFIX}'], n)
    p_r = _col_to_float_array(t['Phase'], n)
    p_m = _col_to_float_array(t[f'Phase{_MEDIUM_TABLE_SUFFIX}'], n)
    t['delta_Z_medium_minus_raw'] = z_m - z_r
    t['delta_Phase_medium_minus_raw'] = p_m - p_r
    sn_r = np.array([str(t['SN_TYPE'][i]).strip() for i in range(n)])
    sn_m = np.array([str(t[f'SN_TYPE{_MEDIUM_TABLE_SUFFIX}'][i]).strip() for i in range(n)])
    t['SN_TYPE_raw_eq_medium'] = sn_r == sn_m
    fam_r = np.array([_coarse_supernova_family_from_fit(t['SN_TYPE'][i]) for i in range(n)])
    fam_m = np.array(
        [_coarse_supernova_family_from_fit(t[f'SN_TYPE{_MEDIUM_TABLE_SUFFIX}'][i]) for i in range(n)]
    )
    t['coarse_SN_TYPE_raw_eq_medium'] = (fam_r == fam_m) & (fam_r != 'unknown') & (fam_m != 'unknown')
    return t


def summarize_medium_vs_raw(merged_tbl, label=''):
    """Print agreement between spectrum-based (raw) and medium-band NGSF fits."""
    n = len(merged_tbl)
    dz = np.asarray(merged_tbl['delta_Z_medium_minus_raw'], dtype=float)
    dp = np.asarray(merged_tbl['delta_Phase_medium_minus_raw'], dtype=float)
    ok_z = np.isfinite(dz)
    ok_p = np.isfinite(dp)
    eq_sn = np.asarray(merged_tbl['SN_TYPE_raw_eq_medium'], dtype=bool)
    eq_coarse = np.asarray(merged_tbl['coarse_SN_TYPE_raw_eq_medium'], dtype=bool)
    print(f'--- medium vs raw {label} ---')
    print(f'joined rows: {n}')
    print(f'SN_TYPE exact match: {np.sum(eq_sn)} / {n} ({100 * np.mean(eq_sn):.1f}%)')
    print(f'SN_TYPE coarse family match: {np.sum(eq_coarse)} / {n} ({100 * np.mean(eq_coarse):.1f}%)')
    if np.any(ok_z):
        print(
            f"ΔZ (med−raw): median |Δ| = {np.nanmedian(np.abs(dz[ok_z])):.4f}, "
            f"mean |Δ| = {np.nanmean(np.abs(dz[ok_z])):.4f} (n={np.sum(ok_z)})"
        )
    if np.any(ok_p):
        print(
            f"ΔPhase (med−raw): median |Δ| = {np.nanmedian(np.abs(dp[ok_p])):.2f} d, "
            f"mean |Δ| = {np.nanmean(np.abs(dp[ok_p])):.2f} d (n={np.sum(ok_p)})"
        )


def plot_ngsf_medium_vs_raw(
    merged_tbl,
    out_dir,
    stem='NGSF_medium_vs_raw',
    title_prefix='NGSF medium vs raw',
    phase_sigma_clip=3.0,
    transient_type='all',
):
    """
    Four panels: Z_med vs Z_raw, Phase_med vs Phase_raw (σ-clipped limits), histograms of ΔZ and ΔPhase.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suf = _stem_suffix_for_type_filter(transient_type)
    stem_out = stem if not suf else f'{stem}_{suf}'
    type_label = _type_filter_label(transient_type)

    n = len(merged_tbl)
    mask_tt = _transient_type_mask(merged_tbl, transient_type)
    n_sel = int(np.sum(mask_tt))

    z_r = _as_float_array(merged_tbl['Z'], n)
    z_m = _as_float_array(merged_tbl[f'Z{_MEDIUM_TABLE_SUFFIX}'], n)
    p_r = _as_float_array(merged_tbl['Phase'], n)
    p_m = _as_float_array(merged_tbl[f'Phase{_MEDIUM_TABLE_SUFFIX}'], n)
    dz = _as_float_array(merged_tbl['delta_Z_medium_minus_raw'], n)
    dp = _as_float_array(merged_tbl['delta_Phase_medium_minus_raw'], n)
    eq_coarse = np.asarray(merged_tbl['coarse_SN_TYPE_raw_eq_medium'], dtype=bool)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0))
    fig.suptitle(f'{title_prefix} ({type_label}, n={n_sel})', fontsize=12, fontweight='bold')

    okz = mask_tt & np.isfinite(z_r) & np.isfinite(z_m)
    if np.any(okz):
        sc = axes[0, 0].scatter(
            z_r[okz], z_m[okz], c=eq_coarse[okz], cmap='RdYlGn', alpha=0.55, s=14, vmin=0, vmax=1,
        )
        fig.colorbar(sc, ax=axes[0, 0], label='coarse SN_TYPE match', shrink=0.8)
        lo = min(np.nanmin(z_r[okz]), np.nanmin(z_m[okz]))
        hi = max(np.nanmax(z_r[okz]), np.nanmax(z_m[okz]))
        if np.isfinite(lo) and np.isfinite(hi):
            axes[0, 0].plot([lo, hi], [lo, hi], 'k--', lw=1)
        axes[0, 0].set_xlabel('Z raw (spectrum)')
        axes[0, 0].set_ylabel('Z medium-band')
        axes[0, 0].set_aspect('equal', adjustable='box')
    else:
        axes[0, 0].text(0.5, 0.5, 'No finite Z pairs', ha='center', va='center', transform=axes[0, 0].transAxes)

    okp = mask_tt & np.isfinite(p_r) & np.isfinite(p_m)
    if np.any(okp):
        sc2 = axes[0, 1].scatter(
            p_r[okp], p_m[okp], c=eq_coarse[okp], cmap='RdYlGn', alpha=0.55, s=14, vmin=0, vmax=1,
        )
        fig.colorbar(sc2, ax=axes[0, 1], label='coarse SN_TYPE match', shrink=0.8)
        lims = _phase_axis_limits_sigma_clip(p_r, p_m, okp, sigma=phase_sigma_clip)
        if lims is not None:
            lo, hi = lims
            axes[0, 1].set_xlim(lo, hi)
            axes[0, 1].set_ylim(lo, hi)
        else:
            lo = min(np.nanmin(p_r[okp]), np.nanmin(p_m[okp]))
            hi = max(np.nanmax(p_r[okp]), np.nanmax(p_m[okp]))
        if np.isfinite(lo) and np.isfinite(hi):
            axes[0, 1].plot([lo, hi], [lo, hi], 'k--', lw=1)
        axes[0, 1].set_xlabel('Phase raw [d]')
        axes[0, 1].set_ylabel('Phase medium-band [d]')
        axes[0, 1].set_aspect('equal', adjustable='box')
    else:
        axes[0, 1].text(0.5, 0.5, 'No finite Phase pairs', ha='center', va='center', transform=axes[0, 1].transAxes)

    okdz = mask_tt & np.isfinite(dz)
    if np.any(okdz):
        axes[1, 0].hist(dz[okdz], bins=40, color='steelblue', alpha=0.85, edgecolor='white')
    axes[1, 0].set_xlabel('ΔZ (medium − raw)')
    axes[1, 0].set_ylabel('count')
    axes[1, 0].axvline(0, color='k', ls='--', lw=0.8)

    okdp = mask_tt & np.isfinite(dp)
    if np.any(okdp):
        axes[1, 1].hist(dp[okdp], bins=40, color='coral', alpha=0.85, edgecolor='white')
    axes[1, 1].set_xlabel('ΔPhase (medium − raw) [d]')
    axes[1, 1].set_ylabel('count')
    axes[1, 1].axvline(0, color='k', ls='--', lw=0.8)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    p_out = out_dir / f'{stem_out}_medium_vs_raw.png'
    fig.savefig(p_out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {p_out}')
    return p_out


def plot_ngsf_medium_vs_raw_by_ngsf_sn_type(
    merged_tbl,
    out_dir,
    stem='NGSF_medium_vs_raw',
    title_prefix='NGSF medium vs raw',
    min_count=3,
    transient_type='all',
    group_by='raw',
):
    """
    Horizontal bar charts grouped by **NGSF fit** ``SN_TYPE``: either the raw-spectrum label
    (``group_by='raw'``) or the medium-band label (``group_by='medium'``). For each class, shows
    the fraction of rows where raw vs medium NGSF agree (coarse family and exact string).

    ``transient_type`` applies a metadata row filter (same as :func:`plot_ngsf_medium_vs_raw`)
    before grouping.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suf = _stem_suffix_for_type_filter(transient_type)
    stem_out = stem if not suf else f'{stem}_{suf}'
    type_label = _type_filter_label(transient_type)
    gb = str(group_by).strip().lower()
    if gb not in ('raw', 'medium'):
        raise ValueError("group_by must be 'raw' or 'medium'")

    mask_tt = _transient_type_mask(merged_tbl, transient_type)
    tbl_f = merged_tbl[mask_tt]
    n_sel = len(tbl_f)
    if n_sel == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No rows in selection', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{title_prefix} — by NGSF SN_TYPE ({type_label})')
        tag = 'raw' if gb == 'raw' else 'med'
        p_out = out_dir / f'{stem_out}_medium_vs_raw_by_ngsf_sn_type_{tag}.png'
        fig.savefig(p_out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {p_out}')
        return p_out

    if gb == 'raw':
        types = np.array([str(t).strip() or 'missing' for t in tbl_f['SN_TYPE']])
        axis_note = 'rows grouped by SN_TYPE (raw / spectrum NGSF)'
    else:
        types = np.array(
            [str(t).strip() or 'missing' for t in tbl_f[f'SN_TYPE{_MEDIUM_TABLE_SUFFIX}']]
        )
        axis_note = 'rows grouped by SN_TYPE (medium-band NGSF)'

    eq_coarse = np.asarray(tbl_f['coarse_SN_TYPE_raw_eq_medium'], dtype=bool)
    eq_ex = np.asarray(tbl_f['SN_TYPE_raw_eq_medium'], dtype=bool)
    uniq, inv = np.unique(types, return_inverse=True)
    counts = np.bincount(inv)
    rate_c = np.array([np.mean(eq_coarse[inv == j]) for j in range(len(uniq))])
    rate_e = np.array([np.mean(eq_ex[inv == j]) for j in range(len(uniq))])
    order = np.argsort(-counts)
    u, rc, re, c = uniq[order], rate_c[order], rate_e[order], counts[order]
    show = c >= min_count
    if not np.any(show):
        show = np.ones(len(u), dtype=bool)

    u_s, rc_s, re_s, c_s = u[show], rc[show], re[show], c[show]
    y = np.arange(len(u_s))
    h = 0.35

    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.32 * len(u_s))))
    fig.suptitle(
        f'{title_prefix} — raw vs medium NGSF SN_TYPE ({type_label}, n={n_sel})',
        fontsize=12,
        fontweight='bold',
    )

    ax.barh(y - h / 2, rc_s, height=h, label='coarse match', color='steelblue', alpha=0.9)
    ax.barh(y + h / 2, re_s, height=h, label='exact match', color='darkorange', alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{ui[:40]} (n={ci})' for ui, ci in zip(u_s, c_s)], fontsize=7)
    ax.set_xlabel('fraction of rows where raw SN_TYPE ≡ medium SN_TYPE')
    ax.set_xlim(0, 1.05)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title(axis_note, fontsize=9)

    if min_count > 1 and np.any(~show):
        fig.text(
            0.5,
            0.01,
            f'NGSF classes with n < {min_count} omitted ({int(np.sum(~show))} labels)',
            ha='center',
            fontsize=8,
            style='italic',
        )

    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    tag = 'raw' if gb == 'raw' else 'med'
    p_out = out_dir / f'{stem_out}_medium_vs_raw_by_ngsf_sn_type_{tag}.png'
    fig.savefig(p_out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {p_out}')
    return p_out


def _metadata_transient_type_column(tbl):
    """Column name for TNS ``transient_type`` after raw–medium join (raw keeps bare name)."""
    if 'transient_type' in tbl.colnames:
        return 'transient_type'
    k = f'transient_type{_MEDIUM_TABLE_SUFFIX}'
    if k in tbl.colnames:
        return k
    raise ValueError('missing transient_type column for metadata grouping')


def plot_ngsf_medium_vs_raw_by_metadata_transient_type(
    merged_tbl,
    out_dir,
    stem='NGSF_medium_vs_raw',
    title_prefix='NGSF medium vs raw',
    min_count=3,
    transient_type='all',
):
    """
    Horizontal bar charts grouped by **metadata** ``transient_type`` (TNS label). For each class,
    shows the fraction of rows where raw vs medium NGSF agree (coarse family and exact ``SN_TYPE``).

    ``transient_type`` (the parameter) applies a metadata row filter (same as
    :func:`plot_ngsf_medium_vs_raw`) before grouping.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suf = _stem_suffix_for_type_filter(transient_type)
    stem_out = stem if not suf else f'{stem}_{suf}'
    type_label = _type_filter_label(transient_type)

    mask_tt = _transient_type_mask(merged_tbl, transient_type)
    tbl_f = merged_tbl[mask_tt]
    n_sel = len(tbl_f)
    col_tt = _metadata_transient_type_column(tbl_f)

    if n_sel == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No rows in selection', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{title_prefix} — by metadata transient_type ({type_label})')
        p_out = out_dir / f'{stem_out}_medium_vs_raw_by_metadata_transient_type.png'
        fig.savefig(p_out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {p_out}')
        return p_out

    types = np.array([str(t).strip() or 'missing' for t in tbl_f[col_tt]])
    axis_note = f'rows grouped by metadata transient_type ({col_tt})'

    eq_coarse = np.asarray(tbl_f['coarse_SN_TYPE_raw_eq_medium'], dtype=bool)
    eq_ex = np.asarray(tbl_f['SN_TYPE_raw_eq_medium'], dtype=bool)
    uniq, inv = np.unique(types, return_inverse=True)
    counts = np.bincount(inv)
    rate_c = np.array([np.mean(eq_coarse[inv == j]) for j in range(len(uniq))])
    rate_e = np.array([np.mean(eq_ex[inv == j]) for j in range(len(uniq))])
    order = np.argsort(-counts)
    u, rc, re, c = uniq[order], rate_c[order], rate_e[order], counts[order]
    show = c >= min_count
    if not np.any(show):
        show = np.ones(len(u), dtype=bool)

    u_s, rc_s, re_s, c_s = u[show], rc[show], re[show], c[show]
    y = np.arange(len(u_s))
    h = 0.35

    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.32 * len(u_s))))
    fig.suptitle(
        f'{title_prefix} — raw vs medium NGSF by metadata type ({type_label}, n={n_sel})',
        fontsize=12,
        fontweight='bold',
    )

    ax.barh(y - h / 2, rc_s, height=h, label='coarse match', color='steelblue', alpha=0.9)
    ax.barh(y + h / 2, re_s, height=h, label='exact match', color='darkorange', alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{ui[:40]} (n={ci})' for ui, ci in zip(u_s, c_s)], fontsize=7)
    ax.set_xlabel('fraction of rows where raw SN_TYPE ≡ medium SN_TYPE')
    ax.set_xlim(0, 1.05)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title(axis_note, fontsize=9)

    if min_count > 1 and np.any(~show):
        fig.text(
            0.5,
            0.01,
            f'metadata types with n < {min_count} omitted ({int(np.sum(~show))} labels)',
            ha='center',
            fontsize=8,
            style='italic',
        )

    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    p_out = out_dir / f'{stem_out}_medium_vs_raw_by_metadata_transient_type.png'
    fig.savefig(p_out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {p_out}')
    return p_out


def plot_ngsf_medium_vs_raw_coarse_confusion(
    merged_tbl,
    out_dir,
    stem='NGSF_medium_vs_raw',
    title_prefix='NGSF medium vs raw',
    transient_type='all',
):
    """
    Heatmap: coarse SN family from **raw** NGSF (rows) vs **medium** NGSF (columns), counts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suf = _stem_suffix_for_type_filter(transient_type)
    stem_out = stem if not suf else f'{stem}_{suf}'
    type_label = _type_filter_label(transient_type)

    mask_tt = _transient_type_mask(merged_tbl, transient_type)
    tbl_f = merged_tbl[mask_tt]
    n_sel = len(tbl_f)
    labels = ['Ia', 'II', 'Ibc', 'unknown']
    nlab = len(labels)

    fam_r = np.array([_coarse_supernova_family_from_fit(tbl_f['SN_TYPE'][i]) for i in range(n_sel)])
    fam_m = np.array(
        [
            _coarse_supernova_family_from_fit(tbl_f[f'SN_TYPE{_MEDIUM_TABLE_SUFFIX}'][i])
            for i in range(n_sel)
        ]
    )

    def _idx(f):
        return labels.index(f) if f in labels else labels.index('unknown')

    mat = np.zeros((nlab, nlab), dtype=float)
    for i in range(n_sel):
        mat[_idx(fam_r[i]), _idx(fam_m[i])] += 1.0

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(mat, aspect='auto', cmap='Blues')
    ax.set_xticks(np.arange(nlab))
    ax.set_yticks(np.arange(nlab))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('medium NGSF coarse family')
    ax.set_ylabel('raw NGSF coarse family')
    ax.set_title(f'{title_prefix} — coarse SN_TYPE (n={n_sel}, {type_label})')

    row_sum = mat.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    pct = 100.0 * mat / row_sum
    for i in range(nlab):
        for j in range(nlab):
            c = int(mat[i, j])
            if c == 0:
                continue
            p = pct[i, j]
            ax.text(j, i, f'{c}\n({p:.0f}%)', ha='center', va='center', color='black', fontsize=8)

    fig.colorbar(im, ax=ax, label='count', shrink=0.8)
    fig.tight_layout()
    p_out = out_dir / f'{stem_out}_medium_vs_raw_coarse_confusion.png'
    fig.savefig(p_out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {p_out}')
    return p_out


def write_medium_vs_raw_analysis(
    tbl_raw_comp,
    path_medium_csv,
    out_dir,
    medium_name='medium0',
    transient_type='all',
):
    """
    Load medium results CSV, join to ``tbl_raw_comp``, write comparison table, print summary, plot.
    """
    path_medium_csv = Path(path_medium_csv)
    out_dir = Path(out_dir)
    if not path_medium_csv.is_file():
        print(f'skip {medium_name}: missing {path_medium_csv}')
        return None
    tbl_m = add_ngsf_metadata_comparison(
        Table.read(path_medium_csv, format='ascii.fixed_width')
    )
    merged = merge_ngsf_raw_and_medium(tbl_raw_comp, tbl_m)
    merged = add_medium_vs_raw_metrics(merged)
    summarize_medium_vs_raw(merged, label=f'{medium_name} ({len(merged)} rows)')
    out_csv = out_dir / f'NGSF_{medium_name}_vs_raw_comparison.csv'
    merged.write(out_csv, format='ascii.fixed_width', overwrite=True)
    print(f'Wrote: {out_csv}')
    plot_ngsf_medium_vs_raw(
        merged,
        out_dir=out_dir,
        stem=f'NGSF_{medium_name}_vs_raw',
        title_prefix=f'NGSF {medium_name} vs raw',
        transient_type=transient_type,
    )
    plot_ngsf_medium_vs_raw_by_metadata_transient_type(
        merged,
        out_dir=out_dir,
        stem=f'NGSF_{medium_name}_vs_raw',
        title_prefix=f'NGSF {medium_name} vs raw',
        transient_type=transient_type,
    )
    plot_ngsf_medium_vs_raw_coarse_confusion(
        merged,
        out_dir=out_dir,
        stem=f'NGSF_{medium_name}_vs_raw',
        title_prefix=f'NGSF {medium_name} vs raw',
        transient_type=transient_type,
    )
    return merged


def summarize_ngsf_comparison(tbl_with_comp, label=''):
    """Print summary stats for rows with finite fit Z and Phase."""
    dz = np.asarray(tbl_with_comp['delta_z'], dtype=float)
    dp = np.asarray(tbl_with_comp['delta_phase'], dtype=float)
    ok_z = np.isfinite(dz)
    ok_p = np.isfinite(dp)
    tm = np.asarray(tbl_with_comp['type_family_match'], dtype=bool)
    print(f'--- NGSF vs metadata {label} ---')
    print(f'rows: {len(tbl_with_comp)}')
    print(f"type_family_match: {np.sum(tm)} / {len(tm)} ({100 * np.mean(tm):.1f}%)")
    if np.any(ok_z):
        print(
            f"delta_z: median |Δz| = {np.nanmedian(np.abs(dz[ok_z])):.4f}, "
            f"mean |Δz| = {np.nanmean(np.abs(dz[ok_z])):.4f} (n={np.sum(ok_z)})"
        )
    if np.any(ok_p):
        print(
            f"delta_phase: median |Δphase| = {np.nanmedian(np.abs(dp[ok_p])):.2f} d, "
            f"mean |Δphase| = {np.nanmean(np.abs(dp[ok_p])):.2f} d (n={np.sum(ok_p)})"
        )


def _as_float_array(col, n):
    return np.asarray(_col_to_float_array(col, n), dtype=float)


def _binned_stats(x, values, weights_match=None, n_bins=10):
    """
    Bin by x; return bin centers, median |values|, and mean match rate per bin (if weights_match given).
    """
    ok = np.isfinite(x) & np.isfinite(values)
    if weights_match is not None:
        ok = ok & np.isfinite(weights_match.astype(float))
    x = np.asarray(x)[ok]
    values = np.asarray(values)[ok]
    if len(x) < 3:
        return None
    if weights_match is not None:
        wm = np.asarray(weights_match, dtype=float)[ok]
    else:
        wm = None
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return None
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    med_abs = []
    match_rate = []
    counts = []
    for i in range(n_bins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if i == n_bins - 1:
            m = (x >= edges[i]) & (x <= edges[i + 1])
        counts.append(np.sum(m))
        if np.sum(m) == 0:
            med_abs.append(np.nan)
            match_rate.append(np.nan)
            continue
        med_abs.append(np.nanmedian(np.abs(values[m])))
        if wm is not None:
            match_rate.append(np.nanmean(wm[m]))
        else:
            match_rate.append(np.nan)
    return centers, np.array(med_abs), np.array(match_rate), np.array(counts, dtype=int)


def _phase_axis_limits_sigma_clip(p_obs, p_fit, okp, sigma=3.0, maxiters=5, pad_frac=0.03):
    """
    Robust square axis limits for phase 1:1 plot from sigma-clipped observed and fit phase.
    Falls back to full range if clipping leaves nothing.
    """
    po = np.asarray(p_obs[okp], dtype=float)
    pf = np.asarray(p_fit[okp], dtype=float)
    if po.size == 0:
        return None
    c1 = sigma_clip(po, sigma=sigma, maxiters=maxiters, cenfunc='median')
    c2 = sigma_clip(pf, sigma=sigma, maxiters=maxiters, cenfunc='median')
    v1 = c1.compressed()
    v2 = c2.compressed()
    if v1.size == 0:
        v1 = po
    if v2.size == 0:
        v2 = pf
    lo = min(float(np.min(v1)), float(np.min(v2)))
    hi = max(float(np.max(v1)), float(np.max(v2)))
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        span = max(np.finfo(float).eps, abs(lo) * 1e-6)
    pad = pad_frac * span
    return lo - pad, hi + pad


_COARSE_FAMILY_KEYS = {
    'ia': 'Ia',
    'ii': 'II',
    'ibc': 'Ibc',
}


def _mask_coarse_family(tbl, family: str):
    """``family`` is 'Ia', 'II', or 'Ibc' (same as ``type_family_meta``)."""
    n = len(tbl)
    if 'type_family_meta' in tbl.colnames:
        return np.array([str(x) == family for x in tbl['type_family_meta']], dtype=bool)
    return np.array(
        [
            _coarse_supernova_family_from_metadata(tbl['transient_type'][i]) == family
            for i in range(n)
        ],
        dtype=bool,
    )


def _transient_type_mask(tbl, transient_type):
    """
    Row mask for plotting.

    * ``'all'`` / ``'any'`` / empty — every row.
    * **Coarse family** (case-insensitive): ``'Ia'``, ``'II'``, ``'Ibc'`` — matches
      ``type_family_meta`` when present, else :func:`_coarse_supernova_family_from_metadata`.
    * **Otherwise** — first try **exact** match on full TNS ``transient_type`` (case-insensitive);
      if no rows match, try **substring** match when the filter has length ≥ 3 (e.g. ``'IIP'``
      matches ``'SN IIP'``).
    """
    key = str(transient_type).strip().lower()
    if key in ('', 'all', 'any'):
        return np.ones(len(tbl), dtype=bool)
    if key in _COARSE_FAMILY_KEYS:
        return _mask_coarse_family(tbl, _COARSE_FAMILY_KEYS[key])

    n = len(tbl)
    key_norm = key
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        t = str(tbl['transient_type'][i]).strip().lower()
        if t == key_norm:
            mask[i] = True
    if np.any(mask):
        return mask
    if len(key_norm) >= 3:
        for i in range(n):
            if key_norm in str(tbl['transient_type'][i]).lower():
                mask[i] = True
    return mask


def _type_filter_label(transient_type):
    """Human-readable label for plot titles."""
    key = str(transient_type).strip().lower()
    if key in ('', 'all', 'any'):
        return 'all types'
    if key in _COARSE_FAMILY_KEYS:
        fam = _COARSE_FAMILY_KEYS[key]
        return f'coarse {fam}'
    return str(transient_type).strip()


def _stem_suffix_for_type_filter(transient_type):
    """Safe fragment for output filenames; empty when ``all``."""
    key = str(transient_type).strip().lower()
    if key in ('', 'all', 'any'):
        return ''
    frag = re.sub(r'[^\w\-.]+', '_', key, flags=re.UNICODE).strip('_')
    return (frag[:80] if frag else 'filter')


def _tns_coarse_family_labels_per_row(tbl_f):
    """
    Per-row TNS **metadata** coarse family (Ia / II / Ibc / unknown) for binning plots.

    Uses ``type_family_meta`` from :func:`add_ngsf_metadata_comparison` when present; otherwise
    derives from ``transient_type`` via :func:`_coarse_supernova_family_from_metadata`.
    This is not the NGSF fit family (``type_family_fit`` / ``SN_TYPE``).
    """
    if 'type_family_meta' in tbl_f.colnames:
        return np.array([str(x) for x in tbl_f['type_family_meta']], dtype=object)
    n = len(tbl_f)
    return np.array(
        [_coarse_supernova_family_from_metadata(tbl_f['transient_type'][i]) for i in range(n)],
        dtype=object,
    )


def _sort_key_tns_coarse_family(label):
    order = {'Ia': 0, 'II': 1, 'Ibc': 2, 'unknown': 3}
    return order.get(str(label), 99)


def plot_ngsf_accuracy_four_panels(
    tbl,
    out_dir,
    stem='NGSF_accuracy',
    title_prefix='NGSF vs metadata',
    n_bins=12,
    phase_sigma_clip=3.0,
    transient_type='all',
):
    """
    Save four separate figures: (1) all data, (2) by TNS metadata coarse family, (3) by phase,
    (4) by redshift.

    Parameters
    ----------
    transient_type : str
        * ``'all'`` — every row.
        * Coarse families (case-insensitive): ``'Ia'``, ``'II'``, ``'Ibc'`` — same bins as
          ``type_family_meta`` / :func:`_coarse_supernova_family_from_metadata`.
        * Any other string — exact TNS ``transient_type`` label, or substring match if no exact
          hit (substring requires length ≥ 3), e.g. ``'SN IIP'`` or ``'IIP'``.

    Panel (2) bins rows by **TNS coarse family** (``type_family_meta``: Ia / II / Ibc / unknown),
    not by NGSF ``SN_TYPE`` or full-string bins that can track the fit.

    For (1), the phase panel x/y limits use sigma-clipped ranges (median-centered, default
    ``phase_sigma_clip`` σ) so a few extreme phases do not stretch the axes; all points are still
    drawn (points outside the window are clipped by matplotlib).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ttf = str(transient_type).strip().lower()
    suf = _stem_suffix_for_type_filter(transient_type)
    stem_out = stem if not suf else f'{stem}_{suf}'
    type_label = _type_filter_label(transient_type)

    n = len(tbl)
    z_meta = _as_float_array(tbl['Redshift'], n)
    z_fit = _as_float_array(tbl['Z'], n)
    p_obs = _as_float_array(tbl['phase_obs_mjd'], n)
    p_fit = _as_float_array(tbl['Phase'], n)
    dz = _as_float_array(tbl['delta_z'], n)
    dp = _as_float_array(tbl['delta_phase'], n)
    match = np.asarray(tbl['type_family_match'], dtype=bool)

    mask_tt = _transient_type_mask(tbl, transient_type)
    n_sel = int(np.sum(mask_tt))

    # 1) All data
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig.suptitle(
        f'{title_prefix} — all data ({type_label}, n={n_sel})',
        fontsize=12,
        fontweight='bold',
    )
    okz = mask_tt & np.isfinite(z_meta) & np.isfinite(z_fit)
    if np.any(okz):
        sc0 = axes[0].scatter(
            z_meta[okz], z_fit[okz], c=match[okz], cmap='RdYlGn', alpha=0.55, s=14, vmin=0, vmax=1,
        )
        fig.colorbar(sc0, ax=axes[0], label='type family match', shrink=0.8)
        lo = min(np.nanmin(z_meta[okz]), np.nanmin(z_fit[okz]))
        hi = max(np.nanmax(z_meta[okz]), np.nanmax(z_fit[okz]))
        if np.isfinite(lo) and np.isfinite(hi):
            axes[0].plot([lo, hi], [lo, hi], 'k--', lw=1)
        axes[0].set_xlabel('Metadata redshift')
        axes[0].set_ylabel('Fit Z')
        axes[0].set_aspect('equal', adjustable='box')
    okp = mask_tt & np.isfinite(p_obs) & np.isfinite(p_fit)
    if np.any(okp):
        sc1 = axes[1].scatter(
            p_obs[okp], p_fit[okp], c=match[okp], cmap='RdYlGn', alpha=0.55, s=14, vmin=0, vmax=1,
        )
        fig.colorbar(sc1, ax=axes[1], label='type family match', shrink=0.8)
        lims = _phase_axis_limits_sigma_clip(
            p_obs, p_fit, okp, sigma=phase_sigma_clip,
        )
        if lims is not None:
            lo, hi = lims
            axes[1].set_xlim(lo, hi)
            axes[1].set_ylim(lo, hi)
        else:
            lo = min(np.nanmin(p_obs[okp]), np.nanmin(p_fit[okp]))
            hi = max(np.nanmax(p_obs[okp]), np.nanmax(p_fit[okp]))
        if np.isfinite(lo) and np.isfinite(hi):
            axes[1].plot([lo, hi], [lo, hi], 'k--', lw=1)

        dp_ok = p_fit[okp] - p_obs[okp]
        dp_clipped = sigma_clip(dp_ok, sigma=3.0, maxiters=5, cenfunc='median')
        clipped_std = float(np.std(dp_clipped.compressed()))
        if np.isfinite(lo) and np.isfinite(hi) and np.isfinite(clipped_std):
            line_x = np.array([lo, hi])
            axes[1].fill_between(
                line_x, line_x - clipped_std, line_x + clipped_std,
                color='royalblue', alpha=0.18, label=rf'$y = x \pm 1\sigma$ ({clipped_std:.1f} d)',
            )
        axes[1].text(
            0.03, 0.97,
            rf'$\sigma_{{clip}}$ = {clipped_std:.1f} d  (n = {int(np.sum(okp))})',
            transform=axes[1].transAxes, fontsize=8,
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8),
        )
        axes[1].legend(fontsize=7, loc='lower right')

        axes[1].set_xlabel('Observed phase (mjd − maxdate) [d]')
        axes[1].set_ylabel('Fit Phase [d]')
        axes[1].set_aspect('equal', adjustable='box')
    fig.tight_layout()
    p1 = out_dir / f'{stem_out}_01_all_data.png'
    fig.savefig(p1, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {p1}')

    # 2) By TNS metadata coarse family (Ia / II / Ibc / unknown), not NGSF SN_TYPE
    tbl_f = tbl[mask_tt]
    match_f = np.asarray(tbl_f['type_family_match'], dtype=bool)
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(tbl_f) > 0:
        types = _tns_coarse_family_labels_per_row(tbl_f)
        uniq, inv = np.unique(types, return_inverse=True)
        counts = np.bincount(inv)
        rates = np.array([np.mean(match_f[inv == j]) for j in range(len(uniq))])
        sort_order = np.argsort([_sort_key_tns_coarse_family(u) for u in uniq])
        u, r, c = uniq[sort_order], rates[sort_order], counts[sort_order]
        n_types = len(u)
        fig.set_size_inches(8, min(28, max(4, 0.28 * max(1, n_types))))
        y = np.arange(len(u))
        ax.barh(y, r, color='steelblue', alpha=0.85)
        for yi, ri in zip(y, r):
            ax.text(ri + 0.01, yi, f'{ri * 100:.1f}%', va='center', ha='left', fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels([f'{str(ui)[:40]} (n={ci})' for ui, ci in zip(u, c)], fontsize=7)
        ax.set_xlabel('Coarse family match rate (TNS metadata bin vs NGSF)')
        ax.set_xlim(0, 1.12)
    else:
        ax.text(0.5, 0.5, 'No rows in selection', ha='center', va='center', transform=ax.transAxes)
    ax.set_title(f'{title_prefix} — by TNS coarse type (metadata) ({type_label})')
    fig.tight_layout()
    p2 = out_dir / f'{stem_out}_02_by_tns_coarse_metadata.png'
    fig.savefig(p2, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {p2}')

    # 3) By phase (same subset as transient_type)
    p_obs_s = p_obs[mask_tt]
    dp_s = dp[mask_tt]
    match_s = match[mask_tt]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax2 = ax.twinx()
    st = _binned_stats(p_obs_s, dp_s, weights_match=match_s.astype(float), n_bins=n_bins)
    if st is not None:
        c, med, mr, _ = st
        ax.plot(c, med, 'o-', color='C0', label='median |Δphase|')
        ax2.plot(c, mr, 's--', color='C1', label='type match')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='best', fontsize=8)
    ax.set_xlabel('Observed phase bin center (mjd − maxdate) [d]')
    ax.set_ylabel('median |Δphase| [d]')
    ax2.set_ylabel('type family match rate')
    ax.set_title(f'{title_prefix} — by observed phase ({type_label}, n={n_sel})')
    if n_sel == 0:
        ax.text(0.5, 0.5, 'No rows in selection', ha='center', va='center', transform=ax.transAxes)
    fig.tight_layout()
    p3 = out_dir / f'{stem_out}_03_by_phase.png'
    fig.savefig(p3, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {p3}')

    # 4) By redshift
    z_meta_s = z_meta[mask_tt]
    dz_s = dz[mask_tt]
    match_z = match[mask_tt]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax2 = ax.twinx()
    st = _binned_stats(z_meta_s, dz_s, weights_match=match_z.astype(float), n_bins=n_bins)
    if st is not None:
        c, med, mr, _ = st
        ax.plot(c, med, '^-', color='C2', label='median |Δz|')
        ax2.plot(c, mr, 'd:', color='C3', label='type match')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='best', fontsize=8)
    ax.set_xlabel('Metadata redshift bin center')
    ax.set_ylabel('median |Δz|')
    ax2.set_ylabel('type family match rate')
    ax.set_title(f'{title_prefix} — by redshift ({type_label})')
    fig.tight_layout()
    p4 = out_dir / f'{stem_out}_04_by_redshift.png'
    fig.savefig(p4, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {p4}')

    return p1, p2, p3, p4


#%%
# NGSF accuracy plots: ``'all'`` | coarse ``Ia``/``II``/``Ibc`` | full TNS label or substring (≥3 chars).
NGSF_PLOT_TRANSIENT_TYPE = 'all'
#%%
# Default path (override to use e.g. /home/hhchoi1022/snal/data/NGSF_all_results_tbl_medium0.csv)
# NGSF_RESULTS_RAW = Path('/home/hhchoi1022/snal/data/NGSF_all_results_tbl_raw.csv')
# if not NGSF_RESULTS_RAW.is_file():
#     NGSF_RESULTS_RAW = Path(SNAL_DIR) / 'NGSF_all_results_tbl_raw.csv'

# all_results_tbl_raw_comp = add_ngsf_metadata_comparison(
#     Table.read(NGSF_RESULTS_RAW, format='ascii.fixed_width')
# )
# all_results_tbl_raw_comp = all_results_tbl_raw_comp[(all_results_tbl_raw_comp['transient_type'] != 'NA/Unknown') & (all_results_tbl_raw_comp['transient_type'] != 'None') & (all_results_tbl_raw_comp['transient_type'] != 'SLSN-I') & (all_results_tbl_raw_comp['transient_type'] != 'SN I') ]
# #%%
# summarize_ngsf_comparison(all_results_tbl_raw_comp, label=str(NGSF_RESULTS_RAW))

# out_comp_path = NGSF_RESULTS_RAW.with_name(
#     NGSF_RESULTS_RAW.stem + '_comparison.csv'
# )
# all_results_tbl_raw_comp.write(out_comp_path, format='ascii.fixed_width', overwrite=True)
# print(f'Wrote: {out_comp_path}')

# plot_ngsf_accuracy_four_panels(
#     all_results_tbl_raw_comp,
#     out_dir=NGSF_RESULTS_RAW.parent,
#     stem='NGSF_accuracy_raw',
#     title_prefix='NGSF raw',
#     transient_type=NGSF_PLOT_TRANSIENT_TYPE,
# )
#%%

# Default path (override to use e.g. /home/hhchoi1022/snal/data/NGSF_all_results_tbl_medium0.csv)
NGSF_RESULTS_MEDIUM0 = Path('/home/hhchoi1022/snal/data/NGSF_all_results_tbl_medium0.csv')
if not NGSF_RESULTS_MEDIUM0.is_file():
    NGSF_RESULTS_MEDIUM0 = Path(SNAL_DIR) / 'NGSF_all_results_tbl_medium0.csv'

all_results_tbl_medium0_comp = add_ngsf_metadata_comparison(
    Table.read(NGSF_RESULTS_MEDIUM0, format='ascii.fixed_width')
)
all_results_tbl_medium0_comp = all_results_tbl_medium0_comp[(all_results_tbl_medium0_comp['transient_type'] != 'NA/Unknown') & (all_results_tbl_medium0_comp['transient_type'] != 'None') & (all_results_tbl_medium0_comp['transient_type'] != 'SLSN-I') & (all_results_tbl_medium0_comp['transient_type'] != 'SN I') ]
#%%
summarize_ngsf_comparison(all_results_tbl_medium0_comp, label=str(NGSF_RESULTS_MEDIUM0))

out_comp_path = NGSF_RESULTS_MEDIUM0.with_name(
    NGSF_RESULTS_MEDIUM0.stem + '_comparison.csv'
)
all_results_tbl_medium0_comp.write(out_comp_path, format='ascii.fixed_width', overwrite=True)
print(f'Wrote: {out_comp_path}')

plot_ngsf_accuracy_four_panels(
    all_results_tbl_medium0_comp,
    out_dir=NGSF_RESULTS_MEDIUM0.parent,
    stem='NGSF_accuracy_medium0',
    title_prefix='NGSF medium0',
    transient_type=NGSF_PLOT_TRANSIENT_TYPE,
)
#%%


# Default path (override to use e.g. /home/hhchoi1022/snal/data/NGSF_all_results_tbl_medium0.csv)
NGSF_RESULTS_MEDIUM1 = Path('/home/hhchoi1022/snal/data/NGSF_all_results_tbl_medium1.csv')
if not NGSF_RESULTS_MEDIUM1.is_file():
    NGSF_RESULTS_MEDIUM1 = Path(SNAL_DIR) / 'NGSF_all_results_tbl_medium1.csv'

all_results_tbl_medium1_comp = add_ngsf_metadata_comparison(
    Table.read(NGSF_RESULTS_MEDIUM1, format='ascii.fixed_width')
)
all_results_tbl_medium1_comp = all_results_tbl_medium1_comp[(all_results_tbl_medium1_comp['transient_type'] != 'NA/Unknown') & (all_results_tbl_medium1_comp['transient_type'] != 'None') & (all_results_tbl_medium1_comp['transient_type'] != 'SLSN-I') & (all_results_tbl_medium1_comp['transient_type'] != 'SN I') ]
#%%
summarize_ngsf_comparison(all_results_tbl_medium1_comp, label=str(NGSF_RESULTS_MEDIUM1))

out_comp_path = NGSF_RESULTS_MEDIUM1.with_name(
    NGSF_RESULTS_MEDIUM1.stem + '_comparison.csv'
)
all_results_tbl_medium1_comp.write(out_comp_path, format='ascii.fixed_width', overwrite=True)
print(f'Wrote: {out_comp_path}')

plot_ngsf_accuracy_four_panels(
    all_results_tbl_medium1_comp,
    out_dir=NGSF_RESULTS_MEDIUM1.parent,
    stem='NGSF_accuracy_medium1',
    title_prefix='NGSF medium1',
    transient_type=NGSF_PLOT_TRANSIENT_TYPE,
)

#%%
# Default path (override to use e.g. /home/hhchoi1022/snal/data/NGSF_all_results_tbl_medium0.csv)
NGSF_RESULTS_MEDIUM2 = Path('/home/hhchoi1022/snal/data/NGSF_all_results_tbl_medium2.csv')
if not NGSF_RESULTS_MEDIUM2.is_file():
    NGSF_RESULTS_MEDIUM2 = Path(SNAL_DIR) / 'NGSF_all_results_tbl_medium2.csv'

all_results_tbl_medium2_comp = add_ngsf_metadata_comparison(
    Table.read(NGSF_RESULTS_MEDIUM2, format='ascii.fixed_width')
)
all_results_tbl_medium2_comp = all_results_tbl_medium2_comp[(all_results_tbl_medium2_comp['transient_type'] != 'NA/Unknown') & (all_results_tbl_medium2_comp['transient_type'] != 'None') & (all_results_tbl_medium2_comp['transient_type'] != 'SLSN-I') & (all_results_tbl_medium2_comp['transient_type'] != 'SN I') ]
summarize_ngsf_comparison(all_results_tbl_medium2_comp, label=str(NGSF_RESULTS_MEDIUM2))

out_comp_path = NGSF_RESULTS_MEDIUM2.with_name(
    NGSF_RESULTS_MEDIUM2.stem + '_comparison.csv'
)
all_results_tbl_medium2_comp.write(out_comp_path, format='ascii.fixed_width', overwrite=True)
print(f'Wrote: {out_comp_path}')

plot_ngsf_accuracy_four_panels(
    all_results_tbl_medium2_comp,
    out_dir=NGSF_RESULTS_MEDIUM2.parent,
    stem='NGSF_accuracy_medium2',
    title_prefix='NGSF medium2',
    transient_type=NGSF_PLOT_TRANSIENT_TYPE,
)

#%%
# Default path (override to use e.g. /home/hhchoi1022/snal/data/NGSF_all_results_tbl_medium0.csv)
NGSF_RESULTS_MEDIUM3 = Path('/home/hhchoi1022/snal/data/NGSF_all_results_tbl_medium3.csv')
if not NGSF_RESULTS_MEDIUM3.is_file():
    NGSF_RESULTS_MEDIUM3 = Path(SNAL_DIR) / 'NGSF_all_results_tbl_medium3.csv'

all_results_tbl_medium3_comp = add_ngsf_metadata_comparison(
    Table.read(NGSF_RESULTS_MEDIUM3, format='ascii.fixed_width')
)
all_results_tbl_medium3_comp = all_results_tbl_medium3_comp[(all_results_tbl_medium3_comp['transient_type'] != 'NA/Unknown') & (all_results_tbl_medium3_comp['transient_type'] != 'None') & (all_results_tbl_medium3_comp['transient_type'] != 'SLSN-I') & (all_results_tbl_medium3_comp['transient_type'] != 'SN I') ]
summarize_ngsf_comparison(all_results_tbl_medium3_comp, label=str(NGSF_RESULTS_MEDIUM3))

out_comp_path = NGSF_RESULTS_MEDIUM3.with_name(
    NGSF_RESULTS_MEDIUM3.stem + '_comparison.csv'
)
all_results_tbl_medium3_comp.write(out_comp_path, format='ascii.fixed_width', overwrite=True)
print(f'Wrote: {out_comp_path}')

plot_ngsf_accuracy_four_panels(
    all_results_tbl_medium3_comp,
    out_dir=NGSF_RESULTS_MEDIUM3.parent,
    stem='NGSF_accuracy_medium3',
    title_prefix='NGSF medium3',
    transient_type=NGSF_PLOT_TRANSIENT_TYPE,
)

#%%
# Default path (override to use e.g. /home/hhchoi1022/snal/data/NGSF_all_results_tbl_medium0.csv)
NGSF_RESULTS_MEDIUM4 = Path('/home/hhchoi1022/snal/data/NGSF_all_results_tbl_medium4.csv')
if not NGSF_RESULTS_MEDIUM4.is_file():
    NGSF_RESULTS_MEDIUM4 = Path(SNAL_DIR) / 'NGSF_all_results_tbl_medium4.csv'

all_results_tbl_medium4_comp = add_ngsf_metadata_comparison(
    Table.read(NGSF_RESULTS_MEDIUM4, format='ascii.fixed_width')
)
all_results_tbl_medium4_comp = all_results_tbl_medium4_comp[(all_results_tbl_medium4_comp['transient_type'] != 'NA/Unknown') & (all_results_tbl_medium4_comp['transient_type'] != 'None') & (all_results_tbl_medium4_comp['transient_type'] != 'SLSN-I') & (all_results_tbl_medium4_comp['transient_type'] != 'SN I') ]
summarize_ngsf_comparison(all_results_tbl_medium4_comp, label=str(NGSF_RESULTS_MEDIUM4))

out_comp_path = NGSF_RESULTS_MEDIUM4.with_name(
    NGSF_RESULTS_MEDIUM4.stem + '_comparison.csv'
)
all_results_tbl_medium4_comp.write(out_comp_path, format='ascii.fixed_width', overwrite=True)
print(f'Wrote: {out_comp_path}')

plot_ngsf_accuracy_four_panels(
    all_results_tbl_medium4_comp,
    out_dir=NGSF_RESULTS_MEDIUM4.parent,
    stem='NGSF_accuracy_medium4',
    title_prefix='NGSF medium4',
    transient_type=NGSF_PLOT_TRANSIENT_TYPE,
)


# %%
NGSF_RESULTS_MEDIUM5 = Path('/home/hhchoi1022/snal/data/NGSF_all_results_tbl_medium5.csv')
if not NGSF_RESULTS_MEDIUM5.is_file():
    NGSF_RESULTS_MEDIUM5 = Path(SNAL_DIR) / 'NGSF_all_results_tbl_medium5.csv'

all_results_tbl_medium5_comp = add_ngsf_metadata_comparison(
    Table.read(NGSF_RESULTS_MEDIUM5, format='ascii.fixed_width')
)
all_results_tbl_medium5_comp = all_results_tbl_medium5_comp[(all_results_tbl_medium5_comp['transient_type'] != 'NA/Unknown') & (all_results_tbl_medium5_comp['transient_type'] != 'None') & (all_results_tbl_medium5_comp['transient_type'] != 'SLSN-I') & (all_results_tbl_medium5_comp['transient_type'] != 'SN I') ]
summarize_ngsf_comparison(all_results_tbl_medium5_comp, label=str(NGSF_RESULTS_MEDIUM5))

out_comp_path = NGSF_RESULTS_MEDIUM5.with_name(
    NGSF_RESULTS_MEDIUM5.stem + '_comparison.csv'
)
all_results_tbl_medium5_comp.write(out_comp_path, format='ascii.fixed_width', overwrite=True)
print(f'Wrote: {out_comp_path}')

plot_ngsf_accuracy_four_panels(
    all_results_tbl_medium5_comp,
    out_dir=NGSF_RESULTS_MEDIUM5.parent,
    stem='NGSF_accuracy_medium5',
    title_prefix='NGSF medium5',
    transient_type=NGSF_PLOT_TRANSIENT_TYPE,
)

# %%
# Compare each medium-band run to spectrum-based (raw) NGSF on matching spectra (objname + mjd).
_RAW_VS_MED_OUT = NGSF_RESULTS_RAW.parent if NGSF_RESULTS_RAW.is_file() else Path(SNAL_DIR)
for _med_idx in range(6):
    _name = f'NGSF_all_results_tbl_medium{_med_idx}.csv'
    _p = Path(f'/home/hhchoi1022/snal/data/{_name}')
    if not _p.is_file():
        _p = Path(SNAL_DIR) / _name
    write_medium_vs_raw_analysis(
        all_results_tbl_raw_comp,
        _p,
        out_dir=_RAW_VS_MED_OUT,
        medium_name=f'medium{_med_idx}',
        transient_type=NGSF_PLOT_TRANSIENT_TYPE,
    )
# %%

raw_tbl = all_results_tbl_raw_comp
target_tbl = all_results_tbl_medium0_comp
clean_idx = (~raw_tbl['SN_TYPE'].mask) & (~target_tbl['SN_TYPE'].mask)
raw_tbl = raw_tbl[clean_idx]
target_tbl = target_tbl[clean_idx]
matched_len = 0
exact_matched_len = 0
for i in range(len(raw_tbl)):
    if raw_tbl['type_family_fit'][i] == target_tbl['type_family_fit'][i]:
        matched_len += 1
    if raw_tbl['SN_TYPE'][i] == target_tbl['SN_TYPE'][i]:
        exact_matched_len += 1
print(f'Matched length: {matched_len}')
print(f'Exact matched length: {exact_matched_len}')
print(f'Matched ratio: {matched_len/len(raw_tbl)}')
print(f'Exact matched ratio: {exact_matched_len/len(raw_tbl)}')

#%%
len(all_results_tbl_medium0_comp)
# %%


import numpy as np
import matplotlib.pyplot as plt

def plot_family_match_rate_raw_vs_medium0(
    raw_tbl,
    medium_tbl,
    type_col='transient_type',
    match_col='type_family_match',
    min_count=1,
    exclude_unknown=True,
    figsize=(10, 8),
    rotation=0,
):
    """
    Compare coarse family match probability by transient type
    between raw and medium0 results.

    Parameters
    ----------
    raw_tbl : astropy.table.Table
    medium_tbl : astropy.table.Table
    type_col : str
        Column containing metadata transient type.
    match_col : str
        Boolean column indicating family match.
    min_count : int
        Minimum number of samples required in either raw or medium
        to keep that transient type in the plot.
    exclude_unknown : bool
        Whether to remove NA/Unknown-like labels.
    """

    def summarize(tbl):
        types = np.array(tbl[type_col].astype(str), dtype=str)
        matches = np.array(tbl[match_col], dtype=bool)

        types = np.char.strip(types)

        bad_types = {'', '--', 'None', 'nan', 'masked', 'NA/Unknown', 'Unknown', 'unknown', 'NA'}
        valid = ~np.isin(types, list(bad_types))

        types = types[valid]
        matches = matches[valid]

        uniq, inv = np.unique(types, return_inverse=True)
        counts = np.bincount(inv)
        success = np.array([matches[inv == i].sum() for i in range(len(uniq))], dtype=int)
        rates = success / counts

        return {
            u: {'count': c, 'success': s, 'rate': r}
            for u, c, s, r in zip(uniq, counts, success, rates)
        }

    raw_stats = summarize(raw_tbl)
    med_stats = summarize(medium_tbl)

    labels = sorted(set(raw_stats.keys()) | set(med_stats.keys()))
    if len(labels) == 0:
        print("No valid transient types found.")
        return

    raw_counts = np.array([raw_stats.get(k, {}).get('count', 0) for k in labels])
    med_counts = np.array([med_stats.get(k, {}).get('count', 0) for k in labels])
    raw_rates  = np.array([raw_stats.get(k, {}).get('rate', np.nan) for k in labels])
    med_rates  = np.array([med_stats.get(k, {}).get('rate', np.nan) for k in labels])

    keep = (raw_counts >= min_count) | (med_counts >= min_count)
    labels = np.array(labels)[keep]
    raw_counts = raw_counts[keep]
    med_counts = med_counts[keep]
    raw_rates = raw_rates[keep]
    med_rates = med_rates[keep]

    # sort by total sample size
    total_counts = raw_counts + med_counts
    order = np.argsort(-total_counts)

    labels = labels[order]
    raw_counts = raw_counts[order]
    med_counts = med_counts[order]
    raw_rates = raw_rates[order]
    med_rates = med_rates[order]

    y = np.arange(len(labels))
    h = 0.38

    plt.figure(figsize=figsize)
    plt.barh(y - h/2, raw_rates, height=h, label='raw')
    plt.barh(y + h/2, med_rates, height=h, label='medium0')

    raw_success = np.array([raw_stats.get(k, {}).get('success', 0) for k in labels])
    med_success = np.array([med_stats.get(k, {}).get('success', 0) for k in labels])

    ylabels = [
        f"{lab}  (raw={sr}/{nr}, med={sm}/{nm})"
        for lab, sr, nr, sm, nm in zip(labels, raw_success, raw_counts, med_success, med_counts)
    ]
    plt.yticks(y, ylabels, rotation=rotation)
    plt.xlabel('Family match probability')
    plt.xlim(0, 1.05)
    plt.ylabel('Transient type')
    plt.legend()
    plt.tight_layout()
    plt.show()
# %%
plot_family_match_rate_raw_vs_medium0(
    all_results_tbl_raw_comp,
    all_results_tbl_medium0_comp,
    min_count=3,
    figsize=(10, 9),
)
#%%
medium_tbls = [all_results_tbl_medium0_comp, all_results_tbl_medium1_comp]
medium_labels = ['medium0', 'medium1']
import numpy as np
import matplotlib.pyplot as plt

def plot_family_match_rate_multi(
    raw_tbl,
    medium_tbls,
    medium_labels=None,
    type_col='transient_type',
    match_col='type_family_match',
    min_count=1,
    exclude_unknown=True,
    figsize=(12, 9),
):
    if medium_labels is None:
        medium_labels = [f'medium{i}' for i in range(len(medium_tbls))]

    def summarize(tbl):
        types = np.array(tbl[type_col].astype(str), dtype=str)
        matches = np.array(tbl[match_col], dtype=bool)

        types = np.char.strip(types)

        bad_types = {'', '--', 'None', 'nan', 'masked'}
        if exclude_unknown:
            bad_types |= {'NA/Unknown', 'Unknown', 'unknown', 'NA'}

        valid = ~np.isin(types, list(bad_types))
        types = types[valid]
        matches = matches[valid]

        uniq, inv = np.unique(types, return_inverse=True)
        counts = np.bincount(inv)
        success = np.array([matches[inv == i].sum() for i in range(len(uniq))], dtype=int)
        rates = success / counts

        return {
            u: {'count': c, 'success': s, 'rate': r}
            for u, c, s, r in zip(uniq, counts, success, rates)
        }

    raw_stats = summarize(raw_tbl)
    medium_stats_list = [summarize(tbl) for tbl in medium_tbls]

    labels = set(raw_stats.keys())
    for stats in medium_stats_list:
        labels |= set(stats.keys())
    labels = sorted(labels)

    raw_counts = np.array([raw_stats.get(k, {}).get('count', 0) for k in labels])
    raw_success = np.array([raw_stats.get(k, {}).get('success', 0) for k in labels])
    raw_rates = np.array([raw_stats.get(k, {}).get('rate', np.nan) for k in labels])

    med_counts_list = [
        np.array([stats.get(k, {}).get('count', 0) for k in labels])
        for stats in medium_stats_list
    ]
    med_success_list = [
        np.array([stats.get(k, {}).get('success', 0) for k in labels])
        for stats in medium_stats_list
    ]
    med_rates_list = [
        np.array([stats.get(k, {}).get('rate', np.nan) for k in labels])
        for stats in medium_stats_list
    ]

    total_counts = raw_counts.copy()
    for c in med_counts_list:
        total_counts += c

    keep = total_counts >= min_count
    labels = np.array(labels)[keep]
    raw_counts = raw_counts[keep]
    raw_success = raw_success[keep]
    raw_rates = raw_rates[keep]

    med_counts_list = [c[keep] for c in med_counts_list]
    med_success_list = [s[keep] for s in med_success_list]
    med_rates_list = [r[keep] for r in med_rates_list]

    total_counts = total_counts[keep]
    order = np.argsort(-total_counts)

    labels = labels[order]
    raw_counts = raw_counts[order]
    raw_success = raw_success[order]
    raw_rates = raw_rates[order]

    med_counts_list = [c[order] for c in med_counts_list]
    med_success_list = [s[order] for s in med_success_list]
    med_rates_list = [r[order] for r in med_rates_list]

    n_groups = 1 + len(medium_tbls)
    y = np.arange(len(labels))
    h = 0.8 / n_groups

    plt.figure(figsize=figsize)

    offsets = np.linspace(-(n_groups-1)/2, (n_groups-1)/2, n_groups) * h

    plt.barh(y + offsets[0], raw_rates, height=h, label='raw')

    for i, (rates, name) in enumerate(zip(med_rates_list, medium_labels), start=1):
        plt.barh(y + offsets[i], rates, height=h, label=name)

    ylabels = []
    for j, lab in enumerate(labels):
        parts = [f"raw={raw_success[j]}/{raw_counts[j]}"]
        for name, succ, cnt in zip(medium_labels, med_success_list, med_counts_list):
            parts.append(f"{name}={succ[j]}/{cnt[j]}")
        ylabels.append(f"{lab}  ({', '.join(parts)})")

    plt.yticks(y, ylabels)
    plt.xlabel('Family match probability')
    plt.xlim(0, 1.05)
    plt.ylabel('Transient type')
    plt.legend()
    plt.tight_layout()
    plt.show()
# %%
plot_family_match_rate_multi(
    raw_tbl=all_results_tbl_raw_comp,
    medium_tbls=[all_results_tbl_medium0_comp, all_results_tbl_medium1_comp, all_results_tbl_medium5_comp ],
    medium_labels=['medium0', 'medium1', 'medium5'],
    min_count=3,
)
# %% # Inverse order
all_ratio = []
for tbl in [all_results_tbl_medium4_comp, all_results_tbl_medium3_comp, all_results_tbl_medium2_comp, all_results_tbl_medium1_comp, all_results_tbl_medium0_comp, all_results_tbl_raw_comp]:
    all_ratio.append(np.sum(tbl['type_family_match'])/len(tbl))
all_ratio = np.asarray(all_ratio, dtype=float)
x_med = np.arange(5)
y_med = all_ratio[:5]
x_spec, y_spec = 5.0, float(all_ratio[5])

fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=120)
ax.plot(
    x_med,
    y_med,
    "o-",
    color="k",
    markersize=8,
    linewidth=1.6,
    markeredgecolor="white",
    markeredgewidth=0.8,
    clip_on=False,
    label="NGSF-7DT (mediumband)",
    zorder=3,
)
ax.scatter(
    [x_spec],
    [y_spec],
    s=140,
    color="#c44e52",
    marker="D",
    edgecolors="white",
    linewidths=1.2,
    zorder=4,
    label="NGSF (spectrum; reference)",
)
ax.axhline(y_spec, color="#c44e52", linestyle="--", linewidth=1.0, alpha=0.35, zorder=1)

xtick_labels = [
    rf"{len(medium_filterset_4)}",
    rf"{len(medium_filterset_3)}",
    rf"{len(medium_filterset_2)}",
    rf"{len(medium_filterset_1)}",
    rf"{len(medium_filterset_0)}",
    "NGSF\nspectrum",
]
ax.set_xticks(np.arange(6))
ax.set_xticklabels(xtick_labels)
ax.set_xlim(-0.35, 5.35)
# ax.set_xlabel("Mediumband filter count (NGSF-7DT); right: NGSF spectrum (reference, not NGSF-7DT)")
# ax.set_ylabel("Family match probability")
ax.set_ylim(0, 1.05)
ax.set_yticks(np.linspace(0, 1, 11))
ax.grid(True, axis="y", alpha=0.35, linestyle="-", linewidth=0.6)
ax.grid(True, axis="x", alpha=0.15, linestyle="-", linewidth=0.4)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="0.85")
fig.tight_layout()
plt.show()
# %%

