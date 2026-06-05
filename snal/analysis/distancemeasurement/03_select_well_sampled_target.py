#%%
"""
In this part, we are going to select the well sampled targets from the photometry and spectra. 
There will be following criteria:

1. The target should have at least 5 photometry points and 1 spectrum point. -> ~3000 targets since 2001.01.01
2. The target should have good well-sampled spectra which has good synthetic photometry consistent with the Light Curve
2.1. To to this, we are going to check consistency between the synthetic photometry and the Light Curve.
2.2. The Light Curve comes from 
2.2.1. SNCOSMO fitting (In case of Type Ia SN)
2.2.2. Light curve flexible fitting

The targets meeting the above criteria will be saved in a table.
Saved contents will be:
1. Object information: objname, ra, dec, hostname, type, distance, redshift, etc
2. Photometry information: mjd, mag, magerr, filter, instrument, telescope, source_names, source_references
3. Spectra information: mjd, flux, fluxerr, wavelength, flux_unit, wavelength_unit
"""
import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
#%%

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
#%% Register filters
def registerfilter(responsefile,
                   name,
                   force = True):
    import astropy.units as u
    tbl = ascii.read(responsefile, format = 'csv')
    band = sncosmo.Bandpass(tbl['wavelength'], tbl['response'], wave_unit = u.AA, name = name)
    sncosmo.register(band, force = force)
    print(f"Registered filter: {name}")
    
helper = Helper()
transmission_dir = Path(helper.config['SYNPHOT_FILTERDIR'])
transmission_files = {file.stem:file for file in list(transmission_dir.glob('*'))}
for filter_name, filter_path in transmission_files.items():
    try:
        registerfilter(filter_path, filter_name)
    except Exception as e:
        print(f"Failed to register filter: {filter_name} with error: {e}")
#%%
def generate_linear_interpolation_model(tbl,
                                        filter_key: str = 'filter',
                                        visualize: bool = False,
                                        color_map: dict = None,
                                        ax = None):
    """
    Generate linear interpolation models for each filter in the photometry table.
    Connects subsequent data points with straight lines.
    
    Parameters
    ----------
    tbl : astropy.table.Table
        Photometry table with columns: mjd, mag, magerr, and filter_key
    filter_key : str, optional
        Column name for filter (default: 'filter')
    order : int, optional
        Not used (kept for compatibility)
    visualize : bool, optional
        If True, plot the light curve data and linear interpolation (default: False)
    color_map : dict, optional
        Dictionary mapping filter names to colors. If None, uses COLOR_MAP from module scope.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None and visualize=True, creates a new figure.
    
    Returns
    -------
    dict
        Dictionary mapping filter names to linear interpolation models (scipy.interpolate.interp1d objects)
    """
    import numpy.ma as ma
    
    # Group table by filter
    if filter_key not in tbl.colnames:
        raise ValueError(f"Filter key '{filter_key}' not found in table columns")

    # Remove masked filters
    f = tbl[filter_key]
    if hasattr(f, "mask"):
        tbl = tbl[~f.mask]
    
    # Group by filter
    tbl_grouped = tbl.group_by(filter_key)
    filter_models = {}
    filter_data = {}  # Store data for visualization
    
    # Set up color map
    if color_map is None:
        color_map = COLOR_MAP
    
    # Set up plotting if visualize is True
    if visualize:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
    
    for group in tbl_grouped.groups:
        filter_name = group[filter_key][0]
        if ma.is_masked(filter_name):
            continue
        
        # Extract data for this filter
        mjd = group['mjd'].astype(float)
        mag = group['mag'].astype(float)
        magerr = group['magerr'].astype(float)
        
        # Replace NaN magerr with 0.1
        magerr = np.where(np.isnan(magerr), 0.1, magerr)
        
        # Remove invalid data points
        valid = np.isfinite(mjd) & np.isfinite(mag)
        mjd_valid = mjd[valid]
        mag_valid = mag[valid]
        magerr_valid = magerr[valid]
        
        # Sort by mjd for linear interpolation
        sort_idx = np.argsort(mjd_valid)
        mjd_sorted = mjd_valid[sort_idx]
        mag_sorted = mag_valid[sort_idx]
        magerr_sorted = magerr_valid[sort_idx]
        
        # Remove duplicate x values
        unique_mask = np.concatenate(([True], np.diff(mjd_sorted) > 1e-10))
        mjd_unique = mjd_sorted[unique_mask]
        mag_unique = mag_sorted[unique_mask]
        magerr_unique = magerr_sorted[unique_mask]
        
        # Skip if not enough points (need at least 2 points for linear interpolation)
        if len(mjd_unique) < 2:
            continue
        
        # Create linear interpolation function
        try:
            from scipy.interpolate import interp1d
            # Use linear interpolation to connect subsequent points
            linear_model = interp1d(mjd_unique, mag_unique, kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
            
            filter_models[filter_name] = linear_model
            filter_data[filter_name] = {
                'mjd': mjd_valid,
                'mag': mag_valid,
                'magerr': magerr_valid
            }
        except (ValueError, TypeError) as e:
            # If fitting fails, skip this filter
            continue
    
    # Visualization
    if visualize and len(filter_models) > 0:
        for filter_name, linear_model in filter_models.items():
            data = filter_data[filter_name]
            color = color_map.get(filter_name, 'gray')
            
            # Plot data points with error bars
            ax.errorbar(data['mjd'], data['mag'], yerr=data['magerr'],
                       fmt='o', label=f'{filter_name} ({len(data["mjd"])} points)',
                       color=color, alpha=0.7, markersize=4)
            
            # Plot linear interpolation (without separate label)
            mjd_min = np.min(data['mjd'])
            mjd_max = np.max(data['mjd'])
            mjd_fit = np.linspace(mjd_min, mjd_max, 200)
            mag_fit = linear_model(mjd_fit)
            ax.plot(mjd_fit, mag_fit, color=color, linestyle='-', linewidth=2,
                   alpha=0.8)
        
        ax.set_xlabel('MJD')
        ax.set_ylabel('Magnitude')
        ax.invert_yaxis()
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    return filter_models

def generate_sncosmo_model(tbl,
                           filter_key: str = 'filter',
                           filters_exclude: list = ['U', 'B', 'I'],
                           maxdate: float = None,
                           phase_range: tuple = [-20, 50],
                           redshift: float = None,
                           visualize: bool = False,
                           color_map: dict = None,
                           ax = None):
    """
    Generate SALT3 light curve models using sncosmo for each filter in the photometry table.
    
    Parameters
    ----------
    tbl : astropy.table.Table
        Photometry table with columns: mjd, mag, magerr, and filter_key
    filter_key : str, optional
        Column name for filter (default: 'filter')
    redshift : float, optional
        Redshift of the supernova. If None, will try to get from metadata or fit.
    visualize : bool, optional
        If True, plot the light curve data and SALT3 fits (default: False)
    color_map : dict, optional
        Dictionary mapping filter names to colors. If None, uses COLOR_MAP from module scope.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None and visualize=True, creates a new figure.
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'model': sncosmo.Model object (fitted SALT3 model)
        - 'result': sncosmo fit result
        - 'filter_models': dict mapping filter names to callable functions for that filter
    """
    import sncosmo
    import numpy.ma as ma
    
    # Set up color map
    if color_map is None:
        color_map = COLOR_MAP
    
    # Set up plotting if visualize is True
    if visualize:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
    
    # Group table by filter
    if filter_key not in tbl.colnames:
        raise ValueError(f"Filter key '{filter_key}' not found in table columns")
    
    # Remove masked filters
    f = tbl[filter_key]
    if hasattr(f, "mask"):
        tbl = tbl[~f.mask]
    
    # Extract and format data for sncosmo
    mjd = tbl['mjd'].astype(float)
    mag = tbl['mag'].astype(float)
    magerr = tbl['magerr'].astype(float)
    filters = tbl[filter_key]
    
    # Remove invalid data points
    valid = np.isfinite(mjd) & np.isfinite(mag) 
    if maxdate is not None:
        mjd_min = maxdate + phase_range[0]
        mjd_max = maxdate + phase_range[1]
        valid = valid & (mjd >= mjd_min) & (mjd <= mjd_max)
    valid_filter = []
    for f in filters:
        if ma.is_masked(f):
            continue
        filter_name = str(f)
        try:
            sncosmo_filter = sncosmo.get_bandpass(filter_name)
            valid_filter.append(True)
        except Exception as e:
            # print(f"Failed to convert filter to string: {f} with error: {e}")
            valid_filter.append(False)
    valid = valid & valid_filter
    mjd_valid = mjd[valid]
    mag_valid = mag[valid]
    magerr_valid = magerr[valid]
    filters_valid = filters[valid]
    magerr_valid = np.where(np.isnan(magerr_valid), 0.1, magerr_valid)
    
    if len(mjd_valid) < 3:
        raise ValueError("Not enough valid data points for SALT3 fitting (need at least 3)")
    
    # Map filters to sncosmo filter names and determine magsys
    magsys_list = []
    for f in filters_valid:
        if ma.is_masked(f):
            continue
        filter_name = str(f)
        # Determine magnitude system
        if filter_name in 'UBVRI':
            magsys = 'vega'
        else:
            magsys = 'ab'
        magsys_list.append(magsys)
    
    # Create sncosmo formatted table
    formatted_tbl = helper.SNcosmo_format(
        mjd_valid, mag_valid, magerr_valid,
        filters_valid, magsys=magsys_list, zp=25.0
    )
    
    # Exclude filters 
    exclude_filter_mask = np.isin(filters_valid, filters_exclude)
    formatted_tbl = formatted_tbl[~exclude_filter_mask]
    
    # Create SALT3 model
    model = sncosmo.Model(source='salt3')
    
    # Set redshift if provided, otherwise try to fit it
    if redshift is not None:
        model.set(z=redshift)
        vparam_names = ['t0', 'x0', 'x1', 'c']
    else:
        # Try to fit redshift as well (requires more data)
        vparam_names = ['z', 't0', 'x0', 'x1', 'c']
        # Set initial redshift guess (can be improved)
        model.set(z=0.01)
    
    # Fit the light curve
    try:
        result, fitted_model = sncosmo.fit_lc(
            formatted_tbl, model,
            vparam_names,
            bounds={'z': (0.001, 0.2)}
        )
    except Exception as e:
        raise RuntimeError(f"SALT3 fitting failed: {str(e)}")
    
    # Create filter-specific model functions
    filter_models = {}
    unique_filters = list(set(filters_valid))
    
    for filter_name in unique_filters:
        if ma.is_masked(filter_name):
            continue
        filter_name_str = str(filter_name)
        
        # Map to sncosmo filter name
        sncosmo_filter = filter_name_str
        # Determine magnitude system
        if filter_name_str in 'UBVRI':
            magsys = 'vega'
        else:
            magsys = 'ab'
        # magsys = 'ab'
        
        # Create a callable function for this filter
        def make_filter_func(band, magsys):
            def filter_func(mjd_values):
                return fitted_model.bandmag(band, magsys, mjd_values)
            return filter_func
        
        filter_models[filter_name_str] = make_filter_func(sncosmo_filter, magsys)
    
    # Visualization
    if visualize:
        # Plot data points
        for filter_name in unique_filters:
            if ma.is_masked(filter_name):
                continue
            filter_name_str = str(filter_name)
            mask = filters_valid == filter_name
            color = color_map.get(filter_name_str, 'gray')
            
            ax.errorbar(mjd_valid[mask], mag_valid[mask], yerr=magerr_valid[mask],
                      fmt='o', label=f'{filter_name_str} ({np.sum(mask)} points)',
                      color=color, alpha=0.7, markersize=4)
        
        # Plot SALT3 model for each filter
        mjd_min = np.min(mjd_valid)
        mjd_max = np.max(mjd_valid)
        mjd_fit = np.linspace(mjd_min, mjd_max, 200)
        
        for filter_name_str, filter_func in filter_models.items():
            # print(filter_name_str)
            color = color_map.get(filter_name_str, 'gray')
            try:
                mag_fit = filter_func(mjd_fit)
                ax.plot(mjd_fit, mag_fit, color=color, linestyle='-', linewidth=2,
                       alpha=0.8)
            except Exception as e:
                print(f"Failed to plot filter: {filter_name_str} with error: {e}")
                continue
        
        ax.set_xlabel('MJD')
        ax.set_ylabel('Magnitude')
        ax.invert_yaxis()
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title('Light Curve with SALT3 Fits')
    
    return {
        'model': fitted_model,
        'result': result,
        'filter_models': filter_models,
        'fig': fig,
        'ax': ax
    }
    
def get_meta_idx(path, objname, prefix: str = 'spectra'):
    name = Path(path).name
    return int(name.replace(f"{objname}_{prefix}_", "").split("_")[0])
# %%
import matplotlib
matplotlib.use("Agg")
result_dict = dict()

for i in range(len(effective_tbl)):
# for i in range(2000, 2500):
    try:
        objname = effective_tbl['objname'][i]
        # tns_meta_path = Path(SNAL_DIR) / objname / f"{objname}_meta_TNS.json"
        osc_photometry_path = Path(SNAL_DIR) / objname / f"{objname}_photometry_OSC.csv"
        osc_spectra_paths = list((Path(SNAL_DIR) / objname).glob(f"{objname}_spectra_*_meta_OSC.json"))
        osc_synphot_paths = list((Path(SNAL_DIR) / objname).glob(f"{objname}_synphot_*_data_OSC.csv"))
        # meta_dict = json.load(open(tns_meta_path, 'r'))
        meta_tbl_osc = oscquerier.get_object(objname = objname)
        # synphot_fig_path = Path(SNAL_DIR) / objname / f"{objname}_synthetic_photometry.png"
        # if synphot_fig_path.exists():
        #     synphot_fig = plt.imread(synphot_fig_path)
        # else:
        #     synphot_fig = None
        # Show figure
        # plt.imshow(synphot_fig)

        # Load Photometry
        photometry_tbl = Table.read(osc_photometry_path, format='ascii.fixed_width')
        linear_interpolation_models = generate_linear_interpolation_model(photometry_tbl, visualize = True)
        tbl = photometry_tbl
        filter_key = 'filter'
        maxdate = meta_tbl_osc['maxdate'][0]
        phase_range = [-10, 40]
        redshift = None
        visualize = True
        color_map = None
        ax = None
        filters_exclude = ['U']
        sncosmo_result = generate_sncosmo_model(photometry_tbl, maxdate = maxdate, phase_range = phase_range, redshift = redshift, visualize = visualize, color_map = color_map, ax = ax, filters_exclude = filters_exclude)
        sncosmo_model = sncosmo_result['model']
        sncosmo_fit_result = sncosmo_result['result']
        sncosmo_filter_models = sncosmo_result['filter_models']
        sncosmo_fig = sncosmo_result['fig']
        sncosmo_ax = sncosmo_result['ax']
        # Load spectra
        spectra_meta_dict = dict()
        for osc_spectra_path in osc_spectra_paths:
            spectra_meta_dict[get_meta_idx(osc_spectra_path, objname)] = json.load(open(osc_spectra_path, 'r'))
            
        # Load synthetic photometry
        synphot_data_dict = dict()
        for osc_synphot_path in osc_synphot_paths:
            synphot_data_dict[get_meta_idx(osc_synphot_path, objname, prefix = 'synphot')] = Table.read(osc_synphot_path, format='ascii.fixed_width')
        
        from astropy.table import vstack
        spectra_result = Table()
        spectra_result_path = Path(SNAL_DIR) / objname / f"{objname}_spectra_quality.csv"
        for idx_spectra in list(synphot_data_dict.keys()):
            synphot_meta = spectra_meta_dict[idx_spectra]
            synphot_data = synphot_data_dict[idx_spectra]
            if 'mjd' not in synphot_data.colnames:
                continue
            mjd = synphot_data['mjd'].astype(float)[0]
            if (mjd > sncosmo_model.maxtime()) or (mjd < sncosmo_model.mintime()):
                continue
            covered_filters = []
            mag_diffs_sncosmo = []
            mag_diffs_linear_interpolation = []
            for filter_name, filter_model in sncosmo_filter_models.items():
                synphot_filter = synphot_data[synphot_data['filter'] == filter_name] 
                if len(synphot_filter) == 0:
                    continue
                mag_from_synphot = synphot_filter['mag'].astype(float)[0]
                mag_from_sncosmo = filter_model(mjd)
                sncosmo_ax.scatter(mjd, mag_from_synphot, facecolor = COLOR_MAP.get(filter_name, 'gray'), edgecolor = 'r', s = 50, marker = 'o', zorder = 100)
                covered_filters.append(filter_name)
                mag_diffs_sncosmo.append(mag_from_synphot - mag_from_sncosmo)
                # print(f"Filter: {filter_name}, Mag: {mag_from_synphot}, Mag from SNCOSMO: {mag_from_sncosmo}, Difference: {mag_from_synphot - mag_from_sncosmo}")# %%
            for filter_name, filter_model in linear_interpolation_models.items():
                synphot_filter = synphot_data[synphot_data['filter'] == filter_name] 
                if len(synphot_filter) == 0:
                    continue
                mag_from_synphot = synphot_filter['mag'].astype(float)[0]
                mag_from_linear_interpolation = filter_model(mjd)
                mag_diffs_linear_interpolation.append(mag_from_synphot - mag_from_linear_interpolation)
                # print(f"Filter: {filter_name}, Mag: {mag_from_synphot}, Mag from Linear Interpolation: {mag_from_linear_interpolation}, Difference: {mag_from_synphot - mag_from_linear_interpolation}")# %%
            mean_mag_diff_sncosmo = np.round(np.mean(np.abs(mag_diffs_sncosmo)), 2)
            mean_mag_diff_linear_interpolation = np.round(np.mean(np.abs(mag_diffs_linear_interpolation)), 2)
            if mean_mag_diff_sncosmo < 0.1 or mean_mag_diff_linear_interpolation < 0.1:
                good_sign = 'Good'
                color_axvline = 'blue'
            elif mean_mag_diff_sncosmo < 0.2 or mean_mag_diff_linear_interpolation < 0.2:
                good_sign = 'Normal'
                color_axvline = 'green'
            elif mean_mag_diff_sncosmo < 0.4 or mean_mag_diff_linear_interpolation < 0.4:
                good_sign = 'Bad'
                color_axvline = 'orange'
            else:
                good_sign = 'Worst'
                color_axvline = 'red'
            sncosmo_ax.axvline(mjd, color = color_axvline, linestyle = '--', alpha = 0.5)

            covered_filters_str = ', '.join(covered_filters)
            mag_diffs_sncosmo_str = ', '.join(np.round(mag_diffs_sncosmo, 2).astype(str))
            mag_diffs_linear_interpolation_str = ', '.join(np.round(mag_diffs_linear_interpolation, 2).astype(str))
            row = dict()
            row['idx_spectra'] = idx_spectra
            row['good_sign'] = good_sign
            row['covered_filters'] = covered_filters_str
            row['mag_diffs_sncosmo'] = mag_diffs_sncosmo_str
            row['mag_diffs_linear_interpolation'] = mag_diffs_linear_interpolation_str
            row['mean_mag_diff_sncosmo'] = mean_mag_diff_sncosmo
            row['mean_mag_diff_linear_interpolation'] = mean_mag_diff_linear_interpolation
            for key, val in synphot_meta.items():
                row[key] = val
                
            for key in row.keys():
                if key not in spectra_result.colnames:
                    spectra_result[key] = None  # create new column
            spectra_result.add_row(row)
            
        spectra_result.write(spectra_result_path, format='ascii.fixed_width', overwrite=True)
        print(f"Saved: {spectra_result_path}")
        sncosmo_fig.savefig(f"{SNAL_DIR}/{objname}/{objname}_sncosmo_fit.png", dpi=300)
        # Close all figure
        plt.close('all')
    except Exception as e:
        result_dict[objname] = e
        
#%%

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
# %%
len(good_objnames)
# %%
good_objnames
# %%
good_objnames[-2]

# %%