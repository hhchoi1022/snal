
#%%

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
#%%
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
        filters_exclude_set = set(map(str, filters_exclude))

        # Plot data points
        for filter_name in unique_filters:
            if ma.is_masked(filter_name):
                continue
            filter_name_str = str(filter_name)

            mask = (filters_valid == filter_name)
            color = color_map.get(filter_name_str, 'gray')

            is_excluded = (filter_name_str in filters_exclude_set)
            alpha = 0.1 if is_excluded else 0.7

            label = f'{filter_name_str} ({np.sum(mask)} points)'
            if is_excluded:
                label += ' [excluded]'

            ax.errorbar(
                mjd_valid[mask], mag_valid[mask], yerr=magerr_valid[mask],
                fmt='o', label=label,
                color=color, alpha=alpha, markersize=4
            )

        
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
# %%
good_objnames
# %%
objname = good_objnames[243]
photometry_tbl = Table.read(Path(SNAL_DIR) / objname / f"{objname}_photometry_OSC.csv", format='ascii.fixed_width')
meta_tbl_osc = oscquerier.get_object(objname = objname)
maxdate = meta_tbl_osc['maxdate'][0]
phase_range = [-20, 50]
redshift = None
visualize = True
color_map = None
ax = None
filters_exclude = ['1','2','H','I1','I2','Ic','J','K','Rc','UVM2','UVW1','UVW2',"g'", 'R']
sncosmo_result = generate_sncosmo_model(photometry_tbl, maxdate = maxdate, phase_range = phase_range, redshift = redshift, visualize = visualize, color_map = color_map, ax = ax, filters_exclude = filters_exclude)
sncosmo_model = sncosmo_result['model']
sncosmo_fit_result = sncosmo_result['result']
sncosmo_filter_models = sncosmo_result['filter_models']
sncosmo_fig = sncosmo_result['fig']
sncosmo_ax = sncosmo_result['ax']
# %% # Fitting result from photometry
for parameter_key, parameter_value in zip(sncosmo_fit_result.param_names, sncosmo_fit_result.parameters):
    print(f"{parameter_key}: {parameter_value}")

# %% Firting result from spectra
spectra_quality_tbl = Table.read(Path(SNAL_DIR) / objname / f"{objname}_spectra_quality.csv", format='ascii.fixed_width')
good_quality = spectra_quality_tbl[spectra_quality_tbl['good_sign'] == 'Good']
good_quality['phase'] = good_quality['mjd'] - maxdate
# %%
from astropy.table import vstack
# idx_list = [59, 63, 20, 80, 51, 61, 37]
# idx_list = [5]
idx_list = good_quality['idx_spectra'].astype(int)[:1]
fit_tbl = Table()
for spectra_idx in idx_list:
    spectra_meta = json.load(open(Path(SNAL_DIR) / objname / f"{objname}_spectra_{spectra_idx}_meta_OSC.json", 'r'))
    spectra_data = Table.read(Path(SNAL_DIR) / objname / f"{objname}_spectra_{spectra_idx}_data_OSC.csv", format='csv')
    synphot_tbl = Table.read(Path(SNAL_DIR) / objname / f"{objname}_synphot_{spectra_idx}_data_OSC.csv", format='ascii.fixed_width')
    synphot_tbl['magerr'] = 0.02
    synphot_tbl.remove_columns(['observatory', 'observer', 'redshift', 'survey','u_errors', 'instrument'])
    fit_tbl = vstack([fit_tbl, synphot_tbl])
print('Length of spectra: ', len(idx_list))
#%%
sncosmo_result_synphot = generate_sncosmo_model(fit_tbl, maxdate = maxdate, phase_range = phase_range, redshift = redshift, visualize = visualize, color_map = color_map, ax = sncosmo_ax, filters_exclude = filters_exclude)
sncosmo_model_synphot = sncosmo_result_synphot['model']
sncosmo_fit_result_synphot = sncosmo_result_synphot['result']
sncosmo_filter_models_synphot = sncosmo_result_synphot['filter_models']
sncosmo_fig_synphot = sncosmo_result_synphot['fig']
sncosmo_ax_synphot = sncosmo_result_synphot['ax']
#%%
sncosmo_fig_synphot
# %%
for parameter_key, parameter_value in zip(sncosmo_fit_result_synphot.param_names, sncosmo_fit_result_synphot.parameters):
    print(f"{parameter_key}: {parameter_value}")



# %%
