
#%%
from bridge.alertmonitor import AlertClassifier
# from NGSF.sf_class import *
from snal.utils import OSCQuerier
from astropy.table import Table
from pathlib import Path
import json
import matplotlib
import numpy as np
from ezphot.dataobjects import Spectrum
from ezphot.dataobjects import PhotometricSpectrum
from astropy.time import Time
import matplotlib.pyplot as plt
# matplotlib.use("Agg")

#%%
oscquerier = OSCQuerier()
SNAL_DIR = oscquerier.config.save_dir
good_meta_tbl = Table.read(Path(SNAL_DIR) / 'good_meta_tbl.ecsv', format='ascii.ecsv')
version = 'v1'
#%%
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
    5: medium_filterset_5,
}
#%%
row = good_meta_tbl[3]
spec_quality_tbl = Table.read(Path(SNAL_DIR) / row['objname'] / f"{row['objname']}_spectra_quality.csv", format='ascii.fixed_width')
good_tbl = spec_quality_tbl[spec_quality_tbl['good_sign'] == 'Good']
for spectra_idx in good_tbl['idx_spectra'].astype(int):
    spectra_meta = json.load(open(Path(SNAL_DIR) / row['objname'] / f"{row['objname']}_spectra_{spectra_idx}_meta_OSC.json", 'r'))
    spectra_data = Table.read(Path(SNAL_DIR) / row['objname'] / f"{row['objname']}_spectra_{spectra_idx}_data_OSC.csv", format='csv')
    synphot_tbl = Table.read(Path(SNAL_DIR) / row['objname'] / f"{row['objname']}_synphot_{spectra_idx}_data_OSC.csv", format='ascii.fixed_width')
    print(spectra_data.colnames)
    print(synphot_tbl['mag_err'])
#%%)
def build_spectrum_tasks(meta_tbl, snal_dir):
    """One task per good spectrum: (snal_dir, objname, objtype, redshift, spectra_idx)."""
    tasks = []
    base = Path(snal_dir)
    for meta_row in meta_tbl:
        objname = meta_row['objname']
        sq_path = base / objname / f"{objname}_spectra_quality.csv"
        if not sq_path.is_file():
            continue
        spectra_quality_tbl = Table.read(sq_path, format='ascii.fixed_width')
        good_tbl = spectra_quality_tbl[spectra_quality_tbl['good_sign'] == 'Good']
        good_tbl.sort('mjd')
        z = meta_row['redshift']
        if hasattr(z, 'item'):
            z = z.item()
        for spectra_idx in good_tbl['idx_spectra'].astype(int):
            tasks.append(
                (
                    str(snal_dir),
                    str(objname),
                    str(meta_row['transient_type']),
                    z,
                    int(spectra_idx),
                )
            )
    return tasks

def process_single(task):
    """Process one spectrum: NGSF fit for each medium-band filter set."""
    try:
        snal_dir, objname, objtype, redshift, spectra_idx = task
        base = Path(snal_dir)
        spectra_meta = json.load(open(base / objname / f"{objname}_spectra_{spectra_idx}_meta_OSC.json", 'r'))
        synphot_tbl = Table.read(base / objname / f"{objname}_synphot_{spectra_idx}_data_OSC.csv", format='ascii.fixed_width')

        mjd = spectra_meta['mjd']
        synphot_tbl['zperr'] = 0
        synphot_tbl['depth'] = 0
        synphot_tbl['telname'] = 'OSC'
        synphot_tbl['obsdate_mjd_group'] = mjd
        synphot_tbl['obsdate_group'] = Time(mjd, format='mjd').isot
        synphot_tbl.remove_column('snr')
        synphot_tbl.remove_column('source_references')
        synphot_tbl.remove_column('instrument')
        synphot_tbl.remove_column('observer')
        synphot_tbl.remove_column('redshift')
        synphot_tbl.remove_column('survey')
        synphot_tbl.remove_column('u_errors')
        if all(np.isnan(synphot_tbl['mag_err'])):
            synphot_tbl['mag_err'] = 0.01

        for filterset_idx in medium_filterset_dict.keys():
            tbl = synphot_tbl.copy()
            tbl = tbl[np.isin(tbl['filter'], medium_filterset_dict[filterset_idx])]

            photspectrum = PhotometricSpectrum()
            photspectrum.plt_params.figure_figsize = (16, 8)
            photspectrum.OFFSET = 0
            photspectrum.plt_params.line_width = 2.5
            photspectrum.plt_params.color_legend_ncols = 1
            photspectrum.data = tbl
            # fig_plot, _, ax, detection_tbl = photspectrum.plot(
            #     flux_key='mag',
            #     fluxerr_key='mag_err',
            #     zperr_key='zperr',
            #     depth_key='depth',
            #     title=f"{objname} ({objtype})",
            # )
            classifier = AlertClassifier()
            save_path = base / objname / f"{objname}_spectra_{spectra_idx}_ngsf_medium{filterset_idx}_{version}_formatted.csv"
            file_path = classifier.ngsf_formatter(
                photspectrum,
                objname=objname,
                mag_key='mag',
                magerr_key='mag_err',
                filter_key='filter',
                objname_key='objname',
                save=True,
                save_path=save_path,
            )[1]
            superfit = classifier.fit(file_path, redshift=redshift)
            fig_fit, ax = superfit.plot_fit_result(0)
            fig_fit.savefig(
                base / objname / f"{objname}_spectra_{spectra_idx}_ngsf_medium{filterset_idx}_{version}_fit.png",
                dpi=300,
                bbox_inches='tight',
            )
            plt.close(fig_fit)
    except Exception as e:
        print(f"Error with task {task}: {e}")
        return None


def process_single_raw(task):
    """Process one spectrum: NGSF fit for each medium-band filter set."""
    try:
        snal_dir, objname, objtype, redshift, spectra_idx = task
        base = Path(snal_dir)
        spectra_meta = json.load(open(base / objname / f"{objname}_spectra_{spectra_idx}_meta_OSC.json", 'r'))
        spec_file_path = base / objname / f"{objname}_spectra_{spectra_idx}_data_OSC.csv"
        spec_data = Table.read(spec_file_path, format='csv')

        classifier = AlertClassifier()
        save_path = base / objname / f"{objname}_spectra_{spectra_idx}_ngsf_raw_{version}_formatted.csv"
        spec_data.write(save_path, format = 'ascii')
        superfit = classifier.fit(save_path, redshift=redshift)
        fig_fit, ax = superfit.plot_fit_result(0)
        fig_fit.savefig(
            base / objname / f"{objname}_spectra_{spectra_idx}_ngsf_raw_{version}_fit.png",
            dpi=300,
            bbox_inches='tight',
        )
        plt.close(fig_fit)s
    except Exception as e:
        print(f"Error with task {task}: {e}")
        return None

#%% Run in multiprocessing
if __name__ == '__main__':
    from multiprocessing import Pool
    from tqdm import tqdm

    spectrum_tasks = build_spectrum_tasks(good_meta_tbl, SNAL_DIR)
    n_tasks = len(spectrum_tasks)
    # imap_unordered + tqdm: bar advances as each spectrum completes (map() gives no progress)
    with Pool(processes=32) as pool:
        list(
            tqdm(
                pool.imap_unordered(process_single, spectrum_tasks),
                total=n_tasks,
                desc='NGSF spectra',
                unit='spectrum',
            )
        )
# %%
