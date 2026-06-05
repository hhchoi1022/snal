
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
import astropy.units as u
import gc
matplotlib.use("Agg")

#%%
def make_json_serializable(obj):
    if isinstance(obj, u.Quantity):
        return obj.value.tolist()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]

    return obj

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

#%%
import glob
from astropy.io import ascii
from ezphot.dataobjects import Spectrum
spec = Spectrum(wavelength = np.linspace(3000, 10000, 1000), flux = np.ones(1000), fluxerr = None, flux_unit = 'flamb', wavelength_unit = 'AA')
_, pyphot_filters, _, _, _ = spec.synphot(filterset='medium', visualize=False, visualize_spectrum=False, visualize_transmission=False)
version = "v1"  # 원하는 버전명으로 수정


SAVE_DIR = Path('/home/hhchoi1022/snal/data')
NGSF_TEMPLATE_DIR = Path('/home/hhchoi1022/code/NGSF_7DT/NGSF_7DT/bank/')

all_templates_dir = glob.glob(
    str(NGSF_TEMPLATE_DIR / 'original_resolution' / 'sne' / '*' / '*'),
    recursive=True
)
#%%
def make_synphot_files_for_obj(target_dir, pyphot_filters):
    """
    One objname에 대해:
    1. 원본 spectrum synphot 저장
    2. medium filterset별 synphot 저장
    3. metadata 저장

    synphot 파일명에는 version을 붙이지 않음.
    """
    target_dir = Path(target_dir)
    objname = target_dir.name
    transient_type = target_dir.parent.name

    meta_path = target_dir / 'wiserep_spectra.csv'
    if not meta_path.exists():
        print(f"[SKIP] No metadata: {meta_path}")
        return []

    try:
        meta_tbl = Table.read(meta_path, format='csv')
    except Exception as e:
        print(f"[ERROR] Cannot read metadata for {objname}: {e}")
        return []

    obj_save_dir = SAVE_DIR / objname
    obj_save_dir.mkdir(parents=True, exist_ok=True)

    original_synphot_paths = []

    for ascii_file in meta_tbl['Ascii file']:
        try:
            spec_path = target_dir / ascii_file
            if not spec_path.exists():
                print(f"[SKIP] Missing spectrum: {spec_path}")
                continue

            spec_data = ascii.read(spec_path)

            spec = Spectrum(
                wavelength=spec_data['col1'],
                flux=spec_data['col2'],
                fluxerr=None,
                flux_unit='flamb',
                wavelength_unit='AA'
            )

            synphot_dict, pyphot_filters, _, _, _ = spec.synphot(
                filterset='medium',
                visualize=False,
                visualize_spectrum=False,
                visualize_transmission=False,
                pyphot_filters=pyphot_filters
            )

            synphot_dict = make_json_serializable(synphot_dict)

            # 원본 synphot 결과 저장: version 붙이지 않음
            original_filename = Path(ascii_file).name
            original_path = obj_save_dir / original_filename

            with open(original_path, 'w') as f:
                for filter_, value in synphot_dict.items():
                    wl_pivot = float(value['wl_pivot']) * 10
                    flux = float(value['flux'])
                    if np.isfinite(flux):
                        f.write(f"{wl_pivot} {flux}\n")

            original_synphot_paths.append(original_path)

            # medium-band subset별 저장: version 붙이지 않음
            for filterset_idx, filterset in medium_filterset_dict.items():
                medium_filename = (
                    Path(ascii_file).stem
                    + f"_medium{filterset_idx}"
                    + Path(ascii_file).suffix
                )
                medium_path = obj_save_dir / medium_filename

                with open(medium_path, 'w') as f:
                    for filter_, value in synphot_dict.items():
                        if filter_ not in filterset:
                            continue

                        wl_pivot = float(value['wl_pivot']) * 10
                        flux = float(value['flux'])

                        if np.isfinite(flux):
                            f.write(f"{wl_pivot} {flux}\n")

        except Exception as e:
            print(f"[ERROR] {objname} / {ascii_file}: {e}")
            continue

    # metadata도 version 붙이지 않음
    out_meta_path = obj_save_dir / 'wiserep_spectra.csv'
    meta_tbl.write(out_meta_path, format='csv', overwrite=True)

    return original_synphot_paths

#%%
def process_single(original_path):
    """
    하나의 원본 synphot spectrum에 대해 medium0~5 NGSF fitting.

    입력 synphot 파일에는 version 없음.
    저장되는 superfit plot 파일에만 version 붙임.
    """
    original_path = Path(original_path)

    try:
        for filterset_idx in medium_filterset_dict.keys():
            medium_path = original_path.with_name(
                original_path.stem
                + f"_medium{filterset_idx}"
                + original_path.suffix
            )

            if not medium_path.exists():
                print(f"[SKIP] Missing medium file: {medium_path}")
                continue

            classifier = AlertClassifier()
            superfit = classifier.fit(medium_path)

            fig_fit, ax = superfit.plot_fit_result(0)

            # superfit 저장파일에만 version suffix 추가
            fig_fit.savefig(
                medium_path.with_name(
                    medium_path.stem + f"_fit_{version}.png"
                ),
                dpi=300,
                bbox_inches='tight',
            )

            plt.close(fig_fit)

    except Exception as e:
        print(f"[ERROR] NGSF failed for {original_path}: {e}")
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
        plt.close(fig_fit)
        del fig_fit, ax, superfit
        gc.collect()
    except Exception as e:
        print(f"Error with task {task}: {e}")
        return None

#%% Run in multiprocessing
if __name__ == '__main__':
    from tqdm import tqdm
    from multiprocessing import Pool
    all_original_synphot_paths = []
    failed_dirs = []

    for target_dir in tqdm(all_templates_dir, desc='Make synphot files', unit='object'):
        paths = make_synphot_files_for_obj(target_dir, pyphot_filters)
        if len(paths) == 0:
            failed_dirs.append(target_dir)
        all_original_synphot_paths.extend(paths)

    print(f"Total original spectra for NGSF: {len(all_original_synphot_paths)}")

    n_tasks = len(all_original_synphot_paths)
    batch_size = 128
    max_workers = 64

    for start in range(0, n_tasks, batch_size):
        end = min(start + batch_size, n_tasks)
        batch_tasks = all_original_synphot_paths[start:end]

        with Pool(processes=max_workers) as pool:
            list(
                tqdm(
                    pool.imap_unordered(process_single, batch_tasks),
                    total=len(batch_tasks),
                    desc=f'NGSF spectra [{start + 1}-{end}/{n_tasks}]',
                    unit='spectrum',
                )
            )

        gc.collect()
# %%
