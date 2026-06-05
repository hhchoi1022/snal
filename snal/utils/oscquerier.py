#%%
import os
import json
import requests
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import fnmatch

import astropy.units as u
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.table import vstack
from tqdm import tqdm

from snal.configuration import Configuration
#%%


class OSCQuerier:
    """
    A class for querying the Open Supernova Catalog (OSC).
    """

    def __init__(self):
        """
        Initialize OSCQuerier.
        """
        self.config = Configuration(config_filenames=['oscquerier.config'])
        self._summary_tbl = None
    
    @property
    def summary_tbl(self):
        if self._summary_tbl is None:
            self._summary_tbl = Table.read(Path(self.config.catalog_dir) / 'summary.ascii_fixed_width', format='ascii.fixed_width')
        return self._summary_tbl

    # ------------------------------------------------------------------
    # MAIN QUERY
    # ------------------------------------------------------------------
    def get_object(self,
                   ra: float | None = None,
                   dec: float | None = None,
                   radius_arcsec: float | None = 10,
                   objname: str | None = None, 
                   exact_match: bool = True):
        """
        Query a single supernova from OSC.

        Parameters
        ----------
        objname : str
            Supernova name (e.g., 'SN2021aefx')
        save : bool
            Save photometry table
        verbose : bool

        Returns
        -------
        tuple
            (raw_json, photometry_table)
        """
        def objname_match(tbl, query, exact_match: bool = True):
            query = query.strip().lower()

            # wildcard 안 썼으면 prefix 검색
            if exact_match:
                pass
            else:
                if not query.endswith('*'):
                    query = query + '*'
                if not query.startswith('*'):
                    query = '*' + query
            
            names = np.char.lower(tbl['objname'].astype(str))
            mask = np.array([fnmatch.fnmatch(n, query) for n in names])

            return tbl[mask]

        if objname is not None:
            sn_data = objname_match(self.summary_tbl, objname, exact_match)
        elif ra is not None and dec is not None:
            target_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            all_coords = SkyCoord(self.summary_tbl['ra']*u.deg, self.summary_tbl['dec']*u.deg)
            distances = target_coord.separation(all_coords).value
            sn_data = self.summary_tbl[distances < radius_arcsec/3600]
            sn_data['separation'] = distances
            sn_data.sort('separation')
        
        return sn_data
    
    def get_photometry(self, 
                       ra: float | None = None,
                       dec: float | None = None,
                       radius_arcsec: float | None = 10,
                       objname: str | None = None,
                       save: bool = True,
                       verbose: bool = True):

        sn_data = self.get_object(ra, dec, radius_arcsec, objname)
        if len(sn_data) == 0:
            raise ValueError("No objects found in the region")
        if len(sn_data) > 1:
            print(f"[WARNING] {len(sn_data)} objects found in the region. Using the first object.")
        sn_data = sn_data[0]
        

        photometry_tbl = None
        if sn_data['num_photometry'] == 0:
            print("No photometry data found for the object")
            return None

        # --------------------------------------------------------------
        # PHOTOMETRY
        # --------------------------------------------------------------
        rows = []
        data_path = Path(self.config.catalog_dir) / sn_data['file']
        with open(data_path, 'r') as f:
            data = json.load(f)
            objname_in_data = sn_data['objname']
        
        photometry_list_all = data[objname_in_data]['photometry']
        source_list_all = data[objname_in_data]['sources']
        source_dict_all = {}
        if len(source_list_all) > 0:
            for source_dict in source_list_all:
                source_name = source_dict.get('alias', np.nan)
                if source_name is not np.nan:
                    source_dict_all[source_name] = source_dict
        
        rows = []
        for p in photometry_list_all:
            try:
                obstime = p.get('time')

                if pd.isna(obstime):
                    continue

                source_names, source_references = self._parse_sources(
                    p.get('source'),
                    source_dict_all
                )

                rows.append({
                    'mjd': obstime,
                    'mag': p.get('magnitude', np.nan),
                    'magerr': p.get('e_magnitude', np.nan),
                    'filter': p.get('band', ''),
                    'instrument': p.get('instrument', ''),
                    'telescope': p.get('telescope', ''),
                    'source_names': source_names,
                    'source_references': source_references,
                })
            except Exception as e:
                pass
        
        photometry_tbl = Table(rows=rows)
        
        # --------------------------------------------------------------
        # SAVE
        # --------------------------------------------------------------
        if save and photometry_tbl is not None:
            outdir = Path(self.config.save_dir) / objname
            outdir.mkdir(parents=True, exist_ok=True)

            outpath = outdir / f"{objname}_photometry_OSC.csv"
            photometry_tbl.write(
                outpath,
                format="ascii.fixed_width",
                overwrite=True
            )

            if verbose:
                print(f"Photometry saved to: {outpath}")

        if verbose and photometry_tbl is not None:
            print(f"Number of photometry points: {len(photometry_tbl)}")

        return photometry_tbl
    
    def get_spectra(self, 
                    ra: float | None = None,
                    dec: float | None = None,
                    radius_arcsec: float | None = 10,
                    objname: str | None = None,
                    save: bool = True,
                    verbose: bool = True,
                    return_instance: bool = False):
        sn_data = self.get_object(ra, dec, radius_arcsec, objname)
        if len(sn_data) == 0:
            raise ValueError("No objects found in the region")
        if len(sn_data) > 1:
            print(f"[WARNING] {len(sn_data)} objects found in the region. Using the first object.")
        sn_data = sn_data[0]
        
        if sn_data['num_spectra'] == 0:
            print("No spectra data found for the object")
            return None
                
        data_path = Path(self.config.catalog_dir) / sn_data['file']
        with open(data_path, 'r') as f:
            data = json.load(f)
            objname_in_data = sn_data['objname']
        
        spectra_list_all = data[objname_in_data]['spectra']
        source_list_all = data[objname_in_data]['sources']
        source_dict_all = {}
        if len(source_list_all) > 0:
            for source_dict in source_list_all:
                source_name = source_dict.get('alias', np.nan)
                if source_name is not np.nan:
                    source_dict_all[source_name] = source_dict
        
        meta_dict = dict()
        data_dict = dict()
        for i,s in enumerate(spectra_list_all):

            data = s.get('data', [])
            if len(data) == 0:
                continue
            
            obstime = s.get('time')
            if pd.isna(obstime):
                continue
            source_names, source_references = self._parse_sources(
                s.get('source'),
                source_dict_all
            )
            
            meta_row = {}
            meta_row['mjd'] = obstime
            meta_row['u_time'] = s.get('u_time', '')
            meta_row['instrument'] = s.get('instrument', '')
            meta_row['observatory'] = s.get('observatory', '')
            meta_row['observer'] = s.get('observer', '')
            meta_row['redshift'] = s.get('redshift', np.nan)
            meta_row['snr'] = s.get('snr', np.nan)
            meta_row['survey'] = s.get('survey', '')
            meta_row['u_fluxes'] = s.get('u_fluxes', '')
            meta_row['u_errors'] = s.get('u_errors', '')
            meta_row['u_wavelengths'] = s.get('u_wavelengths', '')
            meta_row['source_names'] = source_names
            meta_row['source_references'] = source_references
            data_dict[i] = data
            meta_dict[i] = meta_row
            
        if save:
            def nan_to_none(x):
                import math
                # numpy float / python float NaN 처리
                if isinstance(x, float) and math.isnan(x):
                    return None
                if isinstance(x, np.floating) and np.isnan(x):
                    return None
                
                # dict / list / tuple 재귀 처리
                if isinstance(x, dict):
                    return {k: nan_to_none(v) for k, v in x.items()}
                if isinstance(x, list):
                    return [nan_to_none(v) for v in x]
                if isinstance(x, tuple):
                    return [nan_to_none(v) for v in x]  # JSON은 tuple 없음
                
                return x
            
            outdir = Path(self.config.save_dir) / objname
            outdir.mkdir(parents=True, exist_ok=True)
            
            for idx, data in meta_dict.items():
                meta_path = outdir / f"{objname}_spectra_{idx}_meta_OSC.json"
                meta = nan_to_none(meta_dict[idx])
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=4)
                data_path = outdir / f"{objname}_spectra_{idx}_data_OSC.csv"
                data = np.array(data_dict[idx]).astype(float)
                data_tbl = Table()
                if data.shape[1] == 3:
                    err_exists = True
                else:
                    err_exists = False
                data_tbl['wavelength'] = data[:,0]
                data_tbl['flux'] = data[:,1]
                if err_exists:
                    data_tbl['e_flux'] = data[:,2]
                data_tbl.write(data_path, format="csv", overwrite=True)
                if verbose:
                    print(f"Spectra saved\nMeta: {meta_path}\nData: {data_path}")
        
        if verbose and len(meta_dict) > 0:
            print(f"Number of spectra: {len(meta_dict)}")
        
        if return_instance:
            from ezphot.dataobjects import Spectrum
            spectrum_dict = dict()
            for meta, data in zip(meta_dict.values(), data_dict.values()):
                data = np.array(data)
                if data.shape[1] == 3:
                    err_exists = True
                else:
                    err_exists = False
                wl = data[:,0]
                flux = data[:,1]
                if err_exists:
                    fluxerr = data[:,2]
                else:
                    fluxerr = None
                flux_unit = meta.get('u_fluxes', '')
                wavelength_unit = meta.get('u_wavelengths', '')
                if 'angstrom' in flux_unit.lower():
                    flux_unit = 'flamb'
                else:
                    flux_unit = None
                if 'angstrom' in wavelength_unit.lower():
                    wavelength_unit = 'AA'
                else:
                    wavelength_unit = None
                wl = wl.astype(np.float32)
                flux = flux.astype(np.float32)
                if err_exists:
                    fluxerr = fluxerr.astype(np.float32)
                else:
                    fluxerr = None
                wl_sort_map = np.argsort(wl)
                wl_sorted = wl[wl_sort_map]
                flux_sorted = flux[wl_sort_map]
                if err_exists:
                    fluxerr_sorted = fluxerr[wl_sort_map]
                else:
                    fluxerr_sorted = None
                spectrum = Spectrum(path = None, 
                                    wavelength = wl_sorted, 
                                    flux = flux_sorted,
                                    fluxerr = fluxerr_sorted,
                                    flux_unit = flux_unit,
                                    wavelength_unit = wavelength_unit)
                spectrum.header = meta
                spectrum_dict[idx] = spectrum
            return meta_dict, data_dict, spectrum_dict
        
        return meta_dict, data_dict
            
    def update_summary(self):
        from astropy.time import Time
        from datetime import datetime

        data_dir = Path(self.config.catalog_dir)
        summary_path = data_dir / 'summary.ascii_fixed_width'
        
        existing_paths = np.array([], dtype=str)
        summary_tbl_existing = Table()
        if summary_path.exists():
            summary_tbl_existing = Table.read(summary_path, format='ascii.fixed_width')
            existing_paths = np.array(summary_tbl_existing['file'])
            existing_paths = set(existing_paths)
            
        # recursive search for all files in the data_dir
        rows = []
        for file in tqdm(data_dir.rglob('*.json'), desc = 'Updating summary...'):
            filepath_relative = file.relative_to(data_dir)
            if str(filepath_relative) in existing_paths:
                continue            

            if file.is_file():
                try:
                    objname = file.stem
                    file_status = file.stat()
                    filesize = file_status.st_size
                    modified_time = file_status.st_mtime
                    # Read the file (json format)
                    with open(file, 'r') as f:
                        data = json.load(f)
                    
                    observation_info = data[objname]
                    # 
                    ra = observation_info.get('ra', np.nan)
                    if ra is not np.nan:
                        from astropy.coordinates import Angle
                        ras = [
                            Angle(d.get('value'), unit=u.hourangle).deg
                            if d.get('u_value') == 'hours'
                            else Angle(d.get('value'), unit=u.deg).deg
                            for d in ra
                        ]
                        ra = round(np.mean(ras),4)
                    dec = observation_info.get('dec', np.nan)
                    if dec is not np.nan:
                        decs = [
                            Angle(d.get('value'), unit=u.deg).deg
                            for d in dec
                        ]
                        dec = round(np.mean(decs),4)
                    discovery_date = observation_info.get('discoverdate', np.nan)
                    if discovery_date is not np.nan:
                        discovery_dates = [self._parse_time_flexible(d.get('value')).mjd for d in discovery_date]
                        discovery_date = round(np.mean(discovery_dates),4)
                    maxabsmag = observation_info.get('maxabsmag', np.nan)
                    if maxabsmag is not np.nan:
                        maxabsmags = [float(d.get('value')) for d in maxabsmag]
                        maxabsmag = round(np.mean(maxabsmags),4)
                    maxband = observation_info.get('maxband', np.nan)
                    if maxband is not np.nan:
                        maxband = maxband[0].get('value')
                    maxdate = observation_info.get('maxdate', np.nan)
                    if maxdate is not np.nan:
                        maxdate = self._parse_time_flexible(maxdate[0].get('value')).mjd
                    lumdist = observation_info.get('lumdist', np.nan)
                    if lumdist is not np.nan:
                        lumdists = [float(d.get('value')) for d in lumdist]
                        lumdist = round(np.mean(lumdists), 4)
                    photometry = observation_info.get('photometry', np.nan)
                    num_photometry = len(photometry) if photometry is not np.nan else 0
                    spectra = observation_info.get('spectra', np.nan)
                    num_spectra = len(spectra) if spectra is not np.nan else 0
                    
                    row = {}
                    row['objname'] = objname
                    row['ra'] = ra
                    row['dec'] = dec
                    row['discovery_date'] = discovery_date
                    row['maxabsmag'] = maxabsmag
                    row['maxband'] = maxband
                    row['maxdate'] = maxdate
                    row['lumdist'] = lumdist
                    row['num_photometry'] = num_photometry
                    row['num_spectra'] = num_spectra
                    row['file'] = filepath_relative
                    row['filesize'] = filesize
                    row['modified_time'] = modified_time                
                    rows.append(row)
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    continue
                
        summary_tbl_new = Table(rows=rows)
        
        summary_tbl_combined = vstack([summary_tbl_existing, summary_tbl_new])
        
        summary_tbl_combined.sort('modified_time')
        summary_tbl_combined.write(summary_path, format='ascii.fixed_width', overwrite=True)
        print('Summary table updated: ', summary_path)
        
    def _parse_sources(self, source, source_dict_all):
        if pd.isna(source) or source == '':
            return '', ''

        names = []
        refs = []

        for key in source.split(','):
            sdict = source_dict_all.get(key.strip())
            if not sdict:
                continue

            name = sdict.get('name')
            ref = sdict.get('reference')

            if name:
                names.append(name)
            if ref:
                refs.append(ref)

        return ', '.join(names), ', '.join(refs)
    
    def _parse_time_flexible(self, timestr):

        from astropy.time import Time
        from datetime import datetime, timedelta
        import re
        """
        Parse date strings including fractional days (e.g. 2016/04/27.27)
        into astropy Time.
        """
        if timestr is np.nan:
            return np.nan

        s = timestr.strip()

        # ------------------------------------------------------------------
        # Case 1: YYYY/MM/DD.fraction  (TNS / OSC style)
        # ------------------------------------------------------------------
        m = re.match(r"(\d{4})[/-](\d{2})[/-](\d{2})(?:\.(\d+))?$", s)
        if m:
            year, month, day, frac = m.groups()
            dt = datetime(int(year), int(month), int(day))
            if frac is not None:
                frac_day = float(f"0.{frac}")
                dt += timedelta(days=frac_day)
            return Time(dt)

        # ------------------------------------------------------------------
        # Case 2: datetime-like formats
        # ------------------------------------------------------------------
        formats = [
            "%Y",
            "%Y/%m",
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y.%m.%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%m/%d/%Y"
        ]

        for fmt in formats:
            try:
                return Time(datetime.strptime(s, fmt))
            except ValueError:
                pass

        # ------------------------------------------------------------------
        # Case 3: final fallback (JD, ISO, etc.)
        # ------------------------------------------------------------------
        try:
            return Time(s)
        except Exception:
            raise ValueError(f"Unrecognized date format: {timestr}")


#%%
# Example usage
if __name__ == "__main__":

    self = OSCQuerier()
    ra = None
    dec = None
    radius_arcsec = 10
    objname = 'SN2011fe'
    save = True
    verbose = True
    return_instance = True
    result = self.get_object(objname = objname)
    photometry_tbl = self.get_photometry(objname = objname, save = save, verbose = verbose)
    spectra_tbl = self.get_spectra(objname = objname, save = save, verbose = verbose, return_instance = return_instance)
    # self.update_summary()

# %%
