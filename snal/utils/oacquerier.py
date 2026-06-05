#%%
import os
import json
import requests
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import astropy.units as u
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astroquery.oac import OAC

from snal.configuration import Configuration
#%%


class OACQuerier:
    """
    A class for querying the Open Astronomy Catalog (OAC).
    """

    def __init__(self):
        """
        Initialize OACQuerier.
        """
        self.querier = OAC()
        self.config = Configuration(config_filenames=['oacquerier.config'])
        
    def get_object(self,
                   objname: str):
        self.querier.get_object(objname)
        
    def query_lightcurve(self,
                         ra: float | None = None,
                         dec: float | None = None,
                         radius_arcsec: float | None = 10,
                         objname: str | None = None):
        if ra is not None and dec is not None and radius_arcsec is not None:
            object_tbl = self.querier.query_region(coordinates = [ra, dec], radius=  radius_arcsec/3600)
            if len(object_tbl) == 0:
                raise ValueError("No objects found in the region")
            objname = object_tbl['event'][0]
        try:
            photometry_tbl = self.querier.get_photometry(event = objname)
        except Exception as e:
            raise ValueError(f"Failed to get photometry for {objname}: {e}")
        return photometry_tbl
    
    def query_spectra(self,
                      ra: float | None = None,
                      dec: float | None = None,
                      radius_arcsec: float | None = 10,
                      objname: str | None = None,
                      return_instance: bool = False):
        if ra is not None and dec is not None and radius_arcsec is not None:
            object_tbl = self.querier.query_region(coordinates = [ra, dec], radius=  radius_arcsec/3600)
            if len(object_tbl) == 0:
                raise ValueError("No objects found in the region")
            objname = object_tbl['event'][0]
        try:
            spectra_tbl = self.querier.get_spectra(event = objname)
        except Exception as e:
            raise ValueError(f"Failed to get spectra for {objname}: {e}")
        
        if return_instance:
            spec_data = spectra_tbl[objname]['spectra']
            if len(spec_data) == 0:
                raise ValueError("No spectra data found for the object")
            from ezphot.dataobjects import Spectrum
            all_spectrum = []
            for i, data in enumerate(spec_data):
                try:
                    mjd = data[0]
                    data = data[1]
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
                    wl = wl.astype(np.float32)
                    flux = flux.astype(np.float32)
                    if err_exists:
                        fluxerr = fluxerr.astype(np.float32)
                    else:
                        fluxerr = None
                    wl_sort_map = np.argsort(wl)
                    wl_sorted = wl[wl_sort_map]
                    flux_sorted = flux[wl_sort_map]
                    spectrum = Spectrum(path = None, 
                                        wavelength = wl_sorted, 
                                        flux = flux_sorted)
                    meta_row = dict()
                    meta_row['mjd'] = mjd
                    spectrum.header = meta_row
                    all_spectrum.append(spectrum)
                except Exception as e:
                    print(f"Error processing {i}th spectrum: {e}")
                    continue
            return spectra_tbl, all_spectrum
        return spectra_tbl
    
# %%
if __name__ == "__main__":
    self = OACQuerier()
    ra = None
    dec = None
    radius_arcsec = 10
    objname = 'SN2011fe'
    #photometry_tbl = self.query_lightcurve(ra, dec, radius_arcsec, objname)
    # spectra_tbl = self.query_spectra(ra, dec, radius_arcsec, objname, return_instance = True)
# %%
