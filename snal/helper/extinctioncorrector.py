#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 09:25:31 2022

@author: hhchoi1022
"""
#%%
import numpy as np
import extinction
import glob, os
from astropy.table import Table
from astropy.table import vstack
import re
from astropy.io import ascii
from pathlib import Path
from snal.helper import AnalysisHelper
#%%
class ExtinctionCorrector:
    
    def __init__(self,
                 filename : str):
        self.helper = AnalysisHelper()
        self.filename = Path(filename)
        self.original_data = self._load_table()
        self.corrected_data = self.original_data.copy()
        self.correction_history = list()

    def __repr__(self):
        return f'ExtinctionCorrector(corrected_filename = {self.corrected_filename}, correction_history = {self.correction_history})'
    
    @property
    def corrected_filename(self):
        if self.correction_history:
            return self.filename.parent / (self.filename.stem+'_'+'_'.join(self.correction_history) + self.filename.suffix)
        else:
            return self.filename.parent / (self.filename.stem + self.filename.suffix)
    
    def _load_table(self):
        tbl = ascii.read(self.filename, format = 'fixed_width')
        return tbl  

    def correct_host_extinction(self,  
                                ebv : float,
                                Rv : float = 3.1,
                                mag_key: str = 'mag',
                                filter_key: str = 'filter',
                                dereddening = True): 
        corrected_data = self.corrected_data.copy()
        import pyphot
        lib = pyphot.get_library() # Filter library 
        _, _, _, filter_key_dict, _ = self.helper.load_filt_keys()
        filterlist = sorted(list(set(corrected_data[filter_key])))
        Av = Rv * ebv
        if 'A_filter' not in corrected_data.keys():
            corrected_data['A_filter'] = 0.0

        for filtname in filterlist:
            if filtname not in filter_key_dict.keys():
                print(f'filter "{filtname}" is not in filter_key')
                pass
            else:
                filter_ = lib[filter_key_dict[filtname]]
                lpivot_AA = np.array([filter_.lpivot.value])
                Afilt = round(extinction.fitzpatrick99(lpivot_AA, Av, Rv)[0],3)
                
                if dereddening == True:
                    corrected_data[mag_key][corrected_data[filter_key] == filtname] -= Afilt
                    corrected_data['A_filter'][corrected_data[filter_key] == filtname] -= Afilt
                    is_corrected = 'Host_dereddening'
                elif dereddening == False:
                    corrected_data[mag_key][corrected_data[filter_key] == filtname] += Afilt
                    corrected_data['A_filter'][corrected_data[filter_key] == filtname] += Afilt
                    is_corrected = 'Host_reddening'
                else:
                    raise ValueError(f'dereddening = {dereddening}')
        self.correction_history.append(is_corrected)
        corrected_data[mag_key] = corrected_data[mag_key].round(3)
        corrected_data['A_filter'] = corrected_data['A_filter'].round(3)
        self.corrected_data = corrected_data

    def correct_mw_extinction(self, 
                              ra : float = 64.9708333, 
                              dec : float = -54.9480556, 
                              mwRv : float = 3.1, 
                              dereddening = True,
                              mag_key: str = 'mag',
                              filter_key: str = 'filter',
                              SFDmap_path : str = f"{os.path.dirname(os.path.abspath(__file__))}/sfddata-master"): 
        #ebv = 0.097 # Hosseinzadeh 2022
        import sfdmap
        import pyphot
        
        if not os.path.exists(SFDmap_path):
            raise ValueError(f'cannot find SFD map at {SFDmap_path}')
        dustmap = sfdmap.SFDMap(SFDmap_path)
        if ra is not None:
            try:
                mwebv = dustmap.ebv(ra, dec)
            except:
                raise ValueError('cannot calculate extinction with given coordinates')
        
        corrected_data = self.corrected_data.copy()
        lib = pyphot.get_library() # Filter library 
        _, _, _, filter_key_dict, _ = self.helper.load_filt_keys()
        filterlist = sorted(list(set(corrected_data[filter_key])))
        Av = mwRv * mwebv
        if 'A_filter' not in corrected_data.keys():
            corrected_data['A_filter'] = 0.0
            
        for filtname in filterlist:
            if filtname not in filter_key_dict.keys():
                print(f'filter "{filtname}" is not in filter_key')
                pass
            else:
                filter_ = lib[filter_key_dict[filtname]]
                lpivot_AA = np.array([filter_.lpivot.value])
                Afilt = round(extinction.fitzpatrick99(lpivot_AA, Av, mwRv)[0],3)
                if dereddening == True:
                    corrected_data[mag_key][corrected_data[filter_key] == filtname] -= Afilt
                    corrected_data['A_filter'][corrected_data[filter_key] == filtname] -= Afilt
                    is_corrected = 'MW_dereddening'
                    
                elif dereddening == False:
                    corrected_data[mag_key][corrected_data[filter_key] == filtname] += Afilt
                    corrected_data['A_filter'][corrected_data[filter_key] == filtname] += Afilt
                    is_corrected = 'MW_reddening'
                else:
                    raise ValueError(f'dereddening = {dereddening}')
        self.correction_history.append(is_corrected)
        corrected_data[mag_key] = corrected_data[mag_key].round(3)
        corrected_data['A_filter'] = corrected_data['A_filter'].round(3)
        self.corrected_data = corrected_data
        
    def save(self):
        self.corrected_data.write(filename = self.corrected_filename, format = 'ascii.fixed_width', overwrite = True)
        print(f'saved as {self.corrected_filename}')   

#%%
if __name__ == '__main__':
    ebv = 0.097  # Hosseinzadeh 2022
    ra = 64.9708333
    dec = -54.9480556
#%%
# data from Ahsall 2022 is already corrected for MW extinction >>> noextin_dat
if __name__ == '__main__':
    filename = '/home/hhchoi1022/code/SNAL/snal/SN2021aefx/data/SN2021aefx.ascii_fixed_width'
    self = ExtinctionCorrector(filename)
    self.correct_mw_extinction(ra = ra, dec = dec, mwRv = 3.10, dereddening = True)
    self.correct_host_extinction(ebv = ebv, Rv = 2.3, dereddening= True)
    # C.save()
#%% # For A22 table, already MW corrected. Thus reddeing A22 table to make them all same 
if __name__ == '__main__':
    #A22_file = '/data1/supernova_rawdata/SN2021aefx/photometry/Ashall2022.dat'
    #C = CorrectExtinction(A22_file)
    #C.correct_mw_extinction(ra = ra, dec = dec, mwRv = 3.10, dereddening = False)
    #C.save()
    
    A22_file = '/home/hhchoi1022/hhpy/Research/analysis/data/SN2021aefx/Ashall2022.dat'
    C = CorrectExtinction(A22_file)
    #C.correct_mw_extinction(ra = ra, dec = dec, mwRv = 3.10, dereddening = False)
    C.correct_host_extinction(ebv = ebv, Rv = 2.3, dereddening= True)
    C.save()
#%% 
if __name__ == '__main__':
    H22_file = '/home/hhchoi1022/hhpy/Research/analysis/data/SN2021aefx/Hosseinzadeh2022.dat'
    C = CorrectExtinction(H22_file)
    C.correct_mw_extinction(ra = ra, dec = dec, mwRv = 3.10, dereddening = True)
    C.correct_host_extinction(ebv = ebv, Rv = 2.3, dereddening= True)
    C.save()
