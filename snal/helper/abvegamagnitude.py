
#%%
from astropy.table import Table
import numpy as np
#%%


class ABVegaMagnitude:
    '''
    Conversion from Blanton et al. (2007).
    '''
    
    def __init__(self,
                 magnitude,
                 magsys,
                 filter_):
        self.magnitude = magnitude
        self.magsys = magsys
        self.filter_ = filter_
        if not isinstance(self.magnitude, float):
            if len(self.magnitude) > len(self.magsys):
                self.magsys = [self.magsys] * len(self.magnitude)
            if len(self.magnitude) > len(self.filter_):
                self.filter_ = [self.filter_] * len(self.magnitude)
        self.vega = self._vegamag()
        self.AB = self._abmag()
        
    def _abmag(self):
        if not isinstance(self.magnitude, float):
            converted_magnitudelist = []
            for magnitude, magsys, filter_ in zip(self.magnitude, self.magsys, self.filter_):
                if magsys.upper() == 'VEGA':
                    converted_magnitude = self.convert_vega_to_AB(magnitude, filter_)
                else:
                    converted_magnitude = magnitude
                converted_magnitudelist.append(converted_magnitude)
            return np.array(converted_magnitudelist)
        else:
            if self.magsys.upper() == 'VEGA':
                converted_magnitude = self.convert_vega_to_AB(self.magnitude, self.filter_)
            else:
                converted_magnitude = self.magnitude
            return converted_magnitude
    
    def _vegamag(self):
        if not isinstance(self.magnitude, float):
            converted_magnitudelist = []
            for magnitude, magsys, filter_ in zip(self.magnitude, self.magsys, self.filter_):
                if magsys.upper() == 'AB':
                    converted_magnitude = self.convert_AB_to_vega(magnitude, filter_)
                else:
                    converted_magnitude = magnitude
                converted_magnitudelist.append(converted_magnitude)
            return np.array(converted_magnitudelist)
        else:
            if self.magsys.upper() == 'AB':
                converted_magnitude = self.convert_AB_to_vega(self.magnitude, self.filter_)
            else:
                converted_magnitude = self.magnitude
            return converted_magnitude

    def convert_vega_to_AB(self,
                           magnitude_vega,
                           filter_ : str):
        filterset = ['U','B','V','R','I','J','H','K','u','g','r','i','z']
        if not filter_ in filterset:
            raise ValueError(f'{filter_} not registered in the database')
        if filter_ == 'U':
            magnitude_AB = magnitude_vega + 0.79
        if filter_ == 'B':
            magnitude_AB = magnitude_vega + -0.09
        if filter_ == 'V':
            magnitude_AB = magnitude_vega + 0.02
        if filter_ == 'R':
            magnitude_AB = magnitude_vega + 0.21
        if filter_ == 'I':
            magnitude_AB = magnitude_vega + 0.45
        if filter_ == 'J':
            magnitude_AB = magnitude_vega + 0.91
        if filter_ == 'H':
            magnitude_AB = magnitude_vega + 1.39
        if filter_ == 'K':
            magnitude_AB = magnitude_vega + 1.85
        if filter_ == 'u':
            magnitude_AB = magnitude_vega + 0.91
        if filter_ == 'g':
            magnitude_AB = magnitude_vega + -0.08
        if filter_ == 'r':
            magnitude_AB = magnitude_vega + 0.16
        if filter_ == 'i':
            magnitude_AB = magnitude_vega + 0.37
        if filter_ == 'z':
            magnitude_AB = magnitude_vega + 0.54
        return magnitude_AB
    
    def convert_AB_to_vega(self,
                           magnitude_AB,
                           filter_ : str):
        filterset = ['U','B','V','R','I','J','H','K','u','g','r','i','z']
        if not filter_ in filterset:
            raise ValueError(f'{filter_} not registered in the database')
        if filter_ == 'U':
            magnitude_vega = magnitude_AB - 0.79
        if filter_ == 'B':
            magnitude_vega = magnitude_AB - -0.09
        if filter_ == 'V':
            magnitude_vega = magnitude_AB - 0.02
        if filter_ == 'R':
            magnitude_vega = magnitude_AB - 0.21
        if filter_ == 'I':
            magnitude_vega = magnitude_AB - 0.45
        if filter_ == 'J':
            magnitude_vega = magnitude_AB - 0.91
        if filter_ == 'H':
            magnitude_vega = magnitude_AB - 1.39
        if filter_ == 'K':
            magnitude_vega = magnitude_AB - 1.85
        if filter_ == 'u':
            magnitude_vega = magnitude_AB - 0.91
        if filter_ == 'g':
            magnitude_vega = magnitude_AB - -0.08
        if filter_ == 'r':
            magnitude_vega = magnitude_AB - 0.16
        if filter_ == 'i':
            magnitude_vega = magnitude_AB - 0.37
        if filter_ == 'z':
            magnitude_vega = magnitude_AB - 0.54
        return magnitude_vega
            

        
# %%

if __name__ =='__main__':
    # Polin model from Vega to AB mag
    import glob, os
    from astropy.io import ascii
    from astropy.table import Table
    filekey = '/Users/hhchoi1022/Gitrepo/Data/IaSNe_Model/lightcurve/Polin/ddet_Polin2019/*/original/*'
    filelist = glob.glob(filekey)
    file_ = filelist[0]
    for file_ in filelist:
        savepath = os.path.dirname(os.path.dirname(file_)) + '/' + os.path.basename(file_)
        
        tbl = ascii.read(file_, names = ['phase','u','g','r','i','z','U','B','V','R','I'])
        converted_tbl = Table()
        converted_tbl['phase'] = tbl['phase']
        for filter_ in 'ugriz':
            converted_tbl[filter_] = tbl[filter_].round(4)
        for filter_ in 'UBVRI':
            converted_tbl[filter_] = ABVegaMagnitude(tbl[filter_], magsys = 'Vega', filter_ = filter_).AB.round(4)
        converted_tbl.write(savepath, format= 'ascii.fixed_width', overwrite = True)
# %%
