#%%
import os
import requests
from pathlib import Path
from astropy.io.votable import parse
from astropy.table import Table

from snal.configuration import Configuration

#%%
class ZTFQuerier:
    """
    A class for querying ZTF light curves from IRSA.
    """

    def __init__(self):
        self.config = Configuration(config_filenames=['ztfquerier.config'])
        self.save_dir = Path(self.config.save_dir)

    def query_lightcurve(
        self,
        ra: float,
        dec: float,
        radius: float,
        objname: str | None = None,
        save: bool = True,
        verbose: bool = True,
        visualize: bool = False,
    ):
        """
        Query ZTF light curve around a sky position.

        Parameters
        ----------
        ra, dec : float
            Sky position in degrees
        radius : float
            Search radius in degrees
        objname : str, optional
            Object name for saving
        """

        url = (
            "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
            f"?POS=CIRCLE {ra} {dec} {radius}"
            "&BAD_CATFLAGS_MASK=32768"
        )

        r = requests.get(url)
        if r.status_code != 200:
            raise RuntimeError(f"ZTF request failed ({r.status_code})")

        tmpfile = Path(".ztf_tmp.xml")
        tmpfile.write_text(r.text)

        tbl = parse(tmpfile).get_first_table().to_table()
        tmpfile.unlink()

        if objname is None:
            objname = f"{ra:.5f}_{dec:.5f}"

        if save:
            outdir = self.save_dir / "ZTF" / objname
            outdir.mkdir(parents=True, exist_ok=True)
            outfile = outdir / f"{objname}_ZTF.csv"
            tbl.write(outfile, format="ascii.csv", overwrite=True)

            if verbose:
                print(f"ZTF light curve saved to: {outfile}")
        
        if visualize:
            lc = self._plot_lightcurve(tbl, ra, dec)

        return tbl, lc
    
    def _plot_lightcurve(self,
                         tbl: Table,
                         ra: float,
                         dec: float,
                         ):
        from ezphot.dataobjects import LightCurve
        lc = LightCurve()
        lc.data = tbl
        lc.data['zp_err'] = tbl['magzprms']
        lc.data['depth'] = tbl['limitmag']
        lc.data['telname'] = 'ZTF'
        lc.data['observatory'] = 'ZTF'
        filters = [f.split('z')[1] for f in tbl['filtercode']]
        lc.data['filter'] = filters
        groups = [f"{f}|{o}" for f, o in zip(filters, lc.data['observatory'])]
        lc.data['filter_group'] = groups

        lc.plot(ra = ra, dec = dec,
                ra_key = 'ra', dec_key = 'dec',
                flux_key = 'mag', fluxerr_key = 'magerr')
        return lc


# %%
if __name__ == "__main__":
    self = ZTFQuerier()
    ra = 233.857430764 
    dec = 12.0577222937
    tbl, lc = self.query_lightcurve(ra, dec, 5/3600, objname="test", visualize=True)
# %%