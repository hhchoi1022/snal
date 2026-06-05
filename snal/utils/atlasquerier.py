#%%
import os
import time
import io
import requests
import pandas as pd
from pathlib import Path
from astropy.table import Table
import numpy as np
from astropy.table import MaskedColumn

from snal.configuration import Configuration
#%%

class ATLASQuerier:
    """
    A class for querying ATLAS forced photometry.
    """

    BASEURL = "https://fallingstar-data.com/forcedphot"

    def __init__(self):
        self.config = Configuration(config_filenames=['atlasquerier.config'])
        self.token = self.config.token
        self.save_dir = Path(self.config.save_dir)

        self.headers = {
            "Authorization": f"Token {self.token}",
            "Accept": "application/json",
        }

    def query_lightcurve(
        self,
        ra: float,
        dec: float,
        objname: str | None = None,
        mjd_min: float | None = None,
        mjd_max: float | None = None,
        save: bool = True,
        visualize: bool = False,
        verbose: bool = True,
    ):
        """
        Query ATLAS forced photometry.

        Parameters
        ----------
        ra, dec : float
            Sky position in degrees
        mjd_min, mjd_max : float, optional
            Time window
        """

        # ----------------------
        # Submit job
        # ----------------------
        payload = {"ra": ra, "dec": dec}
        if mjd_min is not None:
            payload["mjd_min"] = mjd_min
        if mjd_max is not None:
            payload["mjd_max"] = mjd_max

        r = requests.post(
            f"{self.BASEURL}/queue/",
            headers=self.headers,
            data=payload,
        )

        if r.status_code != 201:
            raise RuntimeError(f"ATLAS queue failed: {r.text}")

        task_url = r.json()["url"]

        if verbose:
            print(f"ATLAS task queued: {task_url}")

        # ----------------------
        # Poll job
        # ----------------------
        while True:
            time.sleep(10)
            status = requests.get(task_url, headers=self.headers).json()

            if status["finishtimestamp"]:
                result_url = status["result_url"]
                break

            if verbose:
                print("ATLAS job running...")

        # ----------------------
        # Fetch result
        # ----------------------
        text = requests.get(result_url, headers=self.headers).text
        df = pd.read_csv(io.StringIO(text.replace("###", "")), delim_whitespace=True)
        tbl = Table.from_pandas(df)

        if objname is None:
            objname = f"{ra:.5f}_{dec:.5f}"

        if save:
            outdir = self.save_dir / "ATLAS" / objname
            outdir.mkdir(parents=True, exist_ok=True)
            outfile = outdir / f"{objname}_ATLAS.csv"
            tbl.write(outfile, format="ascii.csv", overwrite=True)

            if verbose:
                print(f"ATLAS light curve saved to: {outfile}")
        lc = None
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
        lc.data['mjd'] = tbl['MJD']
        detected = (
            np.isfinite(tbl['m']) &
            np.isfinite(tbl['dm']) &
            (tbl['m'] > 5) &
            (tbl['dm'] < 0.5)
        )

        # create masked column: mask = NOT detected
        tbl['m'] = MaskedColumn(
            tbl['m'],
            mask=~detected
        )
        lc.data['mag'] = tbl['m']
        lc.data['magerr'] = tbl['dm']
        lc.data['filter'] = tbl['F']
        lc.data['depth'] = tbl['mag5sig']
        lc.data['zp_err'] = 0
        lc.data['observatory'] = 'ATLAS'
        lc.data['telname'] = 'ATLAS'
        groups = [f"{f}|{o}" for f, o in zip(lc.data['filter'], lc.data['observatory'])]
        lc.data['filter_group'] = groups

        lc.plot(ra = ra, dec = dec,
                ra_key = 'ra', dec_key = 'dec',
                flux_key = 'mag', fluxerr_key = 'magerr')
        return lc

# %%
if __name__ == "__main__":
    self = ATLASQuerier()
    ra = 233.857430764 
    dec = 12.0577222937
    tbl, lc = self.query_lightcurve(ra, dec, objname="test", mjd_min=60100.0, mjd_max=61000, visualize=True)
    print(tbl)
# %%