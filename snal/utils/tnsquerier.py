#%%
import os
import json
import time
import requests
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
from astropy.table import Table
from astropy.io import ascii
import pandas as pd
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import vstack
from astropy.time import Time
import requests
from pathlib import Path


from snal.configuration import Configuration
#%%
class TNSQuerier:
    """
    A class for querying the TNS server and handling results.
    """

    def __init__(self):
        """
        Initialize the TNSQuerier class.

        Parameters:
            api_key (str): The API key for TNS.
            user_id (str): User ID (if querying as a user).
            user_pwd (str): User password (if querying as a user).
        """
        self.config = Configuration(config_filenames=['tnsquerier.config'])
        self.api_key = self.config.api_key
        self.bot_id = self.config.bot_id
        self.bot_name = self.config.bot_name
        self.user_id = self.config.user_id
        self.user_pwd = self.config.user_pwd
        self.min_request_interval = 1.0
        self._last_request_time = 0.0

        self.base_url = "https://www.wis-tns.org"
        self.object_url = f"{self.base_url}/api/get/object"
        self.headers = {'User-Agent': self._set_tns_marker()}
        
    def get_object(self, 
                   objname: str, 
                   with_photometry: bool = True, 
                   with_spectra: bool = True,
                   save: bool = True,
                   verbose: bool = True
                   ):
        """
        Query a single TNS object by name.

        Parameters
        ----------
        objname : str
            TNS object name (e.g., 'SN2023ixf')
        with_photometry : bool
        with_spectra : bool

        Returns
        -------
        dict
            Raw JSON reply from TNS
        """
        data = {
            "objname": objname,
            "photometry": "1" if with_photometry else "0",
            "spectra": "1" if with_spectra else "0"
        }

        payload = {
            "api_key": self.api_key,
            "data": json.dumps(data)
        }

        r = self._post_with_rate_limit(
            self.object_url,
            data=payload,
            stream=False,
            timeout=60,
            verbose=verbose
        )

        if r.status_code != 200:
            raise RuntimeError(f"TNS request failed: {r.text}")

        reply = r.json()
        
        photometry_tbl = Table()
        spectroscopy_tbl = Table()
        if 'data' in reply.keys():
            metadata = reply['data'].copy()
            if 'photometry' in metadata.keys():
                metadata.pop('photometry')
            if 'spectra' in metadata.keys():
                metadata.pop('spectra')
            if save:
                if not os.path.exists(Path(self.config.save_dir) / objname):
                    os.makedirs(Path(self.config.save_dir) / objname)
                metadata_path = Path(self.config.save_dir) / objname / f"{objname}_meta_TNS.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                    if verbose:
                        print('Metadata saved to: ', metadata_path)
            
            # Get photometry
            if 'photometry' in reply['data'].keys():
                phot_list = reply['data']['photometry']
                if verbose:
                    print('Number of photometry points: ', len(phot_list))
                rows = []

                for p in phot_list:
                    mag = p.get("flux")
                    magerr = p.get("fluxerr")
                    limmag = p.get("limflux")

                    is_ul = (mag is None) and (limmag is not None)

                    rows.append({
                        "jd": float(p["jd"]),
                        "mag": float(mag) if mag is not None else np.nan,
                        "magerr": float(magerr) if magerr is not None else np.nan,
                        "limmag": float(limmag) if limmag is not None else np.nan,
                        "is_upper_limit": is_ul,
                        "filter": p.get("filters", {}).get("name"),
                        "mag_system": p.get("flux_unit", {}).get("name"),
                        "instrument": p.get("instrument", {}).get("name"),
                        "telescope": p.get("telescope", {}).get("name"),
                        "exptime": p.get("exptime"),
                        "observer": p.get("observer"),
                        "remarks": p.get("remarks")
                    })
                photometry_tbl = Table(rows=rows)
                photometry_tbl['objname'] = objname
                photometry_tbl['ra'] = reply['data']['radeg']
                photometry_tbl['dec'] = reply['data']['decdeg']
                photometry_tbl.sort("jd")

            if save:
                if len(photometry_tbl) > 0:
                    if not os.path.exists(Path(self.config.save_dir) / objname):
                        os.makedirs(Path(self.config.save_dir) / objname)
                    photometry_path = Path(self.config.save_dir) / objname / f"{objname}_photometry_TNS.csv"
                    photometry_tbl.write(photometry_path, format="ascii.fixed_width", overwrite=True)
                    if verbose:
                        print('Photometry table saved to: ', photometry_path)
            
            if 'spectra' in reply['data'].keys():
                spec_list = reply['data']['spectra']
                if verbose:
                    print('Number of spectra: ', len(spec_list))
                rows = []

                for s in spec_list:
                    ascii_file = s.get("asciifile")
                    fits_file = s.get("fitsfile")
                    ascii_file_path = None
                    fits_file_path = None
                    if ascii_file is not None:
                        ascii_file_path = Path(self.config.save_dir) / objname / 'TNS' / Path(ascii_file).name
                    if fits_file is not None:
                        fits_file_path = Path(self.config.save_dir) / objname / 'TNS' / Path(fits_file).name
                    rows.append({
                        "jd": float(s["jd"]),
                        "exptime": s.get("exptime"),
                        "instrument": s.get("instrument", {}).get("name"),
                        "telescope": s.get("telescope", {}).get("name"),
                        "source_group": s.get("source_group", {}).get("name"),
                        "observer": s.get("observer"),
                        "reducer": s.get("reducer"),
                        "remarks": s.get("remarks"),
                        "ascii_file": ascii_file_path if save else s.get("asciifile"),
                        "fits_file": fits_file_path if save else s.get("fitsfile"),
                        "public": bool(s.get("public", 1)),
                    })
                    if save:
                        if ascii_file_path is not None:
                            ascii_file_path = self._download_file(s.get("asciifile"), ascii_file_path, verbose = verbose)
                        if fits_file_path is not None:
                            fits_file_path = self._download_file(s.get("fitsfile"), fits_file_path, verbose = verbose)

                spectroscopy_tbl = Table(rows=rows)
                spectroscopy_tbl['objname'] = objname
                spectroscopy_tbl['ra'] = reply['data']['radeg']
                spectroscopy_tbl['dec'] = reply['data']['decdeg']
                spectroscopy_tbl.sort("jd")
                if save:
                    if not os.path.exists(Path(self.config.save_dir) / objname):
                        os.makedirs(Path(self.config.save_dir) / objname)
                    spectroscopy_path = Path(self.config.save_dir) / objname /f"{objname}_spectroscopy_TNS.csv"
                    spectroscopy_tbl.write(spectroscopy_path, format="ascii.fixed_width", overwrite=True)
                    if verbose:
                        print('Spectroscopy table saved to: ', spectroscopy_path)
        return metadata, photometry_tbl, spectroscopy_tbl
        
    def search_objects(self,
                       ra: float,
                       dec: float,
                       radius_arcsec: float = 3600,
                       obs_start_date: str = None,
                       obs_end_date: str = None,
                       save_dir: str = None,
                       timeout_seconds: float = 600,
                       verbose: bool = True
                       ):
        url_parameters = {"ra": str(ra), "decl": str(dec), "radius": str(radius_arcsec)}
        if obs_start_date is not None:
            url_parameters['discovery_date_start'] = obs_start_date
        if obs_end_date is not None:
            url_parameters['discovery_date_end'] = obs_end_date
        if verbose:
            print(url_parameters)
        file_ = self._search_tns(url_parameters = url_parameters, save_dir = save_dir, timeout_seconds = timeout_seconds)
        if file_ is not None:
            data = ascii.read(file_, format ='csv')
            return data
        else:
            return None
       
    # function for searching TNS with specified url parameters
    def _search_tns(self, url_parameters, 
                   save_dir : str = None,
                   timeout_seconds : float = 600,
                   verbose: bool = True
                   ):
        #--------------------------------------------------------------------
        # extract keywords and values from url parameters
        url_parameters['format'] = 'csv'
        url_parameters['num_page'] = '100'
        keywords = list(url_parameters.keys())
        values = list(url_parameters.values())
        #--------------------------------------------------------------------
        # flag for checking if url is with correct keywords
        wrong_url = False
        # check if keywords are correct
        for i in range(len(keywords)):
            if keywords[i] not in self._all_url_params:
                if verbose:
                    print ("Unknown url keyword '"+keywords[i]+"'\n")
                wrong_url = True
        if wrong_url == True:
            if verbose:
                print ("TNS search url is not in the correct format.\n")
        #--------------------------------------------------------------------
        # else, if everything is correct
        else:
            # current date and time
            current_datetime = datetime.now()
            current_date_time = current_datetime.strftime("%Y%m%d_%H%M%S")
            # current working directory
            cwd = os.getcwd()        
            extension = ".txt"
            tns_search_file = "tns_search_data_" + current_date_time + extension
            if save_dir is None:
                save_dir = self.config.save_dir
            tns_search_file_path = os.path.join(cwd, save_dir, tns_search_file)       
            if not os.path.exists(os.path.join(cwd, save_dir)):
                os.makedirs(os.path.join(cwd, save_dir))     
            #--------------------------------------------------------------------
            # build TNS search url
            url_par = ['&' + x + '=' + y for x, y in zip(keywords, values)]
            tns_search_url = self.search_url + '?' + "".join(url_par)
            #--------------------------------------------------------------------
            # page number
            page_num = 0
            # searched data
            searched_data = []
            # go trough every page
            start_time = time.time()
            while time.time() - start_time < timeout_seconds:
                # url for download
                url = tns_search_url + "&page=" + str(page_num)        
                headers = self.headers
                # downloading file using request module
                response = requests.post(url, headers=headers, data = {'api_key': self.api_key}, stream=True)
                # chek if response status code is not 200, or if returned data is empty
                if (response.status_code != 200) or (len((response.text).splitlines()) <= 1):
                    if response.status_code != 200:
                        print ("Sending download search request for page num " + str(page_num + 1) + "...")
                        self._print_response(response, page_num + 1, verbose = verbose)
                    break            
                if verbose:
                    print ("Sending download search request for page num " + str(page_num + 1) + "...")
                # print status code of the response
                self._print_response(response, page_num + 1, verbose = verbose)
                # get data
                data = (response.text).splitlines()
                # add to searched data
                if page_num == 0:
                    searched_data.append(data)
                else:
                    searched_data.append(data[1 : ])
                # check reset time
                reset = self._get_reset_time(response)
                if reset != None:
                    # Sleeping for reset + 1 sec
                    if verbose:
                        print("\nSleep for " + str(reset + 1) + " sec and then continue...\n") 
                    time.sleep(reset + 1)
                # increase page num
                page_num = page_num + 1
            #--------------------------------------------------------------------
            # if there is searched data, write to file
            if searched_data != []:            
                searched_data = [j for i in searched_data for j in i]            
                #if merge_to_single_file == 1:
                f = open(tns_search_file_path, 'w')
                for el in searched_data:
                    f.write(el + '\n')
                f.close()
                if len(searched_data) > 2:
                    if verbose:
                        print ("\nTNS searched data returned " + str(len(searched_data) - 1) + " rows. File '" + \
                        tns_search_file + "' is successfully created.\n")
                    return tns_search_file_path
                else: 
                    if verbose:
                        print ("\nTNS searched data returned 1 row. File '" + tns_search_file + "' is successfully created.\n")            
                    return tns_search_file_path
            else:
                if verbose:
                    print ("TNS searched data returned empty list. No file(s) created.\n")
                    
    def _download_file(self, url: str, outpath: str, timeout: int = 60, verbose: bool = True):
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        headers = {'User-Agent': self._set_tns_marker()}
        data = {'api_key': self.api_key}

        r = self._post_with_rate_limit(
            url,
            data=data,
            stream=True,
            timeout=timeout,
            verbose=verbose
        )

        if r.status_code != 200:
            raise RuntimeError(f"TNS download failed ({r.status_code}): {url}")

        with open(outpath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        if verbose:
            print('File downloaded to: ', outpath)

        return outpath

    def _load_config(self, config_path):
        """
        Reads a config file and returns a dictionary.

        Parameters:
            config_path (str): Path to the config file.

        Returns:
            dict: Configuration options as key-value pairs.
        """
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return config_dict
    
    def _set_tns_marker(self):
        """
        Set the TNS marker for API requests.

        Parameters:
            bot (bool): If True, use bot credentials. If False, use user credentials.

        Returns:
            str: TNS marker string.
        """
        return f'tns_marker{{"tns_id": "{self.bot_id}", "type": "bot", "name": "{self.bot_name}"}}'
    

    def _response_status(self, response):
        # external http errors
        ext_http_errors       = [403, 500, 503]
        err_msg               = ["Forbidden", "Internal Server Error: Something is broken", "Service Unavailable"]
        def is_string_json(string):
            try:
                json_object = json.loads(string)
            except Exception:
                return False
            return json_object

        json_string = is_string_json(response.text)
        if json_string != False:
            status = "[ " + str(json_string['id_code']) + " - '" + json_string['id_message'] + "' ]"
        else:
            status_code = response.status_code
            if status_code == 200:
                status_msg = 'OK'
            elif status_code in ext_http_errors:
                status_msg = err_msg[ext_http_errors.index(status_code)]
            else:
                status_msg = 'Undocumented error'
            status = "[ " + str(status_code) + " - '" + status_msg + "' ]"
        return status

    def _print_response(self, response, page_num, verbose: bool = True):
        status = self._response_status(response)
        if response.status_code == 200:     
            stats = 'Page number ' + str(page_num) + ' | return code: ' + status
        
        else:       
            stats = 'Page number ' + str(page_num) + ' | return code: ' + status        
        if verbose:
            print (stats)
                    
    def _send_request(self, parameters):
        """
        Send a POST request to TNS.

        Parameters:
            parameters (dict): Request parameters.

        Returns:
            Response: The HTTP response object.
        """
        search_data = {"api_key": self.api_key, "data": json.dumps(OrderedDict(parameters))}
        return requests.post(self.search_url, headers=self.headers, data=search_data)

    def _get_reset_time(self, response):
        """
        Get the rate-limit reset time from the response headers.

        Parameters:
            response (Response): The HTTP response object.

        Returns:
            int or None: Reset time in seconds, or None if no reset is required.
        """
        # If any of the '...-remaining' values is zero, return the reset time
        for name in response.headers:
            value = response.headers.get(name)
            if name.endswith('-remaining') and value == '0':
                return int(response.headers.get(name.replace('remaining', 'reset')))
        return None     
    
    def _get_skycoord(self,
                     ra : str or float,
                     dec: str or float,
                     ra_unit : str = 'hourangle',
                     dec_unit : str = 'deg'
                     ):
        """
        Parameters
        ==========
        ra : str | float = Right Ascension, if str, it should be in hms format (e.g., "10:20:30"), if float, it should be in decimal degrees
        dec : str | float = Declination, if str, it should be in dms format (e.g., "+20:30:40"), if float, it should be in decimal degrees
        
        Return
        ======
        coord : SkyCoord = SkyCoord object
        """
        
        u_ra = u.hourangle if ra_unit == 'hourangle' else u.deg
        u_dec = u.deg if dec_unit == 'deg' else u.deg
        coord = SkyCoord(ra, dec, unit=(u_ra, u_dec))
        return coord

    def _post_with_rate_limit(self, url, data, stream=False, timeout=60, verbose=True, max_retries=8):
        """
        POST request with polite retry on TNS 429 rate limits.
        """
        self._throttle()
        last_response = None

        for attempt in range(max_retries):
            response = requests.post(
                url,
                headers=self.headers,
                data=data,
                stream=stream,
                timeout=timeout
            )
            last_response = response

            if response.status_code == 200:
                return response

            if response.status_code == 429:
                reset = self._get_reset_time(response)

                # fallback if reset header is absent
                if reset is None:
                    reset = min(60, 2 ** attempt)

                wait_time = reset + 1

                if verbose:
                    print(f"TNS rate limit hit (429). Sleeping {wait_time} sec before retry...")

                time.sleep(wait_time)
                continue

            # for other errors, fail immediately
            raise RuntimeError(f"TNS request failed: {response.text}")

        raise RuntimeError(
            f"TNS request failed after {max_retries} retries: "
            f"{last_response.text if last_response is not None else 'No response'}"
        )

    def _throttle(self):
        dt = time.time() - self._last_request_time
        if dt < self.min_request_interval:
            time.sleep(self.min_request_interval - dt)
        self._last_request_time = time.time()
            
    @property
    def _all_url_params(self):
        URL_PARAMETERS   = ["discovered_period_value", "discovered_period_units", "unclassified_at", "classified_sne", "include_frb",
                            "name", "name_like", "isTNS_AT", "public", "ra", "decl", "radius", "coords_unit", "reporting_groupid[]",
                            "groupid", "classifier_groupid", "objtype", "at_type", "discovery_date_start",
                            "discovery_date_end",  "discovery_mag_min", "discovery_mag_max", "internal_name", "discoverer", "classifier",
                            "spectra_count", "redshift_min", "redshift_max", "hostname", "ext_catid", "ra_range_min", "ra_range_max",
                            "decl_range_min", "decl_range_max", "discovery_instrument[]", "classification_instrument[]",
                            "associated_groups[]", "official_discovery", "official_classification", "at_rep_remarks", "class_rep_remarks",
                            "frb_repeat", "frb_repeater_of_objid", "frb_measured_redshift", "frb_dm_range_min", "frb_dm_range_max",
                            "frb_rm_range_min", "frb_rm_range_max", "frb_snr_range_min", "frb_snr_range_max", "frb_flux_range_min",
                            "frb_flux_range_max", "format", "num_page"]
        return URL_PARAMETERS


#%%
# Example usag
if __name__ == "__main__":
    
    # if len(sys.argv) != 2:
    #     print("Usage: python tnsqurier.py <configfile>")
    #     sys.exit(1)

    # config_file_path = sys.argv[1]
    self = TNSQuerier()
    #tbl = self.query_objects(ra = 64.9725, dec = -54.948081, radius_arcsec = 3600)
    reply, photometry_tbl, spectroscopy_tbl = self.get_object(objname = 'SN2021hpr')
# %%
