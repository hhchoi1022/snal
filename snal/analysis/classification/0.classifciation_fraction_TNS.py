
#%%
from bridge.alertquerier import TNSQuerier
#%%
# =========================================================
# 7. Example usage
# =========================================================

# Replace this with your actual collection code.
# The important structure is: records = [{"search": ..., "detail": ...}, ...]
tnsquerier = TNSQuerier()

tbl_2025 = tnsquerier.search_all(
    discovery_date_start="2025-01-01",
    discovery_date_end="2025-12-31",
    public="all",
    is_tns_at="all",
    num_page=500,
    timeout_seconds=7200,
    verbose=True,
)
#%%
from astropy.table import Table
# %%
tbl_2025 = Table.read('/home/hhchoi1022/bridge/alert/tns/processed/tns_search_data_20260407_183017.txt', format = 'ascii')

import re
import requests
from bs4 import BeautifulSoup


def parse_tnscr_bibcode(bibcode: str):
    """
    Parse ADS-style TNSCR bibcodes such as:
      2026TNSCR1323....1Z
      2026TNSCR.868....1H
      2026TNSCR...3....1M
      2026TNSCR..30....1C
      2025TNSCR4515....1d
      2025TNSCR3202....1v
    """
    s = str(bibcode).strip()

    m = re.match(
        r'^(?P<year>\d{4})TNSCR(?P<ridfield>.{4})\.{4}1(?P<author>[A-Za-z])$',
        s
    )
    if not m:
        raise ValueError(f"Not a recognized TNSCR bibcode: {bibcode}")

    ridfield = m.group('ridfield')
    rid_str = ridfield.replace('.', '')

    if rid_str == '' or not rid_str.isdigit():
        raise ValueError(f"Could not extract TNSCR id from bibcode: {bibcode}")

    return {
        'year': int(m.group('year')),
        'tnscr_id': int(rid_str),
        'author_initial': m.group('author'),
    }

def pick_earliest_bibcode(bibcode_field):
    if bibcode_field is None:
        return None

    try:
        if np.ma.is_masked(bibcode_field):
            return None
    except Exception:
        pass

    s = str(bibcode_field).strip()
    if s == '' or s == '--' or s.lower() in ('none', 'nan'):
        return None

    bibcodes = [x.strip() for x in s.split(',') if x.strip()]
    if len(bibcodes) == 0:
        return None

    bibcodes = list(dict.fromkeys(bibcodes))

    valid = []
    for bib in bibcodes:
        try:
            parse_tnscr_bibcode(bib)
            valid.append(bib)
        except Exception:
            pass

    if len(valid) == 0:
        return None

    return min(
        valid,
        key=lambda bib: (
            parse_tnscr_bibcode(bib)['year'],
            parse_tnscr_bibcode(bib)['tnscr_id']
        )
    )

def estimate_tns_date_from_bibcode(bibcode: str, timeout: int = 30):
    """
    Try to recover the TNS report date from a TNSCR bibcode.

    Returns:
        dict with parsed info and any recovered dates/URLs
    """
    info = parse_tnscr_bibcode(bibcode)
    year = info["year"]
    rid = info["tnscr_id"]

    result = {
        "bibcode": bibcode,
        "year": year,
        "tnscr_id": rid,
        "author_initial": info["author_initial"],
        "tns_url": f"https://www.wis-tns.org/ads/TNSCR-{year}-{rid}",
        "tns_report_date": None,
        "ads_url": f"https://ui.adsabs.harvard.edu/abs/{bibcode}/abstract",
        "ads_pubdate": None,
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    # 1) Try TNS classification report page
    try:
        r = requests.get(result["tns_url"], headers=headers, timeout=timeout)
        if r.ok:
            text = r.text

            # Often appears as: "Transient Classification Report for 2025-01-10"
            m = re.search(
                r"Transient Classification Report for (\d{4}-\d{2}-\d{2})",
                text
            )
            if m:
                result["tns_report_date"] = m.group(1)
            else:
                # fallback: sometimes date may appear as Date Received (UTC)
                m2 = re.search(
                    r"Date Received \(UTC\)\s*[:\-]?\s*(\d{4}-\d{2}-\d{2})",
                    text
                )
                if m2:
                    result["tns_report_date"] = m2.group(1)
    except Exception:
        pass

    # 2) Fallback to ADS abstract page (usually month/year, not exact day)
    try:
        r = requests.get(result["ads_url"], headers=headers, timeout=timeout)
        if r.ok:
            text = r.text
            # Examples often contain "Pub Date: October 2020"
            m = re.search(r"Pub Date:\s*([A-Za-z]+\s+\d{4})", text)
            if m:
                result["ads_pubdate"] = m.group(1)
    except Exception:
        pass

    return result

# %%
tbl_2025_classified = tbl_2025[~tbl_2025['Classification Bibcodes'].mask]
# %%
from tqdm import tqdm
all_bibcodes = []
for bibcode in tqdm(tbl_2025_classified['Classification Bibcodes']):
    try:
        picked_bibcode = pick_earliest_bibcode(bibcode)
        if picked_bibcode is None:
            print(bibcode)
        else:
            all_bibcodes.append(picked_bibcode)
    except:
        all_bibcodes.append(None)
#%%
import requests

ADS_TOKEN = "WpxmeNampwYaVhvisXMnpUoQCKE4dD8azTNvWMgS"
import re
import requests

def get_tns_report_date_from_ads_title(bibcode: str):
    url = "https://api.adsabs.harvard.edu/v1/search/query"
    headers = {"Authorization": f"Bearer {ADS_TOKEN}"}
    params = {
        "q": f'bibcode:"{bibcode}"',
        "fl": "bibcode,title",
        "rows": 1,
    }

    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()

    docs = r.json().get("response", {}).get("docs", [])
    if not docs:
        return None

    doc = docs[0]
    title = doc.get("title")

    # ADS often returns title as a list of one string
    if isinstance(title, list):
        title = " ".join(title)
    if not title:
        return None

    m = re.search(r'(\d{4}-\d{2}-\d{2})', title)
    if m:
        return {
            "bibcode": doc.get("bibcode"),
            "title": title,
            "tns_report_date": m.group(1),
        }

    return {
        "bibcode": doc.get("bibcode"),
        "title": title,
        "tns_report_date": None,
    }
# %%
all_tns_report_dates = []
for bibcode in all_bibcodes[:30]:
    a = get_tns_report_date_from_ads_title(bibcode)
    if a is not None:
        all_tns_report_dates.append(a['tns_report_date'])
# %%
all_tns_report_dates
# %%
# %%
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

def build_ads_session(token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})

    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=50,
        pool_maxsize=50,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_tns_report_date_from_ads_title(
    bibcode: str,
    session: requests.Session,
    timeout: int = 20,
):
    if bibcode is None:
        return bibcode, None

    try:
        url = "https://api.adsabs.harvard.edu/v1/search/query"
        params = {
            "q": f'bibcode:"{bibcode}"',
            "fl": "bibcode,title",
            "rows": 1,
        }

        r = session.get(url, params=params, timeout=timeout)
        r.raise_for_status()

        docs = r.json().get("response", {}).get("docs", [])
        if not docs:
            return bibcode, None

        doc = docs[0]
        title = doc.get("title")

        if isinstance(title, list):
            title = " ".join(title)

        if not title:
            return bibcode, None

        m = re.search(r"(\d{4}-\d{2}-\d{2})", title)
        if m:
            return bibcode, m.group(1)

        return bibcode, None

    except Exception:
        return bibcode, None


def fetch_tns_report_dates_multithread(
    bibcodes,
    ads_token: str,
    max_workers: int = 8,
    timeout: int = 20,
):
    # keep order, remove duplicates, skip None
    unique_bibcodes = list(dict.fromkeys([bib for bib in bibcodes if bib is not None]))
    results_map = {}

    def worker(bibcode):
        session = build_ads_session(ads_token)
        try:
            return get_tns_report_date_from_ads_title(
                bibcode=bibcode,
                session=session,
                timeout=timeout,
            )
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, bibcode) for bibcode in unique_bibcodes]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Fetching ADS title dates"):
            bibcode, tns_report_date = fut.result()
            results_map[bibcode] = tns_report_date

    # return dates in the same order as input bibcodes
    ordered_dates = [results_map.get(bibcode, None) if bibcode is not None else None for bibcode in bibcodes]
    return ordered_dates, results_map


# %%
all_tns_report_dates, results_map = fetch_tns_report_dates_multithread(
    bibcodes=all_bibcodes,
    ads_token=ADS_TOKEN,
    max_workers=32,
    timeout=20,
)
# %%
all_tns_report_dates

#%%
bibcode_map = {}
for bibcode in all_bibcodes:
    if bibcode in results_map:
        if results_map[bibcode] is not None:
            bibcode_map[bibcode] = results_map[bibcode]
        else:
            bibcode_map[bibcode] = estimate_tns_date_from_bibcode(bibcode)['tns_report_date']
    else:
        bibcode_map[bibcode] = None
#%%
bibcode = list(bibcode_map.keys())[40]
# %%
estimate_tns_date_from_bibcode(bibcode)
#%%

tbl_2025_classified['Classification Bibcodes'][120]
#%%
for i in range(len(all_bibcodes)):
    print(all_bibcodes[i], tbl_2025_classified['Classification Bibcodes'][i])
# %%
all_input_dates = [bibcode_map[bibcode] for bibcode in all_bibcodes]
tbl_2025_classified['Classification_date'] = all_input_dates
tbl_2025_classified['Classification_bibcode_applied'] = all_bibcodes
# %%
tbl_2025_classified['Classification Bibcodes']
# %%
tbl_2025_classified['Classification_bibcode_applied']
# %%
tbl_2025_classified.write('TNS_classification_date_applied.csv', format='ascii.fixed_width', overwrite=True)
# %%
tbl_2025_classified = Table.read('TNS_classification_date_applied.csv', format='ascii.fixed_width')
tbl = tbl_2025_classified.copy()
# %%
tbl['Classification_date']
# %%
from astropy.time import Time
from datetime import datetime
# Time format = 2025-01-03 13:55:56.928
discovery_datetime_str = tbl['Discovery Date (UT)']
discovery_mjds = [Time(dt, format = 'iso').mjd for dt in discovery_datetime_str]
classification_mjds = [Time(datetime.strptime(dt, '%Y-%m-%d')).mjd for dt in tbl['Classification_date']]
tbl['Discovery_mjd'] = discovery_mjds
tbl['Classification_mjd'] = classification_mjds
# %%
import numpy as np
delta_mjds = np.array(classification_mjds) - np.array(discovery_mjds)
# %%
plt.hist(delta_mjds, bins = 100)
# %% 
from astropy.stats import sigma_clipped_stats
mean, median, std = sigma_clipped_stats(delta_mjds)
# %%
mean, median, std
# %%
tbl_all = tbl[~tbl['Redshift'].mask]
tbl_near = tbl[~tbl['Redshift'].mask]
tbl_near = tbl_near[(tbl_near['Redshift'] < 0.03)]
tbl_near_sn = tbl_near[~tbl_near['Obj. Type'].mask]
tbl_near_sn = tbl_near_sn[[transient_type.startswith('SN') for transient_type in tbl_near_sn['Obj. Type']]]
#%%
import matplotlib.pyplot as plt
# Time delay
dt_all = np.array(tbl_all['Classification_mjd'] - tbl_all['Discovery_mjd'], dtype=float)
dt = np.array(tbl_near_sn['Classification_mjd'] - tbl_near_sn['Discovery_mjd'], dtype=float)

# Optional: remove NaN / inf
dt_all = dt_all[np.isfinite(dt_all)]
dt = dt[np.isfinite(dt)]

# Sigma-clipped statistics
mean_all, median_all, std_all = sigma_clipped_stats(dt_all, sigma=3.0)
mean, median, std = sigma_clipped_stats(dt, sigma=3.0)

# Plot
fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)

ax.hist(
    dt_all,
    bins=150,
    color = 'k',
    # histtype='stepfilled',
    alpha=0.85,
    edgecolor='black',
    linewidth=0.8,
)
ax.hist(
    dt,
    bins=150,
    color = 'r',
    # histtype='stepfilled',
    alpha=0.85,
    edgecolor='black',
    linewidth=0.8,
)

# Reference lines
ax.axvline(median_all, c = 'k', linestyle='--', linewidth=2, label=f'[all] Median = {median_all:.2f} d')
ax.axvline(median, c = 'r', linestyle='--', linewidth=2, label=f'[z < 0.03] Median = {median:.2f} d')
# Labels
ax.set_xlabel('Classification delay since discovery [days]', fontsize=18)
ax.set_ylabel('Number of supernovae', fontsize=18)
# ax.set_title('Delay Between Discovery and Classification for Supernovae in 2025', fontsize=14, pad=12)

# Improve appearance
ax.tick_params(axis='both', which='major', labelsize=15)
ax.minorticks_on()
ax.grid(alpha=0.25, linestyle=':')
ax.legend(frameon=False, fontsize=16)

# # Add summary text
# ax.text(
#     0.98, 0.95,
#     f'N = {len(dt)}\n'
#     f'$\mu_{{clip}}$ = {mean:.2f} d\n'
#     f'Median = {median:.2f} d\n'
#     f'$\sigma_{{clip}}$ = {std:.2f} d',
#     transform=ax.transAxes,
#     ha='right', va='top',
#     fontsize=10,
#     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='0.8')
# )
ax.set_xlim(-10, 100)
plt.tight_layout()
plt.show()
# %%