

#%%
import numpy as np
import json
import h5py
import csv
import hashlib
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any
from astropy.table import Table
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
matplotlib.use('Agg')
#%%
"""
In this code, we will format the template spectra to be used for the classification.
There are two types of templates:
1. SNID-SAGE templates
2. SCAP templates

These templates will be formatted for NGSF-7DT use
"""

#%% 

"""
First, process SNID-SAGE templates. 

Matched epoch and wiserep spectrum information is stored in the spreadsheet.
"""
#%%
# 1. Load the spreadsheet
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1wwbBlAvOUamYM-l_F-N6xd4UYPEcKhntzLETNtP-m1o/edit?gid=0#gid=0"
import gspread
from astropy.table import Table

gc = gspread.service_account(
    filename="/home/hhchoi1022/code/SNID-SAGE/googlesheet.json"
)

sh = gc.open_by_url(SPREADSHEET_URL)
ws = sh.get_worksheet(0)

rows = ws.get_all_records(numericise_ignore=["all"])

SNID_tbl = Table(rows)
#%%
# 2. Define functions to load the templates and wiserep spectrum
TEMPLATES_DIR = Path("/home/hhchoi1022/code/SNID-SAGE/templates")
TEMPLATES_INDEX_FILE = TEMPLATES_DIR / "template_index.json"
WISEREP_DIR = Path("/home/hhchoi1022/code/SNID-SAGE/wiserep_spectra")


def robust_mad(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return 1.4826 * mad

def _try_parse_obs_datetime(obs_date_raw: str) -> datetime | None:
    """Parse WISeREP Obs-date text into datetime if possible."""
    obs_date_raw = (obs_date_raw or "").strip()
    if obs_date_raw == "":
        return None
    fmts = (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    )
    for fmt in fmts:
        try:
            return datetime.strptime(obs_date_raw, fmt)
        except Exception:
            continue
    return None


def _load_ascii_nested_wiserep_payloads(file_path: Path) -> np.ndarray | None:
    """
    WISeREP nested CSV: each line is like ``"wave,flux,err,skyback",,`` (first cell holds
    comma-separated numbers). Returns (n, 4) float array or None if no data rows found.
    """
    out: list[list[float]] = []
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f):
            if not row:
                continue
            first = (row[0] or "").strip()
            if not first or first.startswith("#"):
                continue
            if first.startswith('"') and first.endswith('"'):
                first = first[1:-1].strip()
            if "," not in first:
                continue
            parts = [p.strip() for p in first.split(",")]
            if len(parts) < 2:
                continue
            u0 = parts[0].upper()
            if u0 == "WAVE" or u0.startswith("WAVE"):
                continue
            try:
                w = float(parts[0])
                fl = float(parts[1])
            except ValueError:
                continue
            fe = float(parts[2]) if len(parts) >= 3 and parts[2] != "" else float("nan")
            sk = float(parts[3]) if len(parts) >= 4 and parts[3] != "" else float("nan")
            out.append([w, fl, fe, sk])
    if not out:
        return None
    return np.asarray(out, dtype=float)

def _load_ascii_wavelength_flux_fluxerr(
    file_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load spectrum columns in order: wavelength, flux, fluxerr, background (optional).

    Supports 2–4 top-level columns, optional trailing empty CSV fields, and WISeREP
    nested CSV (quoted ``wave,flux,err,skyback`` in the first cell). Missing fluxerr /
    background are returned as NaN.
    """
    candidates: list[np.ndarray] = []
    for delimiter, skip_header in ((",", 1), (",", 0), (None, 0)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = np.genfromtxt(
                file_path,
                comments="#",
                delimiter=delimiter,
                skip_header=skip_header,
                invalid_raise=False,
            )
        if data.ndim == 1 and data.size >= 2:
            data = data.reshape(1, -1)
        if data.ndim == 2 and data.shape[1] >= 2 and data.shape[0] > 0:
            candidates.append(data)

    data: np.ndarray | None = None
    for c in candidates:
        wcol = c[:, 0].astype(float)
        fcol = c[:, 1].astype(float)
        if np.any(np.isfinite(wcol) & np.isfinite(fcol)):
            data = c
            break

    if data is None:
        nested = _load_ascii_nested_wiserep_payloads(file_path)
        if nested is not None:
            data = nested

    if data is None or data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Could not parse wavelength/flux from: {file_path}")

    ncols = min(4, int(data.shape[1]))
    wave = data[:, 0].astype(float)
    flux = data[:, 1].astype(float)
    fluxerr = np.full_like(wave, np.nan, dtype=float)
    background = np.full_like(wave, np.nan, dtype=float)
    if ncols >= 3:
        fluxerr = data[:, 2].astype(float)
    if ncols >= 4:
        background = data[:, 3].astype(float)

    finite = np.isfinite(wave) & np.isfinite(flux)
    wave = wave[finite]
    flux = flux[finite]
    fluxerr = fluxerr[finite]
    background = background[finite]
    if wave.size == 0:
        raise ValueError(f"No finite wavelength/flux rows in: {file_path}")
    return wave, flux, fluxerr, background

def mask_invalid_flux(flux):
    """
    Mask 0, nan, and inf values.

    Returns
    -------
    bad_mask : bool array
        True where flux is invalid.
    """
    flux = np.asarray(flux, dtype=float)
    return (~np.isfinite(flux)) | (flux == 0)

def mask_linear_regions(
    flux,
    window=21,
    r2_threshold=0.995,
    min_slope=0.0,
    min_valid_fraction=0.8,
):
    """
    True = suspiciously linear region
    """
    flux = np.asarray(flux, dtype=float)
    n = len(flux)
    x_all = np.arange(n)

    invalid_bad = mask_invalid_flux(flux)
    good = ~invalid_bad

    linear_bad = np.zeros(n, dtype=bool)
    half = window // 2

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)

        local_good = good[start:end]

        if np.mean(local_good) < min_valid_fraction:
            continue

        x = x_all[start:end][local_good]
        y = flux[start:end][local_good]

        if len(y) < 4:
            continue

        coeff = np.polyfit(x, y, deg=1)
        y_fit = np.polyval(coeff, x)

        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            continue

        r2 = 1 - ss_res / ss_tot
        slope = coeff[0]

        if r2 >= r2_threshold and abs(slope) >= min_slope:
            linear_bad[start:end] = True

    return linear_bad

def mask_low_snr(
    flux,
    noise,
    snr_threshold=3.0,
    use_abs_flux=True,
):
    """
    Mask low-SNR pixels.

    Parameters
    ----------
    flux : array-like
        1D flux array.
    noise : array-like
        1D noise/error array. Must have same shape as flux.
    snr_threshold : float
        Pixels with SNR below this value are masked.
    use_abs_flux : bool
        If True, use abs(flux) / noise.
        If False, use flux / noise.

    Returns
    -------
    low_snr_bad : bool array
        True where SNR is too low.
    snr : array
        Calculated SNR array.
    """

    flux = np.asarray(flux, dtype=float)
    noise = np.asarray(noise, dtype=float)

    if flux.shape != noise.shape:
        raise ValueError("flux and noise must have the same shape.")

    bad_noise = (~np.isfinite(noise)) | (noise <= 0)

    snr = np.full_like(flux, np.nan, dtype=float)

    valid = np.isfinite(flux) & np.isfinite(noise) & (noise > 0)

    if use_abs_flux:
        snr[valid] = np.abs(flux[valid]) / noise[valid]
    else:
        snr[valid] = flux[valid] / noise[valid]

    low_snr_bad = bad_noise | (~np.isfinite(snr)) | (snr < snr_threshold)

    return low_snr_bad, snr

def mask_abrupt_peaks(
    flux,
    window=11,
    sigma_threshold=8.0,
):
    """
    True = abrupt isolated peak
    """
    flux = np.asarray(flux, dtype=float)
    n = len(flux)

    invalid_bad = mask_invalid_flux(flux)
    good = ~invalid_bad

    peak_bad = np.zeros(n, dtype=bool)

    if np.sum(good) < 5:
        return peak_bad

    global_sigma = robust_mad(flux[good])

    if not np.isfinite(global_sigma) or global_sigma == 0:
        global_sigma = np.nanstd(flux[good])

    if not np.isfinite(global_sigma) or global_sigma == 0:
        return peak_bad

    if window % 2 == 0:
        window += 1

    half = window // 2

    for i in range(n):
        if not good[i]:
            continue

        start = max(0, i - half)
        end = min(n, i + half + 1)

        local_good = good[start:end]

        if np.sum(local_good) < 5:
            continue

        y = flux[start:end][local_good]

        local_median = np.nanmedian(y)
        local_sigma = robust_mad(y)

        if not np.isfinite(local_sigma) or local_sigma == 0:
            local_sigma = global_sigma

        deviation = abs(flux[i] - local_median)

        if deviation > sigma_threshold * local_sigma:
            peak_bad[i] = True

    return peak_bad

def get_all_flux_masks(
    flux,
    noise=None,
    continuum=None,
    linear_window=21,
    linear_r2_threshold=0.995,
    linear_min_slope=0.0,
    snr_threshold=3.0,
    peak_window=11,
    peak_sigma_threshold=8.0,
):
    """
    Returns all masks separately.

    Individual masks:
        True = bad

    final_good:
        True = usable
    """

    invalid_bad = mask_invalid_flux(flux)

    linear_bad = mask_linear_regions(
        flux,
        window=linear_window,
        r2_threshold=linear_r2_threshold,
        min_slope=linear_min_slope,
    )

    if noise is not None:
        if continuum is None:
            continuum = flux
        low_snr_bad, snr = mask_low_snr(
            continuum,
            noise,
            snr_threshold=snr_threshold,
            use_abs_flux=True,
        )
    else:
        low_snr_bad = np.zeros_like(flux, dtype=bool)
        snr = None

    peak_bad = mask_abrupt_peaks(
        flux,
        window=peak_window,
        sigma_threshold=peak_sigma_threshold,
    )

    total_bad = invalid_bad | linear_bad | low_snr_bad | peak_bad
    final_good = ~total_bad

    return {
        "invalid_bad": invalid_bad,
        "linear_bad": linear_bad,
        "low_snr_bad": low_snr_bad,
        "peak_bad": peak_bad,
        "total_bad": total_bad,
        "final_good": final_good,
        "snr": snr,
    }

import numpy as np
from scipy.ndimage import median_filter

def estimate_noise_from_flux(
    flux,
    continuum_window=31,
    noise_window=31,
):
    """
    Estimate local noise from flux only.

    Parameters
    ----------
    flux : array-like
        1D spectrum.
    continuum_window : int
        Window size for smooth continuum estimate.
    noise_window : int
        Window size for local residual scatter.

    Returns
    -------
    noise : array
        Estimated local noise array.
    continuum : array
        Estimated smooth continuum.
    residual : array
        flux - continuum.
    """

    flux = np.asarray(flux, dtype=float)
    n = len(flux)

    valid = np.isfinite(flux) & (flux != 0)

    noise = np.full(n, np.nan)
    continuum = np.full(n, np.nan)
    residual = np.full(n, np.nan)

    if np.sum(valid) < 5:
        return noise, continuum, residual

    # Fill invalid values temporarily for filtering
    filled_flux = flux.copy()
    filled_flux[~valid] = np.nanmedian(flux[valid])

    # Median filter works best with odd window sizes
    if continuum_window % 2 == 0:
        continuum_window += 1

    if noise_window % 2 == 0:
        noise_window += 1

    # Smooth spectrum
    continuum = median_filter(
        filled_flux,
        size=continuum_window,
        mode="nearest",
    )

    # Residual should mostly contain noise + sharp features
    residual = flux - continuum

    half = noise_window // 2

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)

        local = residual[start:end]
        local_valid = valid[start:end] & np.isfinite(local) & (local != 0)

        if np.sum(local_valid) < 5:
            continue

        noise[i] = robust_mad(local[local_valid])

    return noise, continuum, residual

def plot_flux_masks(wavelength, flux, masks, save_path = None):
    wavelength = np.asarray(wavelength)
    flux = np.asarray(flux)

    continuum = masks.get("continuum")
    residual = masks.get("residual")
    has_continuum = continuum is not None and np.any(np.isfinite(continuum))
    has_residual = residual is not None and np.any(np.isfinite(residual))

    nrows = 1 + has_residual
    fig, axes = plt.subplots(
        nrows, 1, figsize=(14, 4 * nrows + 1),
        sharex=True, gridspec_kw={"height_ratios": [3, 1][:nrows]},
    )
    if nrows == 1:
        axes = [axes]

    ax_flux = axes[0]
    ax_flux.plot(wavelength, flux, color="black", lw=1, label="Flux")

    if has_continuum:
        ax_flux.plot(wavelength, continuum, color="dodgerblue", lw=1.5,
                     ls="--", label="Continuum", zorder=4)

    mask_styles = [
        ("invalid_bad", "red", "0 / NaN / Inf"),
        ("linear_bad", "orange", "Linear region"),
        ("low_snr_bad", "purple", "Low SNR"),
        ("noisy_bad", "green", "Noisy"),
        ("peak_bad", "blue", "Abrupt peak"),
    ]
    for key, color, label in mask_styles:
        m = masks.get(key)
        if m is not None and np.any(m):
            ax_flux.scatter(wavelength[m], flux[m], s=25, color=color,
                            label=label, zorder=5)

    ax_flux.set_ylabel("Flux")
    ax_flux.legend(fontsize=8, ncol=3)

    if has_residual:
        ax_res = axes[1]
        ax_res.plot(wavelength, residual, color="black", lw=0.8, label="Residual")
        ax_res.axhline(0, color="gray", ls=":", lw=0.5)

        local_noise = masks.get("local_noise")
        if local_noise is not None and np.any(np.isfinite(local_noise)):
            ax_res.fill_between(
                wavelength, -local_noise, local_noise,
                color="dodgerblue", alpha=0.2, label="Local noise",
            )

        noisy_bad = masks.get("noisy_bad")
        if noisy_bad is not None and np.any(noisy_bad):
            ax_res.scatter(wavelength[noisy_bad], residual[noisy_bad],
                           s=20, color="green", label="Noisy", zorder=5)

        ax_res.set_ylabel("Residual")
        ax_res.legend(fontsize=8, ncol=3)

    axes[-1].set_xlabel("Wavelength")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    else:
        plt.show()

def plot_snr(wavelength, masks, snr_threshold=3.0, save_path = None):
    snr = masks["snr"]

    if snr is None:
        raise ValueError("No SNR array found. Provide noise to get_all_flux_masks().")

    plt.figure(figsize=(14, 4))
    plt.plot(wavelength, snr, color="black", lw=1, label="SNR")
    plt.axhline(snr_threshold, color="red", ls="--", label=f"SNR = {snr_threshold}")

    plt.scatter(
        wavelength[masks["low_snr_bad"]],
        snr[masks["low_snr_bad"]],
        s=20,
        color="purple",
        label="Low SNR",
        zorder=5,
    )

    plt.xlabel("Wavelength")
    plt.ylabel("SNR")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    else:
        plt.show()

def plot_continuum_noise_residual(
    wavelength,
    flux,
    continuum,
    noise,
    residual=None,
    snr_threshold=3.0,
    title=None,
    save_path = None,
):
    """
    Three-panel diagnostic plot for the continuum fit, noise estimate, and residual.

    Parameters
    ----------
    wavelength : array-like
        Wavelength array.
    flux : array-like
        Original flux array.
    continuum : array-like
        Fitted continuum (e.g. from estimate_noise_from_flux).
    noise : array-like
        Estimated noise array.
    residual : array-like or None
        Flux minus continuum. Computed automatically if None.
    snr_threshold : float
        Threshold drawn on the SNR panel.
    title : str or None
        Optional super-title for the figure.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)
    continuum = np.asarray(continuum, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if residual is None:
        residual = flux - continuum
    else:
        residual = np.asarray(residual, dtype=float)

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 1.5], "hspace": 0.08},
    )

    # --- Panel 1: flux + continuum ---
    ax = axes[0]
    ax.plot(wavelength, flux, color="black", lw=0.8, label="Flux", zorder=2)
    ax.plot(wavelength, continuum, color="dodgerblue", lw=1.5, ls="--",
            label="Continuum", zorder=3)
    ax.fill_between(
        wavelength, continuum - noise, continuum + noise,
        color="dodgerblue", alpha=0.15, label=r"Continuum $\pm$ noise", zorder=1,
    )
    ax.set_ylabel("Flux")
    ax.legend(fontsize=8, loc="upper right")

    # --- Panel 2: residual + noise envelope ---
    ax = axes[1]
    ax.plot(wavelength, residual, color="black", lw=0.7, label="Residual", zorder=2)
    ax.axhline(0, color="gray", ls=":", lw=0.5)
    ax.fill_between(
        wavelength, -noise, noise,
        color="tomato", alpha=0.2, label=r"$\pm$ noise", zorder=1,
    )
    ax.plot(wavelength, noise, color="tomato", lw=1.0, ls="--",
            label="Noise", zorder=3)
    ax.plot(wavelength, -noise, color="tomato", lw=1.0, ls="--", zorder=3)

    outlier = np.abs(residual) > 3 * noise
    if np.any(outlier):
        ax.scatter(wavelength[outlier], residual[outlier], s=15, color="red",
                   marker="x", label=r"|residual| > 3$\sigma$", zorder=4)
    ax.set_ylabel("Residual")
    ax.legend(fontsize=8, loc="upper right")

    # --- Panel 3: SNR ---
    ax = axes[2]
    valid_noise = np.isfinite(noise) & (noise > 0)
    snr = np.full_like(flux, np.nan)
    snr[valid_noise] = np.abs(continuum[valid_noise]) / noise[valid_noise]
    ax.plot(wavelength, snr, color="black", lw=0.8, label="SNR (continuum/noise)")
    ax.axhline(snr_threshold, color="red", ls="--", lw=0.8,
               label=f"SNR = {snr_threshold}")
    ax.set_ylabel("SNR")
    ax.set_xlabel("Wavelength")
    ax.legend(fontsize=8, loc="upper right")

    if title is not None:
        fig.suptitle(title, fontsize=12, y=0.995)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    else:
        plt.show()


def rebin_spectrum_flux_conserving(wave_old, flux_old, wave_new):
    """
    Rebin a spectrum onto a new wavelength array while conserving integrated flux.

    Parameters
    ----------
    wave_old : array
        Original wavelength centers.
    flux_old : array
        Original flux density values, e.g. erg/s/cm^2/Angstrom.
    wave_new : array
        New wavelength centers.

    Returns
    -------
    flux_new : array
        Rebinned flux density on wave_new.
    """

    wave_old = np.asarray(wave_old)
    flux_old = np.asarray(flux_old)
    wave_new = np.asarray(wave_new)

    def centers_to_edges(wave):
        edges = np.empty(len(wave) + 1)
        edges[1:-1] = 0.5 * (wave[1:] + wave[:-1])
        edges[0] = wave[0] - 0.5 * (wave[1] - wave[0])
        edges[-1] = wave[-1] + 0.5 * (wave[-1] - wave[-2])
        return edges

    old_edges = centers_to_edges(wave_old)
    new_edges = centers_to_edges(wave_new)

    # Integrated flux in each old bin
    old_bin_widths = np.diff(old_edges)
    integrated_flux_old = flux_old * old_bin_widths

    # Cumulative integrated flux
    cumulative_flux = np.zeros(len(old_edges))
    cumulative_flux[1:] = np.cumsum(integrated_flux_old)

    # Interpolate cumulative flux onto new bin edges
    cumulative_flux_new = np.interp(
        new_edges,
        old_edges,
        cumulative_flux,
        left=np.nan,
        right=np.nan
    )

    # Flux in each new bin
    integrated_flux_new = np.diff(cumulative_flux_new)
    new_bin_widths = np.diff(new_edges)

    flux_new = integrated_flux_new / new_bin_widths

    return flux_new

def load_template_spectrum(
    template_id: str,
    index_path: Path = TEMPLATES_INDEX_FILE,
) -> tuple[dict[float | None, np.ndarray], dict[float | None, np.ndarray], dict[str, Any]]:
    """
    Load a template spectrum from SNID-SAGE HDF5 storage.

    Returns:
        wavelength, flux, metadata
        - wavelength: dict keyed by `age` -> wavelength grid (np.ndarray)
        - flux: dict keyed by `age` -> flux array (np.ndarray)
        - metadata: template_index.json metadata (common) + computed values, with
          `metadata["epoch"]` containing per-age epoch metadata.
    """
    if not index_path.exists():
        raise FileNotFoundError(f"Template index not found: {index_path}")

    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)

    templates: dict[str, Any] = index.get("templates", {})
    meta = templates.get(template_id)
    if not isinstance(meta, dict):
        raise ValueError(f"Template not found in index: {template_id}")

    storage_file = meta.get("storage_file")
    if not isinstance(storage_file, str) or storage_file.strip() == "":
        raise ValueError(f"No storage_file for template: {template_id}")

    h5_path = TEMPLATES_DIR / storage_file
    if not h5_path.exists():
        raise FileNotFoundError(f"Template HDF5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        if "metadata" not in h5 or "standard_wavelength" not in h5["metadata"]:
            raise KeyError(f"Missing metadata/standard_wavelength in {h5_path}")
        if "templates" not in h5 or template_id not in h5["templates"]:
            raise KeyError(f"Template '{template_id}' not found in {h5_path}")
        h5 = h5py.File(h5_path, "r")

        wavelength = h5["metadata"]["standard_wavelength"][...].astype(float)
        group = h5["templates"][template_id]

        epochs: list[tuple[str, float | None, np.ndarray]] = []
        if "epochs" in group:
            epoch_names = sorted(
                group["epochs"].keys(), key=lambda x: int(x.split("_")[-1])
            )
            for epoch_name in epoch_names:
                ep = group["epochs"][epoch_name]
                age = ep.attrs.get("age")
                age_val = float(age) if age is not None else None
                flux = ep["flux"][...].astype(float)
                epochs.append((epoch_name, age_val, flux))
        else:
            age = group.attrs.get("age")
            age_val = float(age) if age is not None else None
            flux = group["flux"][...].astype(float)
            epochs.append(("epoch_0", age_val, flux))

    # Each epoch has its own flux values, so infer wl_min/wl_max per epoch from
    # where the interpolated flux is actually defined (non-zero / finite).
    epoch_dict: dict[float | None, dict[str, Any]] = {}
    wavelength_dict: dict[float | None, np.ndarray] = {}
    flux_dict: dict[float | None, np.ndarray] = {}
    per_epoch_wl_min: list[float] = []
    per_epoch_wl_max: list[float] = []

    for epoch_name, age_val, flux in epochs:
        abs_flux = np.abs(flux)
        valid = np.isfinite(abs_flux) & (abs_flux != 0)
        if not np.any(valid):
            valid = np.isfinite(abs_flux)

        wl_min_e = float(wavelength[valid].min()) if np.any(valid) else float(wavelength.min())
        wl_max_e = float(wavelength[valid].max()) if np.any(valid) else float(wavelength.max())

        wavelength_dict[epoch_name] = wavelength
        flux_dict[epoch_name] = flux
        epoch_dict[epoch_name] = {
            "epoch": epoch_name,
            "age": age_val,
            "wl_min": wl_min_e,
            "wl_max": wl_max_e,
            "n_points": int(flux.shape[0]),
        }
        per_epoch_wl_min.append(wl_min_e)
        per_epoch_wl_max.append(wl_max_e)

    wl_min = float(min(per_epoch_wl_min)) if per_epoch_wl_min else float(wavelength.min())
    wl_max = float(max(per_epoch_wl_max)) if per_epoch_wl_max else float(wavelength.max())

    metadata_out: dict[str, Any] = dict(meta)
    # Common metadata (not epoch-specific)
    metadata_out.update(
        {
            "n_epochs": len(epoch_dict),
            "epoch_ages": list(epoch_dict.keys()),
        }
    )
    metadata_out["epoch"] = epoch_dict
    return wavelength_dict, flux_dict, metadata_out


def load_wiserep_spectrum(
    wiserep_id: str,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    """
    Load WISeREP spectra for one target.

    Returns:
        wavelength, flux, fluxerr, metadata
        - wavelength: dict keyed by `obsdate` -> wavelength array
        - flux: dict keyed by `obsdate` -> flux array
        - fluxerr: dict keyed by `obsdate` -> flux uncertainty (NaN where absent)
        - metadata: common metadata + `metadata["epoch"]` per obsdate entries.
    """
    wiserep_dir = WISEREP_DIR / wiserep_id
    if not wiserep_dir.exists():
        raise FileNotFoundError(f"WISeREP directory not found: {wiserep_dir}")

    info_path = wiserep_dir / "metadata.csv"
    if not info_path.exists():
        fallback_info_path = wiserep_dir / "downloaded_spectra_info.csv"
        if fallback_info_path.exists():
            info_path = fallback_info_path
        else:
            raise FileNotFoundError(
                f"WISeREP metadata not found in {wiserep_dir} "
                "(expected metadata.csv or downloaded_spectra_info.csv)"
            )

    rows: list[dict[str, str]] = []
    with info_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))

    if not rows:
        raise ValueError(f"No rows in metadata file: {info_path}")

    wavelength_dict: dict[str, np.ndarray] = {}
    flux_dict: dict[str, np.ndarray] = {}
    fluxerr_dict: dict[str, np.ndarray] = {}
    epoch_dict: dict[str, dict[str, Any]] = {}
    redshifts: list[float] = []
    last_wave: np.ndarray | None = None
    last_flux: np.ndarray | None = None
    # Global dedup: if the exact wavelength+flux arrays repeat anywhere in the
    # file, keep the first occurrence and drop subsequent duplicates.
    seen_spectra: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    for row in rows:
        file_name = (row.get("Spectrum ascii File") or "").strip()
        if file_name == "":
            continue
        file_path = wiserep_dir / file_name
        if not file_path.exists():
            continue

        wave, flux, fluxerr, _bg = _load_ascii_wavelength_flux_fluxerr(file_path)
        if np.max(wave) < 1000:
            continue
        obs_date_raw = (row.get("Obs-date (UT)") or "").strip()
        dt = _try_parse_obs_datetime(obs_date_raw)
        obs_date = obs_date_raw if obs_date_raw != "" else None

        # Drop identical duplicates anywhere (not just adjacent rows).
        # Uses an exact digest of the parsed float arrays.
        h = hashlib.blake2b(digest_size=16)
        h.update(wave.tobytes())
        h.update(flux.tobytes())
        digest = h.digest()
        if digest in seen_spectra:
            prev_wave, prev_flux = seen_spectra[digest]
            if prev_wave.shape == wave.shape and prev_flux.shape == flux.shape and np.allclose(
                prev_wave, wave, rtol=1e-7, atol=0.0, equal_nan=True
            ) and np.allclose(prev_flux, flux, rtol=1e-7, atol=0.0, equal_nan=True):
                continue

        # Update the global seen set now that we plan to keep this spectrum.
        # (If the caller later decides to skip for other reasons, the "seen"
        # entry will simply be an unused cached copy.)
        seen_spectra[digest] = (wave, flux)
        # Keep the original timestamp string (including 00:00:00).
        obsdate_key = obs_date_raw

        # Estimate wl_min/wl_max from flux-defined region (interpolated templates
        # often have zeros outside the useful range).
        abs_flux = np.abs(flux)
        valid = np.isfinite(abs_flux) & (abs_flux != 0)
        if not np.any(valid):
            valid = np.isfinite(abs_flux)
        wl_min = float(wave[valid].min()) if np.any(valid) else float(wave.min())
        wl_max = float(wave[valid].max()) if np.any(valid) else float(wave.max())

        wavelength_dict[obsdate_key] = wave
        flux_dict[obsdate_key] = flux
        fluxerr_dict[obsdate_key] = fluxerr

        # Build per-epoch metadata (include all columns from the row).
        row_out: dict[str, Any] = dict(row)
        row_out["Obs-date (UT)"] = obs_date
        row_out["obsdate"] = obs_date
        row_out["wl_min"] = wl_min
        row_out["wl_max"] = wl_max
        row_out["spectrum_file"] = file_name
        epoch_dict[obsdate_key] = row_out

        ztxt = (row.get("Redshift") or "").strip()
        if ztxt != "":
            try:
                zval = float(ztxt)
                if np.isfinite(zval) and zval > 0:
                    redshifts.append(zval)
            except Exception:
                pass

    if not flux_dict:
        raise ValueError(f"No readable spectra found for target: {wiserep_id}")

    redshift: float | None = None
    if redshifts:
        redshift = float(np.median(np.asarray(redshifts, dtype=float)))

    metadata_out: dict[str, Any] = {
        "wiserep_id": wiserep_id,
        "n_spectra": len(flux_dict),
        "redshift": redshift,
        "redshift_values": redshifts,
        "epoch": epoch_dict,
    }
    return wavelength_dict, flux_dict, fluxerr_dict, metadata_out

def mask_lines(
    wavelength,
    lines_dict,
    lines_to_use=None,
    redshift=0.0,
    wavelength_frame="rest",
    padding_A=0.0,
    return_regions=False,
):
    """
    Mask telluric absorption regions.

    Parameters
    ----------
    wavelength : array-like
        1D wavelength array.
        If wavelength_frame="observed", this should be observed-frame wavelength.
        If wavelength_frame="rest", this should be rest-frame wavelength.
    lines_dict : dict
        Dictionary with entries containing "center_A" and "width_A".
    lines_to_use : list[str] or None
        Telluric line keys to use. If None, use all keys in lines_dict.
    redshift : float
        Redshift used when wavelength is rest-frame.
        Telluric bands are fixed in observed frame, so for rest-frame spectra:
            lambda_rest = lambda_obs / (1 + z)
    wavelength_frame : {"observed", "rest"}
        Frame of the input wavelength array.
    padding_A : float
        Extra padding added to each side of every telluric band, in observed-frame Angstrom.
    return_regions : bool
        If True, also return the exact masked wavelength intervals in the same frame
        as the input wavelength.

    Returns
    -------
    telluric_bad : np.ndarray
        Boolean mask. True = inside telluric region.
    regions : dict, optional
        Returned only if return_regions=True.
    """

    wavelength = np.asarray(wavelength, dtype=float)

    if lines_to_use is None:
        lines_to_use = list(lines_dict.keys())

    z = float(redshift)
    if not np.isfinite(z):
        raise ValueError("redshift must be finite.")

    if wavelength_frame not in {"observed", "rest"}:
        raise ValueError("wavelength_frame must be either 'observed' or 'rest'.")

    telluric_bad = np.zeros_like(wavelength, dtype=bool)
    regions = {}

    for line_name in lines_to_use:
        if line_name not in lines_dict:
            raise KeyError(f"{line_name!r} not found in lines_dict.")

        info = lines_dict[line_name]

        center_obs = float(info["center_A"])
        width_obs = float(info["width_A"])

        wl_min_obs = center_obs - width_obs / 2.0 - padding_A
        wl_max_obs = center_obs + width_obs / 2.0 + padding_A

        if wavelength_frame == "observed":
            wl_min = wl_min_obs
            wl_max = wl_max_obs
        else:
            wl_min = wl_min_obs / (1.0 + z)
            wl_max = wl_max_obs / (1.0 + z)

        this_bad = (
            np.isfinite(wavelength)
            & (wavelength >= wl_min)
            & (wavelength <= wl_max)
        )

        telluric_bad |= this_bad

        regions[line_name] = {
            "wl_min": wl_min,
            "wl_max": wl_max,
            "species": info.get("species"),
            "strength": info.get("strength"),
        }

    if return_regions:
        return telluric_bad, regions

    return telluric_bad

def interpolate_masked_regions(
    wavelength,
    flux,
    bad_mask,
    method="linear",
    fill_edges="nearest",
    min_good_points=2,
    return_interpolated_mask=False,
):
    """
    Interpolate over masked spectral regions.

    Parameters
    ----------
    wavelength : array-like
        1D wavelength array.
    flux : array-like
        1D flux array.
    bad_mask : array-like
        Boolean mask. True = bad / should be interpolated.
    method : {"linear"}
        Interpolation method. Currently only linear interpolation is implemented.
    fill_edges : {"nearest", "nan"}
        How to handle masked regions outside the good wavelength range.
        "nearest" fills with the nearest good flux value.
        "nan" leaves edge regions as NaN.
    min_good_points : int
        Minimum number of good pixels required for interpolation.
    return_interpolated_mask : bool
        If True, also return a mask of pixels that were replaced.

    Returns
    -------
    flux_interp : np.ndarray
        Flux with bad pixels replaced by interpolated values.
    interpolated_mask : np.ndarray, optional
        True where flux was replaced.
    """

    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)
    bad_mask = np.asarray(bad_mask, dtype=bool)

    if wavelength.shape != flux.shape:
        raise ValueError("wavelength and flux must have the same shape.")

    if bad_mask.shape != flux.shape:
        raise ValueError("bad_mask and flux must have the same shape.")

    if method != "linear":
        raise ValueError("Only method='linear' is currently supported.")

    if fill_edges not in {"nearest", "nan"}:
        raise ValueError("fill_edges must be either 'nearest' or 'nan'.")

    finite_good = (
        np.isfinite(wavelength)
        & np.isfinite(flux)
        & ~bad_mask
    )

    flux_interp = flux.copy()
    interpolated_mask = bad_mask.copy()

    if np.sum(finite_good) < min_good_points:
        flux_interp[bad_mask] = np.nan
        if return_interpolated_mask:
            return flux_interp, interpolated_mask
        return flux_interp

    order = np.argsort(wavelength)
    wave_sorted = wavelength[order]
    flux_sorted = flux[order]
    bad_sorted = bad_mask[order]
    good_sorted = finite_good[order]

    target_sorted = bad_sorted & np.isfinite(wave_sorted)

    interp_sorted = flux_sorted.copy()

    if fill_edges == "nearest":
        left = flux_sorted[good_sorted][0]
        right = flux_sorted[good_sorted][-1]
    else:
        left = np.nan
        right = np.nan

    interp_sorted[target_sorted] = np.interp(
        wave_sorted[target_sorted],
        wave_sorted[good_sorted],
        flux_sorted[good_sorted],
        left=left,
        right=right,
    )

    # Restore original order
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(len(order))

    flux_interp = interp_sorted[inverse_order]

    if return_interpolated_mask:
        return flux_interp, interpolated_mask

    return flux_interp

# %%
telluric_lines = {
    # Strongest
    "O2_A_band": {
        "center_A": 7625.0,
        "width_A": 130.0,   # ~7545–7665
        "species": "O2",
        "strength": "very strong",
    },

    "H2O_9300_band": {
        "center_A": 9300.0,
        "width_A": 250.0,   # ~9175–9425
        "species": "H2O",
        "strength": "very strong",
    },

    "H2O_8200_band": {
        "center_A": 8200.0,
        "width_A": 180.0,   # ~8110–8290
        "species": "H2O",
        "strength": "strong",
    },

    "H2O_7200_band": {
        "center_A": 7200.0,
        "width_A": 160.0,   # ~7120–7280
        "species": "H2O",
        "strength": "strong",
    },

    "O2_B_band": {
        "center_A": 6900.0,
        "width_A": 80.0,    # ~6830–6910
        "species": "O2",
        "strength": "medium-strong",
    },

    "H2O_5900_band": {
        "center_A": 5900.0,
        "width_A": 80.0,    # ~5860–5940
        "species": "H2O",
        "strength": "medium",
    },

    "H2O_6500_band": {
        "center_A": 6500.0,
        "width_A": 80.0,    # ~6460–6540
        "species": "H2O",
        "strength": "weak-medium",
    },

    "H2O_9700_band": {
        "center_A": 9700.0,
        "width_A": 180.0,   # ~9610–9790
        "species": "H2O",
        "strength": "weak-medium",
    },
}
host_lines = {
    # Very strong
    "Halpha_6563": {
        "center_A": 6562.8,
        "width_A": 50.0,
        "species": "H I",
        "strength": "very strong",
    },

    # Strong
    "OII_3727_doublet": {
        "center_A": 3727.0,
        "width_A": 50.0,     # [O II] λλ3726,3729; unresolved/partly resolved depending on resolution
        "species": "[O II]",
        "strength": "strong",
    },

    "Hbeta_4861": {
        "center_A": 4861.3,
        "width_A": 50.0,
        "species": "H I",
        "strength": "strong",
    },

    "OIII_5007": {
        "center_A": 5006.8,
        "width_A": 50.0,
        "species": "[O III]",
        "strength": "strong",
    },

    "NII_6583": {
        "center_A": 6583.5,
        "width_A": 50.0,
        "species": "[N II]",
        "strength": "strong",
    },

    # Medium-strong
    "OIII_4959": {
        "center_A": 4958.9,
        "width_A": 50.0,
        "species": "[O III]",
        "strength": "medium-strong",
    },

    "SIII_9531": {
        "center_A": 9530.6,
        "width_A": 50.0,
        "species": "[S III]",
        "strength": "medium-strong",
    },

    # Medium
    "NII_6548": {
        "center_A": 6548.1,
        "width_A": 50.0,
        "species": "[N II]",
        "strength": "medium",
    },

    "SII_6717": {
        "center_A": 6716.4,
        "width_A": 50.0,
        "species": "[S II]",
        "strength": "medium",
    },

    "SII_6731": {
        "center_A": 6730.8,
        "width_A": 50.0,
        "species": "[S II]",
        "strength": "medium",
    },

    "Hgamma_4340": {
        "center_A": 4340.5,
        "width_A": 50.0,
        "species": "H I",
        "strength": "medium",
    },

    "SIII_9069": {
        "center_A": 9068.6,
        "width_A": 50.0,
        "species": "[S III]",
        "strength": "medium",
    },

    # Weak-medium
    "Hdelta_4102": {
        "center_A": 4101.7,
        "width_A": 3.0,
        "species": "H I",
        "strength": "weak-medium",
    },

    "NeIII_3869": {
        "center_A": 3868.8,
        "width_A": 3.0,
        "species": "[Ne III]",
        "strength": "weak-medium",
    },

    "HeI_5876": {
        "center_A": 5875.6,
        "width_A": 3.0,
        "species": "He I",
        "strength": "weak-medium",
    },

    "OI_6300": {
        "center_A": 6300.3,
        "width_A": 3.0,
        "species": "[O I]",
        "strength": "weak-medium",
    },

    "ArIII_7136": {
        "center_A": 7135.8,
        "width_A": 3.0,
        "species": "[Ar III]",
        "strength": "weak-medium",
    },

    # Weak
    "OI_6364": {
        "center_A": 6363.8,
        "width_A": 3.0,
        "species": "[O I]",
        "strength": "weak",
    },

    "CaII_7291": {
        "center_A": 7291.5,
        "width_A": 3.0,
        "species": "[Ca II]",
        "strength": "weak",
    },

    "CaII_7324": {
        "center_A": 7323.9,
        "width_A": 3.0,
        "species": "[Ca II]",
        "strength": "weak",
    },
}
absorption_lines = {
    "Na_D_blend": {
        "center_A": 5892.0,
        "width_A": 15.0,
        "species": "[Na I]",
        "strength": "absorption",
    },
}
telluric_lines_to_use = ['O2_A_band', 'H2O_9300_band', 'H2O_8200_band', 'H2O_7200_band', 'O2_B_band', 'H2O_5900_band']
host_lines_to_use = ['Halpha_6563', 'OII_3727_doublet', 'Hbeta_4861', 'OIII_5007', 'NII_6583', 'OIII_4959', 'SIII_9531', 'SII_6731', 'SII_6717']
absorption_lines_to_use = ['Na_D_blend']
#%%
#%%
%matplotlib inline
SAVE_DIR = Path('./specquality')
SAVE_DIR.mkdir(parents=True, exist_ok=True)
failed_history = dict()
objname = 'sn1996cb'
epoch_col = 'epoch_0'
match_info = SNID_tbl[SNID_tbl['Objname'] == objname][0]
redshift = match_info['Redshift']
template_wl_dict, template_flux_dict, template_meta = load_template_spectrum(objname)
wiserep_wl_dict, wiserep_flux_dict, wiserep_fluxerr_dict, wiserep_meta = load_wiserep_spectrum(objname)

num_templates = match_info['n_templates']
num_matched = match_info['n_matched_template']
colnames_epoch = [f'epoch_{i}' for i in range(int(num_templates))]

obsdate_col = match_info[epoch_col]
redshift_col = epoch_col + '_redshift'
redshift_spec = match_info[redshift_col]
w_template = template_wl_dict[epoch_col]
f_template = template_flux_dict[epoch_col]
w_wiserep = wiserep_wl_dict[obsdate_col]
w_wiserep = w_wiserep / (1 + float(redshift_spec))
f_wiserep = wiserep_flux_dict[obsdate_col]
# Rebin the spectrum to the common wavelength grid
w_wiserep_rebinned = np.arange(w_wiserep.min(), w_wiserep.max(), 5)
f_wiserep_rebinned = rebin_spectrum_flux_conserving(w_wiserep, f_wiserep, w_wiserep_rebinned)
ferr_wiserep_rebinned, continuum_wiserep_rebinned, residual_wiserep_rebinned = estimate_noise_from_flux(f_wiserep_rebinned, 
    continuum_window=51, 
    noise_window=31)

masks_wiserep = get_all_flux_masks(
    f_wiserep_rebinned,
    noise=ferr_wiserep_rebinned,
    continuum=continuum_wiserep_rebinned,
    linear_window=21,
    linear_r2_threshold=0.9999,
    snr_threshold=1.0,
    peak_window=5,
    peak_sigma_threshold=100.0,
)

masks_template = get_all_flux_masks(
    f_template,
    noise=None,
    continuum=None,
    linear_window=5,
    linear_r2_threshold=0.9999,
    snr_threshold=1.0,
    peak_window=5,
    peak_sigma_threshold=100.0,
)

fig, ax_wiserep = plt.subplots(figsize = (14, 8))
ax_wiserep.set_title(f'{objname} {epoch_col} template vs WISeREP')
ax_template = ax_wiserep.twinx()
line_wiserep, = ax_wiserep.plot(
    w_wiserep_rebinned, f_wiserep_rebinned, c="k", alpha=1, label="WISeREP"
)
line_template, = ax_template.plot(
    w_template, f_template, c="b", alpha=0.5, label="SNID template"
)

wiserep_ylim = np.nanmax(np.abs(f_wiserep_rebinned))
template_ylim = np.nanmax(np.abs(f_template))

# Plot masked region
ax_wiserep.vlines(w_wiserep_rebinned[masks_wiserep['total_bad']], 0.9*wiserep_ylim, wiserep_ylim, colors = 'r', alpha = 0.5, label = 'Total bad')
ax_wiserep.vlines(w_wiserep_rebinned[masks_wiserep['linear_bad']], 0.8*wiserep_ylim, 0.9*wiserep_ylim, colors = 'yellow', alpha = 0.5, label = 'Linear bad')
ax_wiserep.vlines(w_wiserep_rebinned[masks_wiserep['low_snr_bad']], 0.7*wiserep_ylim, 0.8*wiserep_ylim, colors = 'purple', alpha = 0.5, label = 'Low SNR bad')
ax_wiserep.vlines(w_wiserep_rebinned[masks_wiserep['invalid_bad']], 0.6*wiserep_ylim, 0.7*wiserep_ylim, colors = 'orange', alpha = 0.5, label = 'Invalid bad')

ax_wiserep.vlines(w_template[masks_template['total_bad']], -wiserep_ylim, -0.9*wiserep_ylim, colors = 'r', alpha = 0.5) 
ax_wiserep.vlines(w_template[masks_template['linear_bad']],  -0.9*wiserep_ylim, -0.8*wiserep_ylim, colors = 'yellow', alpha = 0.5)
ax_wiserep.vlines(w_template[masks_template['low_snr_bad']], -0.8*wiserep_ylim, -0.7*wiserep_ylim, colors = 'purple', alpha = 0.5)
ax_wiserep.vlines(w_template[masks_template['invalid_bad']], -0.7*wiserep_ylim, -0.6*wiserep_ylim, colors = 'orange', alpha = 0.5)
for telluric_line in telluric_lines_to_use:
    telluric_line_info = telluric_lines[telluric_line]
    wl_min = telluric_line_info['center_A'] - telluric_line_info['width_A'] / 2
    wl_max = telluric_line_info['center_A'] + telluric_line_info['width_A'] / 2
    ax_wiserep.axvspan(
        wl_min / (1 + float(redshift)),
        wl_max / (1 + float(redshift)),
        color = 'g',
        alpha = 0.3,
    )
    # ax_wiserep.axvline(wl_min / (1 + float(redshift)), color = 'g', alpha = 0.5, linestyle = '--')
    # ax_wiserep.axvline(wl_max / (1 + float(redshift)), color = 'g', alpha = 0.5, linestyle = '--')
    ax_wiserep.text(wl_min / (1 + float(redshift)), 0.9*wiserep_ylim, f'{telluric_line_info["species"]}', color = 'g', alpha = 1, fontsize = 10)
for host_line in host_lines_to_use:
    host_line_info = host_lines[host_line]
    wl_min = host_line_info['center_A'] - host_line_info['width_A'] / 2
    wl_max = host_line_info['center_A'] + host_line_info['width_A'] / 2
    ax_wiserep.axvline(wl_min, color = 'b', alpha = 0.5, linestyle = '--')
    ax_wiserep.axvline(wl_max, color = 'b', alpha = 0.5, linestyle = '--')

    # ax_wiserep.text(wl_min / (1 + float(redshift)), 0.9*wiserep_ylim, f'{host_line_info["species"]}', color = 'b', alpha = 0.3, fontsize = 10)

ax_wiserep.set_xlabel('Wavelength')

custom_lines = [Line2D([0], [0], color='r', lw=4),
                Line2D([0], [0], color='yellow', lw=4),
                Line2D([0], [0], color='purple', lw=4),
                Line2D([0], [0], color='orange', lw=4),
                Line2D([0], [0], color='g', lw=4, linestyle = '--'), 
                Line2D([0], [0], color='b', lw=4, linestyle = '--')]
ax_wiserep.legend(custom_lines, ['Total bad', 'Linear bad', 'Low SNR bad', 'Invalid bad', 'Telluric lines', "Host galaxy lines"], fontsize=10, ncols = 2, loc = 'upper left')
# Save the figure
# fig.savefig(SAVE_DIR / f'{objname}_{epoch_col}.png', dpi=300)
# plt.close(fig)  # <-- add this

# plot_flux_masks(w_wiserep_rebinned, f_wiserep_rebinned, masks_wiserep)
# plot_flux_masks(w_template, f_template, masks_template)
# plot_snr(w_wiserep_rebinned, masks_wiserep, snr_threshold=1.0)
# plot_continuum_noise_residual(w_wiserep_rebinned, f_wiserep_rebinned, continuum_wiserep_rebinned, ferr_wiserep_rebinned, residual_wiserep_rebinned, snr_threshold=1.0)
# %%
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # important for multiprocessing / non-interactive plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SAVE_DIR = Path("./specquality")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def process_one_object(objname):
    """
    Process one object and save all epoch figures.

    Returns
    -------
    objname : str
    failed_obj_history : dict
        Dictionary like {epoch_col: error_message}
    """

    failed_obj_history = {}

    try:
        print(f"Processing {objname}", flush=True)

        match_info = SNID_tbl[SNID_tbl["Objname"] == objname][0]
        redshift = match_info["Redshift"]

        template_wl_dict, template_flux_dict, template_meta = load_template_spectrum(objname)
        wiserep_wl_dict, wiserep_flux_dict, wiserep_fluxerr_dict, wiserep_meta = load_wiserep_spectrum(objname)

        num_templates = match_info["n_templates"]
        colnames_epoch = [f"epoch_{i}" for i in range(int(num_templates))]

        for epoch_col in colnames_epoch:
            try:
                obsdate_col = match_info[epoch_col]

                redshift_col = epoch_col + "_redshift"
                redshift_spec = match_info[redshift_col]

                w_template = template_wl_dict[epoch_col]
                f_template = template_flux_dict[epoch_col]

                w_wiserep = wiserep_wl_dict[obsdate_col]
                w_wiserep = w_wiserep / (1 + float(redshift_spec))

                f_wiserep = wiserep_flux_dict[obsdate_col]

                # Rebin the spectrum to the common wavelength grid
                w_wiserep_rebinned = np.arange(w_wiserep.min(), w_wiserep.max(), 5)

                f_wiserep_rebinned = rebin_spectrum_flux_conserving(
                    w_wiserep,
                    f_wiserep,
                    w_wiserep_rebinned,
                )

                (
                    ferr_wiserep_rebinned,
                    continuum_wiserep_rebinned,
                    residual_wiserep_rebinned,
                ) = estimate_noise_from_flux(
                    f_wiserep_rebinned,
                    continuum_window=51,
                    noise_window=31,
                )

                masks_wiserep = get_all_flux_masks(
                    f_wiserep_rebinned,
                    noise=ferr_wiserep_rebinned,
                    continuum=continuum_wiserep_rebinned,
                    linear_window=21,
                    linear_r2_threshold=0.9999,
                    snr_threshold=1.0,
                    peak_window=5,
                    peak_sigma_threshold=100.0,
                )

                masks_template = get_all_flux_masks(
                    f_template,
                    noise=None,
                    continuum=None,
                    linear_window=5,
                    linear_r2_threshold=0.9999,
                    snr_threshold=1.0,
                    peak_window=5,
                    peak_sigma_threshold=100.0,
                )

                fig, ax_wiserep = plt.subplots(figsize=(14, 8))
                ax_wiserep.set_title(f"{objname} {epoch_col} template vs WISeREP")

                ax_template = ax_wiserep.twinx()

                ax_wiserep.plot(
                    w_wiserep_rebinned,
                    f_wiserep_rebinned,
                    c="k",
                    alpha=1,
                    label="WISeREP",
                )

                ax_template.plot(
                    w_template,
                    f_template,
                    c="b",
                    alpha=0.5,
                    label="SNID template",
                )

                wiserep_ylim = np.nanmax(np.abs(f_wiserep_rebinned))
                template_ylim = np.nanmax(np.abs(f_template))

                # Protect against all-NaN or zero arrays
                if not np.isfinite(wiserep_ylim) or wiserep_ylim == 0:
                    wiserep_ylim = 1.0

                if not np.isfinite(template_ylim) or template_ylim == 0:
                    template_ylim = 1.0

                # Plot WISeREP masked regions
                ax_wiserep.vlines(
                    w_wiserep_rebinned[masks_wiserep["total_bad"]],
                    0.9 * wiserep_ylim,
                    wiserep_ylim,
                    colors="r",
                    alpha=0.5,
                    label="Total bad",
                )

                ax_wiserep.vlines(
                    w_wiserep_rebinned[masks_wiserep["linear_bad"]],
                    0.8 * wiserep_ylim,
                    0.9 * wiserep_ylim,
                    colors="yellow",
                    alpha=0.5,
                    label="Linear bad",
                )

                ax_wiserep.vlines(
                    w_wiserep_rebinned[masks_wiserep["low_snr_bad"]],
                    0.7 * wiserep_ylim,
                    0.8 * wiserep_ylim,
                    colors="purple",
                    alpha=0.5,
                    label="Low SNR bad",
                )

                ax_wiserep.vlines(
                    w_wiserep_rebinned[masks_wiserep["invalid_bad"]],
                    0.6 * wiserep_ylim,
                    0.7 * wiserep_ylim,
                    colors="orange",
                    alpha=0.5,
                    label="Invalid bad",
                )

                # Plot template masked regions
                ax_wiserep.vlines(
                    w_template[masks_template["total_bad"]],
                    -wiserep_ylim,
                    -0.9 * wiserep_ylim,
                    colors="r",
                    alpha=0.5,
                )

                ax_wiserep.vlines(
                    w_template[masks_template["linear_bad"]],
                    -0.9 * wiserep_ylim,
                    -0.8 * wiserep_ylim,
                    colors="yellow",
                    alpha=0.5,
                )

                ax_wiserep.vlines(
                    w_template[masks_template["low_snr_bad"]],
                    -0.8 * wiserep_ylim,
                    -0.7 * wiserep_ylim,
                    colors="purple",
                    alpha=0.5,
                )

                ax_wiserep.vlines(
                    w_template[masks_template["invalid_bad"]],
                    -0.7 * wiserep_ylim,
                    -0.6 * wiserep_ylim,
                    colors="orange",
                    alpha=0.5,
                )

                for telluric_line in telluric_lines_to_use:
                    telluric_line_info = telluric_lines[telluric_line]

                    wl_min = (
                        telluric_line_info["center_A"]
                        - telluric_line_info["width_A"] / 2
                    )
                    wl_max = (
                        telluric_line_info["center_A"]
                        + telluric_line_info["width_A"] / 2
                    )
                    ax_wiserep.axvspan(
                        wl_min / (1 + float(redshift)),
                        wl_max / (1 + float(redshift)),
                        color = 'g',
                        alpha = 0.3,
                    )

                    ax_wiserep.text(
                        wl_min / (1 + float(redshift)),
                        0.9 * wiserep_ylim,
                        f'{telluric_line_info["species"]}',
                        color="g",
                        alpha=1,
                        fontsize=10,
                    )

                for host_line in host_lines_to_use:
                    host_line_info = host_lines[host_line]
                    wl_min = host_line_info['center_A'] - host_line_info['width_A'] / 2
                    wl_max = host_line_info['center_A'] + host_line_info['width_A'] / 2
                    ax_wiserep.axvline(wl_min, color = 'b', alpha = 0.2, linestyle = '--')
                    ax_wiserep.axvline(wl_max, color = 'b', alpha = 0.2, linestyle = '--')
                    # ax_wiserep.text(wl_min / (1 + float(redshift)), 0.9*wiserep_ylim, f'{host_line_info["species"]}', color = 'b', alpha = 0.3, fontsize = 10)

                ax_wiserep.set_xlabel("Wavelength")

                custom_lines = [
                    Line2D([0], [0], color="r", lw=4),
                    Line2D([0], [0], color="yellow", lw=4),
                    Line2D([0], [0], color="purple", lw=4),
                    Line2D([0], [0], color="orange", lw=4),
                    Line2D([0], [0], color="g", lw=4, linestyle="--"),
                    Line2D([0], [0], color="b", lw=4, linestyle="--"),
                ]

                ax_wiserep.legend(
                    custom_lines,
                    [
                        "Total bad",
                        "Linear bad",
                        "Low SNR bad",
                        "Invalid bad",
                        "Telluric lines",
                        "Host galaxy lines",
                    ],
                    fontsize=10,
                    ncols=2,
                    loc="upper left",
                )

                fig.savefig(SAVE_DIR / f"{objname}_{epoch_col}.png", dpi=300)
                plt.close(fig)

            except Exception:
                failed_obj_history[epoch_col] = traceback.format_exc()

    except Exception:
        failed_obj_history["object_level"] = traceback.format_exc()

    return objname, failed_obj_history
# %%
objnames = list(SNID_tbl["Objname"])

failed_history = {}

# Use fewer than all CPUs if memory is heavy.
max_workers = min(64, os.cpu_count())

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = {
        executor.submit(process_one_object, objname): objname
        for objname in objnames
    }

    for future in as_completed(futures):
        objname = futures[future]

        try:
            objname, failed_obj_history = future.result()
            failed_history[objname] = failed_obj_history

            if failed_obj_history:
                print(f"[FAILED] {objname}: {list(failed_obj_history.keys())}")
            else:
                print(f"[DONE] {objname}")

        except Exception:
            failed_history[objname] = {
                "multiprocessing_level": traceback.format_exc()
            }
            print(f"[CRASHED] {objname}")
# %%
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1wwbBlAvOUamYM-l_F-N6xd4UYPEcKhntzLETNtP-m1o/edit?gid=0#gid=0"
import gspread
from astropy.table import Table

gc = gspread.service_account(
    filename="/home/hhchoi1022/code/SNID-SAGE/googlesheet.json"
)

sh = gc.open_by_url(SPREADSHEET_URL)
ws = sh.get_worksheet(0)

rows = ws.get_all_records(numericise_ignore=["all"])

SNID_tbl = Table(rows)
#%% FOR SINGLE OBJECT MASKING
idx = 192
%matplotlib inline
objname = SNID_tbl['Objname'][idx]
template_wl_dict, template_flux_dict, template_meta = load_template_spectrum(objname)
wiserep_wl_dict, wiserep_flux_dict, wiserep_fluxerr_dict, wiserep_meta = load_wiserep_spectrum(objname)
match_info = SNID_tbl[SNID_tbl['Objname'] == objname][0]
redshift = match_info['Redshift']
num_templates = match_info['n_templates']
num_matched = match_info['n_matched_template']
colnames_epoch = [f'epoch_{i}' for i in range(int(num_templates))]
telluric_OII_col = match_info['Telluric_OII']
telluric_H2O_col = match_info['Telluric_H2O']
host_col = match_info['Emission']
if telluric_OII_col == 'all':
    colnames_epoch_telluric_OII_mask = colnames_epoch
else:
    colnames_epoch_telluric_OII_mask = [f'epoch_{i}' for i in telluric_OII_col.split(',')]
if telluric_H2O_col == 'all':
    colnames_epoch_telluric_H2O_mask = colnames_epoch
else:
    colnames_epoch_telluric_H2O_mask = [f'epoch_{i}' for i in telluric_H2O_col.split(',')]
if host_col == 'all':
    colnames_epoch_host_mask = colnames_epoch
else:
    colnames_epoch_host_mask = [f'epoch_{i}' for i in host_col.split(',')]
print('Object name:', objname)
print('Epochs:', colnames_epoch)
print('Telluric mask (OII):', colnames_epoch_telluric_OII_mask)
print('Telluric mask (H2O):', colnames_epoch_telluric_H2O_mask)
print('Host mask:', colnames_epoch_host_mask)

final_dict = {}
for epoch_col in colnames_epoch:
    final_dict[epoch_col] = {}
    obsdate_col = match_info[epoch_col]
    redshift_col = epoch_col + '_redshift'
    redshift_spec = match_info[redshift_col]
    w_template = template_wl_dict[epoch_col]
    f_template = template_flux_dict[epoch_col]
    if obsdate_col not in wiserep_wl_dict:
        continue
    w_wiserep = wiserep_wl_dict[obsdate_col]
    w_wiserep = w_wiserep / (1 + float(redshift_spec))
    f_wiserep = wiserep_flux_dict[obsdate_col]
    # Rebin the spectrum to the common wavelength grid
    w_wiserep_rebinned = np.arange(w_wiserep.min(), w_wiserep.max(), 10)
    f_wiserep_rebinned = rebin_spectrum_flux_conserving(w_wiserep, f_wiserep, w_wiserep_rebinned)

    # Mask telluric lines (OII)
    telluric_mask_OII = [True for _ in range(len(w_wiserep_rebinned))]
    if epoch_col in colnames_epoch_telluric_OII_mask:     
        telluric_lines_to_mask = ['O2_A_band', 'O2_B_band']
        telluric_mask_OII = mask_lines(w_wiserep_rebinned, telluric_lines, lines_to_use = telluric_lines_to_mask, redshift = float(redshift))
        f_wiserep_rebinned_corrected = interpolate_masked_regions(w_wiserep_rebinned, f_wiserep_rebinned, telluric_mask_OII)
    else:
        f_wiserep_rebinned_corrected = f_wiserep_rebinned
    
    # Mask telluric lines (H2O)
    telluric_mask_H2O = [True for _ in range(len(w_wiserep_rebinned))]
    if epoch_col in colnames_epoch_telluric_H2O_mask:     
        telluric_lines_to_mask = ['H2O_5900_band'] #'H2O_8200_band'
        telluric_mask_H2O = mask_lines(w_wiserep_rebinned, telluric_lines, lines_to_use = telluric_lines_to_mask, redshift = float(redshift))
        f_wiserep_rebinned_corrected = interpolate_masked_regions(w_wiserep_rebinned, f_wiserep_rebinned_corrected, telluric_mask_H2O)
    else:
        f_wiserep_rebinned_corrected = f_wiserep_rebinned_corrected
    
    # Mask host galaxy lines
    if epoch_col in colnames_epoch_host_mask:
        host_lines_to_mask = ['Halpha_6563', 'NII_6583', 'SII_6717']#, 'OIII_4959', 'OII_3727_doublet', 'Hbeta_4861', 'OIII_5007', 'SII_6731', 'SII_6717']
        host_mask = mask_lines(w_wiserep_rebinned, host_lines, host_lines_to_mask, redshift = 0)
        f_wiserep_rebinned_corrected = interpolate_masked_regions(w_wiserep_rebinned, f_wiserep_rebinned_corrected, host_mask)
    else:
        f_wiserep_rebinned_corrected = f_wiserep_rebinned_corrected
    
    ferr_wiserep_rebinned, continuum_wiserep_rebinned, residual_wiserep_rebinned = estimate_noise_from_flux(f_wiserep_rebinned, 
        continuum_window=51, 
        noise_window=31)

    if epoch_col in colnames_epoch_telluric_OII_mask:
        ferr_wiserep_rebinned_corrected = interpolate_masked_regions(w_wiserep_rebinned, ferr_wiserep_rebinned, telluric_mask_OII)
    else:
        ferr_wiserep_rebinned_corrected = ferr_wiserep_rebinned

    if epoch_col in colnames_epoch_telluric_H2O_mask:
        ferr_wiserep_rebinned_corrected = interpolate_masked_regions(w_wiserep_rebinned, ferr_wiserep_rebinned, telluric_mask_H2O)
    else:
        ferr_wiserep_rebinned_corrected = ferr_wiserep_rebinned

    if epoch_col in colnames_epoch_host_mask:
        ferr_wiserep_rebinned_corrected = interpolate_masked_regions(w_wiserep_rebinned, ferr_wiserep_rebinned_corrected, host_mask)
    else:
        ferr_wiserep_rebinned_corrected = ferr_wiserep_rebinned_corrected
        
    masks_wiserep = get_all_flux_masks(
        f_wiserep_rebinned_corrected,
        noise=ferr_wiserep_rebinned,
        continuum=continuum_wiserep_rebinned,
        linear_window=21,
        linear_r2_threshold=0.9999,
        snr_threshold=1.0,
        peak_window=5,
        peak_sigma_threshold=100.0,
    )
    masks_wiserep['telluric_bad_OII'] = telluric_mask_OII
    masks_wiserep['telluric_bad_H2O'] = telluric_mask_H2O

    masks_template = get_all_flux_masks(
        f_template,
        noise=None,
        continuum=None,
        linear_window=5,
        linear_r2_threshold=0.9999,
        snr_threshold=1.0,
        peak_window=5,
        peak_sigma_threshold=100.0,
    )

    fig, ax_wiserep = plt.subplots(figsize = (16, 10))
    ax_wiserep.set_title(f'{objname} {epoch_col} template vs WISeREP')
    ax_template = ax_wiserep.twinx()
    line_wiserep, = ax_wiserep.plot(
        w_wiserep_rebinned, f_wiserep_rebinned, c="k", alpha=1, label="WISeREP"
    )
    if epoch_col in colnames_epoch_telluric_OII_mask:     
        line_wiserep_corrected, = ax_wiserep.plot(
            w_wiserep_rebinned, f_wiserep_rebinned_corrected, c="r", alpha=1, label="WISeREP corrected"
        )
    if epoch_col in colnames_epoch_telluric_H2O_mask:     
        line_wiserep_corrected, = ax_wiserep.plot(
            w_wiserep_rebinned, f_wiserep_rebinned_corrected, c="r", alpha=1, label="WISeREP corrected"
        )
    if epoch_col in colnames_epoch_host_mask:
        line_wiserep_corrected, = ax_wiserep.plot(
            w_wiserep_rebinned, f_wiserep_rebinned_corrected, c="g", alpha=1, label="WISeREP corrected"
        )
    line_template, = ax_template.plot(
        w_template, f_template, c="b", alpha=0.2, label="SNID template"
    )

    wiserep_ylim = np.nanmax(np.abs(f_wiserep_rebinned))
    template_ylim = np.nanmax(np.abs(f_template))

    # Plot masked region
    ax_wiserep.vlines(w_wiserep_rebinned[masks_wiserep['total_bad']], 0.9*wiserep_ylim, wiserep_ylim, colors = 'r', alpha = 0.5, label = 'Total bad')
    ax_wiserep.vlines(w_wiserep_rebinned[masks_wiserep['linear_bad']], 0.8*wiserep_ylim, 0.9*wiserep_ylim, colors = 'yellow', alpha = 0.5, label = 'Linear bad')
    ax_wiserep.vlines(w_wiserep_rebinned[masks_wiserep['low_snr_bad']], 0.7*wiserep_ylim, 0.8*wiserep_ylim, colors = 'purple', alpha = 0.5, label = 'Low SNR bad')
    ax_wiserep.vlines(w_wiserep_rebinned[masks_wiserep['invalid_bad']], 0.6*wiserep_ylim, 0.7*wiserep_ylim, colors = 'orange', alpha = 0.5, label = 'Invalid bad')

    ax_wiserep.vlines(w_template[masks_template['total_bad']], -0.5*wiserep_ylim, -0.4*wiserep_ylim, colors = 'r', alpha = 0.5) 
    ax_wiserep.vlines(w_template[masks_template['linear_bad']],  -0.4*wiserep_ylim, -0.3*wiserep_ylim, colors = 'yellow', alpha = 0.5)
    ax_wiserep.vlines(w_template[masks_template['low_snr_bad']], -0.3*wiserep_ylim, -0.2*wiserep_ylim, colors = 'purple', alpha = 0.5)
    ax_wiserep.vlines(w_template[masks_template['invalid_bad']], -0.2*wiserep_ylim, -0.1*wiserep_ylim, colors = 'orange', alpha = 0.5)
    for telluric_line in telluric_lines_to_use:
        telluric_line_info = telluric_lines[telluric_line]
        wl_min = telluric_line_info['center_A'] - telluric_line_info['width_A'] / 2
        wl_max = telluric_line_info['center_A'] + telluric_line_info['width_A'] / 2
        ax_wiserep.axvspan(
            wl_min / (1 + float(redshift)),
            wl_max / (1 + float(redshift)),
            color = 'g',
            alpha = 0.05,
        )
        # ax_wiserep.axvline(wl_min / (1 + float(redshift)), color = 'g', alpha = 0.5, linestyle = '--')
        # ax_wiserep.axvline(wl_max / (1 + float(redshift)), color = 'g', alpha = 0.5, linestyle = '--')
        ax_wiserep.text(wl_min / (1 + float(redshift)), 0.9*wiserep_ylim, f'{telluric_line_info["species"]}', color = 'g', alpha = 1, fontsize = 10)
    for host_line in host_lines_to_use:
        host_line_info = host_lines[host_line]
        wl_min = host_line_info['center_A'] - host_line_info['width_A'] / 2
        wl_max = host_line_info['center_A'] + host_line_info['width_A'] / 2
        ax_wiserep.axvline(wl_min, color = 'b', alpha = 0.1, linestyle = '--')
        ax_wiserep.axvline(wl_max, color = 'b', alpha = 0.1, linestyle = '--')

        # ax_wiserep.text(wl_min / (1 + float(redshift)), 0.9*wiserep_ylim, f'{host_line_info["species"]}', color = 'b', alpha = 0.3, fontsize = 10)

    ax_wiserep.set_xlabel('Wavelength')

    custom_lines = [Line2D([0], [0], color='r', lw=4),
                    Line2D([0], [0], color='yellow', lw=4),
                    Line2D([0], [0], color='purple', lw=4),
                    Line2D([0], [0], color='orange', lw=4),
                    Line2D([0], [0], color='g', lw=4, linestyle = '--'), 
                    Line2D([0], [0], color='b', lw=4, linestyle = '--')]
    ax_wiserep.legend(custom_lines, ['Total bad', 'Linear bad', 'Low SNR bad', 'Invalid bad', 'Telluric lines', "Host galaxy lines"], fontsize=10, ncols = 2, loc = 'upper left')
    final_dict[epoch_col]['w_wiserep_original'] = w_wiserep
    final_dict[epoch_col]['f_wiserep_original'] = f_wiserep
    final_dict[epoch_col]['w_wiserep_rebinned'] = w_wiserep_rebinned
    final_dict[epoch_col]['f_wiserep_rebinned'] = f_wiserep_rebinned
    final_dict[epoch_col]['ferr_wiserep_rebinned'] = ferr_wiserep_rebinned
    final_dict[epoch_col]['f_wiserep_rebinned_corrected'] = f_wiserep_rebinned_corrected
    final_dict[epoch_col]['ferr_wiserep_rebinned_corrected'] = ferr_wiserep_rebinned_corrected
    final_dict[epoch_col]['masks_wiserep'] = masks_wiserep
    final_dict[epoch_col]['masks_wiserep']['telluric_bad_OII'] = telluric_mask_OII
    final_dict[epoch_col]['masks_wiserep']['telluric_bad_H2O'] = telluric_mask_H2O
    final_dict[epoch_col]['masks_wiserep']['host_bad'] = host_mask
#%% SAVE
import json
import numpy as np

def make_json_serializable(obj):
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
TEMPLATE_DIR = Path('./template_result')
for key, value in final_dict.items():
    file_name = f'{objname}_{key}.json'
    with open(TEMPLATE_DIR / file_name, 'w') as f:
        json.dump(make_json_serializable(value), f, indent=4)
# %% FOR SINGLE EPOCH MASKING
epoch_col = 'epoch_4'
final_dict[epoch_col] = {}
obsdate_col = match_info[epoch_col]
redshift_col = epoch_col + '_redshift'
redshift_spec = match_info[redshift_col]
w_template = template_wl_dict[epoch_col]
f_template = template_flux_dict[epoch_col]
w_wiserep = wiserep_wl_dict[obsdate_col]
w_wiserep = w_wiserep / (1 + float(redshift_spec))
f_wiserep = wiserep_flux_dict[obsdate_col]
# Rebin the spectrum to the common wavelength grid
w_wiserep_rebinned = np.arange(w_wiserep.min(), w_wiserep.max(), 10)
f_wiserep_rebinned = rebin_spectrum_flux_conserving(w_wiserep, f_wiserep, w_wiserep_rebinned)

# Mask telluric lines (OII)
telluric_mask_OII = [True for _ in range(len(w_wiserep_rebinned))]
if epoch_col in colnames_epoch_telluric_OII_mask:     
    telluric_lines_to_mask = ['O2_A_band', 'O2_B_band']
    telluric_mask_OII = mask_lines(w_wiserep_rebinned, telluric_lines, lines_to_use = telluric_lines_to_mask, redshift = float(redshift))
    f_wiserep_rebinned_corrected = interpolate_masked_regions(w_wiserep_rebinned, f_wiserep_rebinned, telluric_mask_OII)
else:
    f_wiserep_rebinned_corrected = f_wiserep_rebinned

# Mask manual lines
manual_lines = {
    # Very strong
    "manual_1": {
        "center_A": 5902,
        "width_A":50.0,
        "species": "H I",
        "strength": "very strong",
    },
    "manual_2": {
        "center_A": 6730,
        "width_A": 30.0,
        "species": "H I",
        "strength": "very strong",
    }
    }

# Mask telluric lines (H2O)
telluric_mask_H2O = [True for _ in range(len(w_wiserep_rebinned))]
if epoch_col in colnames_epoch_telluric_H2O_mask:     
    telluric_lines_to_mask = ['H2O_8200_band', 'H2O_5900_band']
    telluric_mask_H2O = mask_lines(w_wiserep_rebinned, telluric_lines, lines_to_use = telluric_lines_to_mask, redshift = float(redshift))
    f_wiserep_rebinned_corrected = interpolate_masked_regions(w_wiserep_rebinned, f_wiserep_rebinned_corrected, telluric_mask_H2O)
else:
    f_wiserep_rebinned_corrected = f_wiserep_rebinned_corrected

# Mask host galaxy lines
if epoch_col in colnames_epoch_host_mask:
    #'OIII_4959', 'OII_3727_doublet', 'Hbeta_4861', 'OIII_5007', 'SII_6731', 'SII_6717']
    host_lines_to_mask = ['SII_6717']# 'NII_6583', 'Hbeta_4861', 'OII_3727_doublet', 'Hbeta_4861', 'OIII_5007', 'NII_6583', 'OIII_4959', 'SII_6717', 'Halpha_6563']
    host_mask = mask_lines(w_wiserep_rebinned, host_lines, host_lines_to_mask, redshift = 0)
    f_wiserep_rebinned_corrected = interpolate_masked_regions(w_wiserep_rebinned, f_wiserep_rebinned_corrected, host_mask)
else:
    f_wiserep_rebinned_corrected = f_wiserep_rebinned_corrected

# Mask manual lines
manual_lines_to_use = ['manual_1']#'manual_1', 'manual_2']
manual_mask = mask_lines(w_wiserep_rebinned, manual_lines, lines_to_use = manual_lines_to_use, redshift = 0)
f_wiserep_rebinned_corrected = interpolate_masked_regions(w_wiserep_rebinned, f_wiserep_rebinned_corrected, manual_mask)

ferr_wiserep_rebinned, continuum_wiserep_rebinned, residual_wiserep_rebinned = estimate_noise_from_flux(f_wiserep_rebinned, 
    continuum_window=51, 
    noise_window=31)

if epoch_col in colnames_epoch_telluric_OII_mask:
    ferr_wiserep_rebinned_corrected = interpolate_masked_regions(w_wiserep_rebinned, ferr_wiserep_rebinned, telluric_mask_OII)
else:
    ferr_wiserep_rebinned_corrected = ferr_wiserep_rebinned

if epoch_col in colnames_epoch_telluric_H2O_mask:
    ferr_wiserep_rebinned_corrected = interpolate_masked_regions(w_wiserep_rebinned, ferr_wiserep_rebinned, telluric_mask_H2O)
else:
    ferr_wiserep_rebinned_corrected = ferr_wiserep_rebinned

if epoch_col in colnames_epoch_host_mask:
    ferr_wiserep_rebinned_corrected = interpolate_masked_regions(w_wiserep_rebinned, ferr_wiserep_rebinned_corrected, host_mask)
else:
    ferr_wiserep_rebinned_corrected = ferr_wiserep_rebinned_corrected

ferr_wiserep_rebinned_corrected = interpolate_masked_regions(w_wiserep_rebinned, ferr_wiserep_rebinned_corrected, manual_mask)
    
masks_wiserep = get_all_flux_masks(
    f_wiserep_rebinned_corrected,
    noise=ferr_wiserep_rebinned,
    continuum=continuum_wiserep_rebinned,
    linear_window=21,
    linear_r2_threshold=0.9999,
    snr_threshold=1.0,
    peak_window=5,
    peak_sigma_threshold=100.0,
)
masks_wiserep['telluric_bad_OII'] = telluric_mask_OII
masks_wiserep['telluric_bad_H2O'] = telluric_mask_H2O

masks_template = get_all_flux_masks(
    f_template,
    noise=None,
    continuum=None,
    linear_window=5,
    linear_r2_threshold=0.9999,
    snr_threshold=1.0,
    peak_window=5,
    peak_sigma_threshold=100.0,
)

fig, ax_wiserep = plt.subplots(figsize = (16, 10))
ax_wiserep.set_title(f'{objname} {epoch_col} template vs WISeREP')
ax_template = ax_wiserep.twinx()
line_wiserep, = ax_wiserep.plot(
    w_wiserep_rebinned, f_wiserep_rebinned, c="k", alpha=1, label="WISeREP"
)
if epoch_col in colnames_epoch_telluric_OII_mask:     
    line_wiserep_corrected, = ax_wiserep.plot(
        w_wiserep_rebinned, f_wiserep_rebinned_corrected, c="r", alpha=1, label="WISeREP corrected"
    )
if epoch_col in colnames_epoch_telluric_H2O_mask:     
    line_wiserep_corrected, = ax_wiserep.plot(
        w_wiserep_rebinned, f_wiserep_rebinned_corrected, c="r", alpha=1, label="WISeREP corrected"
    )
line_wiserep_corrected, = ax_wiserep.plot(
    w_wiserep_rebinned, f_wiserep_rebinned_corrected, c="y", alpha=1, label="WISeREP corrected"
)
if epoch_col in colnames_epoch_host_mask:
    line_wiserep_corrected, = ax_wiserep.plot(
        w_wiserep_rebinned, f_wiserep_rebinned_corrected, c="g", alpha=1, label="WISeREP corrected"
    )
line_template, = ax_template.plot(
    w_template, f_template, c="b", alpha=0.2, label="SNID template"
)

wiserep_ylim = np.nanmax(np.abs(f_wiserep_rebinned))
template_ylim = np.nanmax(np.abs(f_template))

# Plot masked region
ax_wiserep.vlines(w_wiserep_rebinned[masks_wiserep['total_bad']], 0.9*wiserep_ylim, wiserep_ylim, colors = 'r', alpha = 0.5, label = 'Total bad')
ax_wiserep.vlines(w_wiserep_rebinned[masks_wiserep['linear_bad']], 0.8*wiserep_ylim, 0.9*wiserep_ylim, colors = 'yellow', alpha = 0.5, label = 'Linear bad')
ax_wiserep.vlines(w_wiserep_rebinned[masks_wiserep['low_snr_bad']], 0.7*wiserep_ylim, 0.8*wiserep_ylim, colors = 'purple', alpha = 0.5, label = 'Low SNR bad')
ax_wiserep.vlines(w_wiserep_rebinned[masks_wiserep['invalid_bad']], 0.6*wiserep_ylim, 0.7*wiserep_ylim, colors = 'orange', alpha = 0.5, label = 'Invalid bad')

ax_wiserep.vlines(w_template[masks_template['total_bad']], -0.5*wiserep_ylim, -0.4*wiserep_ylim, colors = 'r', alpha = 0.5) 
ax_wiserep.vlines(w_template[masks_template['linear_bad']],  -0.4*wiserep_ylim, -0.3*wiserep_ylim, colors = 'yellow', alpha = 0.5)
ax_wiserep.vlines(w_template[masks_template['low_snr_bad']], -0.3*wiserep_ylim, -0.2*wiserep_ylim, colors = 'purple', alpha = 0.5)
ax_wiserep.vlines(w_template[masks_template['invalid_bad']], -0.2*wiserep_ylim, -0.1*wiserep_ylim, colors = 'orange', alpha = 0.5)
for telluric_line in telluric_lines_to_use:
    telluric_line_info = telluric_lines[telluric_line]
    wl_min = telluric_line_info['center_A'] - telluric_line_info['width_A'] / 2
    wl_max = telluric_line_info['center_A'] + telluric_line_info['width_A'] / 2
    ax_wiserep.axvspan(
        wl_min / (1 + float(redshift)),
        wl_max / (1 + float(redshift)),
        color = 'g',
        alpha = 0.05,
    )
    # ax_wiserep.axvline(wl_min / (1 + float(redshift)), color = 'g', alpha = 0.5, linestyle = '--')
    # ax_wiserep.axvline(wl_max / (1 + float(redshift)), color = 'g', alpha = 0.5, linestyle = '--')
    ax_wiserep.text(wl_min / (1 + float(redshift)), 0.9*wiserep_ylim, f'{telluric_line_info["species"]}', color = 'g', alpha = 1, fontsize = 10)
for host_line in host_lines_to_use:
    host_line_info = host_lines[host_line]
    wl_min = host_line_info['center_A'] - host_line_info['width_A'] / 2
    wl_max = host_line_info['center_A'] + host_line_info['width_A'] / 2
    ax_wiserep.axvline(wl_min, color = 'b', alpha = 0.1, linestyle = '--')
    ax_wiserep.axvline(wl_max, color = 'b', alpha = 0.1, linestyle = '--')

    # ax_wiserep.text(wl_min / (1 + float(redshift)), 0.9*wiserep_ylim, f'{host_line_info["species"]}', color = 'b', alpha = 0.3, fontsize = 10)

ax_wiserep.set_xlabel('Wavelength')

custom_lines = [Line2D([0], [0], color='r', lw=4),
                Line2D([0], [0], color='yellow', lw=4),
                Line2D([0], [0], color='purple', lw=4),
                Line2D([0], [0], color='orange', lw=4),
                Line2D([0], [0], color='g', lw=4, linestyle = '--'), 
                Line2D([0], [0], color='b', lw=4, linestyle = '--')]
ax_wiserep.legend(custom_lines, ['Total bad', 'Linear bad', 'Low SNR bad', 'Invalid bad', 'Telluric lines', "Host galaxy lines"], fontsize=10, ncols = 2, loc = 'upper left')
final_dict[epoch_col]['w_wiserep_original'] = w_wiserep
final_dict[epoch_col]['f_wiserep_original'] = f_wiserep
final_dict[epoch_col]['w_wiserep_rebinned'] = w_wiserep_rebinned
final_dict[epoch_col]['f_wiserep_rebinned'] = f_wiserep_rebinned
final_dict[epoch_col]['ferr_wiserep_rebinned'] = ferr_wiserep_rebinned
final_dict[epoch_col]['f_wiserep_rebinned_corrected'] = f_wiserep_rebinned_corrected
final_dict[epoch_col]['masks_wiserep'] = masks_wiserep
final_dict[epoch_col]['masks_wiserep']['telluric_bad_OII'] = telluric_mask_OII
final_dict[epoch_col]['masks_wiserep']['telluric_bad_H2O'] = telluric_mask_H2O
final_dict[epoch_col]['masks_wiserep']['host_bad'] = host_mask
# %%
