

#%%
import logging
import warnings
import numpy as np
from pathlib import Path
from typing import Any
import json
import h5py
import csv
import hashlib
from datetime import datetime
import json
from collections import Counter
from pathlib import Path
from typing import Any
from astropy.time import Time
from gspread.utils import rowcol_to_a1

TEMPLATES_DIR = Path("/home/hhchoi1022/code/SNID-SAGE/templates")
INDEX_FILE = TEMPLATES_DIR / "template_index.json"
logger = logging.getLogger(__name__)
REQUIRED_SPECTRUM_COLUMNS = {"wave", "flux", "err", "skyback"}

def load_template_object_names(index_path: Path) -> list[str]:
    if not index_path.exists():
        raise FileNotFoundError(f"Template index not found: {index_path}")

    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)

    templates: dict[str, Any] = index.get("templates", {})
    if not isinstance(templates, dict) or len(templates) == 0:
        raise ValueError("No templates found in template_index.json")

    # Prefer explicit object_name when present; otherwise use template id.
    names: list[str] = []
    for template_id, meta in templates.items():
        object_name = None
        if isinstance(meta, dict):
            object_name = meta.get("object_name")
        name = str(object_name).strip() if object_name else str(template_id).strip()
        if name:
            names.append(name)

    # Keep order, deduplicate.
    seen: set[str] = set()
    unique_names: list[str] = []
    for n in names:
        key = n.lower()
        if key not in seen:
            seen.add(key)
            unique_names.append(n)
    return unique_names

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


def _load_ascii_wavelength_flux(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible (wavelength, flux) only."""
    w, f, _, _ = _load_ascii_wavelength_flux_fluxerr(file_path)
    return w, f

def load_template_spectrum(
    template_id: str,
    index_path: Path = INDEX_FILE,
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
            epochs.append(("single", age_val, flux))

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
    wiserep_dir = OUTPUT_ROOT / wiserep_id
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


def apply_wavelength_mask(w, f, ranges=None):
    w = np.asarray(w, dtype=float)
    f = np.asarray(f, dtype=float)

    if ranges is None or len(ranges) == 0:
        return w, f

    keep = np.ones_like(w, dtype=bool)
    for a, b in ranges:
        if b < a:
            raise ValueError(f"mask ({a}, {b}) has b < a")
        keep &= ~((w >= a) & (w <= b))
    return w[keep], f[keep]


def log_rebin_sage_like(wave, flux, num_points=1024, min_wave=2500.0, max_wave=10000.0):
    """
    Flux-conserving log rebin similar to SNID-SAGE.
    Input wave should be REST-FRAME, ascending, in Angstrom.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    good = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    wave = wave[good]
    flux = flux[good]

    if wave.size < 2:
        raise ValueError("Need at least 2 valid points for log rebinning")

    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]

    # remove duplicate wavelengths, keep first
    wave, uniq_idx = np.unique(wave, return_index=True)
    flux = flux[uniq_idx]

    if wave.size < 2:
        raise ValueError("Need at least 2 unique wavelength points")

    nlog = int(num_points)
    w0 = float(min_wave)
    w1 = float(max_wave)
    dwlog = np.log(w1 / w0) / nlog

    log_wave = w0 * np.exp((np.arange(nlog) + 0.5) * dwlog)
    fdest = np.zeros(nlog, dtype=float)

    # source pixel edges in linear wavelength
    s = np.empty(wave.size + 1, dtype=float)
    s[1:-1] = 0.5 * (wave[:-1] + wave[1:])
    s[0] = 1.5 * wave[0] - 0.5 * wave[1]
    s[-1] = 1.5 * wave[-1] - 0.5 * wave[-2]

    # keep edges positive for log()
    tiny = np.finfo(float).tiny
    s = np.maximum(s, tiny)

    # map edges to 1-indexed log-bin coordinates
    slog = np.log(s / w0) / dwlog + 1.0

    for l in range(wave.size):
        s0log = slog[l]
        s1log = slog[l + 1]
        dlam = s[l + 1] - s[l]

        if not (np.isfinite(s0log) and np.isfinite(s1log)):
            continue
        if dlam <= 0:
            continue
        if s1log < s0log:
            s0log, s1log = s1log, s0log

        width_log = s1log - s0log
        if width_log <= 0:
            continue

        i0 = max(1, int(np.floor(s0log)))
        i1 = min(nlog, int(np.floor(s1log)))

        if i1 < 1 or i0 > nlog:
            continue

        for i in range(i0, i1 + 1):
            alen = min(s1log, i + 1.0) - max(s0log, float(i))
            if alen <= 0:
                continue
            frac = alen / width_log
            fdest[i - 1] += flux[l] * frac * dlam

    edges = w0 * np.exp((np.arange(nlog + 1) - 0.5) * dwlog)
    binw = np.diff(edges)
    fdest = fdest / binw

    return log_wave, fdest


def fit_continuum_spline_sage_like(flux, knotnum=13, izoff=0, edge_guard_frac=0.02):
    """
    SNID-SAGE-like continuum fit on the fixed log grid.
    Returns:
        flat = flux/cont - 1
        cont
    """
    flux = np.asarray(flux, dtype=float)
    n = flux.size

    if n < 10 or knotnum < 3:
        return np.zeros_like(flux), np.ones_like(flux)

    # 1) chop off up to one positive pixel at each edge, matching repo logic
    l1 = 0
    nuked = 0
    while l1 < n - 1 and (flux[l1] <= 0 or nuked < 1):
        if flux[l1] > 0:
            nuked += 1
        l1 += 1

    l2 = n - 1
    nuked = 0
    while l2 > 1 and (flux[l2] <= 0 or nuked < 1):
        if flux[l2] > 0:
            nuked += 1
        l2 -= 1

    if (l2 - l1) < 3 * knotnum:
        return np.zeros_like(flux), np.ones_like(flux)

    # 2) edge guard for knot placement
    usable = int(max(0, (l2 - l1 + 1)))
    try:
        eg = float(edge_guard_frac)
    except Exception:
        eg = 0.0
    eg = max(0.0, min(0.2, eg))

    guard = int(round(eg * usable))
    guard = max(0, min(guard, max(0, usable // 3)))

    l1_knot = int(l1 + guard)
    l2_knot = int(l2 - guard)

    if (l2_knot - l1_knot) < 3 * knotnum:
        l1_knot, l2_knot = int(l1), int(l2)

    kwidth = max(1, n // knotnum)
    istart = ((izoff % kwidth) - kwidth) if izoff > 0 else 0

    xknot = []
    yknot = []

    nave = 0.0
    sum_x = 0.0
    sum_flux = 0.0

    for i in range(n):
        if l1_knot < i < l2_knot and flux[i] > 0:
            nave += 1.0
            sum_x += (i - 0.5)
            sum_flux += flux[i]

        if ((i - istart) % kwidth) == 0 and nave > 0:
            xknot.append(sum_x / nave)
            yknot.append(np.log10(sum_flux / nave))
            nave = 0.0
            sum_x = 0.0
            sum_flux = 0.0

    nk = len(xknot)
    if nk < 3:
        return np.zeros_like(flux), np.ones_like(flux)

    xknot = np.asarray(xknot, dtype=float)
    yknot = np.asarray(yknot, dtype=float)

    # 3) natural cubic spline second derivatives
    h = np.diff(xknot)
    rhs = 6.0 * ((yknot[2:] - yknot[1:-1]) / h[1:] - (yknot[1:-1] - yknot[:-2]) / h[:-1])

    y2 = np.zeros(nk, dtype=float)
    if rhs.size > 0:
        A = 2.0 * (h[:-1] + h[1:])
        C = h[1:]

        u = np.empty_like(A)
        z = np.empty_like(rhs)

        u[0] = A[0]
        z[0] = rhs[0]

        for i in range(1, rhs.size):
            li = C[i - 1] / u[i - 1]
            u[i] = A[i] - li * C[i - 1]
            z[i] = rhs[i] - li * z[i - 1]

        y2[-2] = z[-1] / u[-1]
        for i in range(rhs.size - 2, -1, -1):
            y2[i + 1] = (z[i] - C[i] * y2[i + 2]) / u[i]

    # 4) evaluate spline continuum
    cont = np.empty(n, dtype=float)
    for j in range(n):
        xp = j - 0.5
        idx=np.clip(np.searchsorted(xknot, xp) - 1, 0, nk - 2)

        hi = xknot[idx + 1] - xknot[idx]
        a = (xknot[idx + 1] - xp) / hi
        b = (xp - xknot[idx]) / hi

        logc = (
            a * yknot[idx]
            + b * yknot[idx + 1]
            + ((a**3 - a) * y2[idx] + (b**3 - b) * y2[idx + 1]) * (hi**2) / 6.0
        )

        with np.errstate(over="ignore", invalid="ignore"):
            cont[j] = float(np.power(10.0, logc))

    # 5) flat = flux/cont - 1
    flat = np.zeros_like(flux)
    mask = (flux > 0) & np.isfinite(flux) & np.isfinite(cont) & (cont > 0)
    flat[mask] = flux[mask] / cont[mask] - 1.0

    return flat, cont


def fit_continuum_wrapper_sage_like(flux, knotnum=13, izoff=0, edge_guard_frac=0.02):
    """
    Mirror SNID-SAGE wrapper behavior: fit continuum, then zero outside
    the observed-data range on the log grid.
    """
    flat, cont = fit_continuum_spline_sage_like(
        flux,
        knotnum=knotnum,
        izoff=izoff,
        edge_guard_frac=edge_guard_frac,
    )

    valid_indices = np.where((flux != 0) & np.isfinite(flux))[0]
    if valid_indices.size:
        i0, i1 = valid_indices[0], valid_indices[-1]
        flat[:i0] = 0.0
        flat[i1 + 1:] = 0.0
        cont[:i0] = 0.0
        cont[i1 + 1:] = 0.0

    return flat, cont

def apodize_sage_like(arr, n1, n2, percent=5.0):
    out = np.array(arr, dtype=float, copy=True)
    if not (0 <= n1 <= n2 < len(arr)):
        return out
    if percent is None or percent <= 0:
        return out

    valid_len = n2 - n1 + 1
    if valid_len <= 0:
        return out

    ns = int(round(valid_len * percent / 100.0))
    ns = min(ns, valid_len // 2)
    if ns < 1:
        return out

    if ns == 1:
        ramp = np.array([0.0])
    else:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(ns) / (ns - 1.0)))

    out[n1:n1 + ns] *= ramp
    out[n2 - ns + 1:n2 + 1] *= ramp[::-1]
    return out


def preprocess_spectrum_sage_like(
    wave_obs,
    flux_obs,
    z=0.0,
    obs_mask_ranges=None,      # optional observed-frame masks, e.g. telluric/sky/bad regions
    num_points=1024,
    min_wave=2500.0,
    max_wave=10000.0,
    knotnum=13,
    izoff=0,
    edge_guard_frac=0.02,
    apodize_percent=5.0,
):
    """
    SNID-SAGE-like preprocessing starting from observed-frame wavelength and flux.

    Returns:
      wave_obs_used        observed-frame wavelengths after masking
      flux_obs_used        matching observed-frame flux
      wave_rest_used       rest-frame wavelengths used for log rebinning
      wave_log             fixed SNID log grid (rest frame)
      flux_log             rebinned flux on fixed log grid
      continuum            continuum model on fixed log grid
      norm_flux            flux_log / continuum
      flat_flux            norm_flux - 1
      flat_flux_apodized   flat_flux after raised-cosine taper
      flat_flux_ready      apodized flat_flux with zero mean over valid region
      valid_mask           bins with rebinned data on the fixed grid
    """
    wave_obs = np.asarray(wave_obs, dtype=float)
    flux_obs = np.asarray(flux_obs, dtype=float)

    good = np.isfinite(wave_obs) & np.isfinite(flux_obs) & (wave_obs > 0)
    wave_obs = wave_obs[good]
    flux_obs = flux_obs[good]

    if wave_obs.size < 2:
        raise ValueError("Need at least 2 valid points")

    order = np.argsort(wave_obs)
    wave_obs = wave_obs[order]
    flux_obs = flux_obs[order]

    # optional observed-frame masking
    wave_obs_used, flux_obs_used = apply_wavelength_mask(wave_obs, flux_obs, obs_mask_ranges)

    if wave_obs_used.size < 2:
        raise ValueError("Too few points left after observed-frame masking")

    # move to rest frame AFTER observed-frame masking
    wave_rest_used = wave_obs_used / (1.0 + z)

    # overlap with fixed log grid
    if min(np.nanmax(wave_rest_used), max_wave) <= max(np.nanmin(wave_rest_used), min_wave):
        raise ValueError(
            f"No overlap with fixed log grid after de-redshifting: "
            f"{np.nanmin(wave_rest_used):.1f}-{np.nanmax(wave_rest_used):.1f} A"
        )

    # flux-conserving log rebin
    wave_log, flux_log = log_rebin_sage_like(
        wave_rest_used,
        flux_obs_used,
        num_points=num_points,
        min_wave=min_wave,
        max_wave=max_wave,
    )

    # continuum fit
    flat_flux, continuum = fit_continuum_wrapper_sage_like(
        flux_log,
        knotnum=knotnum,
        izoff=izoff,
        edge_guard_frac=edge_guard_frac,
    )

    # normalized flux
    norm_flux = np.zeros_like(flux_log)
    ok = (flux_log > 0) & np.isfinite(flux_log) & np.isfinite(continuum) & (continuum > 0)
    norm_flux[ok] = flux_log[ok] / continuum[ok]

    # valid bins on rebinned grid
    valid_mask = (flux_log != 0) & np.isfinite(flux_log)

    # apodize only over valid data span
    flat_flux_apodized = flat_flux.copy()
    if np.any(valid_mask):
        i0, i1 = np.where(valid_mask)[0][[0, -1]]
        flat_flux_apodized = apodize_sage_like(flat_flux, i0, i1, percent=apodize_percent)

    # zero-mean version for correlation
    flat_flux_ready = flat_flux_apodized.copy()
    if np.any(valid_mask):
        flat_flux_ready[valid_mask] -= np.mean(flat_flux_ready[valid_mask])

    return {
        "wave_obs_used": wave_obs_used,
        "flux_obs_used": flux_obs_used,
        "wave_rest_used": wave_rest_used,
        "wave_log": wave_log,
        "flux_log": flux_log,
        "continuum": continuum,
        "norm_flux": norm_flux,
        "flat_flux": flat_flux,
        "flat_flux_apodized": flat_flux_apodized,
        "flat_flux_ready": flat_flux_ready,
        "valid_mask": valid_mask,
    }


def _annotate_check_axis(ax: Any, message: str, title: str = "") -> None:
    ax.clear()
    ax.text(
        0.5,
        0.5,
        message[:800],
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=7,
        wrap=True,
    )
    if title:
        ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def _safe_plot_stem(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)


def _obsdate_to_mjd(obsdate: Any) -> float:
    """Parse WISeREP / table observation time strings to MJD."""
    if hasattr(obsdate, "mjd"):
        return float(obsdate.mjd)
    s = str(obsdate).strip()
    try:
        return float(Time(s, format="iso").mjd)
    except Exception:
        return float(Time(s).mjd)


def _effective_redshift_for_row(default_redshift: float, row_meta: dict[str, Any] | None) -> float:
    """
    Use z=0 when WISeREP remarks indicate rest-frame correction.
    """
    if not isinstance(row_meta, dict):
        return float(default_redshift)
    remarks = row_meta.get("Remarks")
    text = str(remarks).lower() if remarks is not None else ""
    if "corrected for redshift" in text:
        return 0.0
    return float(default_redshift)


def save_comparison_grid_png(
    out_path: Path,
    suptitle: str,
    entries: list[dict[str, Any]],
    *,
    apply_scale: bool = False,
    corr_tol: float = 0.9,
    nrmse_scaled_tol: float = 0.5,
) -> None:
    """Draw one subplot per entry using the same logic as ``check_spectra_consistency``."""
    import matplotlib.pyplot as plt

    n = len(entries)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No panels to plot", ha="center", va="center", transform=ax.transAxes)
        fig.suptitle(suptitle, fontsize=10)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.9 * nrows), squeeze=False)
    for i, ent in enumerate(entries):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        _mjp = ent.get("mjd_minus_phase", ent.get("mjp"))
        if _mjp is not None:
            _mjp = float(_mjp)
        check_spectra_consistency(
            ent["wl_template"],
            ent["flux_template"],
            ent["wl_wiserep_template"],
            ent["flux_wiserep_template"],
            label1=ent.get("label1", "template"),
            label2=ent.get("label2", "wiserep"),
            apply_scale=apply_scale,
            corr_tol=corr_tol,
            nrmse_scaled_tol=nrmse_scaled_tol,
            plot_only_consistent=False,
            plot=False,
            ax=ax,
            mjp=_mjp,
        )
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    fig.suptitle(suptitle, fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def check_spectra_consistency(
    wl1,
    flux1,
    wl2,
    flux2,
    label1: str = '',
    label2: str = '',
    cut_outer_AA: float = 500.0,
    wl_range_tol_AA: float = 50000.0,
    apply_scale: bool = False,
    corr_tol: float = 0.9,
    nrmse_scaled_tol: float = 0.5,
    plot: bool = False,
    plot_only_consistent: bool = True,
    ax: Any | None = None,
    savefig_path: str | Path | None = None,
    mjp: float | None = None,
) -> dict[str, Any]:
    """
    Compare two spectra on their wavelength overlap using interpolation, not
    element-wise equality (wl1/wl2 need not share the same grid).

    Wavelength consistency (your rules):
      1) Native coverage: if ``|wl1_min - wl2_min| > wl_range_tol_AA`` or
         ``|wl1_max - wl2_max| > wl_range_tol_AA``, spectra are **inconsistent**.
      2) Overlap: crop ``cut_outer_AA`` from **each** spectrum's own ends, then
         take the intersection. If either cropped interval is empty or the
         intersection is empty, **inconsistent**.

    Flux comparison (when wavelength checks pass):
      3) Interpolate spectrum2 onto spectrum1's wavelength samples inside the
         overlap interval.
      4) Pearson r and NRMSE; optional multiplicative scale on flux2.

    **Important:** flux1 and flux2 must be the same kind of quantity (e.g. both
    continuum-normalized, or both raw f_lambda). Comparing raw template flux to
    ``norm_flux - 1`` from preprocessing will not be meaningful.

    ``is_consistent`` is True only when all wavelength rules pass **and**
    Pearson r >= ``corr_tol`` (no abs — negative correlation is not a match)
    and scaled NRMSE <= ``nrmse_scaled_tol``.

    ``mjp``: optional MJD_obs minus template phase (days); shown on each plot title as ``mjp=...``.
    """
    wl1 = np.asarray(wl1, dtype=float)
    flux1 = np.asarray(flux1, dtype=float)
    wl2 = np.asarray(wl2, dtype=float)
    flux2 = np.asarray(flux2, dtype=float)

    # FIX 1: sort wl1 as well as wl2 so the overlap mask and plot are correct.
    o1 = np.argsort(wl1)
    wl1, flux1 = wl1[o1], flux1[o1]

    o2 = np.argsort(wl2)
    wl2s, f2s = wl2[o2], flux2[o2]

    wl1_min = float(wl1.min())
    wl1_max = float(wl1.max())
    wl2_min = float(wl2s.min())
    wl2_max = float(wl2s.max())

    dmin = abs(wl1_min - wl2_min)
    dmax = abs(wl1_max - wl2_max)
    wl_endpoints_mismatch = (dmin > wl_range_tol_AA) or (dmax > wl_range_tol_AA)

    lo1 = wl1_min + cut_outer_AA
    hi1 = wl1_max - cut_outer_AA
    lo2 = wl2_min + cut_outer_AA
    hi2 = wl2_max - cut_outer_AA
    crop1_ok = lo1 < hi1
    crop2_ok = lo2 < hi2
    lo = float(max(lo1, lo2))
    hi = float(min(hi1, hi2))
    overlap_ok = crop1_ok and crop2_ok and (lo < hi)

    wavelength_consistent = (not wl_endpoints_mismatch) and overlap_ok

    if not wavelength_consistent:
        reasons: list[str] = []
        if wl_endpoints_mismatch:
            reasons.append(
                f"native wl endpoints differ by >{wl_range_tol_AA:.0f} Å "
                f"(Δmin={dmin:.1f} Å, Δmax={dmax:.1f} Å)"
            )
        if not crop1_ok:
            reasons.append(
                f"spectrum1 cropped range empty (need span > {2 * cut_outer_AA:.0f} Å)"
            )
        if not crop2_ok:
            reasons.append(
                f"spectrum2 cropped range empty (need span > {2 * cut_outer_AA:.0f} Å)"
            )
        if crop1_ok and crop2_ok and not (lo < hi):
            reasons.append(
                "cropped ranges do not overlap (inconsistent overlapped region)"
            )
        msg = "; ".join(reasons) if reasons else "wavelength checks failed"
        out_fail: dict[str, Any] = {
            "ok": False,
            "reason": msg,
            "is_consistent": False,
            "wavelength_consistent": False,
            "wl_range_tol_AA": wl_range_tol_AA,
            "native_wl1_AA": (wl1_min, wl1_max),
            "native_wl2_AA": (wl2_min, wl2_max),
            "delta_wl_min_AA": dmin,
            "delta_wl_max_AA": dmax,
            "wl_endpoints_mismatch": wl_endpoints_mismatch,
            "cropped_wl1_AA": (lo1, hi1) if crop1_ok else None,
            "cropped_wl2_AA": (lo2, hi2) if crop2_ok else None,
            "overlap_wl_AA": (lo, hi) if overlap_ok else None,
        }
        if ax is not None:
            _am = msg
            if mjp is not None:
                _am = f"{_am}\nmjp={float(mjp):.4f}"
            _annotate_check_axis(ax, _am)
        return out_fail

    m1 = (wl1 >= lo) & (wl1 <= hi) & np.isfinite(flux1)
    wl_c = wl1[m1]
    f1_c = flux1[m1]
    wl_meta = {
        "wavelength_consistent": True,
        "wl_range_tol_AA": wl_range_tol_AA,
        "native_wl1_AA": (wl1_min, wl1_max),
        "native_wl2_AA": (wl2_min, wl2_max),
        "delta_wl_min_AA": dmin,
        "delta_wl_max_AA": dmax,
        "wl_endpoints_mismatch": False,
        "cropped_wl1_AA": (lo1, hi1),
        "cropped_wl2_AA": (lo2, hi2),
        "overlap_wl_AA": (lo, hi),
        # FIX 2: removed duplicate 'wl_range_AA' key that equalled overlap_wl_AA.
    }

    if wl_c.size < 5:
        out = {
            "ok": False,
            "reason": "too few points in overlap",
            "n_points": int(wl_c.size),
            "is_consistent": False,
        }
        out.update(wl_meta)
        if ax is not None:
            _am = out["reason"]
            if mjp is not None:
                _am = f"{_am}\nmjp={float(mjp):.4f}"
            _annotate_check_axis(ax, _am)
        return out

    f2_on1 = np.interp(wl_c, wl2s, f2s, left=np.nan, right=np.nan)
    ok = np.isfinite(f1_c) & np.isfinite(f2_on1)
    n_ok = int(np.count_nonzero(ok))
    if n_ok < 5:
        out = {
            "ok": False,
            "reason": "too few finite pairs after interpolation",
            "n_points": n_ok,
            "is_consistent": False,
        }
        out.update(wl_meta)
        if ax is not None:
            _am = out["reason"]
            if mjp is not None:
                _am = f"{_am}\nmjp={float(mjp):.4f}"
            _annotate_check_axis(ax, _am)
        return out

    x = f1_c[ok]
    y = f2_on1[ok]
    if np.std(x) < 1e-30 or np.std(y) < 1e-30:
        pearson_r = float("nan")
    else:
        pearson_r = float(np.corrcoef(x, y)[0, 1])

    diff = x - y
    rms = float(np.sqrt(np.mean(diff**2)))

    # FIX 5: fall back to range-based normalisation when the RMS of flux1 is
    # near zero (e.g. continuum-normalised spectra centred on 0 such as
    # norm_flux - 1), so the NRMSE denominator is never meaninglessly tiny.
    rms_x = float(np.sqrt(np.mean(x**2)))
    flux_range = float(np.ptp(x))
    if rms_x > 1e-30:
        denom = rms_x
    elif flux_range > 1e-30:
        denom = flux_range
    else:
        denom = 1.0  # last-resort guard; NRMSE will be near-zero anyway
    nrmse = rms / denom

    scale = 1.0
    nrmse_scaled = nrmse
    if apply_scale:
        scale = float(np.dot(x, y) / (np.dot(y, y) + 1e-30))
        diff_s = x - scale * y
        nrmse_scaled = float(np.sqrt(np.mean(diff_s**2)) / denom)

    mad = float(np.median(np.abs(diff)))

    # FIX 4: removed abs() from the Pearson r check.  A strong *negative*
    # correlation means the spectra are anti-correlated — clearly not a match.
    flux_consistent = (
        np.isfinite(pearson_r)
        and pearson_r >= corr_tol
        and nrmse_scaled <= nrmse_scaled_tol
    )
    is_consistent = flux_consistent

    result: dict[str, Any] = {
        "ok": True,
        "n_points": n_ok,
        "pearson_r": pearson_r,
        "nrmse": nrmse,
        "scale_flux2_to_match_flux1": scale,
        "nrmse_scaled": nrmse_scaled,
        "median_abs_diff": mad,
        "is_consistent": is_consistent,
        "flux_consistent": flux_consistent,
        "corr_tol": corr_tol,
        "nrmse_scaled_tol": nrmse_scaled_tol,
        "mjp": mjp,
    }
    result.update(wl_meta)

    want_draw = (plot and ax is None) or (ax is not None)
    if want_draw:
        import matplotlib.pyplot as plt

        created_fig = False
        plot_ax = ax
        if plot_ax is None:
            fig, plot_ax = plt.subplots(figsize=(10, 4))
            created_fig = True

        if plot_only_consistent and not is_consistent:
            logger.debug(
                "plot skipped: spectra not consistent "
                "(pearson_r=%.4f, nrmse_scaled=%.4f)",
                pearson_r,
                nrmse_scaled,
            )
            _sk = (
                f"skipped (plot_only_consistent)\n"
                f"r={pearson_r}\nNRMSE_scaled={nrmse_scaled}"
            )
            if mjp is not None:
                _sk += f"\nmjp={float(mjp):.4f}"
            _annotate_check_axis(plot_ax, _sk)
        else:
            plot_ax.plot(wl_c, f1_c, label=label1, alpha=0.8)
            plot_ax.plot(wl_c, f2_on1, label=label2, alpha=0.8)
            if apply_scale and scale != 1.0:
                plot_ax.plot(
                    wl_c,
                    scale * f2_on1,
                    label=f"flux2 scaled (×{scale:.4f})",
                    alpha=0.7,
                )
            plot_ax.set_xlabel("Wavelength (Å)")
            plot_ax.set_ylabel("Flux")
            if mjp is not None:
                plot_ax.plot(
                    [],
                    [],
                    linestyle="none",
                    marker="",
                    label=f"mjp={float(mjp):.4f}",
                )
            plot_ax.legend(fontsize=8)
            plot_ax.set_title(
                f"r={pearson_r:.4f}, NRMSE={nrmse:.4f}, "
                f"NRMSE_scaled={nrmse_scaled:.4f}, is_consistent={is_consistent}",
                fontsize=9,
            )

        if created_fig:
            if savefig_path is not None:
                fig.savefig(savefig_path, bbox_inches="tight", dpi=150)
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)

    return result

def _sheet_cell_value(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and (not np.isfinite(v)):
        return ""
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v)
    return v


def update_sheet_row_from_table_row(
    ws: Any,
    row: Any,
    key_column: str = "Objname",
    header_row: int = 1,
) -> int:
    """
    Update an existing Google Sheet row matched by key_column from an astropy Row.
    Returns the updated 1-based sheet row index.
    """
    headers = ws.row_values(header_row)
    if not headers:
        raise ValueError("Sheet header row is empty.")
    if key_column not in headers:
        raise KeyError(f"'{key_column}' not found in sheet headers.")
    if key_column not in row.colnames:
        raise KeyError(f"'{key_column}' not found in table row columns.")

    key_value = str(row[key_column]).strip()
    key_col_idx = headers.index(key_column) + 1
    key_col_values = ws.col_values(key_col_idx)

    row_idx = None
    for i, cell_value in enumerate(key_col_values[header_row:], start=header_row + 1):
        if str(cell_value).strip() == key_value:
            row_idx = i
            break
    if row_idx is None:
        raise ValueError(f"Could not find row where {key_column}='{key_value}'.")

    row_values: list[Any] = []
    for col in headers:
        if col in row.colnames:
            row_values.append(_sheet_cell_value(row[col]))
        else:
            row_values.append("")

    start = rowcol_to_a1(row_idx, 1)
    end = rowcol_to_a1(row_idx, len(headers))
    ws.update(range_name=f"{start}:{end}", values=[row_values])
    return row_idx
 #%%
all_targets = load_template_object_names(INDEX_FILE)
#%%
OUTPUT_ROOT = Path("/home/hhchoi1022/code/SNID-SAGE/wiserep_spectra")
PHASE_OFFSET_RESULTS_PATH = OUTPUT_ROOT / "phase_offset_results.json"

#%%
import gspread
gc = gspread.service_account(filename="/home/hhchoi1022/code/SNID-SAGE/googlesheet.json")
sh = gc.open_by_url("https://docs.google.com/spreadsheets/d/1wwbBlAvOUamYM-l_F-N6xd4UYPEcKhntzLETNtP-m1o/edit?gid=0#gid=0")
ws = sh.get_worksheet(0)
rows = ws.get_all_records()
from astropy.table import Table
tbl = Table(rows)
#%%
# tbl.write('spreadsheet.csv', format = 'csv')

# %% 1. Process only clean samples
tbl_clean = tbl[tbl['Quality note'] == 'P']
tbl_check = tbl[tbl['Quality note'] == 'C']
tbl_reject = tbl[tbl['Quality note'] == 'R']

#%%
from bridge.utils import HostGalaxyCatalog
# %%
hg = HostGalaxyCatalog()
# %%




tbl_to_check = tbl_clean[tbl_clean['n_templates'].astype(float) > tbl_clean['n_matched_template'].astype(float)]

idx = 145
row_to_check = tbl_to_check[idx]
objname = row_to_check['Objname']
num_templates = int(row_to_check['n_templates'])
colnames_epoch = [f'epoch_{i}' for i in range(num_templates)]
epochs_to_examine = [colname for colname in colnames_epoch if row_to_check[colname].replace(' ', '') == '']
print('Object name:', objname)
print('Epochs to examine:', epochs_to_examine)
#%%

import glob
from snal.utils.tnsquerier import TNSQuerier
from pathlib import Path
tnsquerier = TNSQuerier()
age_offset = 1
check_consistent = False
FIGURES_MJD_COMPARE_TARGET_DIR = Path('./SNID-SAGE/figures_redshift_compare/check')
TEMPLATE_DIR = Path('./SNID-SAGE/templates/check')
objnames_to_check = dict()
# for i in range(len(tbl_check)):
#     try:
        # row = tbl_check[i]
row = tbl[tbl['Objname'] == objname][0]
objname = row['Objname']
mjd_peak = row['mjd_max_phase']
print(objname, mjd_peak)

if float(mjd_peak) == -99:
    raise ValueError(f'mjd_peak is -99 for {objname}')

wl_phase_dict, flux_phase_dict, template_meta = load_template_spectrum(objname)
template_meta_epoch = template_meta['epoch']
wl_wiserep_dict, flux_wiserep_dict, fluxerr_wiserep_dict, wiserep_meta = load_wiserep_spectrum(
    objname
)
wiserep_meta_epoch = wiserep_meta['epoch']

redshift = template_meta['redshift']
if redshift is None or redshift == 0.0:
    redshift = wiserep_meta['redshift']
if redshift is None or redshift == 0.0:
    print('Redshift is not set, querying TNS')
    tns_result = tnsquerier.get_object(objname)        
    if 'redshift' in tns_result[0].keys():
        redshift = tns_result[0]['redshift']
        if redshift is None or redshift == 0.0:
            ra = tns_result[0]['radeg']
            dec = tns_result[0]['decdeg']
            hg_result = hg.match_host(ra, dec, plot=  False)
            redshift = float(hg_result['z'])
        else:
            redshift = 0
    else:
        redshift = 0
print('redshift:', redshift)

wiserep_obsdate_mjds = [_obsdate_to_mjd(obsdate) for obsdate in wiserep_meta_epoch.keys()]
wiserep_ages = [obsdate_mjd - float(mjd_peak) for obsdate_mjd in wiserep_obsdate_mjds]
template_ages = [float(value['age']) for value in template_meta_epoch.values()]

n_matched = 0
consistency_by_phase: dict[Any, dict[Any, Any]] = {}
import matplotlib.pyplot as plt
phase_items = list(template_meta_epoch.items())
obj_type_default = ""
if len(wiserep_meta_epoch) > 0:
    obj_type_default = str(next(iter(wiserep_meta_epoch.values())).get("Obj Type", ""))
import math

ncols = 2
nrows = math.ceil(len(phase_items) / ncols)
n_phases = len(phase_items)
# ncols = math.ceil(n_phases / nrows)

fig_target, axes_target = plt.subplots(
    nrows,
    ncols * 2,
    figsize=(5 * ncols * 2, nrows * 3.3),
    squeeze=False,
)

num_phases_matched = 0
for phase_idx, (phase, template_meta_single) in enumerate(phase_items):
    try:
        row_idx = phase_idx % nrows
        col_group_idx = phase_idx // nrows

        ax_consistency = axes_target[row_idx, col_group_idx * 2]
        ax_raw = axes_target[row_idx, col_group_idx * 2 + 1]
        wl_template = wl_phase_dict[phase]
        flux_template = flux_phase_dict[phase]
        valid_mask_template = np.where((wl_template > template_meta_single['wl_min']) & (wl_template < template_meta_single['wl_max']))[0]
        wl_template = wl_template[valid_mask_template]
        flux_template = flux_template[valid_mask_template]
        template_age = template_meta_single['age']
        consistency_by_phase[phase] = {}

        age_mask = np.abs(np.array(wiserep_ages) - template_age) <= age_offset
        wiserep_obsdates_to_check = np.array(list(wiserep_meta_epoch.keys()))[age_mask]
        if len(wiserep_obsdates_to_check) == 0:
            ax_consistency.text(0.5, 0.5, f"{phase}: no matched WISeREP age", ha="center", va="center")
            ax_consistency.set_axis_off()
            ax_raw.set_axis_off()
            continue
        
        obsdate_result_dict = {}
        for obsdate in wiserep_obsdates_to_check:
            obsdate_result_dict[obsdate] = {}
            consistency_by_phase[phase][obsdate] = {}
            wiserep_meta_single = wiserep_meta_epoch[obsdate]
            wl_wiserep = wl_wiserep_dict[obsdate]
            flux_wiserep = flux_wiserep_dict[obsdate]
            try:
                # Try with redshift
                wiserep_template = preprocess_spectrum_sage_like(wl_wiserep, flux_wiserep, z=redshift)
                wiserep_template_mask = wiserep_template['valid_mask']
                wl_wiserep_template = wiserep_template['wave_log'][wiserep_template_mask]
                flux_wiserep_template = wiserep_template['norm_flux'][wiserep_template_mask] -1
                result = check_spectra_consistency(wl_template, flux_template, wl_wiserep_template, flux_wiserep_template, label1 = f'template[{template_age}({phase})]', label2 = f'wiserep[{obsdate}]', plot = False, plot_only_consistent = False)
                consistency_by_phase[phase][obsdate]['redshift_nonzero'] = {}
                consistency_by_phase[phase][obsdate]['redshift_nonzero'] = result
                obsdate_result_dict[obsdate]['redshift_nonzero'] = {}
                obsdate_result_dict[obsdate]['redshift_nonzero'] = result
                obsdate_result_dict[obsdate]['redshift_nonzero']['wl_wiserep_template'] = wl_wiserep_template
                obsdate_result_dict[obsdate]['redshift_nonzero']['flux_wiserep_template'] = flux_wiserep_template
                obsdate_result_dict[obsdate]['redshift_nonzero']['wl_template'] = wl_template
                obsdate_result_dict[obsdate]['redshift_nonzero']['flux_template'] = flux_template

                # Try with redshif 0
                wiserep_template_0 = preprocess_spectrum_sage_like(wl_wiserep, flux_wiserep, z=0.0)
                wiserep_template_0_mask = wiserep_template_0['valid_mask']
                wl_wiserep_template_0 = wiserep_template_0['wave_log'][wiserep_template_0_mask]
                flux_wiserep_template_0 = wiserep_template_0['norm_flux'][wiserep_template_0_mask] -1
                result_0 = check_spectra_consistency(wl_template, flux_template, wl_wiserep_template_0, flux_wiserep_template_0, label1 = f'template[{template_age}({phase})]', label2 = f'wiserep[{obsdate}]', plot = False, plot_only_consistent = False)
                consistency_by_phase[phase][obsdate]['redshift_zero'] = {}
                consistency_by_phase[phase][obsdate]['redshift_zero'] = result_0
                obsdate_result_dict[obsdate]['redshift_zero'] = {}
                obsdate_result_dict[obsdate]['redshift_zero'] = result_0
                obsdate_result_dict[obsdate]['redshift_zero']['wl_wiserep_template_0'] = wl_wiserep_template_0
                obsdate_result_dict[obsdate]['redshift_zero']['flux_wiserep_template_0'] = flux_wiserep_template_0
                obsdate_result_dict[obsdate]['redshift_zero']['wl_template'] = wl_template
                obsdate_result_dict[obsdate]['redshift_zero']['flux_template'] = flux_template

            except Exception as e:
                print(f'Error processing wiserep {obsdate}: {e}')
                continue
        
        epoch_pearson_r_nonzero = []
        epoch_pearson_r_zero = []
        epoch_nrmse_nonzero = []
        epoch_nrmse_zero = []
        for obsdate, result_single in obsdate_result_dict.items():
            pr_nonzero = result_single['redshift_nonzero'].get("pearson_r")
            pr_zero = result_single['redshift_zero'].get("pearson_r")
            if pr_nonzero is None or not np.isfinite(pr_nonzero):
                pearson_r_nonzero = float("-inf")
            else:
                pearson_r_nonzero = float(pr_nonzero)
            if pr_zero is None or not np.isfinite(pr_zero):
                pearson_r_zero = float("-inf")
            else:
                pearson_r_zero = float(pr_zero)
            epoch_pearson_r_nonzero.append(pearson_r_nonzero)
            epoch_pearson_r_zero.append(pearson_r_zero)
            epoch_nrmse_nonzero.append(result_single['redshift_nonzero'].get("nrmse_scaled"))
            epoch_nrmse_zero.append(result_single['redshift_zero'].get("nrmse_scaled"))
        max_pearson_r_nonzero = max(epoch_pearson_r_nonzero)
        max_pearson_r_zero = max(epoch_pearson_r_zero)
        # min_nrmse_nonzero = min(epoch_nrmse_nonzero)
        # min_nrmse_zero = min(epoch_nrmse_zero)
        if max_pearson_r_nonzero > max_pearson_r_zero:
            idx_best = epoch_pearson_r_nonzero.index(max_pearson_r_nonzero)
            redshift_input = redshift
        else:
            idx_best = epoch_pearson_r_zero.index(max_pearson_r_zero)
            redshift_input = 0.0
        if len(epoch_pearson_r_nonzero) == 0 or len(epoch_pearson_r_zero) == 0:
            ax_consistency.text(0.5, 0.5, f"{phase}: no valid comparison", ha="center", va="center")
            ax_consistency.set_axis_off()
            ax_raw.set_axis_off()
            continue
        # idx_best = int(np.argmin(epoch_nrmse))
        obsdate_best = list(obsdate_result_dict.keys())[idx_best]
        obsdate_result_best = obsdate_result_dict[obsdate_best]
        wiserep_meta_single = wiserep_meta_epoch[obsdate_best]

        wl_wiserep_best = wl_wiserep_dict[obsdate_best]
        flux_wiserep_best = flux_wiserep_dict[obsdate_best]
        fluxerr_wiserep_best = fluxerr_wiserep_dict[obsdate_best]
        if redshift_input == 0.0:
            wl_wiserep_template_best = obsdate_result_best['redshift_zero']['wl_wiserep_template_0']
            flux_wiserep_template_best = obsdate_result_best['redshift_zero']['flux_wiserep_template_0']
            wl_wiserep_template_nonbest = obsdate_result_best['redshift_nonzero']['wl_wiserep_template']
            flux_wiserep_template_nonbest = obsdate_result_best['redshift_nonzero']['flux_wiserep_template']
            wl_template_best = obsdate_result_best['redshift_zero']['wl_template']
            flux_template_best = obsdate_result_best['redshift_zero']['flux_template']
        else:
            wl_wiserep_template_best = obsdate_result_best['redshift_nonzero']['wl_wiserep_template']
            flux_wiserep_template_best = obsdate_result_best['redshift_nonzero']['flux_wiserep_template']
            wl_wiserep_template_nonbest = obsdate_result_best['redshift_zero']['wl_wiserep_template_0']
            flux_wiserep_template_nonbest = obsdate_result_best['redshift_zero']['flux_wiserep_template_0']
            wl_template_best = obsdate_result_best['redshift_nonzero']['wl_template']
            flux_template_best = obsdate_result_best['redshift_nonzero']['flux_template']
        template_age = template_meta_epoch[phase]['age']
        mjd_minus_phase = _obsdate_to_mjd(obsdate_best) - template_age
        mjd_offset = np.abs(mjd_minus_phase - float(mjd_peak))
        result = check_spectra_consistency(
            wl_template_best,
            flux_template_best,
            wl_wiserep_template_best,
            flux_wiserep_template_best,
            label1=f"template[{template_age}({phase})]",
            label2=f"wiserep[{obsdate_best}, redshift = {redshift_input}]",
            plot=True,
            plot_only_consistent=False,
            mjp=mjd_minus_phase,
            ax=ax_consistency,
        )
        wl_min = np.max(np.array([result['cropped_wl1_AA'][0], result['cropped_wl2_AA'][0]]))
        wl_max = np.min(np.array([result['cropped_wl1_AA'][1], result['cropped_wl2_AA'][1]]))
        wl_mask = (wl_wiserep_template_nonbest > wl_min) & (wl_wiserep_template_nonbest < wl_max)
        ax_consistency.plot(wl_wiserep_template_nonbest[wl_mask], flux_wiserep_template_nonbest[wl_mask]+ np.max(flux_wiserep_template_nonbest[wl_mask]) * 0.2, c = 'r', alpha = 0.2)

        wl_wiserep_rest = wl_wiserep_best /(1 + redshift_input)
        wiserep_meta_single['redshift_input'] = redshift_input
        wiserep_meta_single['snid_sage_epoch'] = template_meta_single['epoch']
        wiserep_meta_single['snid_sage_age'] = template_meta_single['age']
        flux_wiserep_rest = flux_wiserep_best
        fluxerr_wiserep_rest = fluxerr_wiserep_best
        tbl_filename = Path(wiserep_meta_single['Spectrum ascii File']).stem + '_rest.csv'
        (TEMPLATE_DIR / objname).mkdir(parents=True, exist_ok=True)
        tbl_filepath = TEMPLATE_DIR / objname / tbl_filename
        meta_filename = Path(wiserep_meta_single['Spectrum ascii File']).stem + '_meta.json'
        meta_filepath = TEMPLATE_DIR / objname / meta_filename
        
        tbl_rest = Table()
        tbl_rest['wl'] = wl_wiserep_rest
        tbl_rest['flux'] = flux_wiserep_rest
        tbl_rest['fluxerr'] = fluxerr_wiserep_rest
        tbl_rest.write(tbl_filepath, format='csv', overwrite=True)
        json.dump(wiserep_meta_single, open(meta_filepath, 'w'), indent=4)
        num_phases_matched += 1
        
        if mjd_offset > 1:
            color = 'r'
        else:
            color = 'k'
        if redshift != redshift_input:
            color = 'b'
        ax_raw.plot(wl_wiserep_best /(1 + redshift_input), flux_wiserep_best, c = color, alpha = 0.2)
        # Make sure fluxerr_wiserep_best has positive value
        fluxerr_wiserep_best = np.abs(fluxerr_wiserep_best)
        ax_raw.errorbar(wl_wiserep_best /(1 + redshift_input), flux_wiserep_best, yerr=fluxerr_wiserep_best, fmt='none', label=f'wiserep[{obsdate_best}]', c = color, alpha = 0.2)
        ax_raw.set_title(f"{phase} raw+err  MJD-phase={mjd_minus_phase:.2f}")
        ax_raw.set_xlabel("Rest wavelength [A]")
        ax_raw.set_ylabel("Flux")
        ax_raw.legend(loc="best", fontsize=8)
        ax_raw.set_xlim(3500, 9500)
        if check_consistent:
            if result['is_consistent']:
                n_matched += 1
                if phase == 'single':
                    phase = 'epoch_0'
                row[phase] = obsdate_best
        else:
            n_matched += 1
            if phase == 'single':
                phase = 'epoch_0'
            row[phase] = obsdate_best
        print(f"Age: {template_age}, Obsdate: {obsdate_best}, Delta phase: {mjd_minus_phase:.2f}, Redshift: {redshift_input}")
        row[f"{phase}_redshift"] = redshift_input
    except:
        pass

for empty_idx in range(n_phases, nrows * ncols):
    row_idx = empty_idx % nrows
    col_group_idx = empty_idx // nrows

    axes_target[row_idx, col_group_idx * 2].set_axis_off()
    axes_target[row_idx, col_group_idx * 2 + 1].set_axis_off()

num_phases_saved = len(glob.glob(str(TEMPLATE_DIR / objname / '*.csv')))
if num_phases_saved != num_phases_matched:
    objnames_to_check[objname] = 'Duplicated phase'
row['Type'] = obj_type_default
row['Redshift'] = redshift
row['n_matched_template'] = n_matched
fig_target.suptitle(f"{objname}: template-vs-WISeREP matches", fontsize=11)
fig_target.tight_layout(rect=[0, 0.03, 1, 0.96])
path_target_fig = TEMPLATE_DIR / objname / f"{_safe_plot_stem(objname)}_matched_grid.png"
# fig_target.savefig(path_target_fig, dpi=150, bbox_inches="tight")
print(f"Saved target figure: {path_target_fig}")
# plt.close(fig_target)
# %%
# updated_row_idx = update_sheet_row_from_table_row(ws, row, key_column="Objname")
#%%
# Inspect age-matched WISeREP options for one selected template phase

import numpy as np
import matplotlib.pyplot as plt
import math

# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
# phase_to_inspect = "epoch_1"   # Change to "epoch_2", "epoch_3", "single", etc.


for phase_to_inspect in epochs_to_examine:
    try:
        plot_top_n = 24
        save_selected_phase_fig = False
        age_offset = 4

        # ------------------------------------------------------------
        # Basic checks
        # ------------------------------------------------------------
        template_meta_epoch = template_meta["epoch"]
        wiserep_meta_epoch = wiserep_meta["epoch"]

        if phase_to_inspect not in template_meta_epoch:
            raise ValueError(
                f"phase_to_inspect={phase_to_inspect!r} not found. "
                f"Available phases are: {list(template_meta_epoch.keys())}"
            )

        if len(wiserep_meta_epoch) == 0:
            raise ValueError(f"No WISeREP spectra available for {objname}")

        # ------------------------------------------------------------
        # Get selected template phase
        # ------------------------------------------------------------
        template_meta_single = template_meta_epoch[phase_to_inspect]

        wl_template = wl_phase_dict[phase_to_inspect]
        flux_template = flux_phase_dict[phase_to_inspect]

        valid_mask_template = np.where(
            (wl_template > template_meta_single["wl_min"])
            & (wl_template < template_meta_single["wl_max"])
        )[0]

        wl_template = wl_template[valid_mask_template]
        flux_template = flux_template[valid_mask_template]

        template_age = float(template_meta_single["age"])

        # ------------------------------------------------------------
        # Apply the same age_offset filtering as the original code
        # ------------------------------------------------------------
        wiserep_obsdate_list = np.array(list(wiserep_meta_epoch.keys()))

        wiserep_obsdate_mjds = np.array([
            _obsdate_to_mjd(obsdate)
            for obsdate in wiserep_obsdate_list
        ])

        wiserep_ages = wiserep_obsdate_mjds - float(mjd_peak)

        age_mask = np.abs(wiserep_ages - template_age) <= age_offset
        wiserep_obsdates_to_check = wiserep_obsdate_list[age_mask]

        print("=" * 100)
        print(f"Inspecting selected phase: {phase_to_inspect}")
        print(f"Template age: {template_age}")
        print(f"age_offset: {age_offset}")
        print(f"Total WISeREP spectra: {len(wiserep_meta_epoch)}")
        print(f"Age-matched WISeREP spectra: {len(wiserep_obsdates_to_check)}")
        print("=" * 100)

        if len(wiserep_obsdates_to_check) == 0:
            raise RuntimeError(
                f"No WISeREP spectra matched phase={phase_to_inspect!r} "
                f"within age_offset={age_offset}."
            )

        # ------------------------------------------------------------
        # Check only age-matched WISeREP spectra for this selected phase
        # ------------------------------------------------------------
        phase_options = []
        phase_result_dict = {}

        for obsdate in wiserep_obsdates_to_check:
            phase_result_dict[obsdate] = {}

            wl_wiserep = wl_wiserep_dict[obsdate]
            flux_wiserep = flux_wiserep_dict[obsdate]

            for z_key, z_input in [
                ("redshift_nonzero", redshift),
                ("redshift_zero", 0.0),
            ]:
                try:
                    wiserep_template = preprocess_spectrum_sage_like(
                        wl_wiserep,
                        flux_wiserep,
                        z=z_input,
                    )

                    wiserep_template_mask = wiserep_template["valid_mask"]

                    wl_wiserep_template = wiserep_template["wave_log"][wiserep_template_mask]
                    flux_wiserep_template = wiserep_template["norm_flux"][wiserep_template_mask] - 1

                    result = check_spectra_consistency(
                        wl_template,
                        flux_template,
                        wl_wiserep_template,
                        flux_wiserep_template,
                        label1=f"template[{template_age}({phase_to_inspect})]",
                        label2=f"wiserep[{obsdate}, {z_key}, z={z_input}]",
                        plot=False,
                        plot_only_consistent=False,
                        apply_scale = True
                    )

                    result["wl_wiserep_template"] = wl_wiserep_template
                    result["flux_wiserep_template"] = flux_wiserep_template
                    result["wl_template"] = wl_template
                    result["flux_template"] = flux_template

                    phase_result_dict[obsdate][z_key] = result

                    pearson_r = result.get("pearson_r")
                    nrmse_scaled = result.get("nrmse_scaled")
                    is_consistent = result.get("is_consistent", False)

                    pearson_r_score = (
                        float(pearson_r)
                        if pearson_r is not None and np.isfinite(pearson_r)
                        else float("-inf")
                    )

                    nrmse_score = (
                        float(nrmse_scaled)
                        if nrmse_scaled is not None and np.isfinite(nrmse_scaled)
                        else float("inf")
                    )

                    obsdate_mjd = _obsdate_to_mjd(obsdate)
                    wiserep_age = obsdate_mjd - float(mjd_peak)
                    mjd_minus_phase = obsdate_mjd - template_age
                    mjd_offset = abs(wiserep_age - template_age)

                    phase_options.append({
                        "phase": phase_to_inspect,
                        "template_age": template_age,
                        "obsdate": obsdate,
                        "wiserep_age": wiserep_age,
                        "z_key": z_key,
                        "redshift_input": z_input,
                        "is_consistent": bool(is_consistent),
                        "pearson_r": pearson_r_score,
                        "nrmse_scaled": nrmse_score,
                        "mjd_minus_phase": mjd_minus_phase,
                        "mjd_offset": mjd_offset,
                    })

                except Exception as e:
                    print(f"Error processing obsdate={obsdate}, z_key={z_key}, z={z_input}: {e}")
                    continue

        if len(phase_options) == 0:
            raise RuntimeError(
                f"No valid comparison results for phase_to_inspect={phase_to_inspect!r}"
            )

        # ------------------------------------------------------------
        # Rank options
        # Consistent first, then higher Pearson r, then lower NRMSE
        # ------------------------------------------------------------
        phase_options = sorted(
            phase_options,
            key=lambda x: (
                x["is_consistent"],
                x["pearson_r"],
                -x["nrmse_scaled"],
            ),
            reverse=True,
        )

        # ------------------------------------------------------------
        # Print ranked summary
        # ------------------------------------------------------------
        print("\n")
        print("=" * 130)
        print(
            f"{objname}: age-matched WISeREP options for selected phase = "
            f"{phase_to_inspect}"
        )
        print("=" * 130)

        for i, opt in enumerate(phase_options, start=1):
            print(
                f"{i:03d} | "
                f"obsdate={opt['obsdate']} | "
                f"wiserep_age={opt['wiserep_age']:.2f} | "
                f"template_age={opt['template_age']:.2f} | "
                f"age_diff={opt['mjd_offset']:.2f} | "
                f"z_mode={opt['z_key']} | "
                f"z={opt['redshift_input']} | "
                f"consistent={opt['is_consistent']} | "
                f"pearson_r={opt['pearson_r']:.4f} | "
                f"nrmse={opt['nrmse_scaled']:.4f}"
            )

        # Optional: keep this in your existing dictionary
        if "consistency_by_phase" not in globals():
            consistency_by_phase = {}

        consistency_by_phase[phase_to_inspect] = {
            "age_offset": age_offset,
            "age_matched_options_ranked": phase_options,
            "all_results": phase_result_dict,
        }

        # ------------------------------------------------------------
        # Plot top N options
        # ------------------------------------------------------------
        top_options = phase_options[:plot_top_n]

        ncols = 2
        nrows = math.ceil(len(top_options) / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(7 * ncols, 4 * nrows),
            squeeze=False,
        )

        for idx, opt in enumerate(top_options):
            row_idx = idx // ncols
            col_idx = idx % ncols
            ax = axes[row_idx, col_idx]

            obsdate = opt["obsdate"]
            z_key = opt["z_key"]

            result = phase_result_dict[obsdate][z_key]

            check_spectra_consistency(
                result["wl_template"],
                result["flux_template"],
                result["wl_wiserep_template"],
                result["flux_wiserep_template"],
                label1=f"template[{template_age}({phase_to_inspect})]",
                label2=f"wiserep[{obsdate}, {z_key}, z={opt['redshift_input']}]",
                plot=True,
                plot_only_consistent=False,
                mjp=opt["mjd_minus_phase"],
                ax=ax,
                apply_scale = True
            )

            ax.set_title(
                f"Rank {idx + 1}: {obsdate}\n"
                f"{z_key}, z={opt['redshift_input']}, "
                f"consistent={opt['is_consistent']}\n"
                f"age_diff={opt['mjd_offset']:.2f}, "
                f"r={opt['pearson_r']:.3f}, "
                f"nrmse={opt['nrmse_scaled']:.3f}",
                fontsize=9,
            )

        for empty_idx in range(len(top_options), nrows * ncols):
            row_idx = empty_idx // ncols
            col_idx = empty_idx % ncols
            axes[row_idx, col_idx].set_axis_off()

        fig.suptitle(
            f"{objname}: age-matched options for selected phase {phase_to_inspect}",
            fontsize=12,
        )

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_selected_phase_fig:
            selected_phase_fig_path = (
                TEMPLATE_DIR
                / objname
                / f"{_safe_plot_stem(objname)}_{phase_to_inspect}_age_matched_options.png"
            )
            fig.savefig(selected_phase_fig_path, dpi=150, bbox_inches="tight")
            print(f"Saved selected phase inspection figure: {selected_phase_fig_path}")

        plt.show()

        # ------------------------------------------------------------
        # Best-ranked option
        # ------------------------------------------------------------
        best_option = phase_options[0]

        print("\nBest-ranked age-matched option:")
        print(best_option)

        best_obsdate = best_option["obsdate"]
        best_z_key = best_option["z_key"]
        best_redshift_input = best_option["redshift_input"]
        best_result = phase_result_dict[best_obsdate][best_z_key]

        wl_template_best = best_result["wl_template"]
        flux_template_best = best_result["flux_template"]
        wl_wiserep_template_best = best_result["wl_wiserep_template"]
        flux_wiserep_template_best = best_result["flux_wiserep_template"]
    except Exception as e:
        print(f"Error processing obsdate={best_obsdate}, z_key={best_z_key}, z={best_redshift_input}: {e}")
        continue
# %%
# %%
# updated_row_idx = update_sheet_row_from_table_row(ws, row, key_column="Objname")

# %% =====================================================================
# Consistency check from saved spreadsheet.csv
# For each object, plot template[phase] vs WISeREP[obsdate] (SAGE-preprocessed)
# using the obsdate stored in column ``epoch_N`` and the redshift stored in
# column ``epoch_N_redshift``.
# =========================================================================
import re
import math
import pandas as pd
import matplotlib.pyplot as plt

SPREADSHEET_CSV = Path(
    "/home/hhchoi1022/code/SNAL/snal/analysis/classification/spreadsheet.csv"
)
COMPARE_FIG_DIR = Path("./figures_csv_compare")

_EPOCH_COL_RE = re.compile(r"^epoch_(\d+)$")


def read_matched_epochs_from_row(row_dict: dict) -> list[tuple[str, str, float]]:
    """
    Return ``[(epoch_col, obsdate, redshift), ...]`` for every ``epoch_N`` cell
    that has a non-empty obsdate. Sorted by N. Missing/blank redshift -> 0.0.
    """
    out: list[tuple[str, str, float]] = []
    for key, value in row_dict.items():
        m = _EPOCH_COL_RE.match(str(key))
        if not m:
            continue
        if value is None:
            continue
        s_val = str(value).strip()
        if s_val == "" or s_val.lower() == "nan":
            continue
        z_raw = row_dict.get(f"{key}_redshift", "")
        try:
            z_val = float(str(z_raw).strip()) if str(z_raw).strip() != "" else 0.0
        except Exception:
            z_val = 0.0
        out.append((key, s_val, z_val))
    out.sort(key=lambda x: int(_EPOCH_COL_RE.match(x[0]).group(1)))
    return out


def resolve_template_phase_key(
    epoch_col: str, template_phase_keys: list[str]
) -> str | None:
    """
    Map a CSV column name (e.g. ``epoch_3``) to a template phase key.

    Multi-epoch templates store keys like ``epoch_0``, ``epoch_1``, ... so the
    column maps directly. Single-spectrum templates store the key ``single``;
    the matching code wrote those into the ``epoch_0`` column, so we fall back
    to ``single`` when ``epoch_0`` is requested but missing.
    """
    if epoch_col in template_phase_keys:
        return epoch_col
    if epoch_col == "epoch_0" and "single" in template_phase_keys:
        return "single"
    return None


def compare_template_vs_wiserep_from_csv(
    objname: str,
    csv_path: Path = SPREADSHEET_CSV,
    apply_scale: bool = True,
    *,
    save_dir: Path | None = COMPARE_FIG_DIR,
    show: bool = True,
    ncols_pairs: int = 2,
) -> dict[str, dict]:
    """
    Read ``csv_path``, take the row for ``objname``, and for every filled
    ``epoch_N`` column draw two panels side-by-side:

    * **Full spectrum panel** – the entire valid range of the template and
        the SAGE-preprocessed WISeREP spectrum, with no edge crop and no
        overlap intersection.
    * **Comparison panel** – ``check_spectra_consistency`` output (overlap
        region only) with Pearson r / NRMSE in the title.

    ``ncols_pairs`` controls how many *phase pairs* sit on one figure row;
    the actual number of subplot columns is ``ncols_pairs * 2``.

    Returns the per-epoch consistency dict from ``check_spectra_consistency``.
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    sub = df[df["Objname"].astype(str).str.strip() == objname]
    if sub.empty:
        raise ValueError(f"{objname!r} not found in {csv_path}")
    row_dict = sub.iloc[0].to_dict()

    matched = read_matched_epochs_from_row(row_dict)
    if not matched:
        raise ValueError(f"No filled epochs for {objname!r} in {csv_path.name}")

    wl_phase_dict, flux_phase_dict, template_meta = load_template_spectrum(objname)
    template_meta_epoch = template_meta["epoch"]
    template_phase_keys = list(template_meta_epoch.keys())

    wl_wiserep_dict, flux_wiserep_dict, fluxerr_wiserep_dict, _ = load_wiserep_spectrum(
        objname
    )

    n = len(matched)
    nrows = math.ceil(n / ncols_pairs)
    fig, axes = plt.subplots(
        nrows,
        ncols_pairs * 2,
        figsize=(6.0 * ncols_pairs * 2, 3.4 * nrows),
        squeeze=False,
    )

    results: dict[str, dict] = {}
    for idx, (epoch_col, obsdate, z_input) in enumerate(matched):
        row_idx = idx // ncols_pairs
        pair_idx = idx % ncols_pairs
        ax_full = axes[row_idx, pair_idx * 2]
        ax_compare = axes[row_idx, pair_idx * 2 + 1]

        try:
            phase_key = resolve_template_phase_key(epoch_col, template_phase_keys)
            if phase_key is None:
                _annotate_check_axis(
                    ax_full, f"{epoch_col}: phase missing in template", title=epoch_col
                )
                ax_compare.set_axis_off()
                continue

            tmeta = template_meta_epoch[phase_key]
            template_age = float(tmeta["age"]) if tmeta.get("age") is not None else None

            wl_template = wl_phase_dict[phase_key]
            flux_template = flux_phase_dict[phase_key]
            tmask = (wl_template > tmeta["wl_min"]) & (wl_template < tmeta["wl_max"])
            wl_template = wl_template[tmask]
            flux_template = flux_template[tmask]

            if obsdate not in wl_wiserep_dict:
                _annotate_check_axis(
                    ax_full,
                    f"{epoch_col}: obsdate {obsdate!r} not in WISeREP cache",
                    title=epoch_col,
                )
                ax_compare.set_axis_off()
                continue

            wl_wiserep = wl_wiserep_dict[obsdate]
            flux_wiserep = flux_wiserep_dict[obsdate]

            wiserep_proc = preprocess_spectrum_sage_like(
                wl_wiserep, flux_wiserep, z=z_input
            )
            vmask = wiserep_proc["valid_mask"]
            wl_wiserep_template = wiserep_proc["wave_log"][vmask]
            flux_wiserep_template = wiserep_proc["norm_flux"][vmask] - 1.0

            mjp = None
            if template_age is not None:
                mjp = _obsdate_to_mjd(obsdate) - template_age

            template_label = f"template[{template_age}({phase_key})]"
            wiserep_label = f"wiserep[{obsdate}, z={z_input}]"

            line_template, = ax_full.plot(
                wl_template, flux_template, color="C0", alpha=0.8, label=template_label
            )
            line_wiserep, = ax_full.plot(
                wl_wiserep_template,
                flux_wiserep_template + 0.2 * np.max(np.abs(flux_wiserep_template)),
                color="C1",
                alpha=0.8,
                label=wiserep_label,
            )
            ax_full.axhline(0.0, color="k", lw=0.5, alpha=0.3)
            ax_full.set_xlabel("Rest wavelength [Å]")
            ax_full.set_ylabel("flat flux (norm - 1)")

            # Original (un-preprocessed) WISeREP spectrum on a twin y-axis.
            # De-redshifted with the same z used for SAGE preprocessing so the
            # wavelength axis stays consistent with the flat-flux curves.
            ax_full_raw = ax_full.twinx()
            wl_wiserep_rest = np.asarray(wl_wiserep, dtype=float) / (1.0 + float(z_input))
            line_raw, = ax_full_raw.plot(
                wl_wiserep_rest,
                flux_wiserep,
                color="k",
                alpha=0.3,
                lw=0.8,
                label="wiserep raw",
            )
            ax_full_raw.set_ylabel("raw flux", color="k", alpha=0.7)
            ax_full_raw.tick_params(axis="y", colors="k", labelsize=7)

            title_bits = [f"{epoch_col} full"]
            if mjp is not None:
                title_bits.append(f"mjp={mjp:.2f}")
            ax_full.set_title(", ".join(title_bits), fontsize=9)
            ax_full.legend(
                handles=[line_template, line_wiserep, line_raw],
                fontsize=7,
                loc="best",
            )

            res = check_spectra_consistency(
                wl_template,
                flux_template,
                wl_wiserep_template,
                flux_wiserep_template,
                label1=template_label,
                label2=wiserep_label,
                plot=True,
                plot_only_consistent=False,
                apply_scale = apply_scale,
                ax=ax_compare,
                mjp=mjp,
            )
            results[epoch_col] = res
        except Exception as exc:
            _annotate_check_axis(ax_full, f"{epoch_col}: {exc}", title=epoch_col)
            ax_compare.set_axis_off()

    for empty_idx in range(n, nrows * ncols_pairs):
        row_idx = empty_idx // ncols_pairs
        pair_idx = empty_idx % ncols_pairs
        axes[row_idx, pair_idx * 2].set_axis_off()
        axes[row_idx, pair_idx * 2 + 1].set_axis_off()

    fig.suptitle(
        f"{objname}: template vs WISeREP (SAGE-preprocessed) from spreadsheet.csv",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{_safe_plot_stem(objname)}_csv_compare.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison figure: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return results



# %% Run for one object (change ``objname`` as needed)

import gspread
gc = gspread.service_account(filename="/home/hhchoi1022/code/SNID-SAGE/googlesheet.json")
sh = gc.open_by_url("https://docs.google.com/spreadsheets/d/1wwbBlAvOUamYM-l_F-N6xd4UYPEcKhntzLETNtP-m1o/edit?gid=0#gid=0")
ws = sh.get_worksheet(0)
rows = ws.get_all_records()
from astropy.table import Table
tbl = Table(rows)
tbl.write('spreadsheet.csv', format = 'csv', overwrite=True)
#%%
results_csv_compare = compare_template_vs_wiserep_from_csv(
    "PTF11kx",
    csv_path=SPREADSHEET_CSV,
    save_dir=COMPARE_FIG_DIR,
    show=True,
    apply_scale=True)
# %% Optional: batch-run over every object that has at least one filled epoch
df_all = pd.read_csv(SPREADSHEET_CSV, dtype=str, keep_default_na=False)
epoch_cols = [c for c in df_all.columns if _EPOCH_COL_RE.match(c)]
has_match = df_all[epoch_cols].apply(
    lambda s: s.astype(str).str.strip().ne("").any(), axis=1
)
for obj in df_all.loc[has_match, "Objname"].astype(str):
    try:
        compare_template_vs_wiserep_from_csv(
            obj, csv_path=SPREADSHEET_CSV, save_dir=COMPARE_FIG_DIR, show=False, apply_scale=True
        )
    except Exception as exc:
        print(f"[skip] {obj}: {exc}")
# %%
