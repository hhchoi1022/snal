
#%%
from astropy.io import ascii
path_imsng = './data/SN2021aefx_formatted_Host_dereddening_MW_dereddening.ascii_fixed_width'
path_h22 = './data/Hosseinzadeh2022_formatted_Host_dereddening_MW_dereddening.ascii_fixed_width'

tbl_imsng = ascii.read(path_imsng, format = 'fixed_width')
tbl_h22 = ascii.read(path_h22, format = 'fixed_width')
tbl_h22 = tbl_h22[tbl_h22['observatory'] == 'LasCumbres1m']
#%%
from ezphot.dataobjects import LightCurve
lc_imsng = LightCurve()
lc_imsng.data = tbl_imsng
lc_imsng.plt_params.xlim = [59500, 59730]
lc_imsng.plt_params.ylim = [20, 8]
lc_imsng.plt_params.figure_figsize = (12, 8)
lc_imsng.plot(ra = 64.9725, dec= -54.948081, flux_key = 'mag', fluxerr_key = 'e_mag')
#%%
lc_h22 = LightCurve()
lc_h22.data = tbl_h22
lc_h22.plt_params.xlim = [59500, 59730]
lc_h22.plt_params.ylim = [20, 8]
lc_h22.plt_params.figure_figsize = (12, 8)
lc_h22.plot(ra = 64.9725, dec= -54.948081, flux_key = 'mag', fluxerr_key = 'e_mag')
#%%
from astropy.table import vstack
lc_all = LightCurve()
lc_all.data = vstack([tbl_imsng, tbl_h22])
lc_all.plt_params.xlim = [59500, 59730]
lc_all.plt_params.ylim = [20, 8]
lc_all.plt_params.figure_figsize = (12, 8)
lc_all.plot(ra = 64.9725, dec= -54.948081, flux_key = 'mag', fluxerr_key = 'e_mag')

#%%# ---- model ----
def fireball_model(time, amplitude, exptime, alpha):
    dt = np.asarray(time) - exptime
    dt = np.clip(dt, 1e-6, None)  # avoid <=0
    flux = amplitude * (dt**alpha)
    return np.nan_to_num(flux, nan=1e-6, posinf=1e6, neginf=1e-6)

def band_model(params, time, b):
    exptime = params['exptime'].value
    amp     = params[f'amp_{b}'].value
    alpha   = params[f'alpha_{b}'].value
    return fireball_model(time, amp, exptime, alpha)

def residuals(params, x_list, y_list, e_list, band_list):
    res = []
    for i, b in enumerate(band_list):
        mod = band_model(params, x_list[i], b)
        res.append((y_list[i] - mod) / e_list[i])  # unsquared residuals
    return np.concatenate(res)

# --- ADD: build_upper_limits -------------------------------------------------
from snal.helper import AnalysisHelper
from scipy.stats import norm
import numpy as np

helper = AnalysisHelper()

def build_upper_limits(tbl_all, bands, tmin, tmax, nsigma_default=5.0,
                       mjd_window=None):
    """
    tbl_all: 원본 테이블(검출/비검출 모두)
    bands:   반영할 필터 목록 (['g','r','i',...])
    tmin,tmax: 피팅 구간
    nsigma_default: 리미트가 n-sigma일 때의 n (기본 5)
    mjd_window: (center, halfwidth)면 그 근처 상한만 사용. 없으면 구간 전체.
    return: [{mjd, filter, flux_lim, sigma_flux}, ...]
    """
    m = (tbl_all['detected'] == 'False') & np.isin(tbl_all['filter'], bands) & \
        (tbl_all['mjd'] > tmin) & (tbl_all['mjd'] < tmax)
    ul = tbl_all[m].copy()
    if len(ul) == 0:
        return []

    if mjd_window is not None:
        center, half = mjd_window
        ul = ul[np.abs(ul['mjd'] - center) <= half]
        if len(ul) == 0:
            return []

    # 등급 리미트 -> 플럭스 리미트
    ul['flux_lim'] = helper.mag_to_flux(ul['depth'])
    if 'nsigma' not in ul.colnames:
        ul['nsigma'] = nsigma_default
    ul['sigma_flux'] = ul['flux_lim'] / ul['nsigma']

    ul_list = []
    for row in ul:
        ul_list.append({
            'mjd': float(row['mjd']),
            'filter': row['filter'],
            'flux_lim': float(row['flux_lim']),
            'sigma_flux': float(row['sigma_flux']),
        })
    return ul_list

# --- ADD: residuals_with_upper_limits ---------------------------------------
from scipy.stats import norm
import numpy as np

def residuals_with_upper_limits(params,
                                x_list, y_list, e_list, band_list,
                                upper_limits,
                                model_fn):
    """
    x_list, y_list, e_list: 각 밴드별 1D numpy array의 리스트
    band_list: x/y/e와 동일한 순서의 밴드 문자열 리스트 (== bands)
    upper_limits: [{mjd, filter, flux_lim, sigma_flux}, ...]
    model_fn(params, time_array, band) -> model flux array (1D)
    """
    res = []

    # (1) detections: 밴드별 포인트
    for xi, yi, ei, bi in zip(x_list, y_list, e_list, band_list):
        xi = np.asarray(xi, float)
        yi = np.asarray(yi, float)
        ei = np.asarray(ei, float)
        if xi.size == 0:
            continue  # 빈 밴드 건너뛰기
        mod = model_fn(params, xi, bi)        # 1D
        res.append((yi - mod) / ei)           # 1D

    # (2) upper limits: 길이 1짜리 1D 배열로 추가
    if upper_limits:
        for ul in upper_limits:
            t_ul  = np.array([ul['mjd']], dtype=float)
            b_ul  = ul['filter']
            f_lim = float(ul['flux_lim'])
            sig1  = max(float(ul['sigma_flux']), 1e-12)

            mu  = model_fn(params, t_ul, b_ul)[0]   # 스칼라
            z   = (f_lim - mu) / sig1
            cdf = np.clip(norm.cdf(z), 1e-300, 1.0)
            r_ul = np.sqrt(-2.0*np.log(cdf))
            res.append(np.array([r_ul], dtype=float))  # <-- 1D로 감싸기

    return np.hstack(res).astype(float) if res else np.array([], dtype=float)


def band_model_fireball(params, time, b):
    exptime = params['exptime'].value
    amp     = params[f'amp_{b}'].value
    alpha   = params[f'alpha_{b}'].value
    return fireball_model(time, amp, exptime, alpha)

# --- ADD ONCE: PL+G band-model wrapper --------------------------------------
def band_model_plg(params, time, band):
    t0   = params['t0'].value
    tg   = params['tg'].value
    sigg = params['sigg'].value
    A    = params[f'A_{band}'].value
    alpha= params[f'alpha_{band}'].value
    G    = params[f'G_{band}'].value

    t = np.asarray(time, float)
    dt = np.clip(t - t0, 1e-6, None)
    pl = A * (dt ** alpha)
    ga = G * np.exp(-0.5 * ((t - tg) / sigg) ** 2)
    flux = pl + ga
    return np.nan_to_num(flux, nan=1e-6, posinf=1e6, neginf=1e-6)

# %%
def fit_fireball(tbl_fit,
                 start_mjd=59529,
                 end_mjd=None,
                 band='r',
                 ref_band=None,
                 half_on_rising=True):
    """
    Fireball (power-law) fit with shared exptime and per-band amplitude/alpha.
    band: str or list[str] of filter names to include (e.g., 'r' or ['g','r','i'])
    ref_band: which band to use to define the half-flux time when end_mjd is None.
              Defaults to 'r' if available, else the first band in `band`.
    """
    import numpy as np
    from lmfit import Parameters, minimize
    from snal.helper import AnalysisHelper

    # ---- normalize band input ----
    if isinstance(band, str):
        bands = [band]
    else:
        bands = list(band)
    bands = [b for b in bands if b in set(tbl_fit['filter'])]
    if not bands:
        raise ValueError("No requested bands are present in the table.")

    if ref_band is None:
        ref_band = 'r' if 'r' in bands else bands[0]
    elif ref_band not in bands:
        raise ValueError(f"ref_band {ref_band} must be one of {bands}.")

    # ---- prepare table & fluxes (once) ----
    helper = AnalysisHelper()
    tbl = tbl_fit[(tbl_fit['detected'] == 'True') & np.isin(tbl_fit['filter'], bands)].copy()
    tbl['flux']   = helper.mag_to_flux(tbl['mag'])
    tbl['e_flux'] = tbl['e_mag'] * tbl['flux'] * (np.log(10.0)/2.5)  # 0.921...

    # ---- decide end_mjd via half-peak on ref_band (if needed) ----
    tmin = float(start_mjd)
    if end_mjd is None:
        tmax_probe = tmin + 20.0
        sub = tbl[(tbl['filter'] == ref_band) & (tbl['mjd'] > tmin) & (tbl['mjd'] < tmax_probe)].copy()
        sub.sort('mjd')
        t = np.asarray(sub['mjd'])
        f = np.asarray(sub['flux'])
        m = np.isfinite(t) & np.isfinite(f)
        t, f = t[m], f[m]
        if len(f) < 3:
            raise ValueError("Not enough points in ref_band to determine half-peak time.")
        imax  = np.nanargmax(f)
        fmax  = f[imax]
        fhalf = 0.5 * fmax
        if half_on_rising:
            t_rise = t[:imax+1]; f_rise = f[:imax+1]
        else:
            t_rise = t;          f_rise = f
        srt = np.argsort(f_rise)
        t_half = np.interp(fhalf, f_rise[srt], t_rise[srt])
        tmax = float(t_half)
    else:
        tmax = float(end_mjd)

    # ---- final slice for all bands ----
    use = (tbl['mjd'] > tmin) & (tbl['mjd'] < tmax) & np.isin(tbl['filter'], bands)
    tbl_use = tbl[use].copy()
    tbl_use.sort('mjd')
    if len(tbl_use) < 3:
        raise ValueError("Not enough points in the fitting window across selected bands.")

    # build {band: table} safely
    fit_table = {b: tbl_use[tbl_use['filter'] == b] for b in bands}
    # x/y/e lists aligned with bands list order
    x_fit = [np.asarray(fit_table[b]['mjd'])  for b in bands]
    y_fit = [np.asarray(fit_table[b]['flux']) for b in bands]
    e_fit = [np.asarray(fit_table[b]['e_flux']) for b in bands]

    # ---- parameters & fit ----
    from lmfit import fit_report

    p = Parameters()
    # exptime window near start
    p.add('exptime', value=tmin, min=59525, max=59535)
    for b in bands:
        guess_amp = float(np.nanmax(y_fit[bands.index(b)]) if len(y_fit[bands.index(b)]) else 200.0)
        p.add(f'amp_{b}',   value=max(guess_amp, 1.0), min=0.0, max=1e7)
        p.add(f'alpha_{b}', value=2.0, min=0.5, max=6.0)
        
    # upper_limits = build_upper_limits(tbl_fit, bands, tmin-5, tmin+5,
    #                                 nsigma_default=5.0)

    # out = minimize(residuals_with_upper_limits, p,
    #             args=(x_fit, y_fit, e_fit, bands, upper_limits, band_model_fireball),
    #             method='leastsq', max_nfev=30000)

    out = minimize(residuals, p, args=(x_fit, y_fit, e_fit, bands),
                   method='leastsq', max_nfev=30000)
    out.extra = {
        'fit_window': (tmin, tmax),
        'bands': bands,
        'ref_band': ref_band,
        'npts': int(len(tbl_use))
    }
    # quick print if you like:
    # print(fit_report(out))
    return out
# %%
from astropy.table import vstack
table_fit = vstack([tbl_imsng])
filter_key = 'gr'
filter_key = list(filter_key)
outlist = []
start_mjd = 59528
for i in range(7):
    start_mjd += 1
    print(start_mjd)
    out = fit_fireball(table_fit, start_mjd=start_mjd, end_mjd=59538.27207782408, band=filter_key)
    outlist.append(out)

# ---- final slice for all bands ---- (그 아래 부분은 그대로)
# use / tbl_use / x_fit / y_fit / e_fit 생성까지 기존대로 수행
#%%
import numpy as np
import matplotlib.pyplot as plt
chisq_list = [o.redchi for o in outlist]
plt.scatter(np.arange(len(outlist)), chisq_list)
plt.xlabel('Start MJD offset')
plt.ylabel('Reduced $\\chi^2$')
plt.ylim(0, 10)
plt.show()

# %%
from ezphot.dataobjects import LightCurve
lc_fit = LightCurve()
filter_idx = [filter in 'gr' for filter in table_fit['filter']]
table_fit = table_fit[filter_idx]
lc_fit.data = table_fit
lc_fit.plt_params.xlim = [59527, 59537.27207782408]
lc_fit.plt_params.ylim = [22, 10]
lc_fit.plt_params.figure_figsize = (6, 8)
fig, ax, _ = lc_fit.plot(ra = 64.9725, dec= -54.948081, flux_key = 'mag', fluxerr_key = 'e_mag', title = '')
# lc_all.plt_params.xlim = [59527, 59540]
# lc_all.plt_params.ylim = [20, 10]
# lc_all.plt_params.figure_figsize = (6, 8)
# fig, ax, _ = lc_all.plot(ra = 64.9725, dec= -54.948081, flux_key = 'mag', fluxerr_key = 'e_mag')

#%%
import matplotlib.lines as mlines

linestyles = ['-', '--', ':']
handles = []
label_title = r'$\ F \propto (t-t_0)^{\alpha}$'
start_mjd = 59528

for i, (out, ls) in enumerate(zip(outlist, linestyles)):
    result_values = out.params.valuesdict()
    
    # ? Correct f-string syntax for math text
    label = (
        rf'$\alpha_g = {result_values["alpha_g"]:.2f}, '
        rf'\ \alpha_r = {result_values["alpha_r"]:.2f}$'
    )

    phase_min_FB = np.max([59526, result_values['exptime']])
    phase_range_FB = np.arange(phase_min_FB, 59538.27207782408, 0.1)
    start_mjd += 1

    ax.axvline(x = start_mjd, color = 'red', linestyle = ls)

    for filter_ in filter_key:
        exptime = out.params['exptime']
        amp = out.params[f'amp_{filter_}']
        alpha = out.params[f'alpha_{filter_}']
        flux_model = fireball_model(phase_range_FB, amp.value, exptime.value, alpha.value)
        mag_model = helper.flux_to_mag(flux_model)

        ax.plot(
            phase_range_FB,
            mag_model + lc_fit.FILTER_OFFSET[filter_],
            c=lc_fit.FILTER_COLOR[filter_],
            linestyle=ls,
            label=None,
        )

    # Add legend handle (black line for style)
    line_handle = mlines.Line2D([], [], color='black', linestyle=ls, label=label)
    handles.append(line_handle)

# ? Add legend with title
ax.legend(handles=handles, loc=2, title=label_title, fontsize=11, title_fontsize=12)


# %%
fig
#%%
from lmfit import Parameters, minimize

# def fit_powerlaw_gaussian(tbl_all,
#                           bands=('g','r','i'),
#                           start_mjd=59529,
#                           end_mjd=None,
#                           detected_key='detected',
#                           flux_from_mag=True,
#                           use_ul=True,
#                           ul_mjd_window=(59528.3, 0.05),   # ← 상한 창 (원하면 None)
#                           nsigma_default=5.0):
#     import numpy as np
#     from snal.helper import AnalysisHelper
#     helper = AnalysisHelper()

#     # ----- data slice -----
#     tbl = tbl_all.copy()
#     if detected_key in tbl.colnames:
#         tbl = tbl[tbl[detected_key] == 'True']
#     bands = [b for b in bands if b in set(tbl['filter'])]
#     if not bands:
#         raise ValueError("No requested bands exist in the table.")

#     if flux_from_mag:
#         if 'flux' not in tbl.colnames:
#             tbl['flux'] = helper.mag_to_flux(tbl['mag'])
#         if 'e_flux' not in tbl.colnames:
#             tbl['e_flux'] = (np.log(10.0)/2.5) * tbl['flux'] * tbl['e_mag']

#     tmin = float(start_mjd)
#     tmax = float(start_mjd + 20.0) if end_mjd is None else float(end_mjd)

#     use = (tbl['mjd'] > tmin) & (tbl['mjd'] < tmax) & np.isin(tbl['filter'], bands)
#     use = tbl[use].copy(); use.sort('mjd')
#     if len(use) < 5:
#         raise ValueError("Not enough points in the chosen window.")

#     fit_table = {b: use[use['filter'] == b] for b in bands}
#     x_list = [np.asarray(fit_table[b]['mjd'])  for b in bands]
#     y_list = [np.asarray(fit_table[b]['flux']) for b in bands]
#     e_list = [np.asarray(fit_table[b]['e_flux']) for b in bands]

#     # ----- params -----
#     p = Parameters()
#     # 안정적 수렴을 위해 tg를 t0보다 크도록 reparam도 가능: dtg>=0, tg = t0 + dtg
#     # (아래는 간단하게 bound로 처리)
#     p.add('t0',   value=tmin,     min=tmin-3.0, max=tmin+3.0)
#     p.add('tg',   value=tmin+1.5, min=tmin-0.5, max=tmin+10.0)
#     p.add('sigg', value=1.0,      min=0.05,     max=5.0)

#     for b in bands:
#         yb = y_list[bands.index(b)]
#         yb_max = float(np.nanmax(yb)) if len(yb) else 100.0
#         p.add(f'A_{b}',     value=max(0.1*yb_max, 1.0), min=0.0,  max=1e8)
#         p.add(f'alpha_{b}', value=2.0,                 min=0.5,  max=6.0)
#         p.add(f'G_{b}',     value=max(0.1*yb_max, 0.0), min=0.0,  max=1e8)

#     # ----- upper-limits -----
#     upper_limits = []
#     if use_ul:
#         upper_limits = build_upper_limits(tbl_all, bands, tmin-5, tmax+5,
#                                           nsigma_default=nsigma_default)
#         # 디버그: 실제로 들어갔는지 확인
#         print(f"[UL] N_ul = {len(upper_limits)}")

#     # ----- fit (교체 1줄) -----
#     out = minimize(residuals_with_upper_limits, p,
#                    args=(x_list, y_list, e_list, bands, upper_limits, band_model_plg),
#                    method='leastsq', max_nfev=50000)

#     out.extra = {'fit_window': (tmin, tmax), 'bands': bands, 'npts': int(len(use))}
#     return out

def fit_powerlaw_gaussian(tbl_all,
                          bands=('g','r','i'),      # 사용 밴드
                          start_mjd=59529,
                          end_mjd=None,             # None이면 start+20일
                          detected_key='detected',  # 'True' 문자열 사용중
                          flux_from_mag=True):
    """
    Power-law + Gaussian bump (multi-band, shared t0, shared tg,sig_g).
    Returns: lmfit.MinimizerResult (out.extra에 메타 포함)
    """
    import numpy as np
    from lmfit import Parameters, minimize
    from snal.helper import AnalysisHelper

    # -------- 데이터 준비 --------
    tbl = tbl_all.copy()
    if detected_key in tbl.colnames:
        tbl = tbl[tbl[detected_key] == 'True']
    bands = [b for b in bands if b in set(tbl['filter'])]
    if not bands:
        raise ValueError("No requested bands exist in the table.")

    helper = AnalysisHelper()
    if flux_from_mag:
        # flux, e_flux 생성 (이미 있으면 건너뛰어도 됨)
        if 'flux' not in tbl.colnames:
            tbl['flux'] = helper.mag_to_flux(tbl['mag'])
        if 'e_flux' not in tbl.colnames:
            # dF/F = ln(10)/2.5 * dmag
            tbl['e_flux'] = (np.log(10.0)/2.5) * tbl['flux'] * tbl['e_mag']

    tmin = float(start_mjd)
    tmax = float(start_mjd + 20.0) if end_mjd is None else float(end_mjd)

    sel = (tbl['mjd'] > tmin) & (tbl['mjd'] < tmax) & np.isin(tbl['filter'], bands)
    use = tbl[sel].copy()
    use.sort('mjd')
    if len(use) < 5:
        raise ValueError("Not enough points in the chosen window.")

    # 밴드별로 분할
    fit_table = {b: use[use['filter'] == b] for b in bands}
    x_list = [np.asarray(fit_table[b]['mjd'])  for b in bands]
    y_list = [np.asarray(fit_table[b]['flux']) for b in bands]
    e_list = [np.asarray(fit_table[b]['e_flux']) for b in bands]

    # -------- 모델 --------
    def plg_model(time, A, t0, alpha, G, tg, sigg):
        dt = np.asarray(time) - t0
        dt = np.clip(dt, 1e-6, None)
        pl = A * (dt ** alpha)
        ga = G * np.exp(-0.5 * ((np.asarray(time) - tg) / sigg) ** 2)
        flux = pl + ga
        return np.nan_to_num(flux, nan=1e-6, posinf=1e6, neginf=1e-6)

    def band_model(params, time, band):
        t0   = params['t0'].value
        tg   = params['tg'].value
        sigg = params['sigg'].value
        A    = params[f'A_{band}'].value
        alpha= params[f'alpha_{band}'].value
        G    = params[f'G_{band}'].value
        return plg_model(time, A, t0, alpha, G, tg, sigg)

    def residuals(params, x_list, y_list, e_list, band_list):
        res = []
        for i, b in enumerate(band_list):
            mod = band_model(params, x_list[i], b)
            res.append( (y_list[i] - mod) / e_list[i] )   # ← 제곱하지 않음
        return np.concatenate(res)

    # -------- 파라미터 초기값 --------
    p = Parameters()
    # 공통: t0는 창 시작 근처, tg는 대략 1~3일 후, sigg는 ~0.3~3d
    p.add('t0',   value=tmin, min=tmin-3.0, max=tmin+3.0)
    p.add('tg',   value=tmin+1.5, min=tmin-0.5, max=tmin+10.0)
    p.add('sigg', value=1.0, min=0.05, max=5.0)

    for b in bands:
        yb = y_list[bands.index(b)]
        # 대략적인 스케일: 최대 플럭스 기준
        yb_max = float(np.nanmax(yb)) if len(yb) else 100.0
        p.add(f'A_{b}',     value=max(0.1*yb_max, 1.0), min=0.0,  max=1e8)
        p.add(f'alpha_{b}', value=2.0,                 min=0.5,  max=6.0)
        p.add(f'G_{b}',     value=max(0.1*yb_max, 0.0), min=0.0,  max=1e8)

    # -------- 적합 --------
    out = minimize(residuals, p, args=(x_list, y_list, e_list, bands),
                   method='leastsq', max_nfev=50000)

    out.extra = {
        'fit_window': (tmin, tmax),
        'bands': bands,
        'npts': int(len(use))
    }
    return out

# %%
# 단일 밴드
table_fit = vstack([tbl_imsng])
filter_key = 'gr'
filter_key = list(filter_key)

# 다밴드 동시 (t0, tg, sigg 공유; 밴드별 A, alpha, G)
out_with_first = fit_powerlaw_gaussian(table_fit, bands=filter_key, start_mjd=59529, end_mjd=59538.27207782408)
out_without_first = fit_powerlaw_gaussian(table_fit, bands=filter_key, start_mjd=59529.5, end_mjd=59538.27207782408)
from lmfit import fit_report
print(fit_report(out_with_first))
print(fit_report(out_without_first))
# %%

from ezphot.dataobjects import LightCurve
lc_fit = LightCurve()
lc_fit.data = table_fit
lc_fit.plt_params.xlim = [59527, 59538]
lc_fit.plt_params.ylim = [20, 10]
lc_fit.plt_params.figure_figsize = (6, 8)
fig, ax, _ = lc_fit.plot(ra = 64.9725, dec= -54.948081, flux_key = 'mag', fluxerr_key = 'e_mag')
# lc_all.plt_params.xlim = [59527, 59540]
# lc_all.plt_params.ylim = [20, 10]
# lc_all.plt_params.figure_figsize = (6, 8)
# fig, ax, _ = lc_all.plot(ra = 64.9725, dec= -54.948081, flux_key = 'mag', fluxerr_key = 'e_mag')

# 예: r,g,i 결과 플럭스를 마그니튜드로 변환해 오버레이
t0 = out_with_first.params['t0'].value
tg = out_with_first.params['tg'].value
sg = out_with_first.params['sigg'].value

phase = np.arange(59527.5, out_with_first.extra['fit_window'][1], 0.05)

for b in out_with_first.extra['bands']:
    A  = out_with_first.params[f'A_{b}'].value
    al = out_with_first.params[f'alpha_{b}'].value
    G  = out_with_first.params[f'G_{b}'].value
    flux_model = (A*np.clip(phase - t0, 1e-6, None)**al) + G*np.exp(-0.5*((phase - tg)/sg)**2)
    mag_model  = helper.flux_to_mag(flux_model)
    ax.plot(phase, mag_model + lc_fit.FILTER_OFFSET[b],
            c=lc_fit.FILTER_COLOR[b], label=f'{b}: α={al:.2f}')

# t0_without_first = out_without_first.params['t0'].value
# tg_without_first = out_without_first.params['tg'].value
# sg_without_first = out_without_first.params['sigg'].value
# phase_without_first = np.arange(59527.5, out_without_first.extra['fit_window'][1], 0.05)
# for b in out_without_first.extra['bands']:
#     A  = out_without_first.params[f'A_{b}'].value
#     al = out_without_first.params[f'alpha_{b}'].value
#     G  = out_without_first.params[f'G_{b}'].value
#     flux_model = (A*np.clip(phase_without_first - t0_without_first, 1e-6, None)**al) + G*np.exp(-0.5*((phase_without_first - tg_without_first)/sg_without_first)**2)
#     mag_model  = helper.flux_to_mag(flux_model)
#     ax.plot(phase_without_first, mag_model + lc_fit.FILTER_OFFSET[b],
#             c=lc_fit.FILTER_COLOR[b], label=f'{b}: α={al:.2f}', linestyle='--')

ax.legend(loc=2)
fig
