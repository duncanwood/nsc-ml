from collections.abc import Iterable
import gc
import pickle
import time
import os
import re
import shutil
import warnings
import inspect

import pandas as pd
import numpy as np
from scipy.stats import distributions
from scipy import sparse, optimize
from numba import njit



import tqdm

# Explicit public API so `from .nsctools import *` (in __init__) does not leak
# the imported np/pd/os/re/etc. into the package namespace.
__all__ = [
    'color_filter', 'marker_map', 'magstr', 'WS_INTERVAL_DAYS',
    'WS_MAX_REVISIT_DAYS', 'WS_SEQ_LEN', 'OUTLIERS_CUTOFF',
    'OUTLIERS_CUTOFF_DATA', 'Z_THRESHOLD', 'DETECTION_TIMESCALE_DAYS',
    'N_MEASURED', 'DURATION_DAYS', 'N_MIN_OUTSIDE_FIT', 'N_KS_GAUSSIAN',
    'CONTEXT_SIZE_DAYS', 'CROSSING_TIME_GUESS_DAYS', 'TEMPER_ERRORS_FIT',
    'FIT_TIME_PAD_DAYS', 'COND_LIM', 'make_delta_mags_mono', 'make_delta_mags',
    'make_instrument', 'get_default_args', 'convert_to_range_index',
    'well_sampled_region', 'get_well_sampled_objects',
    'get_just_well_sampled_objects', 'microlensing_amplification', 'ml_jac',
    'amp_to_mag', 'synth_objid', 'add_microlensing_event', 'ml_f',
    'generate_synthetic_microlensing_events_from_population', 'ks_weighted',
    'reject_low_error_outliers_args', 'reject_outliers_args', 'reject_outliers',
    'sparse_gaussian_wma', 'sparse_gaussian_wms', 'sparse_gaussian_window_iter',
    'sparse_gaussian_window', 'dense_sparse_gaussian_window', 'gaussian_window',
    'clipped_gaussian_window', 'weighted_avg_and_std',
    'weighted_moving_average', 'compute_weighted_moving_average',
    'weighted_moving_average_gaussian',
    'weighted_moving_average_sparse_gaussian', 'weighted_moving_average_err',
    'weighted_moving_average_scatter', 'weighted_moving_average_df',
    'float_cols_to_double', 'strip_objid', 'find_persistent_excursions',
    'search_files_for_excursions', 'consolidate_search_files_for_excursions',
    'reduce_excursions', 'get_nondetections', 'compute_file_map', 'extend_lc',
    'fit_excursions', 'make_fit_excursions_df', 'search_for_params',
    'common_params', 'default_args_of_functions',
    'search_files_for_microlensing_events', 'cut_by_npoints', 'cut_by_pval',
    'cut_high_points_low_p', 'cut_high_points_inout_low_p', 'cut_pcov',
    'cut_crossing_time', 'split_real_synth_df',
]

# --- Science-bearing default parameters -------------------------------------
# The magic numbers from the function signatures below, gathered in one place.
# The numba kernels keep their own primitive defaults (timescale=2, nclip=10);
# the detector deliberately uses DETECTION_TIMESCALE_DAYS instead.

# Well-sampled region selection (well_sampled_region and its mappers):
WS_INTERVAL_DAYS = 50.          # min baseline a region must span
WS_MAX_REVISIT_DAYS = 10        # max gap between consecutive epochs in a region
WS_SEQ_LEN = 5                  # min epochs in a region

# Persistent-excursion detector (find_persistent_excursions):
OUTLIERS_CUTOFF = 3             # MAD sigma for the baseline-scatter estimate
OUTLIERS_CUTOFF_DATA = 20       # MAD sigma for the optional low-error pre-cut
Z_THRESHOLD = 3                 # excursion significance threshold (sigma)
DETECTION_TIMESCALE_DAYS = 5    # WMA smoothing scale used in detection
N_MEASURED = 4                  # min epochs in an excursion
DURATION_DAYS = 5               # min time span of an excursion (days)

# PSPL fitting (fit_excursions):
N_MIN_OUTSIDE_FIT = 10          # below this, KS uses a synthetic Gaussian reference
N_KS_GAUSSIAN = 10000           # synthetic-reference sample size
CONTEXT_SIZE_DAYS = 100         # padding around an excursion for the fit window
CROSSING_TIME_GUESS_DAYS = 40   # initial Einstein crossing time (days)
TEMPER_ERRORS_FIT = 1           # error-tempering factor applied during fitting
FIT_TIME_PAD_DAYS = 365 * 10    # t0/tE curve_fit bound padding (10 yr)

# Post-fit selection cuts:
COND_LIM = 10**5                # max fit-covariance condition number (cut_pcov)

color_filter = {
    'u':'blue',
    'g':'green',
    'vr': 'yellow',
    'r':'orange',
    'i':'red',
    'z':'brown',
    'y':'black',
    'Y':'black'}

marker_map = {
    'c4d':'x',
    'k4m':'o',
    'ksb':'s',
    'tu1':'_', 'tu2':'_', 'tu3':'_', 'tu4':'_', 'tu':'_'
}

magstr = ", ".join([f'o.{_}mag, o.{_}err' for _ in 'ugrizy']) + ', o.vrmag, o.vrerr'

def make_delta_mags_mono(df: pd.DataFrame):
    newdf = df.copy()
    for f in newdf['filter'].unique():
        f_df = newdf[newdf['filter']==f]
        newdf.loc[f_df.index, 'deltamag'] = f_df['mag_auto']-f_df[f.lower()+'mag']
        newdf.update(f_df)
    return newdf

def make_delta_mags(df: pd.DataFrame, objectinfo: dict):
    newdf = df.copy()
    for f in newdf['filter'].unique():
        f_df = newdf[newdf['filter']==f]
        newdf.loc[f_df.index, 'deltamag'] = f_df['mag_auto']-objectinfo[f.lower()+'mag']
        newdf.update(f_df)
    return newdf

def make_instrument(df: pd.DataFrame):
    
    instrument_list = []
    for expid in df['exposure']:
        if expid[:2] == 'tu':
            instrument_list.append(expid[:2])
        else:
            instrument_list.append(expid[:3])
    
    newdf = df.copy()
    
    newdf.loc[:,'instrument'] = instrument_list
    return newdf

def get_default_args(func):
    """
    From https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }



def convert_to_range_index(idxs):
    if (np.diff(idxs)==1).all():
        return pd.RangeIndex(idxs[0], idxs[-1] + 1)
    else:
        return idxs

def well_sampled_region(df: pd.DataFrame, interval=WS_INTERVAL_DAYS,
                        maxrevisit=WS_MAX_REVISIT_DAYS, seqlen=WS_SEQ_LEN):
    df = df.sort_values('mjd')
    times = df['mjd'].to_numpy()
    valid_regions = []
    for region in np.split(df.index, np.where(np.diff(times) > maxrevisit)[0]+1):
        if (region.shape[0] >= seqlen):
            start, end = df.loc[[region[0], region[-1]], 'mjd']
            if end-start > interval:
                valid_regions.append(convert_to_range_index(region))
    return valid_regions

def get_well_sampled_objects(df, progress=False):
    gb= df.groupby('objectid', observed=True)
    well_sampled_objects = {}
    for obj in tqdm.tqdm(list(gb.groups.keys()), disable=(not progress)):
        objdf = gb.get_group(obj)
        regions = well_sampled_region(objdf, interval=WS_INTERVAL_DAYS,
                                       maxrevisit=WS_MAX_REVISIT_DAYS, seqlen=WS_SEQ_LEN)
        well_sampled_objects[obj] = regions
    return well_sampled_objects

def get_just_well_sampled_objects(df, progress=False):
    gb= df.groupby('objectid')
    well_sampled_objects = {}
    for obj in tqdm.tqdm(list(gb.groups.keys()), disable=(not progress)):
        objdf = gb.get_group(obj)
        regions = well_sampled_region(objdf, interval=WS_INTERVAL_DAYS,
                                       maxrevisit=WS_MAX_REVISIT_DAYS, seqlen=WS_SEQ_LEN)
        if len(regions) > 0:
            well_sampled_objects[obj] = regions
    return well_sampled_objects

@njit
def microlensing_amplification(t, impact_parameter=1, crossing_time=40.0,
                               peak_time=100, blending_factor=1):
    """The microlensing amplification

    Parameters
    ----------
    t : `float`
        The time of observation (days)
    impact_parameter : `float`
        The impact paramter (0 means big amplification)
    crossing_time : `float`
        Einstein crossing time (days)
    peak_time : `float`
        The peak time (days)
    blending_factor: `float`
        The blending factor where 1 is unblended
    """
    # Point-source point-lens (Paczynski) magnification A(u); u is the
    # source-lens separation in Einstein radii. blending_factor f blends the
    # event with constant light: A_obs = f*A + (1 - f).
    lightcurve_u = np.sqrt(impact_parameter**2 
                           + ((t - peak_time) ** 2 / crossing_time**2))
    amplified_mag = (lightcurve_u**2 + 2) / (
        lightcurve_u * np.sqrt(lightcurve_u**2 + 4)
    ) * blending_factor + (1 - blending_factor)

    return amplified_mag

@njit
def ml_jac(t, impact_parameter, crossing_time, peak_time):
    # Analytic Jacobian d(ml_f)/d(impact_parameter, crossing_time, peak_time),
    # columns in curve_fit p0 order. Derivation in the dissertation.
    denominator = (peak_time**2 - 2*peak_time*t + t**2
                   + crossing_time**2 * impact_parameter**2) * \
                   (peak_time**2 - 2*peak_time*t + t**2 
                    + crossing_time**2 * (2+impact_parameter**2)) * \
                  (peak_time**2 - 2*peak_time*t + t**2 
                   + crossing_time**2 * (4+impact_parameter**2)) * np.log(10)
    d0,d1,d2 = (20*crossing_time**6 * impact_parameter * np.ones_like(t),
                    -20*(peak_time-t)**2 * crossing_time**3,
                    20 * (peak_time-t) * crossing_time**4)
    jac = np.zeros(shape=(t.shape[0], 3))
    jac[:,0] = d0/denominator
    jac[:,1] = d1/denominator
    jac[:,2] = d2/denominator

    return jac


@njit
def amp_to_mag(amp):
    return -2.5*np.log10(amp)

def synth_objid(objid, lensing_params):
    return (str(objid) 
               + f"_ml_{lensing_params['peak_time']:.2f}"
               + f"_{lensing_params['crossing_time']:.2f}"
               + f"_{lensing_params['impact_parameter']:.5f}")

def add_microlensing_event(df: pd.DataFrame, **lensing_params):
    """
    Return a copy of input dataframe with a synthesized microlensing event 
    superimposed on the curve, with params given in 'lensing_params'
    """
    lc = df.copy()
    mag_diffs = amp_to_mag(microlensing_amplification(lc['mjd'].to_numpy(), **lensing_params))
    lc['mag_auto'] = (lc['mag_auto'] + mag_diffs).astype(lc.dtypes['mag_auto'])
    lc['deltamag'] = (lc['deltamag'] + mag_diffs).astype(lc.dtypes['deltamag'])

    newobjid = synth_objid(lc.iloc[0]['objectid'], lensing_params)
    lc['originalid'] = lc['objectid']
    lc['objectid'] = newobjid
    
    return lc

@njit
def ml_f(*x):
    return amp_to_mag(microlensing_amplification(*x))

def generate_synthetic_microlensing_events_from_population(
        lcfiles, events_file, ws_regions, outdir, outname, rng=None):
    # rng: pass a seeded numpy Generator for reproducible event/region draws;
    # defaults to a fresh (entropy-seeded) Generator.
    if rng is None:
        rng = np.random.default_rng()

    if isinstance(events_file, str):
        events_df = pd.read_pickle(events_file)
    elif isinstance(events_file,pd.DataFrame):
        events_df = events_file
    else:
        raise ValueError(f'Unsupported type for events_file: {type(events_file)}')

    outsubdir = os.path.join(outdir, f'synth-{outname}')
    os.makedirs(outsubdir, exist_ok=True)
    outinfo = {'lcfiles': lcfiles, 
                 'events_file': events_file, 
                 'outdir': outdir,
                 'outname': outname,
                 'outsubdir': outsubdir}
    object_event_list = []

    for file in tqdm.tqdm(lcfiles):
        filename = os.path.basename(file)
        synthfile = '.'.join(filename.split('.')[:-1]) + f'-synth-{outname}.parquet'
        
        df = pd.read_parquet(file)
        gb = df.groupby('objectid',observed=True)
        mldfs = []
        event_indices = rng.choice(range(events_df.shape[0]), df.shape[0])
        for i, objid in enumerate(tqdm.tqdm(list(gb.groups.keys()), leave=False)):
            lc = gb.get_group(objid)
            regions = [ws_regions[objid][rng.integers(len(ws_regions[objid]))]]
            crossing_time, impact_parameter = events_df[['crossing_time', 'umin']].iloc[event_indices[i]]
            crossing_time = crossing_time /24 # recorded in hours, used here in days

            for region in regions:
                times = lc.loc[region]['mjd'].to_numpy()
                peak_time=np.mean([times[0],times[-1]])
                new_lc = add_microlensing_event(lc, \
                            impact_parameter=impact_parameter, crossing_time=crossing_time, \
                            peak_time=peak_time) 
                mldfs.append(new_lc)
                object_event_list.append({'objectid': objid,
                                          'synthid' : str(new_lc.iloc[0]['objectid']),
                                          'event_index': events_df.index[event_indices[i]],
                                          'crossing_time': crossing_time,
                                          'umin': impact_parameter,
                                          'peak_time': peak_time})



        outpath = os.path.join(outsubdir, synthfile)
        bigdf = pd.concat(mldfs)
        bigdf['exposure'] = bigdf['exposure'].astype('category')
        bigdf['filter'] = bigdf['filter'].astype('category')
        bigdf['objectid'] = bigdf['objectid'].astype('category')
        bigdf['originalid'] = bigdf['originalid'].astype('category')
        bigdf['instrument'] = bigdf['instrument'].astype('category')
        bigdf.to_parquet(outpath)
        del df, bigdf
        
    object_event_df = pd.DataFrame.from_dict(object_event_list)
    with open(os.path.join(outsubdir, f'synth-{outname}-info.pickle'), 'wb') as f:
        pickle.dump((outinfo, object_event_df), f)
    
   

def ks_weighted(data1, data2, wei1, wei2, alternative='two-sided'):
    # Weighted two-sample KS: generalizes scipy.stats.ks_2samp to per-point
    # weights (inverse-variance here) so PSPL-fit residuals can be compared to
    # the out-of-event photometry. p-value via the kstwo survival function.
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)
    data1 = data1[ix1]
    data2 = data2[ix2]
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    data = np.concatenate([data1, data2])
    cwei1 = np.hstack([0, np.cumsum(wei1)/sum(wei1)])
    cwei2 = np.hstack([0, np.cumsum(wei2)/sum(wei2)])
    cdf1we = cwei1[np.searchsorted(data1, data, side='right')]
    cdf2we = cwei2[np.searchsorted(data2, data, side='right')]
    d = np.max(np.abs(cdf1we - cdf2we))
    # calculate p-value
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    m, n = sorted([float(n1), float(n2)], reverse=True)
    en = m * n / (m + n)
    if alternative == 'two-sided':
        prob = distributions.kstwo.sf(d, np.round(en))
    else:
        z = np.sqrt(en) * d
        # Use Hodges' suggested approximation Eqn 5.3
        # Requires m to be the larger of (n1, n2)
        expt = -2 * z**2 - 2 * z * (m + 2*n)/np.sqrt(m*n*(m+n))/3.0
        prob = np.exp(expt)
    return d, prob

def reject_low_error_outliers_args(data, errs, m=3):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/np.sqrt(mdev**2 + errs**2) if mdev else np.zeros(len(d))
    return s<m

def reject_outliers_args(data, m = 3.):
    """
    Return data without outliers. Computed as a factor of median distance from the median.  
    """
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return s<m

def reject_outliers(data, m = 3.):
    """
    Return data without outliers. Computed as a factor of median distance from the median.  
    """
    return data[reject_outliers_args(data, m)]


@njit
def sparse_gaussian_wma(y, t, weights, timescale=2, nclip=10):
    # t must be sorted ascending. windowstart drops epochs older than
    # nclip*timescale (where the Gaussian is truncated), keeping this near
    # O(n * window) instead of O(n^2). Each pair (i, j) is accumulated once and
    # applied to both points; the i==i self term is the weights initialisation.
    wma = np.copy(weights*y)
    wme = weights.copy()
    windows_X_weights = weights.copy()

    windowstart = 0
    for i, ti in enumerate(t):
        dt = ti - t[windowstart]
        while dt > timescale*nclip and dt >= 0:
            windowstart += 1
            dt = ti - t[windowstart]
            continue
        for j in range(windowstart, i):
            dt = ti - t[j]

            window = np.exp(-((dt/timescale)**2)/2)
            window_X_weight_i = window*weights[i]
            window_X_weight_j = window*weights[j]
            wma[i] += window_X_weight_j * y[j]
            wme[i] += window * window_X_weight_j
            windows_X_weights[i] += window_X_weight_j
            wma[j] += window_X_weight_i * y[i]
            wme[j] += window * window_X_weight_i

            windows_X_weights[j] += window_X_weight_i
 
    wma = wma/windows_X_weights
    wme = np.sqrt(wme)/windows_X_weights
    return wma, wme, sparse_gaussian_wms(y, t, weights, wma,  
                                         timescale=timescale, nclip=nclip)
    
@njit
def sparse_gaussian_wms(y, t, weights, wma,  timescale=2, nclip=10):

    wms = np.copy(weights*(y-wma)**2)
    windows_X_weights = weights.copy()

    windowstart = 0
    for i, ti in enumerate(t):
        dt = ti - t[windowstart]
        while dt > timescale*nclip and dt >= 0:
            windowstart += 1
            dt = ti - t[windowstart]
            continue
        for j in range(windowstart, i):
            dt = ti - t[j]

            window = np.exp(-((dt/timescale)**2)/2)
            window_X_weight_i = window*weights[i]
            window_X_weight_j = window*weights[j]
            wms[j] += window_X_weight_i * (y[i] - wma[i])**2
            wms[i] += window_X_weight_j * (y[j] - wma[j])**2
            windows_X_weights[i] += window_X_weight_j
            windows_X_weights[j] += window_X_weight_i

    return np.sqrt(wms/windows_X_weights)

@njit
def sparse_gaussian_window_iter(t, timescale=2, nclip=10):
    rows = []
    cols = []
    vals = [np.float64(x) for x in range(0)]  # typed-empty float64 list (numba cannot infer []'s element type)
    windowstart = 0
    for i, ti in enumerate(t):
        dt = ti - t[windowstart]
        while dt > timescale*nclip and dt >= 0:
            windowstart += 1
            dt = ti - t[windowstart]
            continue
        for j in range(windowstart, i):
            dt = ti - t[j]
            rows.append(i)
            rows.append(j)
            cols.append(j)
            cols.append(i)
            newval = np.exp(-((dt/timescale)**2)/2)
            vals += [newval]*2
        rows.append(i)
        cols.append(i)
        vals += [1]
    return (vals, (rows, cols))

# sadly doesn't work with numba
def sparse_gaussian_window(t, timescale=2, nclip=10):
    sparse_matrix = sparse.csr_array(sparse_gaussian_window_iter(t, timescale, nclip), 
                                     shape=(t.shape[0], t.shape[0]))
    return sparse_matrix
def dense_sparse_gaussian_window(t, timescale=2, nclip=10):
    return sparse_gaussian_window(t, timescale, nclip).todense()
@njit
def gaussian_window(dt, timescale=2):
    return np.exp(-((dt/timescale)**2)/2)

@njit
def clipped_gaussian_window(dt, timescale=2, nclip=5):
    if np.abs(dt) > nclip*timescale:
        return 0.
    return np.exp(-((dt/timescale)**2)/2)

@njit
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they 
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))


def weighted_moving_average(y, t, errors, sparse=True, **kwargs):
    y = y.astype('float64')
    errors = errors.astype('float64')
    if sparse:
        return sparse_gaussian_wma(y, t, 1/errors**2, **kwargs)
    else:
        return compute_weighted_moving_average(y,t,errors,**kwargs)

@njit
def compute_weighted_moving_average(y, t, errors, window_fn=gaussian_window, timescale=2):
    windows = window_fn(t.reshape(-1,1)-t.reshape(1,-1), timescale)
    weights = 1/errors**2
    windowsXweights = (windows @ weights)
    wma = (windows @ (weights*y))/windowsXweights
    return (wma, 
            weighted_moving_average_err(weights, windows, windowsXweights),
            weighted_moving_average_scatter(y, wma, weights, windows, windowsXweights))

@njit
def weighted_moving_average_gaussian(y, t, errors, timescale=2):
    # thin alias: the default window of compute_weighted_moving_average is the gaussian
    return compute_weighted_moving_average(y, t, errors, gaussian_window, timescale)

# not njit: builds a scipy.sparse matrix, which numba does not support
def weighted_moving_average_sparse_gaussian(y, t, errors, timescale=2):
    windows = sparse_gaussian_window(t, timescale, nclip=1)
    weights = 1/errors**2
    windowsXweights = (windows @ weights)
    wma = (windows @ (weights*y))/windowsXweights
    return (wma, 
            weighted_moving_average_err(weights, windows, windowsXweights),
            weighted_moving_average_scatter(y, wma, weights, windows, windowsXweights))

@njit
def weighted_moving_average_err(weights, windows, windowsXweights=None):
    if windowsXweights is None:
        windowsXweights = windows @ weights
    return np.power(windowsXweights, -1)*np.sqrt(windows**2 @ weights)

@njit
def weighted_moving_average_scatter(y, wma, weights, windows, windowsXweights=None):
    if windowsXweights is None:
        windowsXweights = windows @ weights
    return np.sqrt((windows@(weights*np.power((y-wma), 2)))/windowsXweights)


def weighted_moving_average_df(lc,  **kwargs):
    lc = lc.sort_values('mjd')
    y = lc['deltamag'].to_numpy()
    e = lc['magerr_auto'].to_numpy()
    t = lc['mjd'].to_numpy()
    return weighted_moving_average(y,t,e,**kwargs)




def float_cols_to_double(df: pd.DataFrame):
    floatcols = [k for k,v in df.dtypes.items() if v=='float32']
    df[floatcols] = df[floatcols].astype('float64')
    return df

def strip_objid(objid):
    m = re.search(r'(\w+)_ml_', objid)
    return m.group(1)

def find_persistent_excursions(df, outliers_cutoff=OUTLIERS_CUTOFF, cut_outliers=False,
        outliers_cutoff_data=OUTLIERS_CUTOFF_DATA,
        z_threshold=Z_THRESHOLD, timescale=DETECTION_TIMESCALE_DAYS, n_measured=N_MEASURED,
        duration=DURATION_DAYS, restrict_to_indices=None, usescatter=True,
        temper_errors=None):
    df = df.sort_values('mjd')
    if df.shape[0] == 0:
        return []

    no_outliers = df.iloc[reject_outliers_args(df['deltamag'].to_numpy(), outliers_cutoff)]
    if cut_outliers:
        df = df.iloc[reject_low_error_outliers_args(df['deltamag'].to_numpy(), 
                                            df['magerr_auto'].to_numpy(),
                                            outliers_cutoff_data)]


    std = np.std(no_outliers['deltamag'].to_numpy())
    if temper_errors:
        df['magerr_auto'] = np.sqrt(df['magerr_auto']**2 + (std*temper_errors)**2)
    wma, errs, scatter = weighted_moving_average_df(df, timescale=timescale)
    if usescatter:
        errs = np.sqrt(errs**2 + scatter**2)
    # A brightening lowers the magnitude, so a real event appears as a negative
    # delta-mag excursion below -z_threshold sigma (errs includes the local
    # scatter when usescatter is set).
    excursions = wma / np.sqrt(std**2 + errs**2) < -z_threshold

    if restrict_to_indices is not None:
        excursions = excursions & df.index.isin(restrict_to_indices)
    # np.split breaks the index at every True/False transition into alternating
    # runs; take every other run starting at the first True run.
    excursion_regions = np.split(df.index, 
                                 np.where(np.concatenate([[False],
                                          np.diff(excursions)]))[0])[int(not excursions[0])::2]
    valid_regions = []
    for region in excursion_regions:
        n_measured_condition = len(region) >= n_measured
        if not(n_measured_condition):
            continue
        exc_start = df.loc[region[0],'mjd']
        exc_end = df.loc[region[-1],'mjd']
        duration_condition = (exc_end - exc_start >= duration)
        if not duration_condition:
            continue
        valid_regions.append(region)
    return valid_regions

def search_files_for_excursions(lcfiles: Iterable[str], 
        search_domains: dict,  metadata: dict, search_params: dict):

    timestamp = int(time.time())
    tmpfiles = []

    metadata['infiles'] = lcfiles
    if 'outdir' in metadata:
        outdir = metadata['outdir']
    else:
        outdir = os.path.join(os.path.dirname(lcfiles[0]), 'searches')
        metadata['outdir'] = outdir
    tmpdir = os.path.join(metadata['outdir'], f'tmp-{timestamp}')
    os.makedirs(tmpdir, exist_ok=True)
    
    params = get_default_args(find_persistent_excursions)
    params.update(search_params)
    params.pop('restrict_to_indices', None)

    for file in tqdm.tqdm(lcfiles):
        file_excursions = {}
        df = pd.read_parquet(file)
        df = float_cols_to_double(df)
        gb = df.groupby('objectid',observed=True)
        for objid in tqdm.tqdm(list(gb.groups.keys()), leave=False):
            lc = gb.get_group(objid)

            if objid not in search_domains:
                m = re.search(r'(\w+)_ml_', objid)
                original_id = m.group(1)
            else:
                original_id = objid
            excs = find_persistent_excursions(lc, 
                       restrict_to_indices=np.concatenate(search_domains[original_id]),
                       **params)
            file_excursions[objid] = excs
        filename = os.path.basename(file)
        tmpfile = os.path.join(tmpdir, filename + '-search.pickle')
        with open(tmpfile, 'wb') as f:
            pickle_data = (metadata, params, file_excursions)
            pickle.dump(pickle_data, f)
        tmpfiles.append(tmpfile)

        del df, gb
        gc.collect()
    results = consolidate_search_files_for_excursions(tmpfiles)
    shutil.rmtree(tmpdir)
    return results

def consolidate_search_files_for_excursions(partialfiles):
    with open(partialfiles[0], 'rb') as f:
        metadata, search_params, excursions = pickle.load(f)
    for file in partialfiles[1:]:
        with open(file, 'rb') as f:
            file_metadata, file_search_params, file_excursions = pickle.load(f)
        if metadata != file_metadata:
            raise ValueError(f"Metadata doesn't match for {file}; aborting consolidation.")
        if search_params != file_search_params:
            raise ValueError(f"Search parameters don't match for {file}; aborting consolidation.")
        excursions.update(file_excursions)
    with open(os.path.join(metadata['outdir'], metadata['outfile']), 'wb') as f:
        pickle_data = (metadata, search_params, excursions)
        pickle.dump(pickle_data, f)
    return pickle_data

def reduce_excursions(excursions: dict):
    return {k:v for k,v in excursions.items() if len(v)>0}

def get_nondetections(excursions: dict):
    return [k for k,v in excursions.items() if len(v)==0]





def compute_file_map(files):
    objfilemap = {}
    fileenum = {}
    for i, file in enumerate(tqdm.tqdm(files)):
        fileenum[i] = file
        df = pd.read_parquet(file, columns=['objectid'])
        for objid in df['objectid'].unique():
            objfilemap[objid]=i
    return objfilemap, fileenum

def extend_lc(df, region, context_size=CONTEXT_SIZE_DAYS):
    """Return the indices of df within context_size days of the region's time
    span (region assumed sorted by 'mjd')."""
    estart, eend = df.loc[region,'mjd'].min(), df.loc[region,'mjd'].max()
    return df[(df['mjd']> estart-context_size) & (df['mjd'] < eend+context_size)].index



def fit_excursions(excursions, lcfiles,  metadata, params, n_min_outside_fit=N_MIN_OUTSIDE_FIT,
                   outliers_cutoff=OUTLIERS_CUTOFF, temper_errors=TEMPER_ERRORS_FIT, n_ks_gaussian=N_KS_GAUSSIAN,
                   context_size=CONTEXT_SIZE_DAYS, crossing_time_guess=CROSSING_TIME_GUESS_DAYS,
                   rng=None):
    # rng: pass a seeded numpy Generator for a reproducible small-sample KS
    # reference; defaults to a fresh (entropy-seeded) Generator.
    if rng is None:
        rng = np.random.default_rng()
    fitresults = []
    fitfails = []
    fitdups = []
    for lcfile in tqdm.tqdm(lcfiles):
        filedf = pd.read_parquet(lcfile)
        gb = filedf.groupby('objectid',observed=True)
        for objid in tqdm.tqdm(list(gb.groups.keys()), leave=False):

            if (objid not in excursions):
                continue
            df = gb.get_group(objid)

            objfits = []
            for i, region in enumerate(excursions[objid]):
                
                extended_region = extend_lc(df, region, context_size)
                ext_region_full_df = df.loc[extended_region].sort_values('mjd')

                no_outliers = df.iloc[reject_outliers_args(df['deltamag'].to_numpy(), outliers_cutoff)]
                std = np.std(no_outliers['deltamag'].to_numpy())
                ext_region_full_df['magerr_auto'] = np.sqrt(ext_region_full_df['magerr_auto']**2
                                                            + (std*temper_errors)**2)
                ext_region_df = ext_region_full_df

                dms = ext_region_df['deltamag'].to_numpy()
                errs = ext_region_df['magerr_auto'].to_numpy()
                mjds = ext_region_df['mjd'].to_numpy()

                try:
                    with warnings.catch_warnings(action="ignore"):
                        fitresult=optimize.curve_fit(ml_f, mjds, dms, 
                                        p0=(1, crossing_time_guess ,np.mean(mjds)),
                                        sigma=errs,full_output=False,
                                        absolute_sigma=True,
                                        x_scale=[1, crossing_time_guess, 
                                                 np.diff(np.percentile(mjds,(0,100)))],
                                        bounds=([0, 1, mjds[0] - FIT_TIME_PAD_DAYS],
                                                [5, FIT_TIME_PAD_DAYS, mjds[-1] + FIT_TIME_PAD_DAYS]),
                                        jac=ml_jac)
                except RuntimeError:
                    fitfails.append((objid, i))
                    continue
                    
                fitp=fitresult[0]

                if list(fitp) in objfits:
                    fitdups.append((objid, i))
                    continue
                objfits.append(list(fitp))

                fitmags = ml_f(mjds,*fitp)
                outside_fit_df = df.loc[df.index.difference(ext_region_full_df.index)]

                # Compare the PSPL-fit residuals to the out-of-event photometry
                # with a weighted two-sample KS test; when too few points lie
                # outside the event, use a synthetic Gaussian reference of the
                # same weighted residual scatter instead.
                if len(outside_fit_df) > n_min_outside_fit:
                    kstwosided = True
                    outside_fit_dms = outside_fit_df['deltamag'].to_numpy()
                    outside_fit_errs = outside_fit_df['magerr_auto'].to_numpy()
                    ksresult = ks_weighted(dms-fitmags, outside_fit_dms,  errs, outside_fit_errs)
                else:
                    res_ave, res_std = weighted_avg_and_std(dms-fitmags, 1/errs**2)
                    ksresult = ks_weighted(dms-fitmags, 
                                           rng.normal(0, res_std, n_ks_gaussian),
                                           1/errs**2, np.ones(n_ks_gaussian)/res_std**2)
                    kstwosided = False

                fitresults.append([objid, i, ksresult, fitresult, 
                                   ext_region_df.shape[0],len(outside_fit_df), kstwosided])
    outpath = os.path.join(metadata['outdir'], metadata['fitoutfile'])
    os.makedirs(metadata['outdir'], exist_ok=True)
    with open(outpath, 'wb') as f:
        pickle.dump((fitresults, fitfails, fitdups, metadata, params), f)
    return     fitresults, fitfails, fitdups

def make_fit_excursions_df(fitresults):

    data = {
        'objectid': [],
        'excnum': [],
        'pval': [],
        'n_fit': [],
        'n_out': [],
        'cond_num': [],
        'impact_parameter': [],
        'crossing_time': [],
        'peak_time': [],
        'two_sample': []
    }

    for v in fitresults:
        data['objectid'].append(v[0])
        data['excnum'].append(v[1])
        data['pval'].append(v[2][1])
        data['n_fit'].append(v[4])
        data['n_out'].append(v[5])
        data['cond_num'].append(np.linalg.cond(v[3][1]))
        data['impact_parameter'].append(v[3][0][0])
        data['crossing_time'].append(v[3][0][1])
        data['peak_time'].append(v[3][0][2])
        data['two_sample'].append(v[6])

    return pd.DataFrame(data)

def search_for_params(files, params):
    finds = []
    for file in files:
        with open(file, 'rb') as f:
            metadata, search_params, excursions = pickle.load(f)
        if set(params) & set(search_params) == set(params):
            if False in [search_params[k] == v for k,v in params.items()]:
                continue
            finds.append(file)
    return finds

def common_params(f, params):
    args = get_default_args(f)
    return {k:v for k,v in params.items() if k in args}

def default_args_of_functions(fs):
    params = {}
    for f in fs:
        params.update(get_default_args(f))
    return params

def search_files_for_microlensing_events(lcfiles, ws_regions, 
                                         metadata, params):

    all_params = default_args_of_functions([find_persistent_excursions,
                                            fit_excursions])
    if set(all_params) & set(params) != set(params):
        raise ValueError(f'Unknown parameters: {set(params).difference(set(all_params))}') 


    excursion_params = common_params(find_persistent_excursions, params)
    fit_params = common_params(fit_excursions, params)

    excursion_results =  \
        search_files_for_excursions(lcfiles, ws_regions, 
                                             metadata, excursion_params)
    metadata, search_params, excursions = excursion_results

    full_fit_results = fit_excursions(excursions, lcfiles, metadata, 
                                                   fit_params, **fit_params)

    fitresults, fitfails, fitdups = full_fit_results
    
    return excursion_results, full_fit_results

def cut_by_npoints(df, npoints):
    return df[df['n_fit'] + df['n_out']>=npoints]
def cut_by_pval(df, pval: float):
    return df[df['pval'] >= pval]
def cut_high_points_low_p(df, npoints, pval):
    return df[(df['pval'] >= pval) | ((df['n_fit'] + df['n_out'])<npoints)]
def cut_high_points_inout_low_p(df, npoints, pval):
    return df[(df['pval'] >= pval) | ((df['n_fit']<npoints) |  (df['n_out']<npoints))]
def cut_pcov(df, cond_lim=COND_LIM):
    # cond_num is the 2-norm condition number of the fit covariance; a large
    # value flags a degenerate / under-constrained PSPL fit. Default 1e5.
    return df[df['cond_num'] < cond_lim]
def cut_crossing_time(df, timemin=1, timemax=None):
    if timemax is not None:
        return df[(df['crossing_time']>timemin) & (df['crossing_time']<timemax)]

    return df[df['crossing_time']>timemin]

def split_real_synth_df(fitdf):
    # Synthetic objects carry the "_ml_" token in their id (see synth_objid);
    # a bare "ml" substring would also match unrelated ids.
    sfitdf = fitdf[['_ml_' in id for id in fitdf['objectid']]]
    rfitdf = fitdf.loc[fitdf.index.difference(sfitdf.index)]
    return rfitdf, sfitdf
