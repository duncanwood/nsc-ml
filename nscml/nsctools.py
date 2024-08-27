from collections.abc import Iterable
import random
import gc
import pickle
import time
import os
import re
import shutil
import warnings

import matplotlib.pyplot as plt
# import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import pandas as pd
import numpy as np
from scipy.stats import distributions
from scipy import sparse, optimize
from numba import njit, vectorize



import tqdm

# from .plot import *

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


def plot_lc(lc: pd.DataFrame, **kwargs):
    objid = lc['objectid'].unique()
    assert len(objid)==1
    objid = objid[0]
    fig, ax = plt.subplots()
    for f in lc['filter'].unique():
        f_df = lc[lc['filter']==f]
        plt.errorbar(f_df['mjd'], f_df['mag_auto'], f_df['magerr_auto'],
                     c=color_filter[f.lower()],linestyle='None', 
                     marker='o',  label=f)
        plt.hlines(f_df[f.lower()+'mag'], f_df['mjd'].to_numpy().min(),
                   f_df['mjd'].to_numpy().max(), 
                   color=color_filter[f.lower()], linestyle='dashed')
    plt.legend()
    if 'xlims' in kwargs:
        plt.xlim(kwargs['xlims'])
    plt.xlabel('MJD')
    plt.title(f'{objid}')
    plt.gca().invert_yaxis()
    plt.show()

def plot_obj(objid: str, curves: pd.DataFrame, **kwargs):
    plot_lc(curves.get_group(objid), **kwargs)

def plot_deltamags(lc: pd.DataFrame, **kwargs):
    id = lc['objectid'].unique()
    assert len(id)==1
    id = str(id[0])
    # fig, ax = plt.subplots()
    gb = lc.groupby(['filter', 'instrument'], observed=True)
    for f, instrument in gb.groups.keys():
        f_df = lc[lc['filter']==f]
        plt.errorbar(f_df['mjd'], f_df['deltamag'], f_df['magerr_auto'],
                     c=color_filter[f.lower()],linestyle='None', 
                     markersize=5, marker=marker_map[instrument], capsize=0)
        # plt.hlines(f_df[f.lower()+'mag'], f_df['mjd'].to_numpy().min(),f_df['mjd'].to_numpy().max() , color=color_filter[f], linestyle='dashed')
    patches = [ mpatches.Patch(color=color_filter[f.lower()], label=f)  
                for f in lc['filter'].unique() ]
    points = [  Line2D([0], [0], label=instrument, 
                        marker=marker_map[instrument], markersize=10,  
                        markeredgecolor='black', markerfacecolor='black', 
                        linestyle='') 
                for instrument in lc['instrument'].unique()]

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([*patches,  *points])

    plt.legend(handles=handles)
    if 'xlims' in kwargs:
        plt.xlim(kwargs['xlims'])
    plt.xlabel('MJD')
    plt.ylabel('Change from baseline (mag)')
    plt.title(f'{id}')
    plt.gca().invert_yaxis()
    if 'show' in kwargs and kwargs['show']==True:
        plt.show()

def plot_obj_dm(id: str, dmgroupby, **kwargs):
    plot_deltamags(dmgroupby.get_group(id), **kwargs)


def convert_to_range_index(idxs):
    if (np.diff(idxs)==1).all():
        return pd.RangeIndex(idxs[0], idxs[-1] + 1)
    else:
        return idxs

def well_sampled_region(df: pd.DataFrame, interval=50., maxrevisit=10, seqlen=5):
    df = df.sort_values('mjd')
    times = df['mjd'].to_numpy()
    # dtimes = np.diff(times)
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
        regions = well_sampled_region(objdf, interval=50, maxrevisit=10, seqlen=5)
        well_sampled_objects[obj] = regions
    return well_sampled_objects

def get_just_well_sampled_objects(df, progress=False):
    gb= df.groupby('objectid')
    well_sampled_objects = {}
    for obj in tqdm.tqdm(list(gb.groups.keys()), disable=(not progress)):
        objdf = gb.get_group(obj)
        regions = well_sampled_region(objdf, interval=50, maxrevisit=10, seqlen=5)
        if len(regions) > 0:
            well_sampled_objects[obj] = regions
    return well_sampled_objects

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

    lightcurve_u = np.sqrt(impact_parameter**2 
                           + ((t - peak_time) ** 2 / crossing_time**2))
    amplified_mag = (lightcurve_u**2 + 2) / (
        lightcurve_u * np.sqrt(lightcurve_u**2 + 4)
    ) * blending_factor + (1 - blending_factor)

    return amplified_mag

def amp_to_mag(amp):
    return -2.5*np.log10(amp)

def add_microlensing_event(df: pd.DataFrame, **lensing_params):
    """
    Return a copy of input dataframe with a synthesized microlensing event 
    superimposed on the curve, with params given in 'lensing_params'
    """
    lc = df.copy()
    mag_diffs = amp_to_mag(microlensing_amplification(lc['mjd'], **lensing_params))
    lc['mag_auto'] = (lc['mag_auto'] + mag_diffs).astype(lc.dtypes['mag_auto'])
    lc['deltamag'] = (lc['deltamag'] + mag_diffs).astype(lc.dtypes['deltamag'])

    newobjid = str(lc.iloc[0]['objectid']) + f"_ml_{lensing_params['peak_time']:.2f}_{lensing_params['crossing_time']:.2f}_{lensing_params['impact_parameter']:.5f}"
    # lc['objectid'] = lc['objectid'].cat.add_categories(newobjid)
    lc['objectid'] = newobjid
    
    return lc

def ml_f(*x):
    return amp_to_mag(microlensing_amplification(*x))

def generate_synthetic_microlensing_events_from_population(lcfiles, events_file, ws_regions, outdir, outname):
    # outdir = databasedir+datasetname+'/synth/'

    if isinstance(events_file, str):
        events_df = pd.read_pickle(events_file)
    elif isinstance(events_file,pd.DataFrame):
        events_df = events_file
    else:
        raise ValueError(f'Unsupported type for events_file: {type(events_file)}')

    outsubdir = outdir + f'/synth-{outname}/'
    os.makedirs(outsubdir, exist_ok=True)
    outinfo = {'lcfiles': lcfiles, 
                 'events_file': events_file, 
                 'outdir': outdir,
                 'outname': outname,
                 'outsubdir': outsubdir}
    object_event_list = []

    # impact_parameter=1
    # crossing_time=40
    for file in tqdm.tqdm(lcfiles):
        filename = file.split('/')[-1]
        synthfile = '.'.join(filename.split('.')[:-1]) + f'-synth-{outname}.parquet'
        # print(synthfile)
        
        df = pd.read_parquet(file)
        gb = df.groupby('objectid',observed=True)
        mldfs = []
        event_indices = np.random.default_rng().choice(range(events_df.shape[0]), df.shape[0])
        for i, objid in enumerate(tqdm.tqdm(list(gb.groups.keys()), leave=False)):
            lc = gb.get_group(objid)
            regions = [random.choice(ws_regions[objid])]
            crossing_time, impact_parameter = events_df[['crossing_time', 'umin']].iloc[event_indices[i]]
            crossing_time = crossing_time /24 # recorded in hours, used here in days

            for region in regions:
                times = lc.loc[region]['mjd'].to_numpy()
                peak_time=np.mean([times[0],times[-1]])
                mldfs.append(add_microlensing_event(lc, \
                            impact_parameter=impact_parameter, crossing_time=crossing_time, \
                            peak_time=peak_time))
                object_event_list.append({'objectid': objid,
                                          'event_index': events_df.index[event_indices[i]],
                                          'crossing_time': crossing_time,
                                          'umin': impact_parameter,
                                          'peak_time': peak_time})



        outpath =  outsubdir + synthfile
        bigdf = pd.concat(mldfs)
        bigdf['exposure'] = bigdf['exposure'].astype('category')
        bigdf['filter'] = bigdf['filter'].astype('category')
        bigdf['objectid'] = bigdf['objectid'].astype('category')
        bigdf['instrument'] = bigdf['instrument'].astype('category')
        bigdf.to_parquet(outpath)#,append=os.path.exists(outpath))
        del df, bigdf
        
    object_event_df = pd.DataFrame.from_dict(object_event_list)
    with open(outsubdir + f'synth-{outname}-info.pickle', 'wb') as f:
        pickle.dump((outinfo, object_event_df), f)
    
   


def ks_weighted(data1, data2, wei1, wei2, alternative='two-sided'):
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


# @njit
# def sparse_gaussian_window_iter(dts, timescale=2, nclip=10):
#     rows = []
#     cols = []
#     vals = []
#     for i in range(len(dts)):
#         for j in range(len(dts[0])):
#             dt = dts[i,j]
#             if np.abs(dt) > nclip*timescale:
#                 continue
#             else:
#                 rows.append(i)
#                 cols.append(j)
#                 vals.append(np.exp(-((dt/timescale)**2)/2))
#     return (vals, (rows, cols))
    
@njit
def sparse_gaussian_window_iter(t, timescale=2, nclip=10):
    rows = []
    cols = []
    vals = [np.float64(x) for x in range(0)]
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
    sparse_matrix = sparse.csr_array(sparse_gaussian_window_iter(t, timescale, nclip), 
                                     shape=(t.shape[0], t.shape[0]))
    return sparse_matrix.todense()
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


def weighted_moving_average(y, t, errors, **kwargs):
    y = y.astype('float64')
    errors = errors.astype('float64')
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
    windows = gaussian_window(t.reshape(-1,1)-t.reshape(1,-1), timescale)
    weights = 1/errors**2
    windowsXweights = (windows @ weights)
    wma = (windows @ (weights*y))/windowsXweights
    return (wma, 
            weighted_moving_average_err(weights, windows, windowsXweights),
            weighted_moving_average_scatter(y, wma, weights, windows, windowsXweights))

# @njit
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



def plot_weighted_moving_average_df(df, usescatter=True, timescale=2, **kwargs):
    df=df.sort_values('mjd')
    d=df['deltamag'].to_numpy()
    e=df['magerr_auto'].to_numpy()
    t=df['mjd'].to_numpy()
    plot_deltamags(df, **kwargs)
    wma, errs, scatter = weighted_moving_average(d, t, e, timescale=timescale)
    if usescatter:
        confidence = np.sqrt(errs**2 + scatter**2)
    else: 
        confidence = errs
    plt.plot(t,wma, linestyle='dotted')
    plt.fill_between(t,wma-confidence,wma+confidence, alpha=.2)
    # plt.fill_between(t,weighted_moving_average(d-e, t, e, timescale=2), weighted_moving_average(d+e, t, e, timescale=2), alpha=.2)

def float_cols_to_double(df: pd.DataFrame):
    floatcols = [k for k,v in df.dtypes.items() if v=='float32']
    df[floatcols] = df[floatcols].astype('float64')
    return df

def find_persistent_excursions(df, outliers_cutoff=3, cut_outliers=False,
        outliers_cutoff_data=20,
        z_threshold=3, timescale=2, n_measured=4, 
        duration=5, restrict_to_indices=None, usescatter=True,
        temper_errors=None):
    df = df.sort_values('mjd')

    no_outliers = df.iloc[reject_outliers_args(df['deltamag'].to_numpy(), outliers_cutoff)]
    if cut_outliers:
        df = df.iloc[reject_low_error_outliers_args(df['deltamag'].to_numpy(), 
                                            df['deltamag'].to_numpy(),
                                            outliers_cutoff_data)]


    std = np.std(no_outliers['deltamag'].to_numpy())
    if temper_errors:
        df['magerr_auto'] = np.sqrt(df['magerr_auto']**2 + (std*temper_errors)**2)
    wma, errs, scatter = weighted_moving_average_df(df, timescale=timescale)
    if usescatter:
        errs = np.sqrt(errs**2 + scatter**2)
    # excursions = wma < -sigma*std
    excursions = wma / np.sqrt(std**2 + errs**2) < -z_threshold

    if restrict_to_indices is not None:
        excursions = excursions & df.index.isin(restrict_to_indices)
    # print(excursions)
    # print(np.where(np.concatenate([[excursions[0]],np.diff(excursions), [True]]))[0])
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
        # revisit_condition = (np.diff(df.loc[region,'mjd']) <= revisit_time).all()
        # if not revisit_condition:
        #     continue
        # achromatic_condition = is_achromatic(df.loc[region],sigma=achromatic_sigma)
        # if not achromatic_condition:
        #     continue
        valid_regions.append(region)
    return valid_regions

def search_files_for_microlensing_events(lcfiles: Iterable[str], 
        search_domains: dict,  metadata: dict, search_params: dict):

    timestamp = int(time.time())
    tmpfiles = []

    metadata['infiles'] = lcfiles
    if 'outdir' in metadata:
        outdir = metadata['outdir']
    else:
        outdir = '/'.join(lcfiles[0].split('/')[:-1])+'/searches/'
        metadata['outdir'] = outdir
    tmpdir = metadata['outdir']+f'/tmp-{timestamp}/'
    os.makedirs(tmpdir, exist_ok=True)

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
                       **search_params)
            file_excursions[objid] = excs
        filename = file.split('/')[-1]
        tmpfile = tmpdir+filename+'-search.pickle'
        with open(tmpfile, 'wb') as f:
            pickle_data = (metadata, search_params, file_excursions)
            pickle.dump(pickle_data, f)
        tmpfiles.append(tmpfile)

        del df, gb
        gc.collect()
    results = consolidate_search_files_for_microlensing_events(tmpfiles)
    shutil.rmtree(tmpdir)
    return results

def consolidate_search_files_for_microlensing_events(partialfiles):
    with open(partialfiles[0], 'rb') as f:
        metadata, search_params, excursions = pickle.load(f)
    for file in partialfiles[1:]:
        with open(file, 'rb') as f:
            file_metadata, file_search_params, file_excursions = pickle.load(f)
        if metadata != file_metadata:
            print(f"Metadata doesn't match for {file}. Aborting...")
            print(set(metadata.items())^set(file_metadata.items()))
            return
        if search_params != file_search_params:
            print(f"Search parameters don't match for {file}. Aborting...")
            print(set(search_params.items())^set(file_search_params.items()))
            return
        excursions.update(file_excursions)
    with open(metadata['outdir']+metadata['outfile'], 'wb') as f:
        pickle_data = (metadata, search_params, excursions)
        pickle.dump(pickle_data, f)
    return pickle_data

def reduce_excursions(excursions: dict):
    return {k:v for k,v in excursions.items() if len(v)>0}

def get_nondetections(excursions: dict):
    return [k for k,v in excursions.items() if len(v)==0]




def plot_excursion_region(lc, region, timescale=2, context_size=100, **kwargs):
    plot_weighted_moving_average_df(lc,timescale=timescale, xlims=np.percentile(lc.loc[region,'mjd'].to_numpy(),(0,100))+np.array([-context_size,context_size]), **kwargs)
    ymin = np.min(lc.loc[region,'deltamag'].to_numpy() - lc.loc[region,'magerr_auto'].to_numpy())
    
    ymax = np.max(lc.loc[region,'deltamag'].to_numpy() + lc.loc[region,'magerr_auto'].to_numpy())
    plt.vlines(np.percentile(lc.loc[region,'mjd'].to_numpy(),(0,100)), ymin,ymax, linestyle='dashed')
    plt.fill_between(np.percentile(lc.loc[region,'mjd'].to_numpy(),(0,100)), ymin,ymax, alpha=.2)


def compute_file_map(files):
    objfilemap = {}
    fileenum = {}
    for i, file in enumerate(tqdm.tqdm(files)):
        fileenum[i] = file
        df = pd.read_parquet(file, columns=['objectid'])
        for objid in df['objectid'].unique():
            objfilemap[objid]=i
    return objfilemap, fileenum

def extend_lc(df, region, context_size = 100):
    estart, eend = df.loc[region,'mjd'].min(), df.loc[region,'mjd'].max()
    return df[(df['mjd']> estart-context_size) & (df['mjd'] < eend+context_size)].index

def plot_example_fits(fulldf, all_excursions, fitresults, 
                      fileenum, objfilemap, limitnum=10):
    plotidx= np.random.choice(list(fulldf.index), min(limitnum, len(fulldf)), 
                              replace=False)
    for idx in plotidx:
        obj = fulldf.loc[idx]['objectid']
        exc_idx = fulldf.loc[idx]['excnum']
        region = all_excursions[obj][exc_idx]
        file = fileenum[objfilemap[obj]]
        df = pd.read_parquet(file)
        df = df[df['objectid']==obj]
        df = float_cols_to_double(df)
        print(obj)    



        plot_excursion_region(df[df['objectid']==obj], region, context_size=30, timescale=5)
        
        extended_region = extend_lc(df, region)
        ext_region_df = df.loc[extended_region].sort_values('mjd')
        # dms = ext_region_df['deltamag'].to_numpy()
        # errs = ext_region_df['magerr_auto'].to_numpy()
        mjds = ext_region_df['mjd'].to_numpy()
        # filters = ext_region_df['filter'].to_numpy()
        mjds = np.linspace(mjds[0], mjds[-1],200)

        fitresult = [result for result in fitresults if result[0] ==obj][exc_idx]
        fitinfo = fitresult[3][0]
        print(fitresult)
        fitmags = ml_f(mjds,*fitinfo)
        plt.plot(mjds, fitmags,c='black', linestyle='dashed',marker='None', label='PSPL fit')

        plt.show()
        plt.clf()
        plt.close("all")


def fit_excursions(excursions, lcfiles,  n_min_outside_fit = 10, 
                   outliers_cutoff=3, temper_errors=1, n_ks_gaussian=10000):
    
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
                
                extended_region = extend_lc(df, region)
                ext_region_full_df = df.loc[extended_region].sort_values('mjd')

                no_outliers = df.iloc[reject_outliers_args(df['deltamag'].to_numpy(), outliers_cutoff)]
                std = np.std(no_outliers['deltamag'].to_numpy())
                ext_region_full_df['magerr_auto'] = np.sqrt(ext_region_full_df['magerr_auto']**2
                                                            + (std*temper_errors)**2)
                # cut_mask = nsctools.reject_low_error_outliers_args(ext_region_full_df['deltamag'].to_numpy(),
                #                                          ext_region_full_df['magerr_auto'].to_numpy(),20)

                ext_region_df = ext_region_full_df#.iloc[cut_mask]

                dms = ext_region_df['deltamag'].to_numpy()
                errs = ext_region_df['magerr_auto'].to_numpy()
                mjds = ext_region_df['mjd'].to_numpy()
                # filters = ext_region_df['filter'].to_numpy()
                # plt.errorbar(mjds,dms, errs,linestyle='None',marker='.')
                # plt.gca().invert_yaxis()

                try:
                    with warnings.catch_warnings(action="ignore"):
                        fitresult=optimize.curve_fit(ml_f, mjds, dms, 
                                                p0=(1, 40,np.mean(mjds)),
                                                sigma=errs,full_output=False,
                                                bounds=([0,1, mjds[0]-365*10],
                                                        [5,365*10, mjds[-1]+365*10]))
                except RuntimeError:
                    fitfails.append((objid, i))
                    continue
                # except IndexError as e:
                    
                fitp=fitresult[0]

                if list(fitp) in objfits:
                    # print(f'Duplicate found in {objid} with {fitp}')
                    fitdups.append((objid, i))
                    continue
                objfits.append(list(fitp))
                # print(fitp)

                fitmags = ml_f(mjds,*fitp)
                outside_fit_df = df.loc[df.index.difference(ext_region_full_df.index)]

                if len(outside_fit_df) > n_min_outside_fit:
                    kstwosided = True
                    outside_fit_dms = outside_fit_df['deltamag'].to_numpy()
                    outside_fit_errs = outside_fit_df['magerr_auto'].to_numpy()
                    ksresult = ks_weighted(dms-fitmags, outside_fit_dms,  errs, outside_fit_errs)
                else:
                    res_ave, res_std = weighted_avg_and_std(dms-fitmags, 1/errs**2)
                    ksresult = ks_weighted(dms-fitmags, 
                                           np.random.normal(0 , res_std, n_ks_gaussian), 
                                           1/errs**2, np.ones(n_ks_gaussian)/res_std**2)
                    kstwosided = False

                fitresults.append([objid, i, ksresult, fitresult, 
                                   ext_region_df.shape[0],len(outside_fit_df), kstwosided])

def make_fit_excursions_df(fitresults):
    # fitdf = pd.DataFrame({'objectid':[v[0] for v in fitresults], 'excnum': [v[1] for v in fitresults], 
    #                   'pval': [v[2][1] for v in fitresults],'n_fit': [v[4] for v in fitresults], 
    #                   'n_out':[v[5] for v in fitresults], 
    #                   'cond_num': [np.linalg.cond(item[3][1]) for item in fitresults],
    #                   'impact_parameter': [v[3][0][0] for v in fitresults],
    #                   'crossing_time': [v[3][0][1] for v in fitresults],
    #                   'peak_time': [v[3][0][2] for v in fitresults],
    #                   'two_sample': [v[6] for v in fitresults]})


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

    # sfitdf = fitdf[['ml' in id for id in fitdf['objectid']]]
    # rfitdf = fitdf.loc[fitdf.index.difference(sfitdf.index)]
