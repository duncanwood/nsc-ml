import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import pandas as pd
import numpy as np
from numba import njit, vectorize

from collections.abc import Iterable
import gc
import pickle
import time
import os


from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

import tqdm

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
    id = lc['objectid'].unique()
    assert len(id)==1
    id = id[0]
    fig, ax = plt.subplots()
    for f in lc['filter'].unique():
        f_df = lc[lc['filter']==f]
        plt.errorbar(f_df['mjd'], f_df['mag_auto'], f_df['magerr_auto'],c=color_filter[f.lower()],linestyle='None', marker='o',  label=f)
        plt.hlines(f_df[f.lower()+'mag'], f_df['mjd'].to_numpy().min(),f_df['mjd'].to_numpy().max() , color=color_filter[f.lower()], linestyle='dashed')
    plt.legend()
    if 'xlims' in kwargs:
        plt.xlim(kwargs['xlims'])
    plt.xlabel('MJD')
    plt.title(f'{id}')
    plt.gca().invert_yaxis()
    plt.show()

def plot_obj(id: str, curves: pd.DataFrame, **kwargs):
    plot_lc(curves.get_group(id), **kwargs)

def plot_deltamags(lc: pd.DataFrame, **kwargs):
    id = lc['objectid'].unique()
    assert len(id)==1
    id = str(id[0])
    # fig, ax = plt.subplots()
    gb = lc.groupby(['filter', 'instrument'])
    for f, instrument in gb.groups.keys():
        f_df = lc[lc['filter']==f]
        plt.errorbar(f_df['mjd'], f_df['deltamag'], f_df['magerr_auto'],c=color_filter[f.lower()],linestyle='None', markersize=5, marker=marker_map[instrument], capsize=0)
        # plt.hlines(f_df[f.lower()+'mag'], f_df['mjd'].to_numpy().min(),f_df['mjd'].to_numpy().max() , color=color_filter[f], linestyle='dashed')
    patches = [ mpatches.Patch(color=color_filter[f.lower()], label=f)  for f in lc['filter'].unique() ]
    points = [  Line2D([0], [0], label=instrument, marker=marker_map[instrument], markersize=10,  markeredgecolor='black', markerfacecolor='black', linestyle='') for instrument in lc['instrument'].unique()]

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([*patches,  *points])

    plt.legend(handles=handles)
    if 'xlims' in kwargs:
        plt.xlim(kwargs['xlims'])
    plt.xlabel('MJD')
    plt.title(f'{id}')
    plt.gca().invert_yaxis()
    if 'show' in kwargs and kwargs['show']==True:
        plt.show()

def plot_obj_dm(id: str, dmgroupby, **kwargs):
    plot_deltamags(dmgroupby.get_group(id), **kwargs)

def interp1d(items: np.array):
    """
    Add points at even intervals between each point in input array.

    Input array assumed to be sorted
    """
    return np.sort(np.concatenate((items, (items[1:]+items[:-1])/2)))

def segment_times(lc: pd.DataFrame, timescale=40.):
    """
    Segments input dataframe into clusters in time. 
    Scale is set by timescale.

    Returns: A list of segments, each containing the indices of the dataframe within each cluster.
    """
    lc = lc.sort_values('mjd')   
    times = lc['mjd'].to_numpy()
    kde = KernelDensity(kernel='gaussian', bandwidth=timescale).fit(times.reshape((-1,1)))
    s = interp1d(times)
    e = kde.score_samples(s.reshape(-1,1))
    mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    mival, maval = s[mi],s[ma]
    
    mival = np.concatenate([[np.min(times)-timescale], mival, [np.max(times)+timescale]]) 
    segments = [lc[(lc['mjd'] > mival[i]) & (lc['mjd'] <= mival[i+1])].index for i in range(len(mival)-1)]
    return segments

def segment_times_simple(indices: np.array, times: np.array, max_revisit: float = 0.5):
    """
    Assume indicies and times are sorted in the same order on time
    """
    diffs = np.diff(times)
    return np.split(indices, diffs>max_revisit)


@njit
def points_compatible(vals: np.array, errs: np.array, sigma=3.):
    n=len(vals)
    if n <= 1: 
        return True
    for i, val in enumerate(vals):
        others = np.concatenate((vals[:i],vals[i+1:]))
        otherserrs = np.concatenate((errs[:i],errs[i+1:]))
        mean = np.average(others, weights=1/otherserrs**2)
        err = np.power(np.sum(np.power(otherserrs, -2)), -1/2)*np.sqrt(n-1)
        diffsig = np.abs(val-mean) / np.sqrt(err**2 + errs[i]**2)
        if diffsig > sigma:
            return False
    return True

def is_achromatic(lc: pd.DataFrame, timescale=1., sigma=3, magcol='mag_auto', magerrcol='magerr_auto'):
    """
    
    """

    segs =segment_times(lc, timescale=timescale)
    for seg in segs:
        # times = c.loc[[seg[0]]]['mjd']
        times = lc.loc[seg,:]['mjd']
        if not points_compatible(lc.loc[seg,'deltamag'].to_numpy(), lc.loc[seg, 'magerr_auto'].to_numpy(),sigma=sigma):
            return False
    return True

def well_sampled_region(df: pd.DataFrame, interval=50., maxrevisit=10, seqlen=5):
    df = df.sort_values('mjd')
    times = df['mjd'].to_numpy()
    # dtimes = np.diff(times)
    valid_regions = []
    for region in np.split(df.index, np.where(np.diff(times) > maxrevisit)[0]+1):
        if (region.shape[0] >= seqlen):
            start, end = df.loc[[region[0], region[-1]], 'mjd']
            if end-start > interval:
                valid_regions.append(region)
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

def microlensing_amplification(t, impact_parameter=1, crossing_time=40.0, peak_time=100, blending_factor=1):
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

    lightcurve_u = np.sqrt(impact_parameter**2 + ((t - peak_time) ** 2 / crossing_time**2))
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
    lc['mag_auto'] = lc['mag_auto'] + mag_diffs
    lc['deltamag'] = lc['deltamag'] + mag_diffs

    newobjid = str(lc.iloc[0]['objectid']) + f"_ml_{lensing_params['peak_time']}_{lensing_params['crossing_time']}_{lensing_params['impact_parameter']}"
    lc['objectid'] = lc['objectid'].cat.add_categories(newobjid)
    lc['objectid'] = newobjid 
    

    return lc

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
def gaussian_window(dt, timescale=2):
    return np.exp(-((dt/timescale)**2)/2)

@njit
def clipped_gaussian_window(dt, timescale=2, nclip=5):
    if np.abs(dt) > nclip*timescale:
        return 0.
    return np.exp(-((dt/timescale)**2)/2)
@njit
def weighted_moving_average(y, t, errors, window_fn=gaussian_window, timescale=2):
    windows = window_fn(t.reshape(-1,1)-t.reshape(1,-1), timescale)
    weights = 1/errors**2
    wma = np.dot(windows,weights*y)/np.dot(windows, weights)
    return (wma, 
            weighted_moving_average_err(weights, windows),
            weighted_moving_average_scatter(y, wma, weights, windows))

@njit
def weighted_moving_average_err(weights, windows):
    return np.power(np.dot(windows, weights), -1)*np.sqrt(np.dot(windows**2, weights))

@njit
def weighted_moving_average_scatter(y, wma, weights, windows):
    return np.sqrt(np.dot(windows,weights*np.power((y-wma), 2))/np.dot(windows, weights))

@njit
def weighted_moving_average_skewness(y, wma, scatter, weights, windows):
    m3 = np.dot(windows,weights*np.power((y-wma), 3))/np.dot(windows, weights)
    return m3/scatter**(3/2)

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

def find_persistent_excursions(df,
        z_threshold=3, achromatic_sigma=3, timescale=2, n_measured=4, 
        duration=5, restrict_to_indices=None, usescatter=True):
    df = df.sort_values('mjd')
    no_outliers = df.iloc[reject_outliers_args(df['deltamag'].to_numpy())]
    std = np.std(no_outliers['deltamag'].to_numpy())
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

    if 'outdir' in metadata:
        outdir = metadata['outdir']
    else: 
        outdir = '/'.join(lcfiles[0].split('/')[:-1])+'/searches/'
        metadata['outdir'] = outdir
    os.makedirs(metadata['outdir']+f'/tmp-{timestamp}/', exist_ok=True)

    for file in tqdm.tqdm(lcfiles):
        file_excursions = {}
        df = pd.read_parquet(file)
        gb = df.groupby('objectid',observed=True)
        for objid in tqdm.tqdm(list(gb.groups.keys()), leave=False):
            lc = gb.get_group(objid)
            excs = find_persistent_excursions(lc, 
                       restrict_to_indices=np.concatenate(search_domains[objid]),
                       **search_params)
            file_excursions[objid] = excs
        filename = lcfiles[0].split('/')[-1]
        tmpfile = outdir+f'/tmp-{timestamp}/'+filename+'-search.pickle'
        with open(tmpfile, 'wb') as f:
            pickle_data = (metadata, search_params, file_excursions)
            pickle.dump(pickle_data, f)
        tmpfiles.append(tmpfile)

        del df, gb
        gc.collect()
    return consolidate_search_files_for_microlensing_events(tmpfiles)

def consolidate_search_files_for_microlensing_events(partialfiles):
    with open(partialfiles[0], 'rb') as f:
        metadata, search_params, excursions = pickle.load(f)
    for file in partialfiles:
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

def check_segment(lc: pd.DataFrame(), timescale=0.5):
    """Utility to check whether segment_times preserves all the indices 
       in lightcurve lc. 

    Args:
        lc (pd.DataFrame): _description_
        timescale (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """    
    segs = segment_times(lc, timescale)
    indices = pd.Index([])
    fig,ax = plt.subplots()
    for seg in segs:
        indices = indices.union(seg)
        ax.plot(lc.loc[seg, 'mjd'],lc.loc[seg,'deltamag'], linestyle='None', marker='.')
    plt.show()
    print(len(indices))
    print(len(lc.index))
    print(lc.index.difference(indices))
    return segs