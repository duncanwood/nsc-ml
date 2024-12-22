
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity

from .nsctools import *



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
    
@njit
def weighted_moving_average_skewness(y, wma, scatter, weights, windows):
    m3 = np.dot(windows,weights*np.power((y-wma), 3))/np.dot(windows, weights)
    return m3/scatter**(3/2)

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

def get_lc(objid: str, objfilemap: dict, fileenum: dict, **kwargs):
    return pd.read_parquet(fileenum[objfilemap[objid]],filters=[("objectid", "=",objid)], **kwargs)
