from nsctools import *
import matplotlib as mpl
from matplotlib.cm import ScalarMappable

def rows_in_bin(df, col, minval, maxval):
    return df[(df[col] >= minval) & (df[col] < maxval)]
def bin_func(df: pd.DataFrame, col, bins, func, *args):
    bin_results = []
    for i in range(len(bins)-1):
        l, r = bins[i],bins[i+1]
        subdf = rows_in_bin(df, col, l, r)
        bin_results.append(func(subdf, *args))
    return bin_results
def mean_of_col(events, meancol):
    return events[meancol].to_numpy().mean()
def std_of_col(events, meancol):
    return events[meancol].to_numpy().std()
def perc_of_col(events, meancol, percentiles):
    col_vals = events[meancol].to_numpy()
    if col_vals.shape[0] > 0:
        return np.percentile(col_vals, percentiles)
    else:
        return np.full_like(percentiles, np.nan)

def compare_cut_fn(rdf, sdf, cutfn):
    post_rdf = cutfn(rdf)
    post_sdf = cutfn(sdf)
    return compare_cut(rdf, post_rdf, sdf, post_sdf)

def compare_cut(rdf,post_rdf, sdf, post_sdf):
    real_cut_frac = len(post_rdf)/len(rdf)
    synth_cut_frac = len(post_sdf)/len(sdf)
    snr_factor = synth_cut_frac/real_cut_frac
    return real_cut_frac, synth_cut_frac, snr_factor

def compare_cut_2(rdf,post_rdf, sdf, post_sdf):
    real_cut_frac = len(post_rdf)/len(rdf)
    synth_cut_frac = len(post_sdf)/len(sdf)
    rdiff = (len(rdf) - len(post_rdf))/len(rdf)
    sdiff = (len(sdf) - len(post_sdf))/len(sdf)
    purity = rdiff/(sdiff)
    return real_cut_frac, synth_cut_frac, purity

def plot_hist_color(data, bins, colors, clabel='Color', maxcolor=None):
    if maxcolor is None:
        maxcolor = np.max(colors)
    viridis = mpl.colormaps['viridis'].resampled(256)
    fig, ax = plt.subplots(1,1)
    counts, bins = np.histogram(data, bins=bins)
    bars = ax.bar(bins[:-1], counts, width=bins[1:]-bins[:-1], align='edge', 
                color=viridis(colors/maxcolor))
    plt.xscale('log')

    sm = ScalarMappable(cmap=viridis, norm=plt.Normalize(0,maxcolor))
    sm.set_array([])
    cbar = plt.colorbar(sm,ax=ax)
    cbar.set_label(clabel, rotation=270,labelpad=25)
    return fig, ax, cbar