import time
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
import pandas as pd
from . import nsctools
from .nsctools import color_filter, marker_map

# Explicit public API so `from .plot import *` (in __init__) does not leak the
# imported matplotlib/numpy/pandas names into the package namespace.
__all__ = [
    'rows_in_bin', 'bin_func', 'mean_of_col', 'std_of_col', 'perc_of_col',
    'compare_cut_fn', 'compare_cut', 'compare_cut_2', 'plot_hist_color',
    'plot_pval_hist', 'plot_lc', 'plot_obj', 'plot_deltamags', 'plot_obj_dm',
    'plot_weighted_moving_average_df', 'plot_excursion_region',
    'plot_example_fits',
]

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
        # return np.full_like(percentiles, np.nan)
        return np.nan
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

def plot_hist_color(
        data, bins, colors, clabel='Color', maxcolor=None,
        mincolor=None, log=True):
    nanmask = np.isfinite(colors)
    colors = colors[nanmask]
    if maxcolor is None:
        maxcolor = np.max(colors)
    if mincolor is None:
        mincolor = np.min(colors)
    viridis = mpl.colormaps['viridis'].resampled(256)
    fig, ax = plt.subplots(1,1)
    counts, bins = np.histogram(data, bins=bins)

    bars = ax.bar(bins[:-1][nanmask], counts[nanmask], width=(bins[1:]-bins[:-1])[nanmask], align='edge', 
                color=viridis((colors-mincolor)/(maxcolor-mincolor)))
    nanbars = ax.bar(bins[:-1][~nanmask], counts[~nanmask], width=(bins[1:]-bins[:-1])[~nanmask], align='edge', 
                color='white',hatch='/',edgecolor='black')
    if log:
        plt.xscale('log')

    sm = ScalarMappable(cmap=viridis, norm=plt.Normalize(mincolor,maxcolor))
    sm.set_array([])
    cbar = plt.colorbar(sm,ax=ax)
    cbar.set_label(clabel, rotation=270,labelpad=25)
    return fig, ax, cbar

def plot_pval_hist(fitdf: pd.DataFrame, maxcolor=None):

    bins=np.logspace(-5,0,20)
    if maxcolor is None:
        maxcolor = max(np.nan_to_num(bin_func(fitdf, 'pval', bins, 
                                              lambda x: (x['n_fit'] + x['n_out']).mean())))

    fig, axes, cbar = plot_hist_color(fitdf['pval'], bins, 
                    bin_func(fitdf, 'pval',bins, lambda x: (x['n_fit'] + x['n_out']).mean()),
                    clabel='Mean number of measurements per object',
                    maxcolor=maxcolor)
    fig.suptitle('KS test p-value distributions of datapoints' 
                 + ' outside event window\nvs. residuals after'
                 + ' PSPL fit subtraction in synthetic ML events')
    plt.xlabel('p-value')
    plt.ylabel('Counts')
    return fig, axes, cbar


def plot_lc(lc: pd.DataFrame, **kwargs):
    objid = lc['objectid'].unique()
    assert len(objid)==1
    objid = objid[0]
    fig, ax = plt.subplots()
    for f in lc['filter'].unique():
        f_df = lc[lc['filter']==f]
        plt.errorbar(f_df['mjd'], f_df['mag_auto'], f_df['magerr_auto'],
                     c=nsctools.color_filter[f.lower()],linestyle='None', 
                     marker='o',  label=f)
        plt.hlines(f_df[f.lower()+'mag'], f_df['mjd'].to_numpy().min(),
                   f_df['mjd'].to_numpy().max(), 
                   color=nsctools.color_filter[f.lower()], linestyle='dashed')
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
    gb = lc.groupby(['filter', 'instrument'], observed=True)
    for f, instrument in gb.groups.keys():
        f_df = lc[lc['filter']==f]
        plt.errorbar(f_df['mjd'], f_df['deltamag'], f_df['magerr_auto'],
                     c=color_filter[f.lower()],linestyle='None', 
                     markersize=5, marker=marker_map[instrument], capsize=0)
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
    if 'show' in kwargs and kwargs['show']:
        plt.show()

def plot_obj_dm(id: str, dmgroupby, **kwargs):
    plot_deltamags(dmgroupby.get_group(id), **kwargs)

def plot_weighted_moving_average_df(df, usescatter=True, timescale=2, outliers_cutoff=3, **kwargs):
    df=df.sort_values('mjd')
    d=df['deltamag'].to_numpy()
    e=df['magerr_auto'].to_numpy()
    t=df['mjd'].to_numpy()
    plot_deltamags(df, **kwargs)
    wma, errs, scatter = nsctools.weighted_moving_average(d, t, e, timescale=timescale)

    no_outliers = df.iloc[nsctools.reject_outliers_args(df['deltamag'].to_numpy(), outliers_cutoff)]
    std = np.std(no_outliers['deltamag'].to_numpy())

    if usescatter:
        confidence = np.sqrt(errs**2 + scatter**2 + std**2)
    else: 
        confidence = np.sqrt(errs**2 + std**2)
    plt.plot(t,wma, linestyle='dotted')
    plt.fill_between(t,wma-confidence,wma+confidence, alpha=.2)

def plot_excursion_region(lc, region, timescale=2, context_size=100, **kwargs):
    xlims = np.percentile(lc.loc[region,'mjd'].to_numpy(),(0,100)) \
               + np.array([-context_size,context_size])
    plot_weighted_moving_average_df(lc,timescale=timescale, xlims=xlims, **kwargs)
    ymin = np.min(  lc.loc[region,'deltamag'].to_numpy()
                  - lc.loc[region,'magerr_auto'].to_numpy())
    
    ymax = np.max(lc.loc[region,'deltamag'].to_numpy() + lc.loc[region,'magerr_auto'].to_numpy())
    plt.vlines(np.percentile(lc.loc[region,'mjd'].to_numpy(),(0,100)), ymin,ymax, linestyle='dashed')
    plt.fill_between(np.percentile(lc.loc[region,'mjd'].to_numpy(),(0,100)), ymin,ymax, alpha=.2)


def plot_example_fits(fulldf, all_excursions, fitresults, 
                      fileenum, objfilemap, limitnum=10,outdir=None,
                      show=True):
    if limitnum is None:
        plotidx = list(fulldf.index)
    else:
        plotidx= np.random.choice(list(fulldf.index), min(limitnum, len(fulldf)), 
                              replace=False)

    if not (outdir is None):
        if not show:
            raise ValueError('Neither showing nor saving plots')
        os.makedirs(outdir, exist_ok=True)
    for idx in plotidx:
        obj = fulldf.loc[idx]['objectid']
        exc_idx = fulldf.loc[idx]['excnum']
        region = all_excursions[obj][exc_idx]
        file = fileenum[objfilemap[obj]]
        df = pd.read_parquet(file)
        df = df[df['objectid']==obj]
        df = nsctools.float_cols_to_double(df)
        print(obj)    


        fitresult = [result for result in fitresults if result[0] ==obj][exc_idx]
        fitinfo = fitresult[3][0]
        crossingtime = fitinfo[1]
        print(fitinfo)

        plot_excursion_region(df[df['objectid']==obj], region, context_size=max(crossingtime,20), timescale=5)
        
        extended_region = nsctools.extend_lc(df, region)
        ext_region_df = df.loc[extended_region].sort_values('mjd')
        mjds = ext_region_df['mjd'].to_numpy()
        mjds = np.linspace(mjds[0], mjds[-1],200)

        fitmags = nsctools.ml_f(mjds,*fitinfo)
        plt.plot(mjds, fitmags,c='black', linestyle='dashed',marker='None', label='PSPL fit')

        handles = [mpatches.Rectangle((0, 0), 1, 1, fc="white", ec="white", 
                                 lw=0, alpha=0)] * 2

        labels=[]
        labels.append(f'Condition number: {fulldf.loc[idx]["cond_num"]:.2e}')
        labels.append(f'\np-value: {fulldf.loc[idx]["pval"]:.2e}')
        plt.gca().legend(handles, labels, loc='best', fontsize='small', 
          fancybox=True, framealpha=0.7, 
          handlelength=0, handletextpad=0)

        if not (outdir is None):
            plt.savefig(outdir+f'/{obj}_excursion.pdf')
        if show:

            plt.show()
        plt.clf()
        plt.close("all")
