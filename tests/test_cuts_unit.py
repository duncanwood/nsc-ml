"""Unit tests for the post-fit selection cuts.

The cuts are simple boolean masks, but the boundary semantics (strict vs
inclusive) are what decide whether a real event survives, so they are pinned
explicitly here, especially the cond_lim=1e5 degeneracy cut.
"""
import numpy as np
import pandas as pd
import pytest

import swellar


def _fitdf(**cols):
    n = len(next(iter(cols.values())))
    base = {
        'objectid': [f'o{i}' for i in range(n)],
        'excnum': [0] * n,
        'pval': [0.5] * n,
        'n_fit': [20] * n,
        'n_out': [20] * n,
        'cond_num': [1.0] * n,
        'impact_parameter': [0.1] * n,
        'crossing_time': [30.0] * n,
        'peak_time': [100.0] * n,
        'two_sample': [True] * n,
    }
    base.update(cols)
    return pd.DataFrame(base)


def test_cut_pcov_boundary_is_strict():
    df = _fitdf(cond_num=[1e5 - 1, 1e5, 1e5 + 1])
    kept = swellar.cut_pcov(df)  # default cond_lim = 1e5
    assert list(kept['cond_num']) == [1e5 - 1]  # exactly 1e5 is excluded (strict <)


def test_cut_pcov_custom_limit():
    df = _fitdf(cond_num=[10.0, 100.0, 1000.0])
    assert list(swellar.cut_pcov(df, cond_lim=500)['cond_num']) == [10.0, 100.0]


def test_cut_crossing_time_min_only():
    df = _fitdf(crossing_time=[0.5, 1.0, 1.5, 40.0])
    kept = swellar.cut_crossing_time(df, timemin=1)  # strict >, timemax None
    assert list(kept['crossing_time']) == [1.5, 40.0]


def test_cut_crossing_time_both_bounds():
    df = _fitdf(crossing_time=[0.5, 5.0, 40.0, 500.0])
    kept = swellar.cut_crossing_time(df, timemin=1, timemax=100)
    assert list(kept['crossing_time']) == [5.0, 40.0]  # open interval (1, 100)


def test_cut_by_pval_inclusive():
    df = _fitdf(pval=[0.01, 0.05, 0.10])
    assert list(swellar.cut_by_pval(df, 0.05)['pval']) == [0.05, 0.10]  # >= threshold


def test_cut_by_npoints_inclusive():
    df = _fitdf(n_fit=[5, 10, 10], n_out=[4, 0, 5])  # totals 9, 10, 15
    assert list(swellar.cut_by_npoints(df, 10)['objectid']) == ['o1', 'o2']


def test_cut_high_points_low_p_keeps_low_count_or_high_p():
    # keep if pval high OR total points < npoints
    df = _fitdf(pval=[0.5, 0.01, 0.01], n_fit=[20, 20, 1], n_out=[20, 20, 1])
    kept = swellar.cut_high_points_low_p(df, npoints=10, pval=0.05)
    assert set(kept['objectid']) == {'o0', 'o2'}  # o1: low p AND many points -> dropped


def test_cut_high_points_inout_low_p():
    # keep if pval high OR (n_fit < npoints) OR (n_out < npoints)
    df = _fitdf(pval=[0.5, 0.01, 0.01], n_fit=[20, 20, 3], n_out=[20, 20, 20])
    kept = swellar.cut_high_points_inout_low_p(df, npoints=10, pval=0.05)
    assert set(kept['objectid']) == {'o0', 'o2'}


def test_cuts_compose_and_preserve_columns():
    df = _fitdf(pval=[0.5, 0.5], cond_num=[10.0, 1e9], crossing_time=[30.0, 30.0])
    out = swellar.cut_crossing_time(swellar.cut_pcov(swellar.cut_by_pval(df, 0.05)), 1, 100)
    assert list(out['objectid']) == ['o0']  # o1 dropped by cond_num
    assert list(out.columns) == list(df.columns)
