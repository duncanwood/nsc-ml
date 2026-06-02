"""Parity: the live library must reproduce the committed goldens.

Each test re-runs the exact capture routine (redirected to a temp dir) and
compares to golden/. This is the regression guard for the cleanup -- removing
comments and dead code must leave every one of these byte-for-byte unchanged
(kernels, KS, seeded paths) or within solver tolerance (fit floats).
"""
import json
import os

import numpy as np
import pytest

import capture_golden as cg


def _run(monkeypatch, tmp_path, fn):
    monkeypatch.setattr(cg, 'GOLDEN', str(tmp_path))
    fn()
    return tmp_path


def test_kernels_parity(monkeypatch, tmp_path, golden_kernels):
    out = _run(monkeypatch, tmp_path, cg.capture_kernels)
    live = dict(np.load(out / 'kernels.npz'))
    assert set(live) == set(golden_kernels), 'kernel key set changed'
    for k in golden_kernels:
        # numba kernels are deterministic -> bit-exact
        np.testing.assert_array_equal(live[k], golden_kernels[k], err_msg=k)


def test_ks_parity(monkeypatch, tmp_path, golden_ks):
    out = _run(monkeypatch, tmp_path, cg.capture_ks)
    with open(out / 'ks.json') as f:
        live = json.load(f)
    assert live == golden_ks  # D-statistic + kstwo.sf p-value are deterministic


def test_pipeline_parity(monkeypatch, tmp_path, golden_pipeline, fit_cmp):
    out = _run(monkeypatch, tmp_path, cg.capture_pipeline)
    with open(out / 'pipeline.json') as f:
        live = json.load(f)
    # excursion index sets are exact (np.split of an integer index, no float)
    assert live['search_excursions'] == golden_pipeline['search_excursions']
    assert live['direct_excursions'] == golden_pipeline['direct_excursions']
    assert live['fit_meta'] == golden_pipeline['fit_meta']
    fit_cmp(live['fit_df'], golden_pipeline['fit_df'])


def test_pipeline_small_parity(monkeypatch, tmp_path, golden_pipeline_small, fit_cmp):
    out = _run(monkeypatch, tmp_path, cg.capture_pipeline_small)
    with open(out / 'pipeline_small.json') as f:
        live = json.load(f)
    assert live['n_excursions_found'] == golden_pipeline_small['n_excursions_found']
    assert live['fit_meta'] == golden_pipeline_small['fit_meta']
    # the seeded np.random.normal branch must still be the one exercised
    assert all(r['two_sample'] is False for r in live['fit_df'])
    fit_cmp(live['fit_df'], golden_pipeline_small['fit_df'])


def test_synth_parity(monkeypatch, tmp_path, golden_synth):
    out = _run(monkeypatch, tmp_path, cg.capture_synth)
    with open(out / 'synth.json') as f:
        live = json.load(f)
    # seeded (monkeypatched default_rng + stdlib random) -> fully deterministic
    assert live == golden_synth
