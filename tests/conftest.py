"""Shared fixtures, tolerances and helpers for the nscml suite.

Run with the env that holds the pinned deps:
    /Users/duncan/mambaforge/envs/nsc/bin/python -m pytest
"""
import json
import os
import pickle

import numpy as np
import pytest

import fixture_build as fb

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, 'golden')
FIXTURES = os.path.join(HERE, 'fixtures')

# Numba kernels and seeded paths are asserted bit-exact. The fit pipeline
# (scipy curve_fit + weighted KS + np.linalg.cond) is iterative and BLAS-backed:
# bit-identical across reruns on this machine, but a tolerance is allowed for
# last-ULP drift on other platforms. 1e-9 is ~4+ orders below any scientific
# significance here (p-values ~1e-5..1, params O(1e-2..1e2), cond_num to ~1e7).
FIT_RTOL = 1e-9
FIT_ATOL = 1e-12

FIT_FLOAT_FIELDS = ('pval', 'cond_num', 'impact_parameter', 'crossing_time', 'peak_time')
FIT_EXACT_FIELDS = ('objectid', 'excnum', 'n_fit', 'n_out', 'two_sample')


def _load_json(name):
    with open(os.path.join(GOLDEN, name)) as f:
        return json.load(f)


@pytest.fixture(scope='session')
def golden_kernels():
    return dict(np.load(os.path.join(GOLDEN, 'kernels.npz')))


@pytest.fixture(scope='session')
def golden_ks():
    return _load_json('ks.json')


@pytest.fixture(scope='session')
def golden_pipeline():
    return _load_json('pipeline.json')


@pytest.fixture(scope='session')
def golden_pipeline_small():
    return _load_json('pipeline_small.json')


@pytest.fixture(scope='session')
def golden_synth():
    return _load_json('synth.json')


@pytest.fixture(scope='session')
def real_path():
    p = os.path.join(FIXTURES, 'real_objects.parquet')
    if not os.path.exists(p):
        pytest.fail(f'missing committed fixture {p}; run `python fixture_build.py --extract`')
    return p


@pytest.fixture(scope='session')
def working(real_path, tmp_path_factory):
    """Integration fixture: real objects + injected synthetic events, plus the
    search domains, built once per session into a temp dir."""
    outdir = str(tmp_path_factory.mktemp('working'))
    info = fb.build_working_fixtures(real_path, outdir)
    with open(info['ws_regions'], 'rb') as f:
        info['ws_regions_obj'] = pickle.load(f)
    return info


def _compare_fit_records(live, golden, rtol=FIT_RTOL, atol=FIT_ATOL):
    """Field-wise compare of make_fit_excursions_df records: exact on the
    discrete fields, allclose on the solver-derived floats."""
    assert len(live) == len(golden), f'row count {len(live)} != {len(golden)}'
    key = lambda r: (r['objectid'], r['excnum'])
    live = sorted(live, key=key)
    golden = sorted(golden, key=key)
    for lr, gr in zip(live, golden):
        for f in FIT_EXACT_FIELDS:
            assert lr[f] == gr[f], f'{f}: {lr[f]!r} != {gr[f]!r}'
        for f in FIT_FLOAT_FIELDS:
            np.testing.assert_allclose(lr[f], gr[f], rtol=rtol, atol=atol,
                                       err_msg=f'{gr["objectid"]} {f}')


@pytest.fixture
def fit_cmp():
    return _compare_fit_records
