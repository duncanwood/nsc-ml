"""Integration: the real file pipeline end to end on the fixture set.

find_persistent_excursions (via search_files_for_excursions) -> fit_excursions
-> make_fit_excursions_df -> cuts, reading and writing real parquet/pickle.
Assertions are structural/scientific (not float-exact -- that is test_parity).
"""
import os
import pickle

import numpy as np
import pytest

import nscml


def _meta(outdir):
    # outdir must end in '/': consolidate_search_files_for_excursions builds its
    # path as outdir+outfile with no separator (see AUDIT, path construction).
    return {'outdir': str(outdir).rstrip('/') + '/',
            'outfile': 'search.pickle', 'fitoutfile': 'fits.pickle'}


@pytest.fixture
def search_results(working, tmp_path):
    meta = _meta(str(tmp_path))
    results = nscml.search_files_for_excursions(
        [working['mini_lc']], working['ws_regions_obj'], dict(meta), {})
    return results


def test_search_detects_only_injected_objects(search_results, working):
    _, _, excursions = search_results
    detected = set(nscml.reduce_excursions(excursions))
    nondet = set(nscml.get_nondetections(excursions))
    # the four real baseline objects must be clean; the injected ones detected
    for rid in working['real_ids']:
        assert rid in nondet
    for sid in working['synth_ids']:
        assert sid in detected


def test_search_writes_consolidated_pickle(search_results, tmp_path):
    meta, params, excursions = search_results
    out = os.path.join(meta['outdir'], meta['outfile'])
    assert os.path.exists(out)
    with open(out, 'rb') as f:
        m2, p2, e2 = pickle.load(f)
    assert set(e2) == set(excursions)


def test_fit_and_cuts(working, tmp_path, search_results):
    _, _, excursions = search_results
    meta = _meta(str(tmp_path))
    np.random.seed(0)
    fitresults, fitfails, fitdups = nscml.fit_excursions(
        excursions, [working['mini_lc']], dict(meta), {})
    fitdf = nscml.make_fit_excursions_df(fitresults)

    assert list(fitdf.columns) == ['objectid', 'excnum', 'pval', 'n_fit', 'n_out',
                                   'cond_num', 'impact_parameter', 'crossing_time',
                                   'peak_time', 'two_sample']
    assert len(fitdf) >= 1
    # every fit in this fixture has ample data outside the event -> two-sample KS
    assert fitdf['two_sample'].all()

    # cut_pcov must remove ill-conditioned (degenerate) fits and keep good ones.
    # The fixture produces one well-conditioned and one rail-to-the-bound fit.
    assert (fitdf['cond_num'] > 1e5).any(), 'expected a degenerate fit to cut'
    kept = nscml.cut_pcov(fitdf)
    assert (kept['cond_num'] < 1e5).all()
    assert len(kept) < len(fitdf)


def test_fit_results_pickle_roundtrip(working, tmp_path, search_results):
    _, _, excursions = search_results
    meta = _meta(str(tmp_path))
    np.random.seed(0)
    nscml.fit_excursions(excursions, [working['mini_lc']], dict(meta), {})
    out = os.path.join(meta['outdir'], meta['fitoutfile'])
    assert os.path.exists(out)
    with open(out, 'rb') as f:
        fitresults, fitfails, fitdups, m, p = pickle.load(f)
    assert isinstance(fitresults, list)


def test_combined_driver_matches_manual(working, tmp_path):
    """search_files_for_microlensing_events should run search+fit as one step."""
    meta = _meta(str(tmp_path))
    exc_res, fit_res = nscml.search_files_for_microlensing_events(
        [working['mini_lc']], working['ws_regions_obj'], dict(meta), {})
    _, _, excursions = exc_res
    fitresults, fitfails, fitdups = fit_res
    detected = set(nscml.reduce_excursions(excursions))
    assert set(working['synth_ids']) <= detected
    assert isinstance(fitresults, list)


def test_combined_driver_rejects_unknown_param(working, tmp_path):
    with pytest.raises(ValueError):
        nscml.search_files_for_microlensing_events(
            [working['mini_lc']], working['ws_regions_obj'],
            _meta(str(tmp_path)), {'not_a_real_param': 1})
