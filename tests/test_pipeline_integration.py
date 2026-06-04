"""Integration: the real file pipeline end to end on the fixture set.

find_persistent_excursions (via search_files_for_excursions) -> fit_excursions
-> make_fit_excursions_df -> cuts, reading and writing real parquet/pickle.
Assertions are structural/scientific (not float-exact, that is test_parity).
"""
import os
import pickle

import numpy as np
import pytest

import swellar
import fixture_build as fb


def _meta(outdir):
    return {'outdir': str(outdir), 'outfile': 'search.pickle', 'fitoutfile': 'fits.pickle'}


@pytest.fixture
def search_results(working, tmp_path):
    meta = _meta(str(tmp_path))
    results = swellar.search_files_for_excursions(
        [working['mini_lc']], working['ws_regions_obj'], dict(meta), {})
    return results


def test_search_detects_only_injected_objects(search_results, working):
    _, _, excursions = search_results
    detected = set(swellar.reduce_excursions(excursions))
    nondet = set(swellar.get_nondetections(excursions))
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
    fitresults, fitfails, fitdups = swellar.fit_excursions(
        excursions, [working['mini_lc']], dict(meta), {}, rng=np.random.default_rng(0))
    fitdf = swellar.make_fit_excursions_df(fitresults)

    assert list(fitdf.columns) == ['objectid', 'excnum', 'pval', 'n_fit', 'n_out',
                                   'cond_num', 'impact_parameter', 'crossing_time',
                                   'peak_time', 'two_sample']
    assert len(fitdf) >= 1
    # every fit in this fixture has ample data outside the event -> two-sample KS
    assert fitdf['two_sample'].all()

    # cut_pcov must remove ill-conditioned (degenerate) fits and keep good ones.
    # The fixture produces one well-conditioned and one rail-to-the-bound fit.
    assert (fitdf['cond_num'] > 1e5).any(), 'expected a degenerate fit to cut'
    kept = swellar.cut_pcov(fitdf)
    assert (kept['cond_num'] < 1e5).all()
    assert len(kept) < len(fitdf)


def test_fit_results_pickle_roundtrip(working, tmp_path, search_results):
    _, _, excursions = search_results
    meta = _meta(str(tmp_path))
    swellar.fit_excursions(excursions, [working['mini_lc']], dict(meta), {},
                         rng=np.random.default_rng(0))
    out = os.path.join(meta['outdir'], meta['fitoutfile'])
    assert os.path.exists(out)
    with open(out, 'rb') as f:
        fitresults, fitfails, fitdups, m, p = pickle.load(f)
    assert isinstance(fitresults, list)


def test_combined_driver_matches_manual(working, tmp_path):
    """search_files_for_microlensing_events should run search+fit as one step."""
    meta = _meta(str(tmp_path))
    exc_res, fit_res = swellar.search_files_for_microlensing_events(
        [working['mini_lc']], working['ws_regions_obj'], dict(meta), {})
    _, _, excursions = exc_res
    fitresults, fitfails, fitdups = fit_res
    detected = set(swellar.reduce_excursions(excursions))
    assert set(working['synth_ids']) <= detected
    assert isinstance(fitresults, list)


def test_combined_driver_rejects_unknown_param(working, tmp_path):
    with pytest.raises(ValueError):
        swellar.search_files_for_microlensing_events(
            [working['mini_lc']], working['ws_regions_obj'],
            _meta(str(tmp_path)), {'not_a_real_param': 1})


def test_fit_excursions_rng_is_reproducible(tmp_path):
    """The injected Generator controls the small-sample KS reference: same seed
    reproduces, different seed differs (validates the RNG-injection fix)."""
    lc = fb.small_sample_lc()
    p = os.path.join(str(tmp_path), 'small.parquet')
    lc.to_parquet(p)
    excs = {'small_0': swellar.find_persistent_excursions(lc)}
    meta = _meta(tmp_path)

    def run(seed):
        fr, _, _ = swellar.fit_excursions(excs, [p], dict(meta), {},
                                        rng=np.random.default_rng(seed))
        return swellar.make_fit_excursions_df(fr)

    a, b, c = run(0), run(0), run(1)
    assert not a['two_sample'].any()  # confirms the RNG (small-sample) branch is hit
    np.testing.assert_array_equal(a['pval'].to_numpy(), b['pval'].to_numpy())
    assert not np.allclose(a['pval'].to_numpy(), c['pval'].to_numpy())


def test_fit_excursions_context_size_controls_window(tmp_path):
    """context_size now flows through to extend_lc (it was previously ignored,
    so the fit window was always the 100-day default): a larger padding pulls
    more epochs into the fit."""
    lc = fb.small_sample_lc()
    p = os.path.join(str(tmp_path), 'small.parquet')
    lc.to_parquet(p)
    excs = {'small_0': swellar.find_persistent_excursions(lc)}
    meta = _meta(tmp_path)

    def n_fit(cs):
        fr, _, _ = swellar.fit_excursions(excs, [p], dict(meta), {}, context_size=cs,
                                        rng=np.random.default_rng(0))
        return int(swellar.make_fit_excursions_df(fr)['n_fit'].iloc[0])

    assert n_fit(10) < n_fit(30) < n_fit(100)
