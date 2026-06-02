"""Capture golden outputs from the CURRENT nscml code.

Run once before any source edits; the goldens are committed and the parity
tests assert the (cleaned) library still reproduces them. Re-running after a
behaviour-preserving cleanup must leave golden/ unchanged.

    python tests/capture_golden.py
"""
import json
import os
import pickle
import tempfile

import numpy as np

import nscml
import fixture_build as fb

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, 'golden')


def _f(x):
    return float(x)


def capture_kernels():
    ki = fb.kernel_inputs()
    t, y, errs, w = ki['t'], ki['y'], ki['errs'], ki['weights']
    out = {}

    # windowed weighted moving average / scatter (the scientific core)
    wma, wme, wms = nscml.sparse_gaussian_wma(y, t, w, timescale=5.0, nclip=10)
    out['sgw_wma'], out['sgw_wme'], out['sgw_wms'] = wma, wme, wms
    out['sgwms_direct'] = nscml.sparse_gaussian_wms(y, t, w, wma, timescale=5.0, nclip=10)
    out['dense_window'] = np.asarray(
        nscml.dense_sparse_gaussian_window(t, timescale=5.0, nclip=10))

    cwma, cerr, cscat = nscml.compute_weighted_moving_average(y, t, errs, timescale=5.0)
    out['cwma_wma'], out['cwma_err'], out['cwma_scatter'] = cwma, cerr, cscat
    gwma, gerr, gscat = nscml.weighted_moving_average_gaussian(y, t, errs, timescale=5.0)
    out['gauss_wma'], out['gauss_err'], out['gauss_scatter'] = gwma, gerr, gscat

    sp_wma, sp_err, sp_scat = nscml.weighted_moving_average(
        y, t, errs, sparse=True, timescale=5.0, nclip=10)
    out['disp_sparse_wma'] = sp_wma
    de_wma, de_err, de_scat = nscml.weighted_moving_average(
        y, t, errs, sparse=False, timescale=5.0)
    out['disp_dense_wma'] = de_wma

    windows = nscml.gaussian_window(t.reshape(-1, 1) - t.reshape(1, -1), 5.0)
    out['gaussian_window'] = windows
    out['wma_err_direct'] = nscml.weighted_moving_average_err(w, windows)
    out['wma_scatter_direct'] = nscml.weighted_moving_average_scatter(y, cwma, w, windows)
    dts = np.array([-30.0, -7.0, -2.0, 0.0, 1.5, 6.0, 11.0, 26.0])
    out['clipped_window'] = np.array(
        [nscml.clipped_gaussian_window(d, 2.0, 5) for d in dts])

    avg, std = nscml.weighted_avg_and_std(y, w)
    out['wavg'] = np.array([avg, std])

    # PSPL amplification / model / Jacobian
    ai = fb.amplification_inputs()
    at, u0, tE, t0, blend = ai['t'], ai['u0'], ai['tE'], ai['t0'], ai['blend']
    amp = nscml.microlensing_amplification(at, u0, tE, t0)
    out['amp'] = amp
    out['amp_blend'] = nscml.microlensing_amplification(at, u0, tE, t0, blend)
    out['amp_to_mag'] = nscml.amp_to_mag(amp)
    out['ml_f'] = nscml.ml_f(at, u0, tE, t0)
    out['ml_jac'] = nscml.ml_jac(at, u0, tE, t0)

    np.savez(os.path.join(GOLDEN, 'kernels.npz'), **out)
    print('wrote kernels.npz with', len(out), 'arrays')


def capture_ks():
    k = fb.ks_inputs()
    dw, pw = nscml.ks_weighted(k['a'], k['b'], k['wa'], k['wb'])
    de, pe = nscml.ks_weighted(k['a'], k['b'], np.ones_like(k['a']), np.ones_like(k['b']))
    payload = {'weighted': [_f(dw), _f(pw)], 'equal': [_f(de), _f(pe)]}
    with open(os.path.join(GOLDEN, 'ks.json'), 'w') as f:
        json.dump(payload, f, indent=2)
    print('wrote ks.json', payload)


def _df_records(fitdf):
    recs = []
    for _, row in fitdf.iterrows():
        recs.append({
            'objectid': str(row['objectid']),
            'excnum': int(row['excnum']),
            'pval': _f(row['pval']),
            'n_fit': int(row['n_fit']),
            'n_out': int(row['n_out']),
            'cond_num': _f(row['cond_num']),
            'impact_parameter': _f(row['impact_parameter']),
            'crossing_time': _f(row['crossing_time']),
            'peak_time': _f(row['peak_time']),
            'two_sample': bool(row['two_sample']),
        })
    return recs


def capture_pipeline():
    real_path = os.path.join(HERE, 'fixtures', 'real_objects.parquet')
    tmp = tempfile.mkdtemp(prefix='nscml_golden_')
    info = fb.build_working_fixtures(real_path, tmp)
    with open(info['ws_regions'], 'rb') as f:
        ws_regions = pickle.load(f)
    lcfiles = [info['mini_lc']]

    # direct detection on one injected object (unrestricted)
    import pandas as pd
    rb = pd.read_parquet(info['mini_lc'])
    sid = info['synth_ids'][0]
    lc = rb[rb['objectid'].astype(str) == sid]
    direct = nscml.find_persistent_excursions(lc)
    direct_regions = [[int(i) for i in r] for r in direct]

    meta = {'outdir': tmp, 'outfile': 'search.pickle', 'fitoutfile': 'fits.pickle'}
    _, _, excursions = nscml.search_files_for_excursions(
        lcfiles, ws_regions, dict(meta), {})
    exc_ser = {str(k): [[int(i) for i in r] for r in v] for k, v in excursions.items()}

    np.random.seed(0)  # seeds the small-sample np.random.normal branch if hit
    fitresults, fitfails, fitdups = nscml.fit_excursions(
        excursions, lcfiles, dict(meta), {})
    fitdf = nscml.make_fit_excursions_df(fitresults)

    payload = {
        'direct_excursions': {sid: direct_regions},
        'search_excursions': exc_ser,
        'fit_df': _df_records(fitdf),
        'fit_meta': {'n_fail': len(fitfails), 'n_dup': len(fitdups)},
        'synth_ids': info['synth_ids'],
        'real_ids': info['real_ids'],
    }
    with open(os.path.join(GOLDEN, 'pipeline.json'), 'w') as f:
        json.dump(payload, f, indent=2)
    print('wrote pipeline.json:',
          {k: (len(v) if hasattr(v, '__len__') else v) for k, v in payload.items()})


def capture_pipeline_small():
    tmp = tempfile.mkdtemp(prefix='nscml_small_')
    lc = fb.small_sample_lc()
    p = os.path.join(tmp, 'small.parquet')
    lc.to_parquet(p)
    excs = {'small_0': nscml.find_persistent_excursions(lc)}
    meta = {'outdir': tmp, 'outfile': 'search.pickle', 'fitoutfile': 'fits.pickle'}
    np.random.seed(0)  # the small-sample branch draws np.random.normal
    fitresults, fitfails, fitdups = nscml.fit_excursions(excs, [p], dict(meta), {})
    fitdf = nscml.make_fit_excursions_df(fitresults)
    # guard against a vacuous golden: the branch under test must be reached
    assert len(fitdf) > 0 and not fitdf['two_sample'].any(), (
        'small-sample fixture did not exercise the np.random.normal branch')
    payload = {
        'n_excursions_found': len(excs['small_0']),
        'fit_df': _df_records(fitdf),
        'fit_meta': {'n_fail': len(fitfails), 'n_dup': len(fitdups)},
    }
    with open(os.path.join(GOLDEN, 'pipeline_small.json'), 'w') as f:
        json.dump(payload, f, indent=2)
    print('wrote pipeline_small.json:', payload['fit_meta'],
          'two_sample=', [r['two_sample'] for r in payload['fit_df']])


def capture_synth():
    real_path = os.path.join(HERE, 'fixtures', 'real_objects.parquet')
    tmp = tempfile.mkdtemp(prefix='nscml_synth_')
    events = fb.events_table()
    # ws_regions keyed by real id, from the same lcfile we pass in
    import pandas as pd
    rb = pd.read_parquet(real_path)
    rb['objectid'] = rb['objectid'].astype(str)
    ws_regions = {rid: fb._well_sampled(rb[rb['objectid'] == rid])
                  for rid in fb.REAL_OBJECT_IDS}

    with fb.seeded_synth_rng(0):
        nscml.generate_synthetic_microlensing_events_from_population(
            [real_path], events, ws_regions, tmp, 'gold')
    info_pkl = os.path.join(tmp, 'synth-gold', 'synth-gold-info.pickle')
    with open(info_pkl, 'rb') as f:
        _, object_event_df = pickle.load(f)

    recs = []
    for _, row in object_event_df.iterrows():
        recs.append({
            'objectid': str(row['objectid']),
            'synthid': str(row['synthid']),
            'event_index': int(row['event_index']),
            'crossing_time': _f(row['crossing_time']),
            'umin': _f(row['umin']),
            'peak_time': _f(row['peak_time']),
        })
    with open(os.path.join(GOLDEN, 'synth.json'), 'w') as f:
        json.dump({'object_event_df': recs}, f, indent=2)
    print('wrote synth.json with', len(recs), 'rows')


if __name__ == '__main__':
    os.makedirs(GOLDEN, exist_ok=True)
    capture_kernels()
    capture_ks()
    capture_pipeline()
    capture_pipeline_small()
    capture_synth()
    print('GOLDEN CAPTURE COMPLETE')
