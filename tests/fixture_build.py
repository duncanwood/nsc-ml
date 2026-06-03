"""Deterministic fixture construction for the nscml test suite.

Single source of truth shared by ``conftest.py`` (working fixtures for the
live test run) and ``capture_golden.py`` (golden generation). Keeping the
construction here guarantees the goldens and the tests see byte-identical
inputs.

Two data substrates:

* Kernel inputs, fixed analytic arrays defined in code, no I/O, used for
  the WMA/WMS/window/amplification kernel goldens.
* A small real photometry slice (``fixtures/real_objects.parquet``, four
  well-sampled NSC objects extracted once from ``notebooks/test.parquet``)
  from which the integration fixture is built by injecting PSPL events.

The integration fixture deliberately reproduces the real pipeline's index
contract: a synthetic object is a *copy* of its source object's rows, so it
keeps the source's index labels. ``search_domains`` is keyed by the source
id (recovered from the synthetic id via the ``_ml_`` regex in
``search_files_for_excursions``), and restriction works only because the
copy shares those labels. ``build_working_fixtures`` asserts that contract
holds after the parquet round-trip rather than silently yielding zero
detections.
"""
import os
import pickle

import numpy as np
import pandas as pd

import nscml

# Four well-sampled objects (40-54 epochs, ~1500-1900 day baselines, 5-6
# bands), chosen deterministically as the first such ids in test.parquet.
REAL_OBJECT_IDS = ['163053_2831', '163053_2832', '163053_2836', '163053_2837']
# Inject a PSPL event into a copy of each of these two; the other two stay
# as real (baseline) objects to exercise the no-excursion path.
SYNTH_SOURCE_IDS = ['163053_2832', '163053_2837']
CAT_COLS = ['objectid', 'exposure', 'filter', 'instrument']

# Injected-event parameters. crossing_time here is in DAYS (add_microlensing_event
# takes days directly); kept short relative to the ~50+ day well-sampled window so
# the dip lands inside the search region. u0 small => deep, clearly-detectable event.
INJECT_U0 = 0.05
INJECT_TE_DAYS = 15.0


def kernel_inputs():
    """Fixed, RNG-free inputs for the deterministic kernels.

    t is irregularly sampled (microlensing photometry is never on a grid);
    y is a Gaussian dip in delta-mag plus a small deterministic ripple;
    errs are strictly positive and heteroscedastic.
    """
    gaps = np.array([0, 2, 1, 3, 1, 5, 2, 1, 4, 2, 3, 1, 6, 2, 1, 3, 2, 4, 1, 2],
                    dtype='float64')
    t = np.cumsum(gaps)
    tc = t.mean()
    y = -0.5 * np.exp(-0.5 * ((t - tc) / 6.0) ** 2) + 0.01 * np.sin(t)
    errs = 0.02 + 0.005 * np.cos(t) ** 2
    weights = 1.0 / errs ** 2
    return {'t': t, 'y': y, 'errs': errs, 'weights': weights}


def amplification_inputs():
    """Fixed inputs for the PSPL amplification / Jacobian / model kernels."""
    t = np.linspace(80.0, 120.0, 25)
    return {'t': t, 'u0': 0.3, 'tE': 12.0, 't0': 100.0, 'blend': 0.8}


def ks_inputs():
    """Two weighted samples for ks_weighted, plus an equal-weight pair that
    is directly comparable to scipy.stats.ks_2samp."""
    a = np.array([0.10, -0.20, 0.05, 0.30, -0.10, 0.00, 0.22, -0.15, 0.08, 0.12])
    b = np.array([0.40, 0.35, 0.50, 0.20, 0.60, 0.45, 0.33, 0.55])
    wa = np.array([1.0, 2.0, 0.5, 1.5, 1.0, 3.0, 0.7, 1.2, 0.9, 1.1])
    wb = np.array([1.0, 1.0, 2.0, 0.5, 1.3, 0.8, 1.1, 0.6])
    return {'a': a, 'b': b, 'wa': wa, 'wb': wb}


def extract_real_objects(big_parquet, out_path):
    """Extract the four reference objects from the full survey parquet.

    Run once; the result is committed as fixtures/real_objects.parquet so the
    suite does not depend on the multi-GB test.parquet at run time.
    """
    df = pd.read_parquet(big_parquet)
    mask = df['objectid'].astype(str).isin(REAL_OBJECT_IDS)
    sub = df[mask].copy()
    sub['objectid'] = sub['objectid'].astype(str)
    sub = sub.sort_values(['objectid', 'mjd']).reset_index(drop=True)
    for c in CAT_COLS:
        if c in sub.columns:
            sub[c] = sub[c].astype('category')
    sub.to_parquet(out_path)
    return sub


def _well_sampled(sub):
    regs = nscml.well_sampled_region(sub, interval=50, maxrevisit=10, seqlen=5)
    return [np.asarray(r) for r in regs]


def _inject(real_df, source_id):
    lc = real_df[real_df['objectid'].astype(str) == source_id]
    region = _well_sampled(lc)[0]
    rt = lc.loc[region, 'mjd'].to_numpy()
    peak = float(np.mean([rt[0], rt[-1]]))
    synth = nscml.add_microlensing_event(
        lc, impact_parameter=INJECT_U0, crossing_time=INJECT_TE_DAYS, peak_time=peak)
    return synth.drop(columns=[c for c in ['originalid'] if c in synth.columns])


def events_table():
    """Tiny population table for generate_synthetic_*; crossing_time in HOURS
    (the function divides by 24), umin is the impact parameter."""
    return pd.DataFrame({'crossing_time': [240.0, 480.0, 360.0],
                         'umin': [0.10, 0.05, 0.20]})


def small_sample_lc():
    """A short single-object lightcurve engineered so fit_excursions takes its
    small-sample branch: the ~120-day baseline sits inside the +/-100 day
    extend_lc window, leaving <= n_min_outside_fit points outside the fit, so
    the KS reference is drawn from np.random.normal (the legacy-RNG path that
    np.random.seed(0) reproduces). Construction RNG is an isolated Generator
    so it never perturbs the global state the fit seeds.
    """
    rng = np.random.default_rng(0)
    n = 24
    t = np.sort(rng.uniform(0, 120, n))
    dip = nscml.amp_to_mag(nscml.microlensing_amplification(t, 0.06, 8.0, 60.0))
    dm = rng.normal(0, 0.01, n) + dip
    return pd.DataFrame({'objectid': ['small_0'] * n, 'mjd': t,
                         'mag_auto': 18.0 + dm,
                         'magerr_auto': np.full(n, 0.02), 'deltamag': dm})


def build_working_fixtures(real_path, outdir):
    """Build the integration fixture from the committed real slice.

    Writes mini_lc.parquet (real objects + injected synthetic copies sharing
    source index labels), ws_regions.pkl (search domains keyed by source id),
    and events.parquet. Returns a dict of paths and id bookkeeping.
    """
    real_df = pd.read_parquet(real_path)
    real_df['objectid'] = real_df['objectid'].astype(str)

    synths = [_inject(real_df, sid) for sid in SYNTH_SOURCE_IDS]
    synth_ids = [str(s['objectid'].iloc[0]) for s in synths]

    # concat WITHOUT ignore_index: each synthetic object keeps its source's
    # labels, which is what search-domain restriction relies on.
    mini = pd.concat([real_df] + synths)
    mini_path = os.path.join(outdir, 'mini_lc.parquet')
    mini.to_parquet(mini_path)

    rb = pd.read_parquet(mini_path)
    rb_oid = rb['objectid'].astype(str)
    ws_regions = {rid: _well_sampled(rb[rb_oid == rid]) for rid in REAL_OBJECT_IDS}

    # Contract checks: the parquet round-trip must preserve the shared labels,
    # and the injected events must actually be detectable under default params.
    for sid, source in zip(synth_ids, SYNTH_SOURCE_IDS):
        synth_idx = set(rb.index[rb_oid == sid].tolist())
        domain = set(np.concatenate(ws_regions[source]).tolist())
        assert domain <= synth_idx, (
            f'index contract broken: ws domain for {source} not within {sid}')
        lc = rb[rb_oid == sid]
        excs = nscml.find_persistent_excursions(
            lc, restrict_to_indices=np.concatenate(ws_regions[source]))
        assert len(excs) > 0, f'no excursion detected for injected {sid}'

    ws_path = os.path.join(outdir, 'ws_regions.pkl')
    with open(ws_path, 'wb') as f:
        pickle.dump(ws_regions, f)
    events_path = os.path.join(outdir, 'events.parquet')
    events_table().to_parquet(events_path)

    return {
        'mini_lc': mini_path,
        'ws_regions': ws_path,
        'events': events_path,
        'real_ids': list(REAL_OBJECT_IDS),
        'synth_ids': synth_ids,
        'synth_sources': list(SYNTH_SOURCE_IDS),
    }


if __name__ == '__main__':
    import sys
    import tempfile

    here = os.path.dirname(os.path.abspath(__file__))
    real_path = os.path.join(here, 'fixtures', 'real_objects.parquet')
    if '--extract' in sys.argv or not os.path.exists(real_path):
        big = os.path.join(here, os.pardir, 'notebooks', 'test.parquet')
        big = os.path.abspath(big)
        if not os.path.exists(big):
            sys.exit(f'cannot extract: {big} not found')
        print('extracting real objects from', big)
        extract_real_objects(big, real_path)
        print('wrote', real_path)

    tmp = tempfile.mkdtemp(prefix='nscml_fix_')
    info = build_working_fixtures(real_path, tmp)
    print('built working fixtures in', tmp)
    for k, v in info.items():
        print(' ', k, '=', v)
