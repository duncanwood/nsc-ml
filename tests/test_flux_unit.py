"""Flux-space (Phase 2) tests.

Covers the fractional-flux ingest (normalize space='flux'), the Option-1
flux_to_mag adapter, the achromaticity property that makes cross-band pooling
valid in flux, and an end-to-end recovery comparison of the two flux paths.
"""
import os

import numpy as np
import pandas as pd
import pytest

import swellar

FLUX_SCHEMA = swellar.LightcurveSchema(id='objectid', time='mjd', band='filter',
                                     measurement='flux', error='fluxerr',
                                     value=None, space='flux')


def synth_flux_lc(u0=0.2, tE=40.0, t0=200.0, fbase=5000.0, sigma=50.0,
                  span=400.0, n=200, band='r', seed=0):
    """One object's flux light curve with an injected PSPL event:
    F = fbase * A(u(t)) + Gaussian(sigma).  A is the flux ratio. Defaults are a
    realistic, slow event (tE >> the 5-day WMA window), a very fast/strong
    event inflates the in-window scatter above the signal and is suppressed by
    the usescatter self-calibration (true in mag space too)."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, span, n))
    A = swellar.microlensing_amplification(t, u0, tE, t0)   # blend=1 -> flux ratio A
    flux = fbase * A + rng.normal(0, sigma, n)
    return pd.DataFrame({'objectid': ['ev'] * n, 'mjd': t, 'flux': flux,
                         'fluxerr': np.full(n, sigma), 'filter': [band] * n})


# --------------------------------------------------------------------------
# model + ingest math
# --------------------------------------------------------------------------

def test_ml_f_flux_is_amplification_minus_one():
    t = np.linspace(90, 150, 13)
    p = (0.2, 12.0, 120.0)
    np.testing.assert_array_equal(
        swellar.ml_f_flux(t, *p), swellar.microlensing_amplification(t, *p) - 1.0)


def test_normalize_flux_builds_fractional_flux():
    df = pd.DataFrame({
        'objectid': ['a'] * 4,
        'mjd': [1.0, 2.0, 3.0, 4.0],
        'flux': [100.0, 100.0, 300.0, 100.0],   # per-(obj,band) median = 100
        'fluxerr': [5.0, 5.0, 5.0, 5.0],
        'filter': ['r'] * 4,
    })
    out = swellar.normalize(df, FLUX_SCHEMA)
    # s = F/F_ref - 1 with F_ref = 100
    np.testing.assert_allclose(out['deltamag'].to_numpy(), [0.0, 0.0, 2.0, 0.0])
    # sigma_s = sigma_F / F_ref = 5/100
    np.testing.assert_allclose(out['magerr_auto'].to_numpy(), 0.05)


def test_normalize_flux_rejects_nonpositive_baseline():
    df = pd.DataFrame({'objectid': ['a', 'a'], 'mjd': [1.0, 2.0],
                       'flux': [-3.0, 1.0], 'fluxerr': [1.0, 1.0], 'filter': ['r', 'r']})
    with pytest.raises(ValueError, match='F_ref'):
        swellar.normalize(df, FLUX_SCHEMA)


def test_flux_to_mag_drops_nonpositive_and_converts():
    df = pd.DataFrame({
        'objectid': ['a'] * 3,
        'mjd': [1.0, 2.0, 3.0],
        'flux': [swellar.AB_ZEROPOINT_NJY, -10.0, swellar.AB_ZEROPOINT_NJY * 10 ** (-0.4)],
        'fluxerr': [1e7, 1e7, 1e7],
        'filter': ['r'] * 3,
    })
    with pytest.warns(UserWarning, match='non-positive'):
        out = swellar.flux_to_mag(df, FLUX_SCHEMA)
    assert len(out) == 2                                   # the negative-flux epoch dropped
    # AB mag of the zeropoint flux is 0; of zp*10**-0.4 is +1 -> baseline-subtracted
    # (median of {0, 1} = 0.5): deltamag = {-0.5, +0.5}
    np.testing.assert_allclose(sorted(out['deltamag']), [-0.5, 0.5], atol=1e-9)


def test_fractional_flux_is_achromatic_across_bands():
    """Same achromatic A(t) in three bands with very different baseline fluxes:
    fractional flux s = A/median(A) - 1 is band-independent (raw flux is not).
    This is what makes pooling bands valid in flux space."""
    t = np.linspace(80, 160, 30)
    A = swellar.microlensing_amplification(t, 0.2, 12.0, 120.0)
    rows = []
    for band, fbase in (('g', 2000.0), ('r', 5000.0), ('i', 9000.0)):
        for ti, Ai in zip(t, A):
            rows.append({'objectid': 'ev', 'mjd': ti, 'flux': fbase * Ai,
                         'fluxerr': 0.01 * fbase, 'filter': band})
    out = swellar.normalize(pd.DataFrame(rows), FLUX_SCHEMA)
    for ti in t[::6]:
        s = out[np.isclose(out['mjd'], ti)]['deltamag'].to_numpy()
        assert len(s) == 3
        np.testing.assert_allclose(s, s[0], atol=1e-9)     # identical across bands


# --------------------------------------------------------------------------
# end-to-end detection + fit, and the two-path comparison
# --------------------------------------------------------------------------

def _run(space, frame, tmp_path, label):
    """Detect + fit one object's canonical frame in the given space."""
    p = os.path.join(str(tmp_path), f'{label}.parquet')
    frame.to_parquet(p)
    excs = {oid: swellar.find_persistent_excursions(frame[frame['objectid'] == oid], space=space)
            for oid in frame['objectid'].unique()}
    meta = {'outdir': str(tmp_path), 'outfile': f'{label}_s.pkl', 'fitoutfile': f'{label}_f.pkl'}
    fr, _, _ = swellar.fit_excursions(excs, [p], dict(meta), {}, space=space,
                                    rng=np.random.default_rng(0))
    return excs, swellar.make_fit_excursions_df(fr)


def test_flux_pipeline_detects_and_recovers(tmp_path):
    lc = synth_flux_lc()                               # u0=0.2, tE=40, t0=200
    frame = swellar.normalize(lc, FLUX_SCHEMA)           # deltamag now carries s>0 bump
    excs, fitdf = _run('flux', frame, tmp_path, 'flux')
    assert len(excs['ev']) >= 1                        # positive bump detected
    assert len(fitdf) >= 1
    row = fitdf.iloc[0]
    assert row['crossing_time'] == pytest.approx(40.0, rel=0.25)
    assert row['peak_time'] == pytest.approx(200.0, abs=5.0)


def test_flux_and_flux_to_mag_recover_consistent_params(tmp_path):
    """Option 1 (flux->mag) and Option 2 (space='flux') run on the SAME injected
    event recover consistent crossing time and peak time."""
    lc = synth_flux_lc()

    flux_frame = swellar.normalize(lc, FLUX_SCHEMA)
    _, flux_df = _run('flux', flux_frame, tmp_path, 'fx')

    mag_frame = swellar.flux_to_mag(lc, FLUX_SCHEMA)     # bright source -> nothing dropped
    _, mag_df = _run('mag', mag_frame, tmp_path, 'mg')

    assert len(flux_df) >= 1 and len(mag_df) >= 1
    fr, mr = flux_df.iloc[0], mag_df.iloc[0]
    # both paths see the same event: peak time within a couple of days,
    # crossing time within ~15%
    assert fr['peak_time'] == pytest.approx(mr['peak_time'], abs=2.0)
    assert fr['crossing_time'] == pytest.approx(mr['crossing_time'], rel=0.15)


# --------------------------------------------------------------------------
# synthetic injection (multiplicative, flux space), the recovery yardstick
# --------------------------------------------------------------------------

def _quiescent_flux_lc(fbase=5000.0, sigma=50.0, span=400.0, n=200, band='r', seed=1):
    """A flat (event-free) flux light curve, F = fbase + Gaussian(sigma): the
    real-data baseline an event is injected into."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, span, n))
    return pd.DataFrame({'objectid': ['ev'] * n, 'mjd': t,
                         'flux': fbase + rng.normal(0, sigma, n),
                         'fluxerr': np.full(n, sigma), 'filter': [band] * n})


def test_add_microlensing_event_flux_is_multiplicative():
    """flux mode multiplies the observed flux by A: the stored s = F/F_ref - 1
    becomes (s+1)*A - 1, and sigma_s scales by A (holds sigma_F/F invariant)."""
    n = 60
    t = np.linspace(0.0, 400.0, n)
    s0 = np.linspace(-0.05, 0.05, n)              # non-flat baseline: distinguishes (s+1)A-1 from A-1
    sig0 = np.full(n, 0.02)
    frame = pd.DataFrame({'objectid': ['ev'] * n, 'mjd': t,
                          'deltamag': s0, 'magerr_auto': sig0, 'filter': ['r'] * n})
    params = dict(impact_parameter=0.2, crossing_time=40.0, peak_time=200.0)
    out = swellar.add_microlensing_event(frame, space='flux', **params)
    A = swellar.microlensing_amplification(t, **params)
    np.testing.assert_allclose(out['deltamag'].to_numpy(), (s0 + 1.0) * A - 1.0)
    np.testing.assert_allclose(out['magerr_auto'].to_numpy(), sig0 * A)
    assert (out['originalid'] == 'ev').all()
    assert '_ml_' in str(out.iloc[0]['objectid'])     # relabeled synth id
    np.testing.assert_array_equal(frame['deltamag'].to_numpy(), s0)   # input not mutated


def test_add_microlensing_event_flux_is_a_positive_bump():
    """A flux event is a POSITIVE excursion (s = A-1 >= 0), the sign-flip from the
    magnitude dip, and it peaks at t0."""
    n = 200
    t = np.sort(np.random.default_rng(0).uniform(0, 400, n))
    frame = pd.DataFrame({'objectid': ['ev'] * n, 'mjd': t, 'deltamag': np.zeros(n),
                          'magerr_auto': np.full(n, 0.01), 'filter': ['r'] * n})
    out = swellar.add_microlensing_event(frame, space='flux',
                                       impact_parameter=0.1, crossing_time=30.0, peak_time=200.0)
    s = out['deltamag'].to_numpy()
    assert s.max() > 0.1
    assert s.min() > -1e-9                              # A >= 1 everywhere => s >= 0
    assert t[np.argmax(s)] == pytest.approx(200.0, abs=10.0)


def test_add_microlensing_event_rejects_unknown_space():
    frame = pd.DataFrame({'objectid': ['ev'], 'mjd': [1.0], 'deltamag': [0.0],
                          'magerr_auto': [0.01], 'filter': ['r']})
    with pytest.raises(ValueError, match='space'):
        swellar.add_microlensing_event(frame, space='nonsense',
                                     impact_parameter=0.2, crossing_time=40.0, peak_time=1.0)


def test_add_microlensing_event_mag_and_flux_are_same_event():
    """The same A injected in both spaces is one physical (multiplicative) event:
    mag adds -2.5 log10(A); flux takes s+1 -> (s+1)*A. On a flat baseline
    10**(-dmag/2.5) == s+1 == A."""
    n = 50
    t = np.linspace(0.0, 400.0, n)
    params = dict(impact_parameter=0.2, crossing_time=40.0, peak_time=200.0)
    mag_frame = pd.DataFrame({'objectid': ['ev'] * n, 'mjd': t,
                              'mag_auto': np.full(n, 20.0), 'deltamag': np.zeros(n)})
    flux_frame = pd.DataFrame({'objectid': ['ev'] * n, 'mjd': t, 'deltamag': np.zeros(n),
                               'magerr_auto': np.full(n, 0.01), 'filter': ['r'] * n})
    dmag = swellar.add_microlensing_event(mag_frame, space='mag', **params)['deltamag'].to_numpy()
    s = swellar.add_microlensing_event(flux_frame, space='flux', **params)['deltamag'].to_numpy()
    A = swellar.microlensing_amplification(t, **params)
    np.testing.assert_allclose(10.0 ** (-dmag / 2.5), s + 1.0, rtol=1e-9)
    np.testing.assert_allclose(s + 1.0, A, rtol=1e-12)


def test_flux_injection_recovered_by_detector(tmp_path):
    """End-to-end yardstick: inject a known event into a quiescent flux LC with
    add_microlensing_event(space='flux'), then detect + fit and recover it."""
    frame = swellar.normalize(_quiescent_flux_lc(), FLUX_SCHEMA)     # canonical flux frame, s ~ 0
    inj = swellar.add_microlensing_event(frame, space='flux',
                                       impact_parameter=0.2, crossing_time=40.0, peak_time=200.0)
    excs, fitdf = _run('flux', inj, tmp_path, 'inj')
    sid = inj.iloc[0]['objectid']
    assert len(excs[sid]) >= 1                          # injected bump detected
    assert len(fitdf) >= 1
    row = fitdf.iloc[0]
    assert row['crossing_time'] == pytest.approx(40.0, rel=0.25)
    assert row['peak_time'] == pytest.approx(200.0, abs=5.0)


def test_generate_synthetic_flux_mode_runs_and_injects(tmp_path):
    """generate_synthetic...(space='flux') injects multiplicative events into a
    canonical flux frame and writes the synth parquet without the NSC-only
    columns (exposure/instrument)."""
    rng = np.random.default_rng(3)
    t = np.sort(np.concatenate([np.arange(0.0, 250.0, 2.0), [300.0, 350.0]]))  # dense => one WS region
    n = len(t)
    frame = pd.DataFrame({'objectid': ['ev'] * n, 'mjd': t,
                          'deltamag': rng.normal(0, 0.01, n), 'magerr_auto': np.full(n, 0.01),
                          'filter': ['r'] * n}).reset_index(drop=True)
    lcpath = os.path.join(str(tmp_path), 'lc.parquet')
    frame.to_parquet(lcpath)
    regions = swellar.well_sampled_region(frame, interval=50, maxrevisit=10, seqlen=5)
    assert regions, 'need a well-sampled region to inject into'
    ws_regions = {'ev': regions}
    events = pd.DataFrame({'crossing_time': [40.0 * 24], 'umin': [0.2]})    # tE in HOURS (gen divides by 24)
    swellar.generate_synthetic_microlensing_events_from_population(
        [lcpath], events, ws_regions, str(tmp_path), 'fxsynth',
        rng=np.random.default_rng(0), space='flux')
    base = os.path.join(str(tmp_path), 'synth-fxsynth')
    assert os.path.exists(os.path.join(base, 'synth-fxsynth-info.pickle'))
    synth = pd.read_parquet(os.path.join(base, 'lc-synth-fxsynth.parquet'))
    assert synth['deltamag'].max() > 0.1                # positive flux bump injected
    assert (synth['originalid'].astype(str) == 'ev').all()


# --------------------------------------------------------------------------
# the high-level in-memory detect(df, schema) API
# --------------------------------------------------------------------------

def test_detect_flux_recovers_event():
    """detect() end-to-end on a raw flux table: normalize -> find -> fit."""
    out = swellar.detect(synth_flux_lc(), FLUX_SCHEMA, rng=np.random.default_rng(0))
    assert len(out) >= 1
    row = out.iloc[0]
    assert row['crossing_time'] == pytest.approx(40.0, rel=0.25)
    assert row['peak_time'] == pytest.approx(200.0, abs=5.0)


def test_detect_mag_space_recovers_event():
    """detect() with the default NSC (magnitude) schema on an injected mag event."""
    n = 200
    t = np.sort(np.random.default_rng(5).uniform(0, 400, n))
    base = pd.DataFrame({'objectid': ['ev'] * n, 'mjd': t, 'mag_auto': np.full(n, 20.0),
                         'deltamag': np.zeros(n), 'magerr_auto': np.full(n, 0.02),
                         'filter': ['r'] * n})
    lc = swellar.add_microlensing_event(base, space='mag',
                                      impact_parameter=0.2, crossing_time=40.0, peak_time=200.0)
    out = swellar.detect(lc, rng=np.random.default_rng(0))        # default schema = NSC mag
    assert len(out) >= 1
    row = out.iloc[0]
    assert row['crossing_time'] == pytest.approx(40.0, rel=0.3)
    assert row['peak_time'] == pytest.approx(200.0, abs=5.0)


def test_detect_unknown_param_raises():
    with pytest.raises(ValueError, match='Unknown parameters'):
        swellar.detect(synth_flux_lc(n=10), FLUX_SCHEMA, not_a_real_param=1)


def test_detect_empty_when_no_well_sampled_region():
    """A sparse light curve has no well-sampled region to search -> an empty
    result with the canonical columns (so callers can rely on the schema)."""
    lc = pd.DataFrame({'objectid': ['ev'] * 4, 'mjd': [0.0, 100.0, 200.0, 300.0],
                       'flux': [5000.0, 5010.0, 4990.0, 5005.0], 'fluxerr': [50.0] * 4,
                       'filter': ['r'] * 4})
    out = swellar.detect(lc, FLUX_SCHEMA)
    assert len(out) == 0
    for c in ('objectid', 'excnum', 'crossing_time', 'peak_time', 'pval'):
        assert c in out.columns


def test_detect_matches_manual_file_pipeline(tmp_path):
    """detect() reproduces the file-based pipeline it wraps (normalize -> find ->
    fit -> make_df); the fitted parameters are curve_fit outputs, so they match
    independent of the KS rng."""
    lc = synth_flux_lc()
    frame = swellar.normalize(lc, FLUX_SCHEMA)
    _, manual = _run('flux', frame, tmp_path, 'man')                   # file-backed pipeline
    auto = swellar.detect(lc, FLUX_SCHEMA, restrict_well_sampled=False,   # match _run (no ws restriction)
                        rng=np.random.default_rng(0))
    assert len(auto) == len(manual) >= 1
    for col in ('crossing_time', 'peak_time', 'impact_parameter'):
        np.testing.assert_allclose(sorted(auto[col]), sorted(manual[col]), rtol=1e-6)
