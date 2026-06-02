"""Flux-space (Phase 2) tests.

Covers the fractional-flux ingest (normalize space='flux'), the Option-1
flux_to_mag adapter, the achromaticity property that makes cross-band pooling
valid in flux, and an end-to-end recovery comparison of the two flux paths.
"""
import os

import numpy as np
import pandas as pd
import pytest

import nscml

FLUX_SCHEMA = nscml.LightcurveSchema(id='objectid', time='mjd', band='filter',
                                     measurement='flux', error='fluxerr',
                                     value=None, space='flux')


def synth_flux_lc(u0=0.2, tE=40.0, t0=200.0, fbase=5000.0, sigma=50.0,
                  span=400.0, n=200, band='r', seed=0):
    """One object's flux light curve with an injected PSPL event:
    F = fbase * A(u(t)) + Gaussian(sigma).  A is the flux ratio. Defaults are a
    realistic, slow event (tE >> the 5-day WMA window) -- a very fast/strong
    event inflates the in-window scatter above the signal and is suppressed by
    the usescatter self-calibration (true in mag space too)."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, span, n))
    A = nscml.microlensing_amplification(t, u0, tE, t0)   # blend=1 -> flux ratio A
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
        nscml.ml_f_flux(t, *p), nscml.microlensing_amplification(t, *p) - 1.0)


def test_normalize_flux_builds_fractional_flux():
    df = pd.DataFrame({
        'objectid': ['a'] * 4,
        'mjd': [1.0, 2.0, 3.0, 4.0],
        'flux': [100.0, 100.0, 300.0, 100.0],   # per-(obj,band) median = 100
        'fluxerr': [5.0, 5.0, 5.0, 5.0],
        'filter': ['r'] * 4,
    })
    out = nscml.normalize(df, FLUX_SCHEMA)
    # s = F/F_ref - 1 with F_ref = 100
    np.testing.assert_allclose(out['deltamag'].to_numpy(), [0.0, 0.0, 2.0, 0.0])
    # sigma_s = sigma_F / F_ref = 5/100
    np.testing.assert_allclose(out['magerr_auto'].to_numpy(), 0.05)


def test_normalize_flux_rejects_nonpositive_baseline():
    df = pd.DataFrame({'objectid': ['a', 'a'], 'mjd': [1.0, 2.0],
                       'flux': [-3.0, 1.0], 'fluxerr': [1.0, 1.0], 'filter': ['r', 'r']})
    with pytest.raises(ValueError, match='F_ref'):
        nscml.normalize(df, FLUX_SCHEMA)


def test_flux_to_mag_drops_nonpositive_and_converts():
    df = pd.DataFrame({
        'objectid': ['a'] * 3,
        'mjd': [1.0, 2.0, 3.0],
        'flux': [nscml.AB_ZEROPOINT_NJY, -10.0, nscml.AB_ZEROPOINT_NJY * 10 ** (-0.4)],
        'fluxerr': [1e7, 1e7, 1e7],
        'filter': ['r'] * 3,
    })
    with pytest.warns(UserWarning, match='non-positive'):
        out = nscml.flux_to_mag(df, FLUX_SCHEMA)
    assert len(out) == 2                                   # the negative-flux epoch dropped
    # AB mag of the zeropoint flux is 0; of zp*10**-0.4 is +1 -> baseline-subtracted
    # (median of {0, 1} = 0.5): deltamag = {-0.5, +0.5}
    np.testing.assert_allclose(sorted(out['deltamag']), [-0.5, 0.5], atol=1e-9)


def test_fractional_flux_is_achromatic_across_bands():
    """Same achromatic A(t) in three bands with very different baseline fluxes:
    fractional flux s = A/median(A) - 1 is band-independent (raw flux is not).
    This is what makes pooling bands valid in flux space."""
    t = np.linspace(80, 160, 30)
    A = nscml.microlensing_amplification(t, 0.2, 12.0, 120.0)
    rows = []
    for band, fbase in (('g', 2000.0), ('r', 5000.0), ('i', 9000.0)):
        for ti, Ai in zip(t, A):
            rows.append({'objectid': 'ev', 'mjd': ti, 'flux': fbase * Ai,
                         'fluxerr': 0.01 * fbase, 'filter': band})
    out = nscml.normalize(pd.DataFrame(rows), FLUX_SCHEMA)
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
    excs = {oid: nscml.find_persistent_excursions(frame[frame['objectid'] == oid], space=space)
            for oid in frame['objectid'].unique()}
    meta = {'outdir': str(tmp_path), 'outfile': f'{label}_s.pkl', 'fitoutfile': f'{label}_f.pkl'}
    fr, _, _ = nscml.fit_excursions(excs, [p], dict(meta), {}, space=space,
                                    rng=np.random.default_rng(0))
    return excs, nscml.make_fit_excursions_df(fr)


def test_flux_pipeline_detects_and_recovers(tmp_path):
    lc = synth_flux_lc()                               # u0=0.2, tE=40, t0=200
    frame = nscml.normalize(lc, FLUX_SCHEMA)           # deltamag now carries s>0 bump
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

    flux_frame = nscml.normalize(lc, FLUX_SCHEMA)
    _, flux_df = _run('flux', flux_frame, tmp_path, 'fx')

    mag_frame = nscml.flux_to_mag(lc, FLUX_SCHEMA)     # bright source -> nothing dropped
    _, mag_df = _run('mag', mag_frame, tmp_path, 'mg')

    assert len(flux_df) >= 1 and len(mag_df) >= 1
    fr, mr = flux_df.iloc[0], mag_df.iloc[0]
    # both paths see the same event: peak time within a couple of days,
    # crossing time within ~15%
    assert fr['peak_time'] == pytest.approx(mr['peak_time'], abs=2.0)
    assert fr['crossing_time'] == pytest.approx(mr['crossing_time'], rel=0.15)
