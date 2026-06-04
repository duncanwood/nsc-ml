"""Unit / edge-case tests for find_persistent_excursions (the detector).

The gating (n_measured, duration, restrict_to_indices) is tested by driving
the parameters rather than hand-tuning lightcurves, so the tests are robust
to small changes in the synthetic data.
"""
import numpy as np
import pandas as pd
import pytest

import swellar


def make_lc(t, dm, err=0.02, obj='o', bands=None):
    t = np.asarray(t, float)
    dm = np.asarray(dm, float)
    d = {'objectid': [obj] * len(t), 'mjd': t, 'mag_auto': 18.0 + dm,
         'magerr_auto': np.full(len(t), err), 'deltamag': dm}
    if bands is not None:
        d['filter'] = list(bands)
    return pd.DataFrame(d)


def injected_lc(n=80, seed=0, u0=0.05, tE=20.0, t0=100.0, span=200.0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, span, n))
    dm = rng.normal(0, 0.01, n) + swellar.amp_to_mag(
        swellar.microlensing_amplification(t, u0, tE, t0))
    return make_lc(t, dm)


# --------------------------------------------------------------------------
# degenerate inputs
# --------------------------------------------------------------------------

def test_empty_lightcurve_returns_empty():
    # empty curve is now guarded (was an unguarded IndexError before the audit fix)
    assert swellar.find_persistent_excursions(make_lc([], [])) == []


@pytest.mark.parametrize('n', [1, 2, 3])
def test_too_short_returns_no_regions(n):
    t = np.linspace(0, 50, n)
    assert swellar.find_persistent_excursions(make_lc(t, np.zeros(n))) == []


def test_flat_curve_no_false_positive():
    t = np.linspace(0, 200, 60)
    assert swellar.find_persistent_excursions(make_lc(t, np.zeros(60))) == []


# --------------------------------------------------------------------------
# detection + gating
# --------------------------------------------------------------------------

def test_injected_event_is_detected():
    regions = swellar.find_persistent_excursions(injected_lc())
    assert len(regions) >= 1
    assert all(len(r) >= 4 for r in regions)  # default n_measured


def test_n_measured_gate_rejects_when_too_high():
    lc = injected_lc()
    assert swellar.find_persistent_excursions(lc, n_measured=1000) == []


def test_duration_gate_rejects_when_too_high():
    lc = injected_lc()
    assert swellar.find_persistent_excursions(lc, duration=1e6) == []


def test_z_threshold_controls_sensitivity():
    lc = injected_lc(u0=0.2)  # shallower event
    loose = swellar.find_persistent_excursions(lc, z_threshold=2)
    strict = swellar.find_persistent_excursions(lc, z_threshold=50)
    assert len(strict) <= len(loose)
    assert strict == []


def test_restrict_to_indices_masks_outside_window():
    lc = injected_lc().sort_values('mjd')
    full = swellar.find_persistent_excursions(lc)
    assert len(full) >= 1
    event_idx = np.concatenate([np.asarray(r) for r in full])
    # restricting to the event indices keeps it; restricting to the complement drops it
    kept = swellar.find_persistent_excursions(lc, restrict_to_indices=event_idx)
    assert len(kept) >= 1
    complement = lc.index.difference(event_idx).to_numpy()
    assert swellar.find_persistent_excursions(lc, restrict_to_indices=complement) == []


def test_detection_is_band_agnostic():
    """Detection runs on delta-mag only; assigning bands must not change it."""
    base = injected_lc()
    t, dm = base['mjd'].to_numpy(), base['deltamag'].to_numpy()
    one = make_lc(t, dm, bands=['g'] * len(t))
    multi = make_lc(t, dm, bands=(['g', 'r', 'i'] * len(t))[:len(t)])
    r_one = swellar.find_persistent_excursions(one)
    r_multi = swellar.find_persistent_excursions(multi)
    assert [list(map(int, r)) for r in r_one] == [list(map(int, r)) for r in r_multi]


def test_temper_errors_suppresses_detection():
    """Inflating errors by the global scatter should weaken marginal events."""
    lc = injected_lc(u0=0.3)  # weak event
    base = swellar.find_persistent_excursions(lc, temper_errors=None)
    tempered = swellar.find_persistent_excursions(lc, temper_errors=5)
    assert len(tempered) <= len(base)


def test_cut_outliers_path_runs():
    # exercise the optional low-error-outlier pre-cut branch
    lc = injected_lc()
    regions = swellar.find_persistent_excursions(lc, cut_outliers=True)
    assert isinstance(regions, list)
