"""Unit tests for the numerical kernels (the scientific core).

These assert properties and cross-identities directly, independent of the
golden snapshots: a behaviour change that still happened to match a stale
golden would still trip these.
"""
import numpy as np
import pytest
from scipy.stats import ks_2samp

import swellar
import fixture_build as fb

TS = 5.0


# --------------------------------------------------------------------------
# windowed weighted moving average / scatter
# --------------------------------------------------------------------------

def test_sparse_wma_constant_signal():
    t = np.linspace(0, 100, 12)
    y = np.full(12, 0.7)
    w = np.full(12, 2500.0)
    wma, wme, wms = swellar.sparse_gaussian_wma(y, t, w, timescale=TS, nclip=10)
    np.testing.assert_allclose(wma, 0.7, atol=1e-12)
    np.testing.assert_allclose(wms, 0.0, atol=1e-12)  # zero scatter on a flat curve


def test_sparse_wma_single_point():
    t = np.array([5.0])
    y = np.array([0.3])
    w = np.array([400.0])  # err = 0.05
    wma, wme, wms = swellar.sparse_gaussian_wma(y, t, w, timescale=TS, nclip=10)
    np.testing.assert_allclose(wma, [0.3], atol=1e-12)
    np.testing.assert_allclose(wme, [0.05], atol=1e-12)  # 1/sqrt(w)
    np.testing.assert_allclose(wms, [0.0], atol=1e-12)


def test_sparse_equals_dense_when_unclipped():
    """With nclip large enough that no pair is dropped, the O(n^2) sparse
    kernel and the dense matrix kernel are the same estimator, mean, error
    and scatter must agree to round-off. This is the core consistency check."""
    ki = fb.kernel_inputs()
    y, t, e, w = ki['y'], ki['t'], ki['errs'], ki['weights']
    s_wma, s_err, s_scat = swellar.sparse_gaussian_wma(y, t, w, timescale=TS, nclip=100)
    d_wma, d_err, d_scat = swellar.compute_weighted_moving_average(y, t, e, timescale=TS)
    np.testing.assert_allclose(s_wma, d_wma, rtol=0, atol=1e-12)
    np.testing.assert_allclose(s_err, d_err, rtol=0, atol=1e-12)
    np.testing.assert_allclose(s_scat, d_scat, rtol=0, atol=1e-12)


def test_nclip_actually_clips():
    """Small nclip must change the result (far pairs excluded), i.e. clipping
    is not a no-op."""
    ki = fb.kernel_inputs()
    y, t, w = ki['y'], ki['t'], ki['weights']
    wide = swellar.sparse_gaussian_wma(y, t, w, timescale=TS, nclip=100)[0]
    tight = swellar.sparse_gaussian_wma(y, t, w, timescale=TS, nclip=1)[0]
    assert not np.allclose(wide, tight)


def test_dense_gaussian_matches_default_window():
    ki = fb.kernel_inputs()
    y, t, e = ki['y'], ki['t'], ki['errs']
    a = swellar.weighted_moving_average_gaussian(y, t, e, timescale=TS)
    b = swellar.compute_weighted_moving_average(y, t, e, timescale=TS)
    for x, z in zip(a, b):
        np.testing.assert_array_equal(x, z)


def test_dispatcher_matches_backends():
    ki = fb.kernel_inputs()
    y, t, e, w = ki['y'], ki['t'], ki['errs'], ki['weights']
    sp = swellar.weighted_moving_average(y, t, e, sparse=True, timescale=TS, nclip=10)
    de = swellar.weighted_moving_average(y, t, e, sparse=False, timescale=TS)
    np.testing.assert_array_equal(sp[0], swellar.sparse_gaussian_wma(y, t, w, timescale=TS, nclip=10)[0])
    np.testing.assert_array_equal(de[0], swellar.compute_weighted_moving_average(y, t, e, timescale=TS)[0])


def test_weighted_avg_and_std_scale_invariance():
    v = np.array([1.0, 2.0, 3.0, 10.0])
    w = np.array([1.0, 1.0, 2.0, 0.5])
    a0, s0 = swellar.weighted_avg_and_std(v, w)
    a1, s1 = swellar.weighted_avg_and_std(v, 1000.0 * w)  # scaling weights is a no-op
    np.testing.assert_allclose([a0, s0], [a1, s1], rtol=0, atol=1e-12)
    # matches the explicit definition
    avg = np.average(v, weights=w)
    std = np.sqrt(np.average((v - avg) ** 2, weights=w))
    np.testing.assert_allclose([a0, s0], [avg, std], atol=1e-12)


# --------------------------------------------------------------------------
# window functions
# --------------------------------------------------------------------------

def test_gaussian_window_shape():
    assert swellar.gaussian_window(0.0, 2.0) == pytest.approx(1.0)
    # symmetric and equal to the analytic gaussian
    for dt in (1.0, -1.0, 3.5):
        assert swellar.gaussian_window(dt, 2.0) == pytest.approx(np.exp(-(dt / 2.0) ** 2 / 2))
    assert swellar.gaussian_window(1.0, 2.0) == swellar.gaussian_window(-1.0, 2.0)


def test_clipped_window_truncates_beyond_nclip():
    ts, nclip = 2.0, 5
    assert swellar.clipped_gaussian_window(0.0, ts, nclip) == pytest.approx(1.0)
    # just inside the clip radius: equals the gaussian; just outside: exactly 0
    assert swellar.clipped_gaussian_window(9.9, ts, nclip) == pytest.approx(np.exp(-(9.9 / ts) ** 2 / 2))
    assert swellar.clipped_gaussian_window(10.1, ts, nclip) == 0.0
    assert swellar.clipped_gaussian_window(-10.1, ts, nclip) == 0.0


# --------------------------------------------------------------------------
# PSPL amplification / model / Jacobian
# --------------------------------------------------------------------------

def test_amplification_physics():
    t = np.linspace(50, 150, 201)
    u0, tE, t0 = 0.2, 12.0, 100.0
    amp = swellar.microlensing_amplification(t, u0, tE, t0)
    assert np.all(amp >= 1.0 - 1e-12)                 # unblended A >= 1
    assert np.argmax(amp) == np.argmin(np.abs(t - t0))  # peak at t0
    assert amp[0] == pytest.approx(1.0, abs=1e-2)     # returns to baseline far out
    # smaller impact parameter -> stronger peak
    amp_close = swellar.microlensing_amplification(np.array([t0]), 0.05, tE, t0)[0]
    assert amp_close > amp.max()


def test_blending_factor():
    t = np.linspace(80, 120, 41)
    u0, tE, t0, f = 0.3, 10.0, 100.0, 0.8
    full = swellar.microlensing_amplification(t, u0, tE, t0)
    blended = swellar.microlensing_amplification(t, u0, tE, t0, f)
    np.testing.assert_allclose(blended, full * f + (1 - f), atol=1e-12)


def test_amp_to_mag_and_ml_f():
    assert swellar.amp_to_mag(1.0) == pytest.approx(0.0)
    assert swellar.amp_to_mag(10.0) == pytest.approx(-2.5)
    t = np.linspace(90, 110, 9)
    u0, tE, t0 = 0.3, 12.0, 100.0
    np.testing.assert_array_equal(
        swellar.ml_f(t, u0, tE, t0),
        swellar.amp_to_mag(swellar.microlensing_amplification(t, u0, tE, t0)))


def test_ml_jac_matches_finite_difference():
    """The analytic Jacobian (columns d/du0, d/dtE, d/dt0) supplied to
    curve_fit must match a central finite difference of ml_f."""
    t = np.linspace(85, 115, 13)
    p = np.array([0.3, 12.0, 100.0])  # u0, tE, t0
    analytic = swellar.ml_jac(t, *p)
    steps = np.array([1e-6, 1e-4, 1e-4])
    for j in range(3):
        pp, pm = p.copy(), p.copy()
        pp[j] += steps[j]
        pm[j] -= steps[j]
        num = (swellar.ml_f(t, *pp) - swellar.ml_f(t, *pm)) / (2 * steps[j])
        np.testing.assert_allclose(analytic[:, j], num, rtol=2e-4, atol=1e-7,
                                   err_msg=f'jac column {j}')


# --------------------------------------------------------------------------
# weighted KS
# --------------------------------------------------------------------------

def test_ks_weighted_matches_scipy_statistic():
    """With equal weights the D statistic must equal scipy.stats.ks_2samp."""
    k = fb.ks_inputs()
    d, _ = swellar.ks_weighted(k['a'], k['b'], np.ones_like(k['a']), np.ones_like(k['b']))
    ref = ks_2samp(k['a'], k['b'])
    assert d == pytest.approx(ref.statistic, abs=1e-12)


def test_ks_weighted_identical_samples():
    x = np.array([0.1, 0.2, 0.3, 0.4])
    w = np.array([1.0, 2.0, 1.5, 0.5])
    d, p = swellar.ks_weighted(x, x, w, w)
    assert d == pytest.approx(0.0, abs=1e-12)
    assert p == pytest.approx(1.0, abs=1e-9)


def test_ks_weighted_weight_sensitivity():
    """Reweighting one sample changes the statistic, weights are honoured."""
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([0.5, 1.5, 2.5])
    d_flat, _ = swellar.ks_weighted(a, b, np.ones(4), np.ones(3))
    d_skew, _ = swellar.ks_weighted(a, b, np.array([10.0, 1.0, 1.0, 1.0]), np.ones(3))
    assert d_flat != d_skew


# --------------------------------------------------------------------------
# outlier rejection helpers
# --------------------------------------------------------------------------

def test_reject_outliers_flags_extreme_point():
    # tight cluster + one gross outlier so the MAD cleanly isolates the last point
    data = np.array([0.0, 0.01, -0.01, 0.02, -0.02, 5.0])
    keep = swellar.reject_outliers_args(data, m=3.0)
    assert keep[-1] == False
    assert keep[:-1].all()
    np.testing.assert_array_equal(swellar.reject_outliers(data, 3.0), data[keep])


def test_reject_outliers_zero_mad_keeps_all():
    """When the median absolute deviation is 0 the guard returns all-True
    rather than dividing by zero."""
    data = np.full(6, 3.0)
    assert swellar.reject_outliers_args(data, 3.0).all()


def test_reject_low_error_outliers_uses_errors():
    # non-zero MAD (avoids the all-True zero-MAD guard); last point is the outlier
    data = np.array([0.0, 0.05, -0.05, 0.1, 0.3])
    small = np.full(5, 0.01)
    big = np.array([0.01, 0.01, 0.01, 0.01, 1.0])  # last point has huge error
    # with a small error the deviation is significant -> rejected;
    # with a large error the same deviation is consistent with noise -> kept
    assert swellar.reject_low_error_outliers_args(data, small, m=3)[-1] == False
    assert swellar.reject_low_error_outliers_args(data, big, m=3)[-1] == True
