"""Unit tests for the Rubin LSST adapter (swellar.surveys.lsst).

Synthetic LSST-shaped flux light curves (ForcedSource direct flux and DiaSource
difference flux). Real-data validation should use the public simulated DP0.2 (no
local Rubin data here). Mirrors the flux-space patterns in test_flux_unit.py.
"""
import numpy as np
import pandas as pd
import pytest

import swellar
from swellar.surveys import lsst

FBASE = 5000.0       # quiescent (template) flux, nJy
SIGMA = 50.0


def _synth(kind='forced', n=200, band='r', seed=0,
           u0=0.2, tE=40.0, t0=200.0, span=400.0):
    """One object's LSST-shaped light curve with an injected PSPL event.
    `total` flux = FBASE * A(u(t)) + noise; ForcedSource reports `total`,
    DiaSource reports the difference `total - FBASE` plus a `template` column."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, span, n))
    A = swellar.microlensing_amplification(t, u0, tE, t0)        # flux ratio (blend=1)
    total = FBASE * A + rng.normal(0, SIGMA, n)
    if kind == 'forced':
        return pd.DataFrame({'objectId': ['ev'] * n, 'expMidptMJD': t, 'band': [band] * n,
                             'psfFlux': total, 'psfFluxErr': np.full(n, SIGMA)})
    return pd.DataFrame({'diaObjectId': ['ev'] * n, 'midpointMjdTai': t, 'band': [band] * n,
                         'psfFlux': total - FBASE, 'psfFluxErr': np.full(n, SIGMA),
                         'template': np.full(n, FBASE)})


def test_from_lsst_forcedsource_builds_canonical_fractional_flux():
    out = lsst.from_lsst(_synth('forced'))
    for c in ('objectid', 'mjd', 'deltamag', 'magerr_auto', 'filter'):
        assert c in out.columns
    assert out['deltamag'].max() > 0.1          # the magnification bump (s = A - 1 > 0)
    # sigma_s = sigma_F / F_ref ~ SIGMA / FBASE
    np.testing.assert_allclose(np.median(out['magerr_auto']), SIGMA / FBASE, rtol=0.1)


def test_from_lsst_diasource_with_template():
    out = lsst.from_lsst(_synth('dia'), schema=lsst.LSST_DIASOURCE_SCHEMA,
                         template_flux_col='template')
    assert out['deltamag'].max() > 0.1


def test_from_lsst_diasource_without_template_rejected():
    # difference flux with a non-positive per-band median: no valid F_ref
    df = pd.DataFrame({'diaObjectId': ['a', 'a', 'a'], 'midpointMjdTai': [1., 2., 3.],
                       'band': ['r', 'r', 'r'], 'psfFlux': [-3.0, -1.0, 1.0],
                       'psfFluxErr': [1., 1., 1.]})
    with pytest.raises(ValueError, match='F_ref'):
        lsst.from_lsst(df, schema=lsst.LSST_DIASOURCE_SCHEMA)


def test_from_lsst_forced_and_dia_agree():
    # same underlying light curve via direct vs difference photometry -> same signal
    f = lsst.from_lsst(_synth('forced'))
    d = lsst.from_lsst(_synth('dia'), schema=lsst.LSST_DIASOURCE_SCHEMA,
                       template_flux_col='template')
    np.testing.assert_allclose(np.sort(f['deltamag'].to_numpy()),
                               np.sort(d['deltamag'].to_numpy()), rtol=1e-9)


def test_from_lsst_output_detects_event():
    out = lsst.from_lsst(_synth('forced'))
    excs = swellar.find_persistent_excursions(out, space='flux')
    assert len(excs) >= 1                        # the injected event is recovered
