"""Rubin LSST quickstart: find a microlensing event in LSST-shaped photometry.

The same detector that runs on NOIRLab Source Catalog magnitudes runs on Rubin
LSST flux, unchanged, only the *schema* (column names + ``space='flux'``)
differs. This script builds synthetic LSST-shaped light curves with a known
injected event, so it runs anywhere with no data access, then recovers the event
two ways:

  1. ForcedSource (direct photometry): ``detect(df, LSST_FORCEDSOURCE_SCHEMA)``.
     detect() normalizes the per-band flux to the achromatic fractional flux
     ``s = F/F_ref - 1`` and runs the flux-space detector in one call.
  2. DiaSource (difference photometry): ``from_lsst(df, LSST_DIASOURCE_SCHEMA,
     template_flux_col=...)`` folds the positive template flux back in (a
     difference-flux median is ~0 and is not a valid reference ``F_ref``), then
     the same detector runs on the resulting canonical frame.

On real data, prototype against the public, simulated DP0.2 (DP1 is access-gated);
confirm the exact column names against the data release and override the schema.
See PORTABILITY.md sec. 2-3.

Run:  python examples/lsst_quickstart.py
"""
import os

os.environ.setdefault('TQDM_DISABLE', '1')   # quiet the pipeline's progress bars

import numpy as np
import pandas as pd

import swellar
from swellar.surveys.lsst import (from_lsst, LSST_FORCEDSOURCE_SCHEMA,
                                LSST_DIASOURCE_SCHEMA)

# Injected event (the ground truth we try to recover).
U0, TE, T0 = 0.15, 40.0, 100.0
# Per-band quiescent flux in nanojansky, deliberately very different across
# bands so that pooling *raw* flux would be dominated by the bright band; the
# fractional flux s = A - 1 is achromatic and pools correctly.
BANDS_FBASE = {'g': 2000.0, 'r': 5000.0, 'i': 9000.0}
SNR = 80.0                       # per-epoch flux SNR at baseline
SPAN, CADENCE = 200.0, 2.0       # ~one season, dense (well-sampled) revisit


def synthetic_forcedsource(rng):
    """An LSST ForcedSource-shaped table (objectId, expMidptMJD, band, psfFlux,
    psfFluxErr in nJy) for one object, with the injected PSPL event in every band."""
    rows = []
    for band, f_base in BANDS_FBASE.items():
        # dense, slightly jittered cadence -> one well-sampled region per band
        t = np.sort(np.arange(0.0, SPAN, CADENCE) + rng.uniform(-0.4, 0.4,
                    size=int(SPAN // CADENCE)))
        amp = swellar.microlensing_amplification(t, U0, TE, T0)   # achromatic flux ratio A(t)
        sigma = f_base / SNR
        flux = f_base * amp + rng.normal(0, sigma, size=t.size)
        rows.append(pd.DataFrame({'objectId': 'LSST-0001', 'expMidptMJD': t,
                                  'band': band, 'psfFlux': flux, 'psfFluxErr': sigma}))
    return pd.concat(rows, ignore_index=True)


def to_diasource(forced):
    """Turn the direct-flux table into a DiaSource-shaped difference-flux table:
    psfFlux becomes (flux - template) and a positive `template` column carries the
    per-band reference flux."""
    template = forced['band'].map(BANDS_FBASE).to_numpy()
    return pd.DataFrame({
        'diaObjectId': forced['objectId'].to_numpy(),
        'midpointMjdTai': forced['expMidptMJD'].to_numpy(),
        'band': forced['band'].to_numpy(),
        'psfFlux': forced['psfFlux'].to_numpy() - template,   # difference flux (median ~ 0)
        'psfFluxErr': forced['psfFluxErr'].to_numpy(),
        'template': template,
    })


# The canonical frame from from_lsst already carries the fractional-flux signal
# `s` in `deltamag`, so detect() runs with a "signal present" flux schema
# (normalize is then just a rename, no re-baselining).
SIGNAL_SCHEMA = swellar.LightcurveSchema(value='deltamag', error='magerr_auto',
                                       band='filter', space='flux')


def _report(label, fits):
    if len(fits) == 0:
        print(f"  {label}: no event detected")
        return
    row = fits.sort_values('pval').iloc[0]          # most significant candidate
    print(f"  {label}: recovered tE={row['crossing_time']:.1f} d  "
          f"t0={row['peak_time']:.1f} d  (injected tE={TE} d, t0={T0} d)  "
          f"p={row['pval']:.1e}")


def main():
    rng = np.random.default_rng(0)
    forced = synthetic_forcedsource(rng)
    print(f"Synthetic LSST light curve: {len(forced)} epochs across "
          f"{forced['band'].nunique()} bands; injected u0={U0}, tE={TE} d, t0={T0} d.\n")

    # 1) ForcedSource (direct flux) -> detect() normalizes + detects in one call.
    forced_fits = swellar.detect(forced, schema=LSST_FORCEDSOURCE_SCHEMA, rng=rng)
    _report("ForcedSource (detect, direct flux)", forced_fits)

    # 2) DiaSource (difference flux) -> from_lsst folds in the template, then detect.
    canonical = from_lsst(to_diasource(forced), schema=LSST_DIASOURCE_SCHEMA,
                          template_flux_col='template')
    dia_fits = swellar.detect(canonical, schema=SIGNAL_SCHEMA, rng=rng)
    _report("DiaSource  (from_lsst + detect)    ", dia_fits)

    return forced_fits


if __name__ == '__main__':
    main()
