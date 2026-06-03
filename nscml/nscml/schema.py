"""Survey-agnostic light-curve schema for the nscml detector.

The detector reads four canonical per-epoch columns and groups by object:

    objectid     object identifier
    mjd          observation time (days)
    deltamag     detection signal: magnitude relative to the per-band baseline
                 (an event is a negative dip)
    magerr_auto  1-sigma error on the signal

plus an optional ``filter`` (band) column used by the diagnostics.

``LightcurveSchema`` maps an arbitrary survey's column names onto those, and
``normalize`` applies the mapping, renaming columns and, when the survey gives
raw measurements rather than a baseline-subtracted signal, computing ``deltamag``
as (measurement - per-object/per-band median). This keeps the numba kernels and
the file pipeline survey-agnostic: hand them ``normalize(df, schema)``.

``space='mag'`` subtracts a baseline magnitude (the NSC default). ``space='flux'``
builds the achromatic **fractional flux** ``s = F/F_ref - 1`` with error
``sigma_F/F_ref``, the correct signal for flux surveys such as Rubin LSST,
where flux can be negative. In flux mode the canonical ``deltamag`` /
``magerr_auto`` columns carry ``s`` / ``sigma_s`` (same names, flux meaning) and
the detector is run with ``space='flux'``. ``flux_to_mag`` is the alternative,
lossy flux->mag ingest. See PORTABILITY.md.
"""
import warnings
from dataclasses import dataclass

import numpy as np

# canonical column names the detector consumes
ID, TIME, VALUE, ERROR, BAND = 'objectid', 'mjd', 'deltamag', 'magerr_auto', 'filter'

__all__ = ['LightcurveSchema', 'normalize', 'from_nsc', 'flux_to_mag',
           'NSC_SCHEMA', 'AB_ZEROPOINT_NJY']


@dataclass(frozen=True)
class LightcurveSchema:
    """Column-name mapping from a survey table to the canonical detector frame.

    Set ``value`` when the survey already provides the signal (NSC's
    ``deltamag``, or a precomputed fractional flux); otherwise set
    ``measurement`` (``mag_auto`` for ``space='mag'``, or a flux column for
    ``space='flux'``) and ``normalize`` builds the signal from a per-object,
    per-band baseline. ``band`` may be None for single-band data.
    """
    id: str = ID
    time: str = TIME
    error: str = ERROR
    band: str = BAND
    value: str = VALUE          # the signal column, if the survey already has it
    measurement: str = None     # else: raw per-epoch measurement to baseline against
    space: str = 'mag'          # 'mag' (baseline-subtracted) or 'flux' (s = F/F_ref - 1)


NSC_SCHEMA = LightcurveSchema()   # NSC tables already use the canonical names


def normalize(df, schema=NSC_SCHEMA):
    """Return a copy of ``df`` with canonical columns (objectid, mjd,
    magerr_auto, filter, deltamag), per ``schema``.

    For ``space='flux'`` the ``deltamag`` / ``magerr_auto`` columns carry the
    fractional flux ``s = F/F_ref - 1`` and its error ``sigma_F/F_ref`` (same
    names, flux meaning); run the detector with ``space='flux'``.
    """
    if schema.space not in ('mag', 'flux'):
        raise ValueError(f"unknown space {schema.space!r}; use 'mag' or 'flux'")
    if schema.value is None and schema.measurement is None:
        raise ValueError("LightcurveSchema needs either `value` or `measurement`")

    rename = {}
    for src, canon in ((schema.id, ID), (schema.time, TIME), (schema.error, ERROR)):
        if src and src != canon:
            rename[src] = canon
    if schema.band and schema.band != BAND:
        rename[schema.band] = BAND
    if schema.value is not None and schema.value != VALUE:
        rename[schema.value] = VALUE

    out = df.rename(columns=rename).copy()
    if schema.value is not None:
        return out                       # survey already provides the signal + error

    # build the signal from a raw measurement and a per-object/per-band baseline
    groups = [ID] + ([BAND] if BAND in out.columns else [])
    grouped = out.groupby(groups, observed=True)[schema.measurement]
    if schema.space == 'mag':
        out[VALUE] = out[schema.measurement] - grouped.transform('median')
    else:                                # flux: achromatic fractional flux
        f_ref = grouped.transform('median')
        if (f_ref <= 0).any():
            raise ValueError(
                "flux baseline (per-object/per-band median flux) is <= 0 for some "
                "groups; fractional-flux normalization needs a positive reference "
                "flux F_ref (e.g. a template flux). See PORTABILITY.md.")
        out[VALUE] = out[schema.measurement] / f_ref - 1.0
        out[ERROR] = out[ERROR] / f_ref      # sigma_F -> sigma_s = sigma_F / F_ref
    return out


def from_nsc(df):
    """NSC adapter: NSC already uses the canonical names, so this is identity
    (a normalize with the default schema)."""
    return normalize(df, NSC_SCHEMA)


# AB system: m = -2.5 log10(F / 3631 Jy); 3631 Jy = 3.631e9 nanojansky.
AB_ZEROPOINT_NJY = 3.631e9
_MAG_PER_FRAC_FLUX = 2.5 / np.log(10)        # ~1.0857: sigma_mag = this * sigma_F / F


def flux_to_mag(df, schema, zeropoint=AB_ZEROPOINT_NJY):
    """Option-1 flux ingest: convert a flux table to the canonical MAGNITUDE
    frame so the standard ``space='mag'`` pipeline runs unchanged.

    Non-positive-flux epochs are dropped (magnitude is undefined there), lossy,
    and biased near the noise floor; for the negative-flux-safe path use
    ``normalize(df, schema)`` with ``schema.space='flux'``. ``schema.measurement``
    is the flux column and ``schema.error`` its 1-sigma error, in the same flux
    units as ``zeropoint`` (default nanojansky, AB).
    """
    flux, ferr = schema.measurement, schema.error
    if flux is None:
        raise ValueError("flux_to_mag needs schema.measurement set to the flux column")
    keep = df[flux] > 0
    dropped = int((~keep).sum())
    if dropped:
        warnings.warn(
            f"flux_to_mag dropped {dropped} non-positive-flux epoch(s); "
            f"use normalize(space='flux') to keep them", stacklevel=2)
    out = df[keep].copy()
    out['_mag'] = -2.5 * np.log10(out[flux] / zeropoint)
    out['_magerr'] = _MAG_PER_FRAC_FLUX * out[ferr] / out[flux]
    mag_schema = LightcurveSchema(id=schema.id, time=schema.time, band=schema.band,
                                  error='_magerr', measurement='_mag', value=None, space='mag')
    return normalize(out, mag_schema)
