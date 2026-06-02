"""Survey-agnostic light-curve schema for the nscml detector.

The detector reads four canonical per-epoch columns and groups by object:

    objectid     object identifier
    mjd          observation time (days)
    deltamag     detection signal: magnitude relative to the per-band baseline
                 (an event is a negative dip)
    magerr_auto  1-sigma error on the signal

plus an optional ``filter`` (band) column used by the diagnostics.

``LightcurveSchema`` maps an arbitrary survey's column names onto those, and
``normalize`` applies the mapping -- renaming columns and, when the survey gives
raw measurements rather than a baseline-subtracted signal, computing ``deltamag``
as (measurement - per-object/per-band median). This keeps the numba kernels and
the file pipeline survey-agnostic: hand them ``normalize(df, schema)``.

Phase 1 is magnitude-space only (``space='mag'``). Flux-space ingest (Rubin
LSST) is Phase 2 -- see PORTABILITY.md.
"""
from dataclasses import dataclass

# canonical column names the detector consumes
ID, TIME, VALUE, ERROR, BAND = 'objectid', 'mjd', 'deltamag', 'magerr_auto', 'filter'

__all__ = ['LightcurveSchema', 'normalize', 'NSC_SCHEMA', 'from_nsc']


@dataclass(frozen=True)
class LightcurveSchema:
    """Column-name mapping from a survey table to the canonical detector frame.

    Set ``value`` when the survey already provides a baseline-subtracted signal
    (e.g. NSC's ``deltamag``); otherwise set ``measurement`` (e.g. ``mag_auto``)
    and ``normalize`` builds the signal by subtracting the per-object, per-band
    median. ``band`` may be None for single-band data.
    """
    id: str = ID
    time: str = TIME
    error: str = ERROR
    band: str = BAND
    value: str = VALUE          # baseline-subtracted signal column, if present
    measurement: str = None     # else: raw per-epoch measurement to baseline-subtract
    space: str = 'mag'          # 'mag' today; 'flux' reserved for Phase 2


NSC_SCHEMA = LightcurveSchema()   # NSC tables already use the canonical names


def normalize(df, schema=NSC_SCHEMA):
    """Return a copy of ``df`` with canonical columns (objectid, mjd,
    magerr_auto, filter, deltamag), per ``schema``."""
    if schema.space != 'mag':
        raise NotImplementedError(
            "flux-space normalization is Phase 2; see PORTABILITY.md")
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

    if schema.value is None:
        # build the signal: measurement minus its per-object, per-band median
        groups = [ID] + ([BAND] if BAND in out.columns else [])
        baseline = out.groupby(groups, observed=True)[schema.measurement].transform('median')
        out[VALUE] = out[schema.measurement] - baseline
    return out


def from_nsc(df):
    """NSC adapter: NSC already uses the canonical names, so this is identity
    (a normalize with the default schema)."""
    return normalize(df, NSC_SCHEMA)
