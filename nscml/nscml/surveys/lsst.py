"""Rubin LSST adapter (flux space).

Maps an LSST flux light-curve table -- ForcedSource (direct photometry) or
DiaSource (difference photometry) -- onto the canonical fractional-flux frame the
detector consumes (``nscml.schema``, ``space='flux'``). LSST reports flux in
nanojansky and difference fluxes can be negative, so detection runs on the
achromatic fractional flux ``s = F/F_ref - 1`` (error-safe at low flux), not in
magnitudes. See PORTABILITY.md sec. 2-3.

Column names follow the DP1 schema [1][2]; they shift slightly between data
previews, so confirm against the release you use (and override via the schema).
ForcedSource needs the visit mid-time (``expMidptMJD``) joined from the Visit
table upstream; DiaSource carries ``midpointMjdTai`` natively.

Validated here on synthetic LSST-shaped light curves (tests/test_lsst_adapter.py);
end-to-end validation on real data should use the public, simulated DP0.2 (DP1 is
access-gated).

[1] https://dp1.lsst.io/tutorials/notebook/105/notebook-105-3.html
[2] https://dp1.lsst.io/tutorials/notebook/201/notebook-201-3.html
"""
from ..schema import LightcurveSchema, normalize

__all__ = ['from_lsst', 'LSST_FORCEDSOURCE_SCHEMA', 'LSST_DIASOURCE_SCHEMA']

# Direct photometry: psfFlux is total (positive) flux; its per-band median is a
# valid reference flux F_ref.
LSST_FORCEDSOURCE_SCHEMA = LightcurveSchema(
    id='objectId', time='expMidptMJD', band='band',
    measurement='psfFlux', error='psfFluxErr', value=None, space='flux')

# Difference photometry: psfFlux is difference flux (can be negative; per-band
# median ~ 0). Needs a positive template flux -- see from_lsst(template_flux_col).
LSST_DIASOURCE_SCHEMA = LightcurveSchema(
    id='diaObjectId', time='midpointMjdTai', band='band',
    measurement='psfFlux', error='psfFluxErr', value=None, space='flux')


def from_lsst(df, schema=LSST_FORCEDSOURCE_SCHEMA, template_flux_col=None):
    """Adapt an LSST flux table to the canonical fractional-flux frame.

    ForcedSource (direct flux): leave ``template_flux_col=None``; the per-band
    median ``psfFlux`` is the reference flux ``F_ref``.

    DiaSource (difference flux): its per-band median is ~0, which ``normalize``
    rejects (``F_ref <= 0``). Pass ``template_flux_col`` -- a column of positive
    per-epoch template/reference flux (the object's quiescent flux in that band,
    e.g. from the coadd) -- so the science flux ``F = difference + template`` is
    normalized, and its per-band median (~ the template) is ``F_ref``. Use
    ``schema=LSST_DIASOURCE_SCHEMA`` for the DiaSource column names.

    Returns the canonical frame (``objectid, mjd, deltamag, magerr_auto, filter``,
    where ``deltamag``/``magerr_auto`` carry ``s``/``sigma_s``); run the detector
    with ``space='flux'``.
    """
    if template_flux_col is not None:
        df = df.assign(**{schema.measurement: df[schema.measurement] + df[template_flux_col]})
    return normalize(df, schema)
