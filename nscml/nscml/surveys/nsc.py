"""NOIRLab Source Catalog (NSC) adapter.

NSC tables already use the canonical column names, so ``from_nsc`` is identity.
This module is also the labelled home for the NSC-specific helpers: they remain
defined in ``nsctools`` for backward compatibility (notebooks import them from
there) and are re-exported here to mark them as NSC-specific rather than part of
the survey-agnostic detection core. A future cleanup may relocate the
definitions once those callers are migrated.
"""
from ..schema import LightcurveSchema, NSC_SCHEMA, normalize, from_nsc
from .. import nsctools

# NSC catalog-access / naming conventions (defined in nsctools):
make_instrument = nsctools.make_instrument          # DECam/NSC exposure-id -> instrument
make_delta_mags = nsctools.make_delta_mags          # per-band baseline subtraction (<band>mag)
make_delta_mags_mono = nsctools.make_delta_mags_mono
magstr = nsctools.magstr                            # NSC BigQuery column list
color_filter = nsctools.color_filter                # band -> plot colour (ugrizy + VR/Y)
marker_map = nsctools.marker_map                    # instrument -> plot marker

__all__ = [
    'NSC_SCHEMA', 'normalize', 'from_nsc', 'make_instrument', 'make_delta_mags',
    'make_delta_mags_mono', 'magstr', 'color_filter', 'marker_map',
]
