"""DP1 microlensing search on the Rubin Science Platform (paste into an RSP notebook).

Pulls *real* DP1 (LSSTComCam) forced-source light curves via TAP, runs the nscml
flux-space detector, and ranks PSPL candidates. Cells are marked with `# %%` so
this runs top-to-bottom as a script or pastes cell-by-cell into a Jupyter notebook.

REQUIRES the Rubin Science Platform: `lsst.rsp` / the TAP service, and DP1 data
rights. It does NOT run locally (no Rubin data here), see RUNNING_ON_RSP.md.

Caveats for this first run (engineering shakedown, not a science result):
- DP1 is ComCam: ~7 fields, short time baseline. Expect variable stars / artifacts,
  not real microlensing, the point is to prove the pipeline on real photometry.
- The detector's well-sampled-region gate (WS_INTERVAL_DAYS=50) is an NSC default
  and is almost certainly longer than ComCam's per-field baseline, so we pass
  restrict_well_sampled=False here (search the whole light curve). Tuning the
  cadence parameters for LSST is the deliberately-deferred research item.
- Detection runs on the achromatic fractional flux s = F/F_ref - 1 (space='flux').
"""

# %% 1. setup -------------------------------------------------------------------
from dataclasses import replace
from lsst.rsp import get_tap_service        # newer RSP; older images: get_tap_service()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nscml
from nscml.surveys.lsst import (from_lsst, LSST_FORCEDSOURCE_SCHEMA,
                                LSST_DIASOURCE_SCHEMA)

service = get_tap_service("tap")

# %% 2. select variable DiaObjects to search -----------------------------------
# nDiaSources = number of difference-image detections; >10 picks objects that
# actually varied (and so have epochs worth searching). This biases toward
# variables, fine for a shakedown; for an unbiased search, select by sky region
# or by forced-source epoch count instead.
N_OBJECTS = 200
targets = service.search(f"""
    SELECT TOP {N_OBJECTS} diaObjectId, ra, dec, nDiaSources
    FROM   dp1.DiaObject
    WHERE  nDiaSources > 10
    ORDER  BY nDiaSources DESC
""").to_table().to_pandas()                 # if TOP is rejected: use maxrec= or LIMIT
print(f"{len(targets)} objects; nDiaSources "
      f"{targets.nDiaSources.min()}-{targets.nDiaSources.max()}")
id_list = ", ".join(str(i) for i in targets["diaObjectId"])

# %% 3. pull their forced-source light curves (joined to Visit for the time) ----
# psfFlux is the direct (total) forced flux; psfDiffFlux is the difference flux.
# diao.<band>_psfFluxMean is the per-band reference (template) flux F_ref.
lc = service.search(f"""
    SELECT fsodo.diaObjectId, fsodo.band, vis.expMidptMJD,
           fsodo.psfFlux, fsodo.psfFluxErr,
           fsodo.psfDiffFlux, fsodo.psfDiffFluxErr,
           diao.u_psfFluxMean, diao.g_psfFluxMean, diao.r_psfFluxMean,
           diao.i_psfFluxMean, diao.z_psfFluxMean, diao.y_psfFluxMean
    FROM   dp1.ForcedSourceOnDiaObject AS fsodo
    JOIN   dp1.Visit                   AS vis  ON vis.visit = fsodo.visit
    JOIN   dp1.DiaObject               AS diao ON diao.diaObjectId = fsodo.diaObjectId
    WHERE  fsodo.diaObjectId IN ({id_list})
""").to_table().to_pandas()
print(f"{len(lc)} forced-source epochs across {lc.diaObjectId.nunique()} objects")

# %% 4. detect, direct flux ---------------------------------------------------
# psfFlux is total flux, so its per-band median is a valid F_ref; detect()
# normalizes to fractional flux and runs the flux detector in one call. Reuse the
# survey's tested ForcedSource schema (value=None + space='flux'); only the id
# column differs for ForcedSourceOnDiaObject.
direct_schema = replace(LSST_FORCEDSOURCE_SCHEMA, id="diaObjectId")
cand = nscml.detect(lc, schema=direct_schema, restrict_well_sampled=False)
cand = cand.sort_values("pval").reset_index(drop=True)
print(f"{len(cand)} PSPL candidates (direct flux)")
cand.head(15)

# %% 5. detect, difference flux (optional; more sensitive on bright hosts) ----
# Difference flux has a per-band median ~0, so fold in the DiaObject mean as F_ref.
# Keep only (object, band) groups whose science flux (psfDiffFlux + template) has a
# positive per-band median -- a non-positive median can't be normalized to
# fractional flux (real DP1 difference photometry has some of these).
lc["template"] = lc.apply(lambda r: r[f"{r['band']}_psfFluxMean"], axis=1)
lc["F_sci"] = lc["psfDiffFlux"] + lc["template"]
med = lc.groupby(["diaObjectId", "band"], observed=True)["F_sci"].transform("median")
lc_ok = lc[med > 0].copy()
diff_schema = replace(LSST_DIASOURCE_SCHEMA, time="expMidptMJD",
                      measurement="psfDiffFlux", error="psfDiffFluxErr")
canonical = from_lsst(lc_ok, diff_schema, template_flux_col="template")
cand_diff = nscml.detect(
    canonical,
    schema=nscml.LightcurveSchema(value="deltamag", error="magerr_auto",
                                  band="filter", space="flux"),
    restrict_well_sampled=False).sort_values("pval").reset_index(drop=True)
print(f"{len(cand_diff)} PSPL candidates (difference flux)")
cand_diff.head(15)

# %% 6. eyeball the top candidate ----------------------------------------------
if len(cand):
    top = cand.iloc[0]["objectid"]
    obj = lc[lc["diaObjectId"] == top]
    for band, g in obj.groupby("band"):
        plt.errorbar(g["expMidptMJD"], g["psfFlux"], g["psfFluxErr"],
                     fmt="o", ms=3, label=band)
    plt.xlabel("MJD"); plt.ylabel("psfFlux (nJy)")
    plt.title(f"DiaObject {top}  (p={cand.iloc[0]['pval']:.1e})")
    plt.legend(); plt.show()
    print(cand.iloc[0])          # tE (crossing_time), t0 (peak_time), u0, pval
