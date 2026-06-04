"""DP1 microlensing search run LOCALLY over the RSP TAP API (no RSP notebook needed).

Same search as ``examples/rsp_dp1_search.py`` -- the ADQL and the ``detect()`` calls
are the same -- but instead of running inside an RSP notebook this pulls the DP1
light curves over the network with an RSP user token and runs the detector on this
machine. Detection then happens in the environment swellar was *validated* on
(numpy 1.23.5, the version the goldens were captured against), so the numpy-2.x
caveat in RUNNING_ON_RSP.md never applies: only the catalog query goes remote.

The first run pulls from TAP and caches the light curves to ``examples/results/``;
later runs reuse that cache (set ``RSP_REPULL=1`` to force a fresh pull), so you can
iterate on the *detection* -- target selection, schema, thresholds -- without
re-querying. The pull is small (a few hundred objects), so there's no reason to hold
compute on the platform for a search this size; for a survey-scale sweep the calculus
flips and co-located compute wins (see RUNNING_ON_RSP.md).

Setup (one time):
  1. Create an RSP user token with scope ``read:tap``. DP1's documented home is the
     IDF (https://data.lsst.cloud/); the USDF (https://usdf-rsp.slac.stanford.edu/)
     serves the same API -- make the token on whichever your data rights are on, and
     set RSP_TAP_URL accordingly.
  2. Put it in the environment (never in a tracked file). On macOS the login keychain
     is the natural store:
       security add-generic-password -s rsp-usdf-tap -w        # paste at the prompt
       export RSP_TOKEN=$(security find-generic-password -s rsp-usdf-tap -w)
       export RSP_TAP_URL='https://usdf-rsp.slac.stanford.edu/api/tap'   # USDF
  3. Run it in an env that has swellar + pyvo + pandas (the project's ``nsc`` env does):
       python examples/dp1_search_local.py

Cells are marked ``# %%`` so this also pastes top-to-bottom into a notebook.
"""

# %% 1. setup ------------------------------------------------------------------
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless: write plots to disk instead of opening a window
import matplotlib.pyplot as plt

import swellar
from swellar.surveys.lsst import (from_lsst, LSST_FORCEDSOURCE_SCHEMA,
                                LSST_DIASOURCE_SCHEMA)

# DP1 lives on the IDF by default; override to point at the USDF deployment.
RSP_TAP_URL = os.environ.get("RSP_TAP_URL", "https://data.lsst.cloud/api/tap")
N_OBJECTS = int(os.environ.get("N_OBJECTS", "200"))

try:
    _BASE = Path(__file__).resolve().parent
except NameError:  # pasted into a notebook, where __file__ is undefined
    _BASE = Path.cwd()
RESULTS = _BASE / "results"
RESULTS.mkdir(exist_ok=True)
CACHE = RESULTS / "dp1_lightcurves.parquet"


def _tap_service():
    """An authenticated TAP service (token from RSP_TOKEN, else ~/.rsp/tap.token)."""
    import requests
    import pyvo

    tok = os.environ.get("RSP_TOKEN")
    if not tok:
        token_file = Path.home() / ".rsp" / "tap.token"
        if token_file.exists():
            tok = token_file.read_text().strip()
    if not tok:
        host = RSP_TAP_URL.split("/api")[0]
        sys.exit(
            "No RSP token found.\n"
            f"  Create one (scope read:tap) at {host} -> Security tokens, then:\n"
            "      export RSP_TOKEN=$(security find-generic-password -s rsp-usdf-tap -w)\n"
            "  or write it to ~/.rsp/tap.token. Never commit the token."
        )
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {tok.strip()}"
    return pyvo.dal.TAPService(RSP_TAP_URL, session=session)


# %% 2-3. pull the light curves (or reuse a cached pull) -----------------------
# nDiaSources = number of difference-image detections; >10 picks objects that
# actually varied. NOTE: this biases toward the *most variable* objects, whose
# baseline scatter tends to swamp any single excursion -- fine to prove the
# pipeline, but for a real search select mostly-quiescent objects (by sky region
# or forced-source epoch count) instead. psfFlux is the direct (total) forced
# flux; psfDiffFlux is the difference flux; diao.<band>_psfFluxMean is the
# per-band reference (template) flux F_ref.
if CACHE.exists() and not os.environ.get("RSP_REPULL"):
    print(f"reusing cached light curves: {CACHE}  (set RSP_REPULL=1 to re-pull)")
    lc = pd.read_parquet(CACHE)
else:
    service = _tap_service()
    print(f"TAP service: {RSP_TAP_URL}")
    targets = service.search(f"""
        SELECT TOP {N_OBJECTS} diaObjectId, ra, dec, nDiaSources
        FROM   dp1.DiaObject
        WHERE  nDiaSources > 10
        ORDER  BY nDiaSources DESC
    """).to_table().to_pandas()
    print(f"{len(targets)} objects; nDiaSources "
          f"{targets.nDiaSources.min()}-{targets.nDiaSources.max()}")
    id_list = ", ".join(str(i) for i in targets["diaObjectId"])
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
    lc.to_parquet(CACHE)
print(f"{len(lc)} forced-source epochs across {lc.diaObjectId.nunique()} objects")

# %% 4. detect, direct flux ----------------------------------------------------
# Reuse the survey's tested ForcedSource schema (it carries value=None + space=
# 'flux', so detect() normalizes psfFlux to fractional flux s = F/F_ref - 1 and
# runs the flux detector in one call); only the id column differs for
# ForcedSourceOnDiaObject. restrict_well_sampled=False: ComCam's baseline (~32 d)
# is shorter than the 50-day well-sampled default, so search the whole curve.
direct_schema = replace(LSST_FORCEDSOURCE_SCHEMA, id="diaObjectId")
cand = swellar.detect(lc, schema=direct_schema, restrict_well_sampled=False)
cand = cand.sort_values("pval").reset_index(drop=True)
print(f"{len(cand)} PSPL candidates (direct flux)")
cand.to_parquet(RESULTS / "dp1_candidates_direct.parquet")
cand.head(15).to_csv(RESULTS / "dp1_candidates_direct_top15.csv", index=False)

# %% 5. detect, difference flux (optional; more sensitive on bright hosts) -----
# Difference flux has a per-band median ~0, so fold in the DiaObject mean as F_ref.
# Real DP1 difference photometry is messy: some objects' science flux
# (psfDiffFlux + template) has a non-positive per-band median, which can't be
# normalized to fractional flux -- drop those (object, band) groups, and guard the
# whole path so a data-quality issue degrades gracefully instead of aborting.
cand_diff = pd.DataFrame()
try:
    lc["template"] = lc.apply(lambda r: r[f"{r['band']}_psfFluxMean"], axis=1)
    lc["F_sci"] = lc["psfDiffFlux"] + lc["template"]
    med = lc.groupby(["diaObjectId", "band"], observed=True)["F_sci"].transform("median")
    lc_ok = lc[med > 0].copy()
    print(f"difference flux: {len(lc_ok)}/{len(lc)} epochs with positive median "
          f"science flux ({lc_ok.diaObjectId.nunique()} objects)")
    diff_schema = replace(LSST_DIASOURCE_SCHEMA, time="expMidptMJD",
                          measurement="psfDiffFlux", error="psfDiffFluxErr")
    canonical = from_lsst(lc_ok, diff_schema, template_flux_col="template")
    signal_schema = swellar.LightcurveSchema(value="deltamag", error="magerr_auto",
                                           band="filter", space="flux")
    cand_diff = swellar.detect(canonical, schema=signal_schema,
                             restrict_well_sampled=False).sort_values("pval").reset_index(drop=True)
    print(f"{len(cand_diff)} PSPL candidates (difference flux)")
    cand_diff.to_parquet(RESULTS / "dp1_candidates_diff.parquet")
    cand_diff.head(15).to_csv(RESULTS / "dp1_candidates_diff_top15.csv", index=False)
except Exception as e:
    print(f"difference-flux path skipped ({type(e).__name__}: {e})")

# %% 6. eyeball + save the top candidate ---------------------------------------
if len(cand):
    top = cand.iloc[0]["objectid"]
    obj = lc[lc["diaObjectId"] == top]
    for band, g in obj.groupby("band"):
        plt.errorbar(g["expMidptMJD"], g["psfFlux"], g["psfFluxErr"],
                     fmt="o", ms=3, label=band)
    plt.xlabel("MJD"); plt.ylabel("psfFlux (nJy)")
    plt.title(f"DiaObject {top}  (p={cand.iloc[0]['pval']:.1e})")
    plt.legend()
    plt.savefig(RESULTS / "dp1_top_candidate.png", dpi=130, bbox_inches="tight")
    print(cand.iloc[0])          # tE (crossing_time), t0 (peak_time), u0, pval

# %% 7. run summary ------------------------------------------------------------
span = float(lc["expMidptMJD"].max() - lc["expMidptMJD"].min())
summary = (
    f"DP1 local search run {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
    f"  endpoint     : {RSP_TAP_URL}\n"
    f"  objects      : {lc.diaObjectId.nunique()} (nDiaSources>10, top {N_OBJECTS})\n"
    f"  epochs        : {len(lc)}; baseline ~{span:.0f} d\n"
    f"  candidates   : direct={len(cand)}  diff={len(cand_diff)}\n"
)
(RESULTS / "dp1_run_summary.txt").write_text(summary)
print(summary)
