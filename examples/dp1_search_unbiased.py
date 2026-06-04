"""Unbiased DP1 microlensing search: forced photometry on point sources, chunked.

``examples/dp1_search_local.py`` is a shakedown biased toward the *most variable*
objects (``dp1.DiaObject`` ordered by ``nDiaSources``), whose baseline scatter
swamps any single excursion. A real microlensing search is the opposite: monitor
*all point sources* for the rare one that brightens. This selects stars from
``dp1.Object`` (``refExtendedness < 0.5``) and pulls their ``dp1.ForcedSource``
light curves -- the unbiased sample, the DP1 analog of the NSC search's
``class_star > 0.9, ndet > 10`` cut.

SCALE -- DP1 is laptop-sized; this stays local. DP1 has ~378k point sources brighter
than ~200 nJy (median 23 forced epochs, a well-sampled tail to ~800), ~21M rows total,
~0.3 GB -- the full unbiased DP1 search runs locally in chunks. The ~1e6 sources that
optical depth (~1e-6) implies for an expected event is an LSST-WFD number, not DP1;
that run -- billions of sources, years of baseline -- is the USDF/batch job. Detection
ignores sparse objects (too few epochs -> no excursion), so no explicit epoch cut is
needed, though MIN_PSF_FLUX or a sky-region cut can focus the search.

Memory stays bounded at any N: it pulls and detects one chunk of stars at a time and
keeps only the candidate rows (a 200-object run produced a 6 KB candidate table). On
the USDF the same script runs against the local TAP service (faster, no egress).

SCHEMA -- verified against the live DP1 schema (2026-06-03): ``dp1.ForcedSource`` has
``objectId, band, psfFlux, psfFluxErr, visit``; it joins ``dp1.Visit`` on ``visit``
(which carries ``expMidptMJD``). ``dp1.Object`` has no bare ``psfFlux`` -- fluxes are
per-band (``r_psfFlux`` etc.) -- and ``refExtendedness`` (0 = point, 1 = extended).

Setup is the same as dp1_search_local.py (RSP_TOKEN, RSP_TAP_URL). Run:
    python examples/dp1_search_unbiased.py             # small-N test
    N_STARS=400000 CHUNK=2000 python examples/dp1_search_unbiased.py    # full DP1 point sources
"""

# %% 1. setup ------------------------------------------------------------------
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import swellar
from swellar.surveys.lsst import LSST_FORCEDSOURCE_SCHEMA

RSP_TAP_URL = os.environ.get("RSP_TAP_URL", "https://data.lsst.cloud/api/tap")
N_STARS = int(os.environ.get("N_STARS", "2000"))      # small default = local selection test
CHUNK = int(os.environ.get("CHUNK", "500"))           # objectIds per light-curve query
MIN_PSF_FLUX = float(os.environ.get("MIN_PSF_FLUX", "200"))  # nJy; skip the faintest (noise-floor)
REF_FLUX_COL = os.environ.get("REF_FLUX_COL", "r_psfFlux")   # dp1.Object band flux for the cut
VISIT_JOIN = os.environ.get("VISIT_JOIN", "visit")    # ForcedSource->Visit FK (verified: visit)

try:
    _BASE = Path(__file__).resolve().parent
except NameError:
    _BASE = Path.cwd()
RESULTS = _BASE / "results"
RESULTS.mkdir(exist_ok=True)


def _tap_service():
    import requests
    import pyvo
    tok = os.environ.get("RSP_TOKEN")
    if not tok:
        f = Path.home() / ".rsp" / "tap.token"
        tok = f.read_text().strip() if f.exists() else None
    if not tok:
        host = RSP_TAP_URL.split("/api")[0]
        sys.exit(f"No RSP token. Create one (read:tap) at {host} -> Security tokens, then "
                 "export RSP_TOKEN=$(security find-generic-password -s rsp-usdf-tap -w)")
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {tok.strip()}"
    return pyvo.dal.TAPService(RSP_TAP_URL, session=session)


# %% 2. select point sources (unbiased: stars, not the most-variable objects) --
service = _tap_service()
print(f"TAP: {RSP_TAP_URL}  | selecting up to {N_STARS:,} point sources")
star_ids = service.search(f"""
    SELECT TOP {N_STARS} objectId
    FROM   dp1.Object
    WHERE  refExtendedness < 0.5          -- point sources (0 = star, 1 = extended)
      AND  {REF_FLUX_COL} > {MIN_PSF_FLUX}   -- bright enough to monitor (per-band ref flux)
    ORDER  BY objectId
""").to_table().to_pandas()["objectId"].tolist()
print(f"{len(star_ids):,} point sources selected")

# %% 3. chunked pull + detect; keep only candidates ----------------------------
# One chunk of light curves in memory at a time -> bounded RAM at any N. psfFlux is
# total flux (positive), so detect() normalizes to fractional flux s = F/F_ref - 1.
# restrict_well_sampled=False: ComCam's ~32-day baseline is shorter than the 50-day
# well-sampled default.
schema = replace(LSST_FORCEDSOURCE_SCHEMA, id="objectId")
cands, n_epochs, n_done = [], 0, 0
for i in range(0, len(star_ids), CHUNK):
    batch = star_ids[i:i + CHUNK]
    ids = ", ".join(str(x) for x in batch)
    lc = service.search(f"""
        SELECT fs.objectId, fs.band, vis.expMidptMJD, fs.psfFlux, fs.psfFluxErr
        FROM   dp1.ForcedSource AS fs
        JOIN   dp1.Visit        AS vis ON vis.{VISIT_JOIN} = fs.{VISIT_JOIN}
        WHERE  fs.objectId IN ({ids})
    """).to_table().to_pandas()
    n_epochs += len(lc); n_done += len(batch)
    if len(lc):
        # Forced PSF flux scatters negative for faint stars; keep only (object, band)
        # groups with a positive median (a valid reference flux F_ref to normalize by).
        med = lc.groupby(["objectId", "band"], observed=True)["psfFlux"].transform("median")
        lc = lc[med > 0]
    if len(lc):
        c = swellar.detect(lc, schema=schema, restrict_well_sampled=False)
        if len(c):
            cands.append(c)
    got = sum(len(c) for c in cands)
    print(f"  chunk {i // CHUNK + 1}/{-(-len(star_ids) // CHUNK)}: "
          f"{n_done:,}/{len(star_ids):,} stars, {n_epochs:,} epochs, {got} candidate(s)")

cand = (pd.concat(cands, ignore_index=True).sort_values("pval").reset_index(drop=True)
        if cands else pd.DataFrame())

# %% 4. capture results --------------------------------------------------------
cand.to_parquet(RESULTS / "dp1_unbiased_candidates.parquet")
cand.head(50).to_csv(RESULTS / "dp1_unbiased_candidates_top50.csv", index=False)
summary = (
    f"DP1 unbiased search {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
    f"  endpoint   : {RSP_TAP_URL}\n"
    f"  selection  : dp1.Object refExtendedness<0.5, psfFlux>{MIN_PSF_FLUX} nJy\n"
    f"  stars      : {len(star_ids):,} (epochs pulled {n_epochs:,})\n"
    f"  candidates : {len(cand)}\n"
)
(RESULTS / "dp1_unbiased_summary.txt").write_text(summary)
print("\n" + summary)
if len(cand):
    print(cand[["objectid", "pval", "crossing_time", "peak_time",
                "impact_parameter"]].head(20).to_string(index=False))
