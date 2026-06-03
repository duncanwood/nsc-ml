# Running nsc-ml on the Rubin Science Platform (RSP)

How to drop this detector onto the [Rubin Science Platform](https://rsp.lsst.io/),
pull photometry out of a data release with the **Butler** (or **TAP**), and run
`detect()` on it. Written against the public docs as of June 2026; the data
model shifts slightly between releases, so confirm column names against the
release you actually use and override the schema accordingly.

> **Status:** this is a deployment recipe, not yet an end-to-end validated run.
> No Rubin data lives on the dev machine, so the steps below are assembled from
> the public DP0.2 / DP1 documentation (sources at the bottom). The last mile,
> running it in an RSP notebook against real tables, is the open task.

## TL;DR

| Question | Answer |
|---|---|
| Python compatible? | **Yes.** The RSP stack is Science Pipelines **v29.2 / Python 3.12**; nscml needs `>=3.11`. |
| Which data? | **DP0.2** (simulated DC2) is the public sandbox; **DP1** (real, ComCam, released 2025-06-30) is **data-rights-gated** (US/Chile scientists + students qualify, you do). |
| How to install? | `pip install --user` in an RSP terminal (no root; pip wraps conda over `rubin-env`). |
| How to get light curves? | **TAP/ADQL** for per-object light curves (Qserv-backed, ergonomic); **Butler** for tract/patch table retrieval. |
| Detector entry point | `nscml.detect(df, schema)` with a flux `LightcurveSchema` (`space='flux'`). |
| Main caveat | nscml was validated on **numpy 1.23.5**; the stack is **numpy 2.x**, install *without* the pinned `requirements.txt` and run the test suite first. |

## 1. Account and data access

Get an RSP account at <https://rsp.lsst.io/> (sign-in via your institutional
identity). **DP0.2** is available to anyone with an account. **DP1**, the first
release of *real* Rubin data (LSSTComCam, ~3.5 TB, 2025-06-30), is restricted
to Rubin data-rights holders: all scientists and students in the US and Chile,
plus named international in-kind members. As a US-based physicist you qualify;
request access through the RSP sign-up.

Prototype on **DP0.2** first (it's public and free of NDA friction), then point
the same code at DP1 by swapping the catalog namespace.

## 2. Install nscml on the RSP

Open a **Terminal** in the Notebook Aspect.

**Recommended (try it in the stack kernel).** Install against the stack's own
numpy/scipy/numba (i.e. *don't* pin), so you don't perturb `rubin-env`:

```bash
pip install --user "git+https://github.com/duncanwood/nsc-ml.git#subdirectory=nscml"
```

The package lives in the `nscml/` subdirectory of the repo (the `--subdirectory`
is required). `--user` drops it in `~/.local/...`, visible from the LSST kernel.

Then **validate before trusting it**, the stack is numpy 2.x and nscml's
goldens were captured on numpy 1.23.5:

```bash
git clone https://github.com/duncanwood/nsc-ml.git
cd nsc-ml && python -m pytest        # expect 100 passing; investigate any numpy-2 breakage
```

> **Do not** `pip install --user -r requirements.txt`. Those pins (numpy==1.23.5,
> scipy==1.11.4, ...) are for reproducing the dissertation results; forcing them
> into the stack kernel would downgrade numpy and break the Butler/`lsst.*`
> packages. Use the lower-bound install above, or option B.

**Option B (isolated kernel).** If the numpy-2 validation surfaces breakage,
build a dedicated env with the pinned deps and register it as a Jupyter kernel:

```bash
conda create -y -n nscml python=3.11
conda activate nscml
pip install -r requirements-dev.txt && pip install -e nscml --config-settings editable_mode=compat
python -m ipykernel install --user --name nscml --display-name "nscml (pinned)"
```

Then **access data in the stack kernel** (Butler/TAP live there), save the light
curves to parquet, and **detect in the `nscml` kernel**, nscml's file pipeline
already reads parquet, so the two kernels hand off cleanly through disk.

## 3. Pull a light curve

The detector wants one tidy table per object: an id, a time (MJD), a band, a
flux, and a flux error. Two ways to get there.

### 3a. TAP / ADQL (best for per-object light curves)

Qserv is optimized for catalog queries; the forced-source tables join to the
visit table for the time. On the RSP:

```python
from lsst.rsp import get_tap_service
import pandas as pd

service = get_tap_service()        # the RSP TAP endpoint

# DP0.2 (simulated). Forced PSF photometry on a DiaObject, joined to CcdVisit
# for the observation time. psfFlux/psfFluxErr are in nanojansky (nJy).
query = """
SELECT  fs.diaObjectId, fs.band, fs.psfFlux, fs.psfFluxErr, cv.expMidptMJD
FROM    dp02_dc2_catalogs.ForcedSourceOnDiaObject AS fs
JOIN    dp02_dc2_catalogs.CcdVisit               AS cv  ON cv.ccdVisitId = fs.ccdVisitId
WHERE   fs.diaObjectId = {oid}
""".format(oid=MY_DIAOBJECT_ID)

df = service.search(query).to_table().to_pandas()
```

For **DP1** (real data) the namespace is `dp1.*` and the time comes from the
`Visit` table on the `visit` key:

```sql
SELECT fs.diaObjectId, fs.band, fs.psfFlux, fs.psfFluxErr, v.expMidptMJD
FROM   dp1.ForcedSourceOnDiaObject AS fs
JOIN   dp1.Visit                   AS v  ON v.visit = fs.visit
WHERE  fs.diaObjectId = <id>
```

DP1 also carries `psfDiffFlux` / `psfDiffFluxErr` (forced flux on the *difference*
image), that's difference photometry; see the schema note in step 4.

### 3b. Butler (best for tract/patch sweeps)

The Butler serves the same photometry as per-tract parquet tables:

```python
from lsst.daf.butler import Butler
butler = Butler('dp02', collections='2.2i/runs/DP0.2')

# forced PSF photometry at Object positions (direct flux), one row per source
fsrc = butler.get('forcedSourceTable', dataId={'tract': 4431, 'patch': 17})
# or forced on DiaObject positions (carries direct + difference flux):
# butler.get('forcedSourceOnDiaObjectTable', dataId={'tract': 4431, 'patch': 16})
```

The forced-source tables don't carry the visit time directly, join on the
visit key (`ccdVisitId` / `visit`) to the `CcdVisit` / `Visit` table to add
`expMidptMJD`, same as the ADQL join above. The Butler is organized by
tract/patch, so it's the right tool for an area sweep; for a single known object,
TAP is simpler. (Useful Butler catalog `datasetType`s: `objectTable`,
`sourceTable`, `forcedSourceTable`, `diaObjectTable_tract`, `diaSourceTable`,
`forcedSourceOnDiaObjectTable`.)

## 4. Run the detector

The columns map straight onto a flux `LightcurveSchema`; nscml ships two starting
points and `detect()` does the rest in memory:

```python
import nscml
from nscml.surveys.lsst import from_lsst, LSST_FORCEDSOURCE_SCHEMA, LSST_DIASOURCE_SCHEMA

# Direct flux (ForcedSource on Object): psfFlux is total flux; its per-band median
# is a valid reference F_ref. One call -- detect normalizes to fractional flux
# s = F/F_ref - 1 (achromatic, poolable across bands) and runs the flux detector.
events = nscml.detect(df, schema=LSST_FORCEDSOURCE_SCHEMA)   # df has objectId, expMidptMJD, band, psfFlux, psfFluxErr
```

Adjust the schema to whatever columns you actually pulled, e.g. for
`ForcedSourceOnDiaObject` with direct `psfFlux`, the id is `diaObjectId`:

```python
schema = nscml.LightcurveSchema(id='diaObjectId', time='expMidptMJD', band='band',
                                measurement='psfFlux', error='psfFluxErr', space='flux')
events = nscml.detect(df, schema=schema)
```

**Difference photometry** (`psfDiffFlux`, or any DiaSource difference flux) has a
per-band median ~0, which is not a valid `F_ref`. Fold in a positive template
flux first (the object's quiescent/coadd flux in that band) via `from_lsst`:

```python
df['template'] = <per-epoch positive template flux, e.g. the coadd Object psfFlux in that band>
canonical = from_lsst(df, LSST_DIASOURCE_SCHEMA, template_flux_col='template')  # F = diff + template
events = nscml.detect(canonical, schema=nscml.LightcurveSchema(
    value='deltamag', error='magerr_auto', band='filter', space='flux'))
```

See [`examples/lsst_quickstart.py`](examples/lsst_quickstart.py) for the same flow
end-to-end on synthetic LSST-shaped data (runs locally, no RSP needed), and
**[`examples/rsp_dp1_search.py`](examples/rsp_dp1_search.py)** for a ready-to-paste
DP1 notebook: TAP-select variable `DiaObject`s, pull their `ForcedSourceOnDiaObject`
light curves (joined to `Visit` for the time), and run `detect()` over real data.

## 5. Caveats and the open last mile

- **numpy 1.x -> 2.x.** nscml's goldens were captured on numpy 1.23.5; the RSP
  stack is numpy 2.x. Run `python -m pytest` on the RSP before trusting results;
  fall back to the isolated kernel (option B) if anything breaks.
- **Fractional flux, not raw flux.** Detection runs on `s = F/F_ref - 1`
  (achromatic, so bands pool); never pool raw `dF` across bands. This is handled
  for you by `space='flux'`. See [PORTABILITY.md](PORTABILITY.md) sec. 2.
- **F_ref must be positive.** Direct flux (`psfFlux`) is fine; difference flux
  needs a template (step 4).
- **Cadence / event-population defaults are NSC's.** The well-sampled-region,
  detection-timescale, and crossing-time defaults were tuned with Rubin in mind
  and are kept as-is; quantifying how well they suit the LSST WFD cadence (and an
  LSST `(tE, u0)` population) is deliberate future research, not yet done.
- **Confirm the schema against the release.** Column and table names shift
  between data previews; the DP1 names (`expMidptMJD`, `psfFlux`, `diaObjectId`,
  the `Visit` join) are the current best reference, verify in the release's
  schema browser and override the `LightcurveSchema` if they differ.

## Sources

- RSP Notebook Aspect / Python env: <https://nb.lsst.io/environment/python.html>, <https://rsp.lsst.io/guides/notebooks/index.html>
- Stack version / Python 3.12 (v29.2, `rubin-env`): <https://pipelines.lsst.io/>, <https://developer.lsst.io/stack/conda.html>
- DP0.2 Butler tutorial (instantiation, dataset types): <https://github.com/rubin-dp0/tutorial-notebooks/blob/main/DP02_04a_Introduction_to_the_Butler.ipynb>
- DP0.2 data access / ADQL recipes / TAP service: <https://dp0-2.lsst.io/data-access-analysis-tools/index.html>, <https://dp0-2.lsst.io/data-access-analysis-tools/adql-recipes.html>
- DP0.2 forced-photometry light curves: <https://dp0-2.lsst.io/tutorials-examples/Portal-5.html>
- DP1 (release, access, ForcedSource/Visit schema): <https://dp1.lsst.io/>, <https://dp1.lsst.io/tutorials/notebook/201/notebook-201-3.html>, <https://rubinobservatory.org/for-scientists/data-products/recent-data-releases>
- DP1 data rights / access: <https://noirlab.edu/public/announcements/rubinann25009/>
