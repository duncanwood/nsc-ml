# nscml -- portability & generalization audit

How to take this from "my NSC dissertation pipeline" to "a tool another person,
telescope, or dataset (Rubin LSST in particular) can run." This is an audit /
roadmap of written proposals, not applied changes.

## Where it stands

The detector is survey-agnostic in spirit -- a weighted-moving-average
excursion test plus a PSPL fit -- but the code is wired to one concrete schema:
the NOIRLab Source Catalog (NSC), DECam-era band/instrument conventions, and
**magnitudes**. Two coupling layers have to be loosened, in order of difficulty:

1. **Column/band naming** (mechanical) -- the schema is hardcoded as bare
   string literals everywhere.
2. **Magnitude vs flux** (substantive) -- the method is formulated in
   magnitudes; LSST photometry is flux in nanojansky and can be negative. This
   is the real Rubin blocker.

## 1. Schema coupling -- a data contract + adapter protocol

Hardcoded assumptions, found by grep:

- Column names as bare literals across the core: `'mjd'` (28x), `'objectid'`
  (22x), `'deltamag'` (19x), `'magerr_auto'` (13x), `'filter'` (11x),
  `'mag_auto'` (6x), `'instrument'` (5x), `'exposure'` (3x), `'ra'`/`'dec'`.
  There is no single place that says "these are the columns."
- The detection signal is `deltamag` = magnitude relative to a **per-band
  baseline**. The only code that builds it (`make_delta_mags`) assumes the
  baseline lives in per-band columns named `<band>mag` (e.g. `gmag`) and that
  the measured column is `mag_auto`.
- Band/instrument conventions: `color_filter` (ugrizy + VR + Y), `marker_map`
  (DECam instrument codes c4d/k4m/ksb/tu), `make_instrument` (parses the NSC
  exposure-id prefix), and `magstr` (an NSC BigQuery column list).
- Units: magnitudes throughout; `crossing_time` is in **hours** in the event
  population table (divided by 24 internally) but **days** everywhere else.
- ID conventions: `synth_objid` / `strip_objid` assume `"<id>_ml_<t0>_<tE>_<u0>"`.

### Proposed: a canonical frame + per-survey adapters

Define one internal "canonical lightcurve" the core operates on, and push every
survey-specific name/convention into an adapter that produces it. Concretely:

```python
# nscml/schema.py
@dataclass(frozen=True)
class LightcurveSchema:
    id: str          # object id column
    time: str        # MJD (days)
    band: str        # filter/band label
    value: str       # the detection signal column after normalization
    error: str       # 1-sigma error on value
    space: str = "mag"   # "mag" or "flux" (see section 2)

def normalize(df, schema, baseline="median") -> pd.DataFrame:
    """Rename to canonical columns (id, mjd, band, value, err) and compute the
    per-band baseline-subtracted signal the detector consumes."""
```

Then the core (`find_persistent_excursions`, `fit_excursions`, ...) reads only
the canonical names. Survey knowledge lives in small adapters:

```python
# nscml/surveys/nsc.py   -> from_nsc(df)     (mag_auto, <band>mag baselines, ...)
# nscml/surveys/lsst.py  -> from_lsst(df)    (psfFlux/psfFluxErr in nJy, band, ...)
```

Two implementation paths, smallest-diff first:

- **Minimal (no kernel changes):** a `normalize(df, schema)` shim at each
  pipeline entry point that renames the caller's columns to the canonical ones
  the code already uses and computes `deltamag`. This de-hardcodes ~90% of the
  coupling without touching the numba kernels. The `color_filter` / `marker_map`
  / `make_instrument` / `magstr` NSC-isms move into `surveys/nsc.py`.
- **Cleaner (medium):** make the kernels and cuts take a `schema` (or operate on
  the renamed canonical frame) end to end, and ship adapters per survey.

Recommendation: do the minimal shim + extract NSC-isms first (Phase 1 below);
it is low-risk and immediately lets someone point the tool at their own columns.

## 2. Magnitude vs flux -- the Rubin blocker

nscml is a magnitude-space method: the signal is a **negative** delta-magnitude
excursion, `amp_to_mag = -2.5 log10(A)`, and the PSPL model is fit in mags.

LSST is different in a way that matters:

- Photometry is reported as **flux in nanojansky (nJy)**, and forced fluxes
  **can be negative** (faint or difference-image sources). Rubin deliberately
  plots light curves in flux, not magnitude, because the mag conversion drops
  negative-flux epochs. [1][2]

So a naive "convert LSST flux to mag" adapter throws away exactly the faint
epochs that matter and is ill-defined near zero flux. Two options:

- **(a) Flux->mag adapter (lossy, quick):** keep only `flux > 0`, convert to AB
  mag, build `deltamag` against a per-band reference. Fine for bright,
  well-measured stars; wrong for anything near the noise floor. A stopgap.
- **(b) Flux-space detector (correct, scoped change):** microlensing multiplies
  the baseline flux, `F_obs = A(u) * F_base`, so the natural detection signal is
  a **positive** fractional flux excursion `delta_flux = F_obs - F_base`. The
  WMA/scatter kernels are signal-agnostic -- they already work on any
  (value, error, time) series. The only mag-specific pieces are: the sign of the
  excursion threshold, `amp_to_mag`, and the fit model `ml_f`. A flux model is
  just `F_base * microlensing_amplification(...)` -- and
  `microlensing_amplification` already returns the flux amplification `A`;
  `amp_to_mag` is the only conversion. So a `space="flux"` mode is a contained
  change: swap the model function and the excursion sign, reuse everything else.

Recommendation: add a `space` toggle (`"mag"` default for NSC, `"flux"` for
LSST), golden-tested. This is the single most important change for Rubin.

## 3. Running on Rubin LSST -- concrete recipe

- **Data access:** DP1 (released June 2025) is access-gated to US/Chilean
  institutions and named affiliates; the public, simulated **DP0.2** is the
  easiest place to prototype an adapter without credentials. (See brain memory
  M0000085 on Rubin access.) Plan the adapter against DP0.2, validate on DP1/DP2
  when access is available.
- **Tables / columns:** build per-object light curves from `ForcedSource` /
  `ForcedSourceOnDiaObject` (direct photometry) or `DiaSource` (difference
  photometry). Relevant fields: `objectId`, `band` (u/g/r/i/z/y),
  `psfFlux`/`psfFluxErr` in nJy, `coord_ra`/`coord_dec`. Time is
  `midpointMjdTai` for DiaSource; for ForcedSource the visit mid-time
  (`expMidptMJD`) is obtained by joining the Visit table. [1][2] (Confirm exact
  names against the DP1 schema at release time -- they shift slightly between
  data previews.)
- **Adapter sketch** (`from_lsst`): join ForcedSource to Visit for the time,
  map `objectId->id`, `band->band`, `expMidptMJD->mjd`, `psfFlux->value`,
  `psfFluxErr->err`, set `space="flux"`, and let `normalize` compute
  `delta_flux` against the per-band median (the baseline).
- **Cadence parameters:** `WS_INTERVAL_DAYS=50`, `WS_MAX_REVISIT_DAYS=10`, the
  detection `timescale`, and the crossing-time prior are tuned to NSC's cadence.
  LSST WFD has different revisit (~few days) and season gaps -- retune these
  (now that they are named constants, this is a config change, not a code hunt).
- **Event population:** `generate_synthetic_microlensing_events_from_population`
  takes a population table; supply an LSST-appropriate (tE, u0) distribution for
  the recovery/contamination yardstick.

## 4. General usability (any user, any dataset)

- **README** with install + quickstart + the data contract (done alongside this
  audit; see README.md).
- **Documented data contract:** exact required columns, dtypes, units, and sort
  order, in one place (the `LightcurveSchema` above formalizes it).
- **A high-level in-memory API:** today the user wires `search -> fit -> cuts`
  by hand over parquet files with pickle intermediates. A
  `detect(lightcurves_df, schema, **params) -> fit_df` that runs the whole
  pipeline in memory would make small datasets and notebook use trivial; keep
  the file pipeline for scale.
- **Separate library from NSC specifics:** move `magstr`, `make_instrument`,
  `color_filter`, `marker_map` into `nscml/surveys/nsc.py`. The core should not
  import DECam instrument codes.
- **Plotting is schema-coupled too:** `plot.py` indexes `color_filter[band]` and
  `marker_map[instrument]` and reads `<band>mag` baseline columns -- drive these
  from the schema/config so a new survey's bands render.
- **Tooling:** type hints + numpydoc docstrings on the public API; a CI workflow
  (GitHub Actions) running the suite on 3.11/3.12; `logging` instead of
  `print`/`tqdm` for library use (or a `progress=` flag, which some functions
  already have).
- **Robustness gaps already noted in AUDIT.md** (curve_fit u0 bound, the
  `'_ml_'` id convention) become more important once arbitrary survey ids flow
  through `strip_objid`/`split_real_synth_df`.

## 5. Suggested phased plan

- **Phase 1 (low-risk, high-value):** README + data-contract; extract NSC-isms
  into `surveys/nsc.py`; add `LightcurveSchema` + `normalize(df, schema)` at the
  pipeline entries. No kernel changes; goldens unaffected.
- **Phase 2 (Rubin):** `space="flux"` mode (flux PSPL model + excursion sign);
  `from_lsst` adapter; retune cadence/population params; validate on DP0.2.
- **Phase 3:** high-level `detect()` in-memory API; LSST example notebook; CI;
  docstrings/docs.

---

Sources for the LSST specifics:
- [1] DP1 -- Forced photometry tutorial: https://dp1.lsst.io/tutorials/notebook/105/notebook-105-3.html
- [2] DP1 -- ForcedSource table / variable-star light curves:
  https://dp1.lsst.io/tutorials/notebook/201/notebook-201-3.html ,
  https://dp1.lsst.io/tutorials/notebook/305/notebook-305-1.html
- Rubin nanojansky calibration note: https://community.lsst.org/t/photocalib-has-replaced-calib-welcoming-our-nanojansky-overlords/3648
