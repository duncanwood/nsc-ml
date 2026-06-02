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

## 2. Magnitude vs flux -- the Rubin blocker (DONE: both implemented)

nscml is a magnitude-space method: the detection signal is a **negative**
delta-magnitude excursion, `amp_to_mag = -2.5 log10(A)`, and the PSPL model is
fit in mags. LSST reports **flux in nanojansky (nJy)**, and forced/difference
fluxes **can be negative** at faint flux; Rubin plots light curves in flux, not
mag, precisely because the mag conversion drops negative-flux epochs. [1][2]

### Why this is subtle: the cross-band coherence is a magnitude-space property

The detector's power comes from pooling all bands into one delta series: PSPL
magnification `A(u(t))` is wavelength-independent, so in MAGNITUDES every band
shows the *same* dip and they reinforce. Concretely, unblended:

    mag:             dm        = -2.5 log10(A)     # same for every band  (achromatic)
    raw flux:        dF        = (A - 1) * F_base  # scales with per-band baseline (CHROMATIC)
    fractional flux: dF/F_base = A - 1             # same for every band  (achromatic again)

So pooling *raw* flux across bands is wrong (a bright band dominates) -- the
concern is real. But **fractional flux** (equivalently the flux ratio
`F_obs/F_base = A`) is achromatic again, and is the faithful flux-space twin of
delta-mag. Under blending, both `dm` and fractional flux pick up the *same* weak
chromaticity through the per-band blend fraction `b = F_source/F_base`
(`dm = -2.5 log10(A*b + 1 - b)`, `dF/F_base = (A-1)*b`), so the band pooling is
already an achromatic approximation in either space.

### The two real options

- **(A) Flux -> mag at the top (quick, lossy).** Convert `F -> AB mag` where
  `F > 0`, build delta-mag, and run the existing, fully-tested mag pipeline
  unchanged. Pros: zero core/kernel changes; reuses the validated pipeline and
  goldens; cross-band pooling intact. Cons: mag is undefined for `F <= 0`, so the
  faint and difference-image epochs (common in LSST) must be dropped or clipped
  -- losing exactly the low-SNR points and biasing the baseline; and flux->mag
  error is asymmetric and diverges as `F -> 0`, breaking the Gaussian-error
  assumption the WMA and KS rely on. Good for bright, high-SNR events; a fast
  first look.

- **(B) Fractional-flux throughout (correct, scoped change).** Detect on
  `s = F_obs/F_ref - 1` (per-band reference flux `F_ref`, e.g. the median): a
  *positive* excursion; model `s = A(u) - 1` (`microlensing_amplification`
  already returns `A`); errors `sigma_s = sigma_F / F_ref`, symmetric and finite
  even at `F = 0`. The numba WMA/scatter kernels are signal-agnostic and stay
  unchanged; the only mag-specific pieces are the excursion sign, `amp_to_mag`,
  and the model `ml_f`. Pros: correct for all flux incl. negatives; preserves the
  achromatic pooling; proper errors; the natural LSST space. Cons: real changes
  (a `space` toggle, a flux baseline, a flux model, the flipped sign) plus new
  goldens for the flux path, while keeping the mag/NSC path bit-identical. Good
  for the actual Rubin (faint/low-SNR) regime.

**Decision (implemented):** (B) as the real path, (A) as the comparison
baseline. After a thorough audit of every magnitude assumption in the codepath:
- (B) `normalize(space='flux')` builds `s = F/F_ref - 1` (error `sigma_F/F_ref`,
  `F_ref` a positive per-band reference flux; guarded against `F_ref<=0`);
  `find_persistent_excursions(space='flux')` flips to a one-sided positive-bump
  test; `fit_excursions(space='flux')` fits `ml_f_flux = microlensing_amplification - 1`
  with `jac=None` (the analytic `ml_jac` carries the magnitude `-2.5/ln10` chain
  factor -- *verified* wrong for flux). All WMA/scatter kernels, the KS ranking,
  the cuts, and outlier rejection are reused unchanged; `space='mag'` is the
  default everywhere, so the mag goldens are byte-identical.
- (A) `flux_to_mag(df, schema)` converts flux->AB mag (`sigma_mag = 1.0857
  sigma_F/F`), drops non-positive-flux epochs (lossy; warns), and runs the
  standard mag pipeline -- the comparison baseline.
On a synthetic event (u0=0.2, tE=40, t0=200) the two paths recover consistent
parameters (tE 38.2 vs 37.6, t0 200.0 vs 200.1) -- see `tests/test_flux_unit.py`.
Remaining gotcha for real data: `F_ref` must be a positive baseline (template)
flux, not a difference-image-flux median (the achromaticity assumption's failure
mode). Multiplicative synthetic injection in flux is implemented:
`add_microlensing_event(space='flux')` and
`generate_synthetic_microlensing_events_from_population(space='flux')` map the
observed flux `F -> F*A` (stored signal `s -> (s+1)*A - 1`) with
`sigma_s -> sigma_s*A` -- the flux twin of the magnitude path's additive
`-2.5 log10(A)` (which leaves `sigma_mag` unchanged).

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

- **Phase 1 (low-risk, high-value) -- DONE.** README + data contract;
  `LightcurveSchema` + `normalize(df, schema)` (`nscml/schema.py`); NSC adapter
  namespace (`nscml/surveys/nsc.py`) re-exporting the NSC-specific helpers. No
  kernel changes; goldens unaffected; 8 new tests. (Physical relocation of the
  NSC helpers out of `nsctools` is deferred -- notebooks still import them
  there.)
- **Phase 2 (flux/Rubin) -- DONE (detection core).** `space='flux'`
  fractional-flux detection + fit, and the `flux_to_mag` baseline (section 2);
  mag goldens byte-identical; 7 new tests. The `from_lsst` adapter
  (`nscml/surveys/lsst.py`: ForcedSource/DiaSource -> fractional flux, with a
  `template_flux_col` supplying a positive `F_ref` for difference flux) is built
  and synthetic-tested (5 tests). Multiplicative synthetic injection in flux
  (`add_microlensing_event`/`generate_synthetic` `space='flux'`: `F -> F*A`, i.e.
  `s -> (s+1)*A - 1` with `sigma_s -> sigma_s*A`) is built and tested (6 tests;
  the recovery yardstick round-trips a known event back through the detector).
  **Still TODO:** cadence/event-population retune for LSST, and end-to-end
  validation on the public DP0.2 (RSP access -- no local Rubin data on the dev
  machine).
- **Phase 3:** high-level `detect()` in-memory API; LSST example notebook; CI;
  docstrings/docs.

---

Sources for the LSST specifics:
- [1] DP1 -- Forced photometry tutorial: https://dp1.lsst.io/tutorials/notebook/105/notebook-105-3.html
- [2] DP1 -- ForcedSource table / variable-star light curves:
  https://dp1.lsst.io/tutorials/notebook/201/notebook-201-3.html ,
  https://dp1.lsst.io/tutorials/notebook/305/notebook-305-1.html
- Rubin nanojansky calibration note: https://community.lsst.org/t/photocalib-has-replaced-calib-welcoming-our-nanojansky-overlords/3648
