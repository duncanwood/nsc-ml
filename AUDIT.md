# nscml -- audit

## Verdict

The code is in good shape. It runs end to end on this machine, the science is
sound, and the detection pipeline recovers injected PSPL events with sensible
parameters (a fixture event injected at u0=0.05, tE=15d is recovered at
u0=0.046, tE=14.9d; a degenerate fit is correctly flagged by the
condition-number cut). The two `numba` weighted-moving-average kernels -- the
scientific core -- agree with an independent dense implementation to machine
epsilon, and `ks_weighted` reproduces `scipy.stats.ks_2samp`'s D statistic
exactly. Output is bit-for-bit reproducible across runs.

What is here is typical, healthy research code. The findings below are
almost all "would make this a cleaner reusable library," not "this is wrong."
The one genuine defect is a small cluster of unfinished `fit_stats*` stubs
that are not on any live path. Nothing was changed for any item in this file;
every item is a proposal.

## What was changed in this branch

Only deletions of dead/commented code, sparse research comments, and trailing-
whitespace fixes -- no logic, no renames, no restructuring. Specifically:

- `nsctools.py`: removed the commented-out duplicate `ml_jac`, disabled debug
  `print`s, the dead `cut_mask` / achromatic / revisit blocks, stale inline
  kwargs, and the IDE-placeholder docstring on `extend_lc`; added ~8 short
  comments at non-obvious spots (PSPL units, Jacobian column order, the
  sliding-window kernel, the numba typed-empty-list idiom, the excursion
  threshold and `np.split` run selection, the two-branch KS, the cond_num cut).
- `plot.py`: removed dead commented-out plotting calls only.
- `__deprecated.py`: removed (import-dead; see below).

A parity test suite captured from the pre-cleanup code stays green across
every cleanup commit, so these edits are verified behaviour-preserving.

The branch's first commit snapshots pre-existing uncommitted working-tree
changes to `plot.py` and the egg-info `SOURCES.txt` that were present before
this audit; they are not part of the cleanup.

## Findings (proposals, not applied)

### 1. Reproducibility / RNG -- inject a Generator (highest value)

Two functions draw random numbers from process-global state, and they do it
two different ways:

- `generate_synthetic_microlensing_events_from_population` uses
  `np.random.default_rng().choice(...)` **and** stdlib `random.choice(...)`.
  `np.random.default_rng()` is seeded from OS entropy and is **not** affected
  by `np.random.seed(...)`, so this path cannot be reproduced by seeding the
  legacy global RNG at all. (The test harness reproduces it only by
  monkeypatching `np.random.default_rng` to a seeded factory *and* calling
  `random.seed` -- see `tests/fixture_build.py:seeded_synth_rng`.)
- `fit_excursions` uses `np.random.normal(0, res_std, n_ks_gaussian)` for the
  small-sample KS reference. This *is* reproducible via `np.random.seed`.

Proposal: give both functions an `rng: np.random.Generator = None` parameter
(default `np.random.default_rng()`), and use `rng.choice` / `rng.normal`
throughout, replacing the stdlib `random.choice` with `rng.choice`. Then
reproducibility is a caller concern (`pass rng=np.random.default_rng(0)`), the
library carries no global-RNG dependence, and the mixed numpy/stdlib RNG
sources collapse to one. Seeding stays out of the library (the tests seed; the
library does not).

### 2. Star imports define no public API

`__init__.py` does `from .nsctools import *; from .plot import *`, re-exporting
~100 names including internal helpers, `color_filter`, and the SQL constant
`magstr`. `plot.py` and `nsctools.py` can bind the same name (both import
`numpy`, `pandas`); the second wildcard wins silently.

Proposal: declare `__all__` in each module (or re-export named symbols in
`__init__.py`) so the public surface is explicit and internal helpers stop
leaking. This is also what makes the package safe to `import *` from notebooks.

### 3. Magic numbers worth naming

Defaults are scattered and sometimes inconsistent between callers:

- `cond_lim = 1e5` (`cut_pcov`) -- the fit-degeneracy threshold.
- `timescale` -- the WMA smoothing scale in days: default `2` in the kernels
  but `5` in `find_persistent_excursions`.
- `nclip` -- window truncation in units of timescale: `10` in the sparse
  kernels, `5` in `clipped_gaussian_window`, `1` in
  `weighted_moving_average_sparse_gaussian`.
- detection gates: `z_threshold=3`, `n_measured=4`, `duration=5`.
- `n_ks_gaussian=10000`, `context_size=100`, `crossing_time_guess=40`.
- `curve_fit` bounds: `u0 in [0, 5]`, `tE in [1, 3650]`,
  `t0 in [mjd0-3650, mjdN+3650]` with `365*10` hardcoded (a 10-year window);
  `outliers_cutoff=3`, `outliers_cutoff_data=20`.
- `well_sampled_region`: `interval=50`, `maxrevisit=10`, `seqlen=5`.

Proposal: lift these into named module-level constants (or a small config
dataclass) with one-line justifications, and reconcile the `timescale`/`nclip`
defaults so the same physical scale is not spelled two ways.

### 4. Hardcoded / fragile output paths

- `search_files_for_excursions` defaults `outdir` to
  `'/'.join(lcfiles[0].split('/')[:-1]) + '/searches/'` -- a hardcoded
  `/searches/` suffix and POSIX-only path splitting.
- Path strings are built by concatenation, **inconsistently**:
  `consolidate_search_files_for_excursions` writes
  `metadata['outdir'] + metadata['outfile']` (no separator -- silently writes
  to the wrong place unless `outdir` ends in `/`), while `fit_excursions`
  writes `metadata['outdir'] + '/' + metadata['fitoutfile']`. The test suite
  has to pass a trailing-slash `outdir` to work around this.

Proposal: use `os.path.join` / `pathlib.Path` everywhere and take output
locations as explicit parameters rather than deriving them from the first
input file.

### 5. Unfinished `fit_stats*` stubs (the one real defect)

`fit_stats`, `fit_stats_df`, `fit_stats_split` (end of `nsctools.py`) are
incomplete and not called by any pipeline function:

- `fit_stats(full_fit_results)` recurses as `fit_stats(rfitdf)`, passing a
  DataFrame where a `(fitresults, fitfails, fitdups)` tuple is expected -- the
  unpack raises `ValueError` whenever both real and synthetic rows are present.
  The function also has no `return` (returns `None`) and appears truncated.
- `fit_stats_df` repeats `info.update({'n_excursions': df.shape[0]})` twice (a
  no-op copy-paste).
- `fit_stats_split` is a stub: `info.update({})`, returns `{}`.

Flagged, not touched (deleting/fixing live definitions is out of scope here).
Proposal: finish or delete them; if kept, add a test that exercises the
real+synth branch so the recursive bug cannot resurface.

### 6. Suspected numerical / robustness risks (flagged, not fixed)

- **Empty lightcurve**: `find_persistent_excursions` raises `IndexError`
  (`excursions[0]` with no empty guard). Short curves (< `seqlen`) return `[]`
  gracefully; only the truly-empty case is unguarded.
- **curve_fit `u0` lower bound = 0** permits degenerate fits that rail to
  `u0 ~ 0` with enormous covariance condition number (observed `cond_num ~
  1.4e7` on a fixture object). `cut_pcov` is what removes these downstream; a
  small positive lower bound would prevent the degeneracy at the source.
- **`split_real_synth_df`** classifies synthetic events with `'ml' in id`,
  which would misclassify any real id containing the substring "ml"; the
  `_ml_` token (as in `strip_objid`) is the safe test.
- **`consolidate_search_files_for_excursions`** prints and `return`s `None` on
  a metadata/param mismatch; callers unpack the result directly and would crash
  on `None`. Prefer raising.
- **`ml_jac`** assumes `blending_factor = 1` (the live 3-parameter fit); fine
  as used, but it does not cover a blended fit if blending is ever fit.

### 7. Near-duplicate functions (simplification)

- `weighted_moving_average_gaussian` duplicates `compute_weighted_moving_average`
  with the window hardcoded (the latter's `window_fn=gaussian_window` default
  already covers it).
- `get_well_sampled_objects` vs `get_just_well_sampled_objects` differ only by
  `observed=True` and an empty-region filter.
- `sparse_gaussian_window` vs `dense_sparse_gaussian_window` differ only by
  `.todense()`.

### 8. Packaging

`setup.py` uses `distutils` (deprecated; removed from the stdlib in Python
3.12) and declares `py_modules=['nsctools', 'nscml.plot']`, which does not
match the actual package (`nscml` with submodules). Proposal: replace with a
`pyproject.toml` (setuptools backend, automatic package discovery,
`python_requires=">=3.11"`, deps from `requirements.txt`).

## Dependencies and version sensitivities

Pinned in `requirements.txt`, validated against conda env `nsc` (CPython
3.11.9):

| package | version | role |
|---|---|---|
| numpy | 1.23.5 | arrays |
| pandas | 2.2.2 | DataFrames, parquet I/O |
| scipy | 1.11.4 | curve_fit, sparse, kstwo |
| numba | 0.59.1 | the WMA/WMS/PSPL kernels |
| pyarrow | 16.1.0 | parquet backend |
| tqdm | 4.66.2 | progress bars |
| matplotlib | 3.8.4 | plot.py only |

- **Python >= 3.11 is required**, not optional: `fit_excursions` calls
  `warnings.catch_warnings(action="ignore")`, and the `action=` keyword was
  added in 3.11. (Consistent with the committed cpython-311 `__pycache__`.)
- **numba <-> numpy** are ABI-coupled; numba 0.59 supports numpy < 1.27, so
  the pinned numpy 1.23.5 is safe. Bumping either without the other risks an
  import-time failure.
- **scipy** `curve_fit` (LM/TRF) and `kstwo.sf` are deterministic for fixed
  inputs but only guaranteed bit-identical on the same BLAS/platform; the fit
  parity tests allow a tight tolerance (1e-9) for cross-platform last-ULP
  drift, while the pure-numba kernels are asserted bit-exact.
- **scikit-learn** is intentionally absent: it was used only by the removed
  `__deprecated.py` KDE time-segmentation, not by the detection pipeline.
- I/O uses **pickle** for intermediate search/fit results -- convenient but
  version-fragile and unsafe on untrusted input; fine for a single-user
  research workflow.

## `__deprecated.py` removal

Verified import-dead before removal: not imported by `__init__.py`,
`nsctools.py`, or `plot.py`, and not imported by any notebook (the notebook
references are stale cProfile output that predates the refactor, locally
redefined `get_lc`, or `nsctools.segment_times(...)` calls that already fail
against the current module because those functions were moved out). It held
parked-but-real functionality -- KDE-based time segmentation, the
achromaticity discriminant (`is_achromatic` / `points_compatible`), and a
weighted-moving-average skewness. It is recoverable from git history; if any of
it is wanted as a reference artifact, `git revert` of the removal restores it.

## Reproducibility notes

- The pipeline is fully deterministic except the two RNG paths in finding 1.
  Capturing the goldens twice produced byte-identical kernels, KS results,
  curve_fit outputs, and (under the seeded-RNG context) synthetic event
  assignments.
- Goldens live in `tests/golden/`; `tests/capture_golden.py` regenerates them
  and the parity tests re-run it into a temp dir and diff against the committed
  copies. To intentionally accept a numerical change, re-run capture and review
  the golden diff.
- The integration fixture (`tests/fixtures/real_objects.parquet`, 4 real NSC
  objects) is committed so the suite needs neither the multi-GB `test.parquet`
  nor a network.

## Test coverage

64 tests: kernel parity (bit-exact) + properties (sparse==dense, PSPL physics,
analytic-vs-finite-difference Jacobian, ks_weighted vs scipy), cut boundary
semantics (incl. cond_lim=1e5), detector edge cases (empty/short/flat,
gating, restriction, band-agnosticism), the seeded small-sample KS branch, and
the full file pipeline end to end (search -> fit -> cuts, with the degenerate
fit dropped by cut_pcov). plot.py is not covered (no display backend).
