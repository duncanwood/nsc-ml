# nscml -- code map

A multi-band time-series outlier-detection method for finding gravitational
microlensing events in survey photometry (NOIRLab Source Catalog, NSC). The
detector flags persistent achromatic brightenings ("excursions") in a
weighted moving average of each star's delta-magnitude lightcurve, fits a
point-source point-lens (PSPL) model to each candidate, and ranks candidates
with a weighted two-sample Kolmogorov-Smirnov test against the out-of-event
photometry. Synthetic events injected from a population model provide the
recovery/contamination yardstick for tuning the selection cuts.

The scientific core is a pair of `numba`-compiled windowed weighted moving
average / scatter kernels; the rest is a parquet-backed batch pipeline.

## Package layout

```
nscml/
  pyproject.toml           build config (setuptools; requires-python >=3.11)
  nscml/
    __init__.py            from .nsctools import *; from .plot import *
    nsctools.py            detection core + file pipeline (this map)
    plot.py                matplotlib diagnostics (lightcurves, fits, p-value hists)
requirements.txt           pinned runtime deps (Python 3.11)
tests/                     unit + parity + integration + golden fixtures
```

`__init__.py` re-exports `nsctools` and `plot` via `import *`, bounded by each
module's `__all__` (so the imported numpy/pandas/etc. do not leak into the
package namespace). `__deprecated.py` (parked KDE segmentation, achromaticity
test, weighted skewness) was import-dead and has been removed; it survives in
git history.

## Data model

A lightcurve is a `pandas.DataFrame`, one row per epoch, columns:

| column | meaning |
|---|---|
| `objectid` | star id (categorical); synthetic ids are `"<id>_ml_<t0>_<tE>_<u0>"` |
| `mjd` | epoch (days) |
| `mag_auto`, `magerr_auto` | instrumental magnitude and error |
| `deltamag` | magnitude relative to the per-band baseline (the detection signal) |
| `filter`, `instrument`, `exposure` | band / camera / exposure id (categorical) |

Detection runs on `deltamag` vs `mjd` only and is band-agnostic; a brightening
appears as a NEGATIVE `deltamag` excursion. Files are parquet; intermediate
search/fit results are pickled.

In flux mode (`space='flux'`, for surveys like Rubin LSST) the same canonical
columns instead carry the achromatic **fractional flux** `s = F/F_ref - 1` and
its error, and an event is a POSITIVE excursion (`nscml.normalize(df,
schema_flux)` builds it; pass `space='flux'` to the detector/fit). See
PORTABILITY.md.

## Numba kernels (`@njit`)

The scientific core. All operate on plain float64 arrays with `mjd` sorted
ascending.

| kernel | returns | notes |
|---|---|---|
| `microlensing_amplification(t, u0, tE, t0, blend=1)` | A(t) | PSPL/Paczynski magnification; `u` = source-lens separation in Einstein radii |
| `amp_to_mag(amp)` | -2.5 log10(amp) | magnitude offset of an amplification |
| `ml_f(*x)` | mag | mag model fit function = `amp_to_mag(microlensing_amplification(*x))` |
| `ml_f_flux(*x)` | flux | flux model = `microlensing_amplification(*x) - 1` (fractional flux; used when `space='flux'`) |
| `ml_jac(t, u0, tE, t0)` | (n,3) | analytic Jacobian in `curve_fit` p0 order |
| `sparse_gaussian_wma(y, t, w, timescale, nclip)` | (wma, err, scatter) | **windowed weighted moving average**; sliding window truncated at `nclip*timescale`, ~O(n*window) |
| `sparse_gaussian_wms(y, t, w, wma, ...)` | scatter | **windowed weighted moving scatter** (called by the above) |
| `sparse_gaussian_window_iter(t, ...)` | COO ingredients | builds the sparse window matrix (consumed by the non-njit `sparse_gaussian_window`) |
| `compute_weighted_moving_average(y, t, e, window_fn, timescale)` | (wma, err, scatter) | dense O(n^2) reference; `window_fn` must itself be njit |
| `weighted_moving_average_gaussian(y, t, e, timescale)` | (wma, err, scatter) | thin alias of `compute_weighted_moving_average` with the gaussian window |
| `weighted_moving_average_err / _scatter` | array | the error / scatter pieces of the dense form |
| `gaussian_window(dt, timescale)` / `clipped_gaussian_window(dt, ts, nclip)` | weight | the window itself (clipped truncates beyond `nclip*timescale`) |
| `weighted_avg_and_std(values, weights)` | (avg, std) | weight-normalised |

Dispatch/wrappers (not njit): `weighted_moving_average(y, t, e, sparse=True, **kw)`
picks the sparse or dense backend; `weighted_moving_average_df(lc, **kw)` pulls
`deltamag/magerr_auto/mjd` out of a DataFrame. `weighted_moving_average_sparse_gaussian`
and `sparse_gaussian_window` use `scipy.sparse` and therefore cannot be njit.

The sparse and dense kernels are the same estimator when `nclip` is large
enough that no pair is dropped; `tests/test_kernels_unit.py` asserts they
agree to round-off.

## Detection pipeline (file level)

Data flow, single object -> batch of files:

1. `well_sampled_region(df, interval, maxrevisit, seqlen)` -> the dense,
   long-baseline time windows worth searching. `get_well_sampled_objects` /
   `get_just_well_sampled_objects` map this over a file's objects to build the
   `ws_regions` / search-domain dict.
2. `find_persistent_excursions(df, z_threshold, timescale, n_measured,
   duration, restrict_to_indices, usescatter, temper_errors, ...)` -> list of
   index runs where `wma / sqrt(std^2 + errs^2) < -z_threshold`, gated by
   minimum point count (`n_measured`) and time span (`duration`), optionally
   restricted to a search domain. **The core detector.**
3. `search_files_for_excursions(lcfiles, search_domains, metadata, params)` ->
   reads each parquet, runs the detector per object (synthetic ids mapped to
   their source domain via the `_ml_` regex), writes per-file pickles, and
   consolidates them (`consolidate_search_files_for_excursions`).
4. `fit_excursions(excursions, lcfiles, metadata, params, ...)` -> for each
   excursion, fits the PSPL model with `scipy.optimize.curve_fit` (analytic
   `ml_jac`, bounded, `x_scale`d) over an extended window (`extend_lc`,
   `context_size` days of padding), then scores the fit residuals with
   `ks_weighted` -- a two-sample test against the out-of-event photometry, or
   (too few outside points) against a synthetic Gaussian of the same weighted
   scatter, drawn from the injected `rng` (default: a fresh Generator; pass a
   seeded one for reproducibility). Returns `(fitresults, fitfails, fitdups)`.
5. `make_fit_excursions_df(fitresults)` -> tidy DataFrame: `objectid, excnum,
   pval, n_fit, n_out, cond_num, impact_parameter, crossing_time, peak_time,
   two_sample`.
6. Selection cuts (below) on that DataFrame.

`search_files_for_microlensing_events(lcfiles, ws_regions, metadata, params)`
runs steps 3-4 as one call and validates the parameter names against the two
functions' defaults.

Synthetic-event generation (the recovery yardstick):
`generate_synthetic_microlensing_events_from_population(lcfiles, events_file,
ws_regions, outdir, outname, rng=None)` injects events drawn from a population
table (`crossing_time` in hours, `umin`) into copies of real objects via
`add_microlensing_event` and writes synthetic parquet files. `rng` controls the
event/region draws (pass a seeded Generator for reproducibility).

## Selection cuts

All take the fit DataFrame and return a filtered copy.

| cut | keeps |
|---|---|
| `cut_by_npoints(df, n)` | `n_fit + n_out >= n` |
| `cut_by_pval(df, p)` | `pval >= p` |
| `cut_pcov(df, cond_lim=1e5)` | `cond_num < cond_lim` -- drops degenerate/under-constrained fits |
| `cut_crossing_time(df, timemin=1, timemax=None)` | `timemin < crossing_time (< timemax)` |
| `cut_high_points_low_p`, `cut_high_points_inout_low_p` | high-p OR low-count escape hatches |

`split_real_synth_df` separates injected (`_ml_`) from real objects for
recovery/contamination accounting.

## Other helpers

`make_delta_mags` / `make_delta_mags_mono` (baseline subtraction),
`make_instrument` (camera from exposure id), `compute_file_map`
(objectid -> file index), `get_lc`-style parquet filtering, `strip_objid` /
`synth_objid` (id <-> event params), `float_cols_to_double`, `magstr` (a SQL
column-list constant used by the data-pull notebooks). `reduce_excursions` /
`get_nondetections` summarise a search result dict.

## plot.py

Matplotlib diagnostics: `plot_lc`, `plot_deltamags`,
`plot_weighted_moving_average_df`, `plot_excursion_region`,
`plot_example_fits` (overlays the PSPL fit + reports cond_num/p-value),
`plot_pval_hist` / `plot_hist_color`, and cut-comparison helpers
(`compare_cut*`). Not exercised by the tests (no display backend).

## Running it

Python 3.11 with the pinned `requirements.txt` (validated against conda env
`nsc`, CPython 3.11.9). Build config is `nscml/pyproject.toml`; install editable
with `pip install -e nscml --config-settings editable_mode=compat` (the nested
`nscml/nscml` layout needs develop-style path resolution). Tests:
`pip install -r requirements-dev.txt && python -m pytest`.
