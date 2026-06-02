# nsc-ml

[![tests](https://github.com/duncanwood/nsc-ml/actions/workflows/tests.yml/badge.svg)](https://github.com/duncanwood/nsc-ml/actions/workflows/tests.yml)

Multi-band time-series outlier detection for finding gravitational
**microlensing** events in survey photometry. The detector flags persistent,
achromatic brightenings ("excursions") in a weighted moving average of each
star's delta-magnitude light curve, fits a point-source point-lens (PSPL)
model to each candidate, and ranks candidates with a weighted two-sample
Kolmogorov-Smirnov test against the out-of-event photometry. Synthetic events
injected from a population model provide the recovery/contamination yardstick
for tuning the selection cuts.

Developed for the NOIRLab Source Catalog (NSC) as part of a physics PhD. The
scientific core is a pair of `numba`-compiled windowed weighted-moving-average
kernels; the rest is a parquet-backed batch pipeline.

> Status: research code. It is currently coupled to the NSC schema and works in
> magnitudes. See [PORTABILITY.md](PORTABILITY.md) for what it takes to run on
> other surveys (e.g. Rubin LSST).

## Install

Requires **Python 3.11** (uses `warnings.catch_warnings(action=...)`).
Validated against the pinned versions in `requirements.txt` (numpy 1.23.5,
pandas 2.2.2, scipy 1.11.4, numba 0.59.1, pyarrow 16.1.0, tqdm, matplotlib).

```bash
pip install -r requirements.txt
# editable install; compat mode is needed because the distribution root and the
# package share the name "nscml" (see nscml/pyproject.toml):
pip install -e nscml --config-settings editable_mode=compat
```

## Quickstart

Detection runs on a single object's light curve. It reads three columns:

| column | meaning |
|---|---|
| `mjd` | observation time (days) |
| `deltamag` | magnitude relative to the per-band baseline; an event is a **negative** dip |
| `magerr_auto` | 1-sigma magnitude error per epoch |

```python
import numpy as np, pandas as pd, nscml

rng = np.random.default_rng(0)
n, t0, tE, u0 = 80, 100.0, 20.0, 0.1
t = np.sort(rng.uniform(0, 200, n))

# inject a PSPL event: amplification A(u) -> magnitude dip
dip = nscml.amp_to_mag(nscml.microlensing_amplification(t, u0, tE, t0))
lc = pd.DataFrame({
    'mjd': t,
    'deltamag': rng.normal(0, 0.01, n) + dip,
    'magerr_auto': np.full(n, 0.02),
})

regions = nscml.find_persistent_excursions(lc, z_threshold=3, timescale=5)
print(len(regions), "excursion region(s):", [list(r) for r in regions])
```

## Full pipeline (many objects, many files)

For a survey-scale run the work is read from parquet (one or more files,
grouped by `objectid`) and proceeds search -> fit -> cuts:

```python
# excursion search + PSPL fit over a list of parquet files:
exc_results, fit_results = nscml.search_files_for_microlensing_events(
    lcfiles, ws_regions, metadata, params)            # ws_regions: per-object search windows

fitdf = nscml.make_fit_excursions_df(fit_results[0])  # tidy results table
kept = nscml.cut_pcov(nscml.cut_by_pval(fitdf, 0.05)) # apply selection cuts
```

`fit_excursions` and `generate_synthetic_microlensing_events_from_population`
take an `rng=` argument (a `numpy.random.Generator`); pass a seeded one for
reproducible results. See [MAP.md](MAP.md) for the full public API, the kernels,
the cuts, and the file-pipeline functions.

## Flux surveys (e.g. Rubin LSST)

The detector defaults to magnitudes (NSC). For flux data (LSST forced
photometry is nanojansky and can be negative) use the fractional-flux mode,
which keeps the achromatic cross-band pooling that makes the method work. The
high-level `detect()` runs the whole pipeline in memory straight from a flux
schema:

```python
from nscml.surveys.lsst import from_lsst, LSST_FORCEDSOURCE_SCHEMA, LSST_DIASOURCE_SCHEMA

# ForcedSource (direct flux): one call -- the schema carries space='flux'
events = nscml.detect(forcedsource_df, schema=LSST_FORCEDSOURCE_SCHEMA)

# DiaSource (difference flux): fold in a positive template F_ref first, then detect
canonical = from_lsst(diasource_df, LSST_DIASOURCE_SCHEMA, template_flux_col='template')
```

Under the hood that is `normalize` (`s = F/F_ref - 1`, an achromatic *positive*
bump) -> `find_persistent_excursions(space='flux')` -> `fit_excursions(space='flux')`;
call those directly for finer control. `nscml.flux_to_mag(flux_df, schema)` is the
alternative (lossy) flux->mag ingest, kept as a comparison baseline.

See **[examples/lsst_quickstart.py](examples/lsst_quickstart.py)** for a runnable
end-to-end demo (synthetic LSST data, no data access needed) and
[PORTABILITY.md](PORTABILITY.md) for the design and the achromaticity argument.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest                # unit, golden parity, integration
```

The suite captures golden outputs from the committed code and asserts the
kernels and seeded paths reproduce them exactly (and the float pipeline within a
tight tolerance). `tests/fixtures/real_objects.parquet` (four real NSC objects)
keeps the suite self-contained -- no large data file or network needed.

## Repository map

| file | what |
|---|---|
| [MAP.md](MAP.md) | code map: public API, numba kernels, cuts, file pipeline |
| [AUDIT.md](AUDIT.md) | code-quality audit + resolution of the implemented fixes |
| [PORTABILITY.md](PORTABILITY.md) | roadmap to other surveys / Rubin LSST |
| `nscml/nscml/nsctools.py` | detection core + file pipeline |
| `nscml/nscml/plot.py` | matplotlib diagnostics |
| `tests/` | unit + parity + integration + golden fixtures |

## Reproducing the original (dissertation) behavior

The git tag **`original-behavior`** marks the last commit bit-identical to the
pre-refactor pipeline. A single later change (injecting a `numpy` Generator)
alters only the small-sample KS reference; everything else is unchanged. See the
tag annotation (`git show original-behavior`) for details.

## Use

Research code; no license set yet -- please ask before reuse.
