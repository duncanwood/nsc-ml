# Handoff

State as of 2026-06-02. Active branch: **`refactor/nscml-proposals`** (off
`cleanup/nscml-audit`, off `main`). Not merged. `main` is untouched.

## Setup & run

Python 3.11 (the env this was validated in is conda env `nsc`, CPython 3.11.9).

```bash
pip install -r requirements-dev.txt
pip install -e nscml --config-settings editable_mode=compat   # nested nscml/nscml needs compat
python -m pytest                                              # 83 tests, must stay green
```

The test suite is the safety net. `tests/golden/` holds outputs captured from
the original code; the parity tests assert they still reproduce. **Keep parity
green** — `space='mag'` is the default everywhere precisely so the original
(NSC, magnitude) behavior is byte-identical. Only one golden ever changed on
this branch (an intentional RNG re-capture, `pipeline_small.json`).

## What's done

- **Audit + safe cleanup** (`AUDIT.md`): dead code removed, sparse comments,
  bugs fixed (empty-LC guard, consolidate raise, `_ml_` id check, `context_size`
  wired through), magic numbers named, `os.path.join` paths, `__all__`, RNG
  injected (`rng=`), `pyproject.toml`. Tag **`original-behavior`** marks the last
  commit bit-identical to the pre-refactor pipeline.
- **Phase 1 portability** (`PORTABILITY.md` §1): `LightcurveSchema` +
  `normalize(df, schema)` (`nscml/schema.py`) decouple the detector from
  hardcoded column names; `nscml/surveys/nsc.py` is the NSC adapter.
- **Phase 2 flux** (`PORTABILITY.md` §2): `space='flux'` fractional-flux
  detection (`s = F/F_ref - 1`) + `flux_to_mag` baseline. Both recover a
  synthetic event consistently (`tests/test_flux_unit.py`).

## What's next (see PORTABILITY.md §5)

1. ~~**`from_lsst` adapter** (`nscml/surveys/lsst.py`)~~ — **DONE** (on `main`):
   ForcedSource/DiaSource -> flux `LightcurveSchema`, with `template_flux_col`
   supplying a positive `F_ref` for difference flux; 5 synthetic tests. DP0.2
   validation still pending (no local Rubin data / RSP access).
2. ~~Multiplicative synthetic injection in flux~~ — **DONE** (`feat/flux-injection`):
   `add_microlensing_event(space='flux')` and `generate_synthetic_..._from_population(space='flux')`
   map `F -> F*A` (`s -> (s+1)*A - 1`, `sigma_s -> sigma_s*A`); 6 tests incl. the
   recovery yardstick. mag goldens byte-identical (94 tests green).
3. Mostly done. **`detect(df, schema)` API, a runnable LSST example, and CI —
   DONE.** `nscml.detect` (normalize → find → fit → df, no temp files;
   `restrict_well_sampled=True` mirrors the NSC search; 5 tests).
   `examples/lsst_quickstart.py` (synthetic LSST → `from_lsst`/`detect`,
   smoke-tested). GitHub Actions on **macos-latest/arm64** — the goldens' capture
   arch; the bit-exact numba-kernel tests need the matching platform, so don't
   move CI to ubuntu/x86 without tolerance-comparing those. Full suite 100 green,
   mag goldens byte-identical. Remaining: (1) cadence/event-population retune —
   **deferred by decision** (keep the NSC defaults; they were chosen Rubin-aware,
   justification is future research); (2) DP0.2/DP1 end-to-end validation on the
   RSP — recipe in **RUNNING_ON_RSP.md** (Butler/TAP access; stack is Python 3.12
   so nscml installs via `pip install --user`; the numpy 1.x→2.x jump is the open
   risk -- run the suite on the RSP first).

## Things to know (the non-obvious bits)

- The detection's power is **achromatic cross-band pooling**, which holds in
  *magnitudes* and in *fractional flux* (`s = A-1`) but **not** raw flux excess.
  Use `space='flux'` (fractional), never raw `dF`.
- `ml_jac` is the **magnitude** Jacobian (carries `-2.5/ln10`); flux fits use
  `jac=None`. `microlensing_amplification` already returns the flux ratio.
- Read order: `MAP.md` (code map) -> `AUDIT.md` (findings/resolutions) ->
  `PORTABILITY.md` (generalization roadmap + the flux design).
