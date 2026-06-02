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

1. **`from_lsst` adapter** (`nscml/surveys/lsst.py`): map ForcedSource /
   DiaSource columns to a flux `LightcurveSchema`; **supply a positive template
   `F_ref`** (a difference-flux median is not valid — `normalize` guards
   `F_ref<=0`). Validate on the public, simulated **DP0.2** (DP1 is access-gated).
2. Multiplicative synthetic injection in flux (`add_microlensing_event` /
   `generate_synthetic` flux mode) for the recovery yardstick.
3. Cadence/event-population retune for LSST; a high-level in-memory
   `detect(df, schema)` API; CI.

## Things to know (the non-obvious bits)

- The detection's power is **achromatic cross-band pooling**, which holds in
  *magnitudes* and in *fractional flux* (`s = A-1`) but **not** raw flux excess.
  Use `space='flux'` (fractional), never raw `dF`.
- `ml_jac` is the **magnitude** Jacobian (carries `-2.5/ln10`); flux fits use
  `jac=None`. `microlensing_amplification` already returns the flux ratio.
- Read order: `MAP.md` (code map) -> `AUDIT.md` (findings/resolutions) ->
  `PORTABILITY.md` (generalization roadmap + the flux design).
