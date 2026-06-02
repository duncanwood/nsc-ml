"""Smoke test for examples/lsst_quickstart.py so the example can't silently rot.

Runs the example end-to-end (synthetic LSST ForcedSource -> detect via the LSST
schema) and asserts the injected event is recovered. Deterministic (the example
seeds default_rng(0))."""
import importlib.util
import os

import pytest

EXAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'examples', 'lsst_quickstart.py')


def _load_example():
    spec = importlib.util.spec_from_file_location('lsst_quickstart', EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lsst_quickstart_recovers_injected_event(capsys):
    mod = _load_example()
    fits = mod.main()                                  # runs the ForcedSource + DiaSource demo
    assert len(fits) >= 1
    row = fits.sort_values('pval').iloc[0]             # most significant candidate
    assert row['peak_time'] == pytest.approx(mod.T0, abs=5.0)
    assert row['crossing_time'] == pytest.approx(mod.TE, rel=0.3)
    assert 'recovered' in capsys.readouterr().out      # the report actually printed
