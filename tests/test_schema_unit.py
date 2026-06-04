"""Unit tests for the survey-agnostic schema layer (swellar.schema)."""
import numpy as np
import pandas as pd
import pytest

import swellar
import fixture_build as fb


def test_schema_symbols_exposed():
    assert hasattr(swellar, 'LightcurveSchema')
    assert hasattr(swellar, 'normalize')
    assert hasattr(swellar, 'NSC_SCHEMA')
    assert hasattr(swellar, 'from_nsc')


def test_normalize_identity_for_nsc(real_path):
    df = pd.read_parquet(real_path)
    df['objectid'] = df['objectid'].astype(str)
    out = swellar.normalize(df)  # default NSC schema
    # canonical columns present and the detector signal unchanged
    for col in ('objectid', 'mjd', 'deltamag', 'magerr_auto', 'filter'):
        assert col in out.columns
    np.testing.assert_array_equal(out['deltamag'].to_numpy(), df['deltamag'].to_numpy())


def test_normalize_renames_custom_columns_preserves_detection():
    lc = fb.small_sample_lc()
    orig = swellar.find_persistent_excursions(lc)
    assert len(orig) >= 1  # the fixture has an injected event

    renamed = lc.rename(columns={'objectid': 'oid', 'mjd': 't',
                                 'magerr_auto': 'err', 'deltamag': 'sig'})
    schema = swellar.LightcurveSchema(id='oid', time='t', error='err',
                                    value='sig', band=None)
    norm = swellar.normalize(renamed, schema)
    new = swellar.find_persistent_excursions(norm)
    assert [list(map(int, r)) for r in orig] == [list(map(int, r)) for r in new]


def test_normalize_computes_deltamag_from_measurement():
    df = pd.DataFrame({
        'oid': ['a', 'a', 'a', 'b', 'b'],
        't': [1.0, 2.0, 3.0, 1.0, 2.0],
        'band': ['g', 'g', 'r', 'g', 'g'],
        'mag': [10.0, 12.0, 20.0, 5.0, 7.0],
        'err': [0.1] * 5,
    })
    schema = swellar.LightcurveSchema(id='oid', time='t', band='band',
                                    error='err', measurement='mag', value=None)
    out = swellar.normalize(df, schema)
    # baseline = per (objectid, band) median: a/g=11, a/r=20, b/g=6
    np.testing.assert_allclose(out['deltamag'].to_numpy(), [-1.0, 1.0, 0.0, -1.0, 1.0])
    for col in ('objectid', 'mjd', 'filter', 'magerr_auto', 'deltamag'):
        assert col in out.columns


def test_normalize_rejects_unknown_space():
    with pytest.raises(ValueError):
        swellar.normalize(pd.DataFrame({'mjd': [1.0]}),
                        swellar.LightcurveSchema(space='lumens', measurement='x'))


def test_normalize_requires_value_or_measurement():
    with pytest.raises(ValueError):
        swellar.normalize(pd.DataFrame({'mjd': [1.0]}),
                        swellar.LightcurveSchema(value=None, measurement=None))


def test_from_nsc_identity_matches_normalize():
    df = fb.small_sample_lc()
    a = swellar.from_nsc(df)
    b = swellar.normalize(df, swellar.NSC_SCHEMA)
    pd.testing.assert_frame_equal(a, b)


def test_surveys_nsc_adapter():
    from swellar.surveys import nsc
    assert nsc.make_instrument is swellar.make_instrument
    assert nsc.magstr == swellar.magstr
    assert nsc.color_filter is swellar.color_filter
    assert nsc.NSC_SCHEMA is swellar.NSC_SCHEMA
    assert callable(nsc.from_nsc)
