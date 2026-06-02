"""Unit tests for the pure helper functions in nsctools."""
import numpy as np
import pandas as pd
import pytest

import nscml


def test_strip_and_synth_objid_roundtrip():
    params = {'peak_time': 100.0, 'crossing_time': 40.0, 'impact_parameter': 0.1}
    sid = nscml.synth_objid('163053_2831', params)
    assert sid.startswith('163053_2831_ml_')
    assert nscml.strip_objid(sid) == '163053_2831'


def test_convert_to_range_index():
    out = nscml.convert_to_range_index(np.array([3, 4, 5, 6]))
    assert isinstance(out, pd.RangeIndex)
    assert list(out) == [3, 4, 5, 6]
    gapped = np.array([3, 4, 7])
    assert list(nscml.convert_to_range_index(gapped)) == [3, 4, 7]  # unchanged


def test_get_default_args_and_common_params():
    defaults = nscml.get_default_args(nscml.find_persistent_excursions)
    assert defaults['z_threshold'] == 3
    assert defaults['timescale'] == 5
    common = nscml.common_params(nscml.find_persistent_excursions,
                                 {'z_threshold': 9, 'bogus': 1})
    assert common == {'z_threshold': 9}


def test_default_args_of_functions_union():
    params = nscml.default_args_of_functions(
        [nscml.find_persistent_excursions, nscml.fit_excursions])
    assert 'z_threshold' in params and 'crossing_time_guess' in params


def test_make_instrument_prefixes():
    df = pd.DataFrame({'exposure': ['c4d_181108_x', 'k4m_1_y', 'tu1869989', 'ksb_2_z']})
    out = nscml.make_instrument(df)
    assert list(out['instrument']) == ['c4d', 'k4m', 'tu', 'ksb']


def test_extend_lc_window():
    t = np.array([0.0, 10.0, 50.0, 60.0, 200.0, 400.0])
    df = pd.DataFrame({'mjd': t}, index=range(len(t))).sort_values('mjd')
    region = [2, 3]  # mjd 50..60
    ext = nscml.extend_lc(df, region, context_size=100)
    # within [50-100, 60+100] = [-50, 160] -> indices 0,1,2,3
    assert set(ext) == {0, 1, 2, 3}


def test_float_cols_to_double():
    df = pd.DataFrame({'a': np.array([1.0, 2.0], dtype='float32'),
                       'b': np.array([1, 2], dtype='int64')})
    out = nscml.float_cols_to_double(df)
    assert out['a'].dtype == np.float64
    assert out['b'].dtype == np.int64  # untouched


def test_add_microlensing_event_darkens_and_relabels():
    n = 30
    t = np.linspace(0, 200, n)
    df = pd.DataFrame({'objectid': ['src'] * n, 'mjd': t,
                       'mag_auto': np.full(n, 18.0), 'deltamag': np.zeros(n)})
    out = nscml.add_microlensing_event(df, impact_parameter=0.05,
                                       crossing_time=20.0, peak_time=100.0)
    assert (out['objectid'] == nscml.synth_objid('src', {
        'peak_time': 100.0, 'crossing_time': 20.0, 'impact_parameter': 0.05})).all()
    assert (out['originalid'] == 'src').all()
    assert out['deltamag'].min() < -0.1          # brightening => negative delta-mag
    assert out['mag_auto'].idxmin() == np.argmin(np.abs(t - 100.0))  # peak at t0


def test_well_sampled_region_detects_dense_cluster():
    # a dense, evenly-sampled run spanning > interval(50) days, then a long gap
    dense = np.arange(0, 80, 4.0)            # 20 points, revisit 4 < maxrevisit 10
    sparse = np.array([500.0, 1000.0])
    t = np.concatenate([dense, sparse])
    df = pd.DataFrame({'mjd': t}, index=range(len(t)))
    regions = nscml.well_sampled_region(df, interval=50, maxrevisit=10, seqlen=5)
    assert len(regions) == 1
    assert len(regions[0]) == len(dense)


def test_well_sampled_region_rejects_short_run():
    t = np.array([0.0, 1.0, 2.0])  # too few, too short
    df = pd.DataFrame({'mjd': t}, index=range(3))
    assert nscml.well_sampled_region(df, interval=50, maxrevisit=10, seqlen=5) == []


def test_compute_file_map(working, real_path, tmp_path):
    objfilemap, fileenum = nscml.compute_file_map([working['mini_lc'], real_path])
    assert set(fileenum) == {0, 1}
    # every mapped object resolves to a file that exists
    assert all(idx in fileenum for idx in objfilemap.values())
    # a known real object is present in the map
    assert any(str(k).startswith('163053_2831') for k in objfilemap)


def test_reduce_and_nondetections():
    exc = {'a': [np.array([1, 2])], 'b': [], 'c': [np.array([3])]}
    assert set(nscml.reduce_excursions(exc)) == {'a', 'c'}
    assert nscml.get_nondetections(exc) == ['b']


def test_split_real_synth_df_uses_ml_token():
    df = pd.DataFrame({'objectid': ['163053_2831',
                                    '163053_2831_ml_57990.00_15.00_0.05000',
                                    'ml_galaxy_7']})  # contains 'ml' but not '_ml_'
    rdf, sdf = nscml.split_real_synth_df(df)
    assert list(sdf['objectid']) == ['163053_2831_ml_57990.00_15.00_0.05000']
    assert set(rdf['objectid']) == {'163053_2831', 'ml_galaxy_7'}  # not misclassified


def test_consolidate_raises_on_metadata_mismatch(tmp_path):
    import pickle
    f1, f2 = tmp_path / 'a.pkl', tmp_path / 'b.pkl'
    with open(f1, 'wb') as f:
        pickle.dump(({'outdir': str(tmp_path), 'tag': 1}, {}, {'o1': []}), f)
    with open(f2, 'wb') as f:
        pickle.dump(({'outdir': str(tmp_path), 'tag': 2}, {}, {'o2': []}), f)
    with pytest.raises(ValueError):
        nscml.consolidate_search_files_for_excursions([str(f1), str(f2)])
