import importlib.util

import numpy as np
import pandas as pd
import pytest

import brightwind as bw
from brightwind.analyse import shear


@pytest.mark.parametrize('mode', [None, 'warn', 'raise'])
def test_shear_import_preserves_chained_assignment(mode):
    with pd.option_context('mode.chained_assignment', mode):
        module = importlib.util.module_from_spec(shear.__spec__)
        shear.__spec__.loader.exec_module(module)
        assert pd.options.mode.chained_assignment == mode


def _cleaning_input(tmp_path, windographer, sensor='Spd40m'):
    if windographer:
        cleaning = pd.DataFrame({
            'Data Column': [sensor],
            'Flag Name': ['Bad data'],
            'Start Time': ['2020-01-01 00:00'],
            'End Time': ['2020-01-01 02:00'],
        })
        path = tmp_path / 'cleaning.tsv'
        cleaning.to_csv(path, sep='\t', index=False)
        return bw.apply_cleaning_windographer, str(path)
    cleaning = pd.DataFrame({
        'Sensor': [sensor],
        'Start': [pd.Timestamp('2020-01-01 00:00')],
        'Stop': [pd.Timestamp('2020-01-01 02:00')],
    })
    return bw.apply_cleaning, cleaning


@pytest.mark.parametrize('mode', [None, 'warn', 'raise'])
@pytest.mark.parametrize('windographer', [False, True])
@pytest.mark.parametrize('inplace', [False, True])
def test_cleaning_preserves_chained_assignment(tmp_path, mode, windographer, inplace):
    data = pd.DataFrame({'Spd40m_avg': [4., 5., 6.], 'Spd60m_avg': [7., 8., 9.]},
                        index=pd.date_range('2020-01-01', periods=3, freq='h'))
    original = data.copy(deep=True)
    expected = data.copy(deep=True)
    expected.loc[expected.index[:2], 'Spd40m_avg'] = np.nan
    clean, cleaning = _cleaning_input(tmp_path, windographer)

    with pd.option_context('mode.chained_assignment', mode):
        result = clean(data, cleaning, inplace=inplace)
        assert pd.options.mode.chained_assignment == mode
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(data, expected if inplace else original)


@pytest.mark.parametrize('mode', [None, 'warn', 'raise'])
@pytest.mark.parametrize('windographer', [False, True])
def test_cleaning_restores_chained_assignment_after_error(tmp_path, mode, windographer):
    data = pd.DataFrame({'Spd40m_avg': [4., 5., 6.]},
                        index=pd.date_range('2020-01-01', periods=3, freq='h'))
    clean, cleaning = _cleaning_input(tmp_path, windographer, sensor=123)

    with pd.option_context('mode.chained_assignment', mode):
        with pytest.raises(TypeError, match='must be str, not'):
            clean(data, cleaning)
        assert pd.options.mode.chained_assignment == mode


@pytest.mark.parametrize('mode', [None, 'warn', 'raise'])
def test_cleaning_all_sensors_preserves_chained_assignment(tmp_path, mode):
    data = pd.DataFrame({'Spd40m_avg': [4., 5., 6.], 'Spd60m_avg': [7., 8., 9.]},
                        index=pd.date_range('2020-01-01', periods=3, freq='h'))
    expected = data.copy(deep=True)
    expected.iloc[:2, :] = np.nan
    clean, cleaning = _cleaning_input(tmp_path, False, sensor='All')
    with pd.option_context('mode.chained_assignment', mode):
        result = clean(data, cleaning)
        assert pd.options.mode.chained_assignment == mode
    pd.testing.assert_frame_equal(result, expected)
