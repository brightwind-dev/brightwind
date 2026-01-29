import pytest
import pandas as pd
import numpy as np

import brightwind as bw

DATA = bw.load_csv(bw.demo_datasets.demo_data)
DATA = bw.apply_cleaning(DATA, bw.demo_datasets.demo_cleaning_file)
WSPD_COLS = ['Spd80mN', 'Spd80mS', 'Spd60mN', 'Spd60mS', 'Spd40mN', 'Spd40mS']
WDIR_COLS = ['Dir78mS', 'Dir58mS', 'Dir38mS']


def test_slice_data():
    data_sliced = bw.utils.utils.slice_data(DATA, date_from='2016-11-23 00:30', date_to='2017-10-23 12:20')

    assert data_sliced.index[0] == pd.Timestamp('2016-11-23 00:30')
    assert data_sliced.index[-1] == pd.Timestamp('2017-10-23 12:10')

    data_sliced = bw.utils.utils.slice_data(DATA, date_from='2016-11-23', date_to='2017-10-23')

    assert data_sliced.index[0] == pd.Timestamp('2016-11-23 00:00')
    assert data_sliced.index[-1] == pd.Timestamp('2017-10-22 23:50')

    data_sliced = bw.utils.utils.slice_data(DATA, date_from='2016-11-23')
    assert data_sliced.index[-1] == DATA.index[-1]

    data_sliced = bw.utils.utils.slice_data(DATA, date_to='2017-10-23')
    assert data_sliced.index[0] == DATA.index[0]


def test_assert_function_variable_type():
    # Correct type
    bw.utils.utils.assert_function_variable_type(5, int, 'var1')
    bw.utils.utils.assert_function_variable_type(5.0, (int, float), 'var2')
    bw.utils.utils.assert_function_variable_type('test', str, 'var3')
    bw.utils.utils.assert_function_variable_type([1, 2, 3], list, 'var4')
    bw.utils.utils.assert_function_variable_type((1, 2), tuple, 'var5')
    bw.utils.utils.assert_function_variable_type({'a': 1}, dict, 'var6')

    # Incorrect type
    with pytest.raises(TypeError):
        bw.utils.utils.assert_function_variable_type(5, str, 'var1')

    with pytest.raises(TypeError):
        bw.utils.utils.assert_function_variable_type(5.0, int, 'var2')

    with pytest.raises(TypeError):
        bw.utils.utils.assert_function_variable_type('test', list, 'var3')

    with pytest.raises(TypeError):
        bw.utils.utils.assert_function_variable_type([1, 2, 3], dict, 'var4')

    with pytest.raises(TypeError):
        bw.utils.utils.assert_function_variable_type((1, 2), str, 'var5')

    with pytest.raises(TypeError):
        bw.utils.utils.assert_function_variable_type({'a': 1}, list, 'var6')


