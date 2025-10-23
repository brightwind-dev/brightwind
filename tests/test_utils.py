import pytest
import pandas as pd
import numpy as np

import brightwind as bw
from brightwind.utils import utils

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

def test_apply_scale_factor():
    assert utils.apply_scale_factor(3, 0.5) == 1.5
    assert (utils.apply_scale_factor(np.array([0, 1, 2]), 0.5) == [0, 0.5, 1]).all()
    assert (utils.apply_scale_factor(pd.Series([10, 20, 30, 40]), -10) == [-100, -200, -300, -400]).all()
    df = pd.DataFrame({'a':[0.5, 1.2], 'b':[3, 4], 'c':['a', 'b']})
    result_df = pd.DataFrame({'a':[1.0, 2.4], 'b':[6, 8], 'c':['a', 'b']})
    assert result_df.equals(bw.utils.utils.apply_scale_factor(df, 2))
