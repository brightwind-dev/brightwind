import pytest
import brightwind as bw
import pandas as pd

DATA = bw.load_csv(bw.demo_datasets.demo_data)
DATA = bw.apply_cleaning(DATA, bw.demo_datasets.demo_cleaning_file)
WSPD_COLS = ['Spd80mN', 'Spd80mS', 'Spd60mN', 'Spd60mS', 'Spd40mN', 'Spd40mS']
WDIR_COLS = ['Dir78mS', 'Dir58mS', 'Dir38mS']

TEST_WIND_MAP_EXTRACTION_GDF = 


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


def test_get_country_code_for_geometry(geom):