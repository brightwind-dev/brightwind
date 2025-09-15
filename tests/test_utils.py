import pytest
import brightwind as bw
import pandas as pd

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

def test_vertically_extrapolate_at_constant_rate():
    assert bw.utils.utils.vertically_extrapolate_at_constant_rate(variable_reference_value=10, reference_height=5, target_height=10, lapse_rate=-0.5) == 7.5
    pd.testing.assert_series_equal(bw.utils.utils.vertically_extrapolate_at_constant_rate(DATA.T2m.loc['2016-01-09 17:10':'2016-01-09 18:00'], reference_height = 2, target_height = 20, lapse_rate = -0.0065),
        pd.Series(data = [0.837, 0.746, 0.614, 0.735, 0.654, 0.796], 
        index = pd.to_datetime(['2016-01-09 17:10:00', '2016-01-09 17:20:00', '2016-01-09 17:30:00', '2016-01-09 17:40:00',
                                '2016-01-09 17:50:00', '2016-01-09 18:00:00'])), check_names=False)