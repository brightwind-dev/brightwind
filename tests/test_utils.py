import pytest
import brightwind as bw
import pandas as pd
import numpy as np

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


def test_linear_transform():
    # test with float and int inputs
    assert bw.utils.utils.linear_transform(x_target=10, x_ref=5, y_ref=10, slope=-0.5) == 7.5
    
    # test with array input for y_ref
    assert (bw.utils.utils.linear_transform(
        x_target=20,
        x_ref=2,
        y_ref=DATA.T2m.loc['2016-01-09 17:10':'2016-01-09 18:00'].values,
        slope=-0.0065) == np.array([0.837, 0.746, 0.614, 0.735, 0.654, 0.796])).all()

    # test with pandas Series input for y_ref
    pd.testing.assert_series_equal(bw.utils.utils.linear_transform(
        x_target=DATA.T2m.loc['2016-01-09 17:10':'2016-01-09 18:00'],
        x_ref=2,
        y_ref=20,
        slope=-0.0065).round(3),
        pd.Series(data = [20.007, 20.007, 20.008, 20.007, 20.008, 20.007],
                  index = pd.to_datetime(['2016-01-09 17:10:00', '2016-01-09 17:20:00', '2016-01-09 17:30:00',
                                          '2016-01-09 17:40:00', '2016-01-09 17:50:00', '2016-01-09 18:00:00'])),
                  check_names=False)
