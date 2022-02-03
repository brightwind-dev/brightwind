import pytest
import brightwind as bw
import pandas as pd
import numpy as np
import datetime


wndspd = 8
wndspd_df = pd.DataFrame([2, 13, np.NaN, 5, 8])
wndspd_series = pd.Series([2, 13, np.NaN, 5, 8])
current_slope = 0.045
current_offset = 0.235
new_slope = 0.046
new_offset = 0.236
wndspd_adj = 8.173555555555556
wndspd_adj_df = pd.DataFrame([2.0402222222222224, 13.284666666666668, np.NaN, 5.106888888888888, 8.173555555555556])
wndspd_adj_series = pd.Series([2.0402222222222224, 13.284666666666668, np.NaN, 5.106888888888888, 8.173555555555556])
ref_date = pd.to_datetime('2000-01-01')

DATA = bw.load_campbell_scientific(bw.demo_datasets.demo_campbell_scientific_data)
DATA_CLND = bw.apply_cleaning(DATA, bw.demo_datasets.demo_cleaning_file)
STATION = bw.MeasurementStation(bw.demo_datasets.demo_wra_data_model)
DATA_ADJUSTED = bw.load_csv(bw.demo_datasets.demo_data_adjusted_for_testing)
WSPD_COLS = ['Spd80mN', 'Spd80mS', 'Spd60mN', 'Spd60mS', 'Spd40mN', 'Spd40mS']
WDIR_COLS = ['Dir78mS', 'Dir58mS', 'Dir38mS']
MERRA2 = bw.load_csv(bw.demo_datasets.demo_merra2_NE)


def np_array_equal(a, b):
    # nan's don't compare so use this instead
    try:
        np.testing.assert_equal(a, b)
    except AssertionError:
        return False
    return True


def test_selective_avg():
    date_today = datetime.datetime(2019, 6, 1)
    days = pd.date_range(date_today, date_today + datetime.timedelta(24), freq='D')
    data = pd.DataFrame({'DTM': days})
    data = data.set_index('DTM')
    data['Spd1'] = [1, np.NaN, 1, 1, 1, 1, 1, 1, 1, np.NaN, 1, 1, 1, 1, np.NaN, 1, 1, np.NaN, 1, 1, 1, 1, np.NaN, 1, 1]
    data['Spd2'] = [2, 2, np.NaN, 2, 2, 2, 2, 2, np.NaN, 2, 2, 2, 2, np.NaN, 2, 2, 2, np.NaN, 2, 2, 2, 2, 2, np.NaN, 2]
    data['Dir'] = [0, 15, 30, 45, np.NaN, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300,
                   315, np.NaN, 345, 360]

    # Test Case 1: Neither boom is near 0-360 crossover
    result = np.array([1.5, 2, 1, 1.5, 1.5, 1.5, 1.5, 2, 1, 2, 2, 2, 1.5, 1, 2, 1.5, 1.5, np.NaN,
                       1.5, 1, 1, 1, 2, 1, 1.5])
    bw.selective_avg(data[['Spd1']], data[['Spd2']], data[['Dir']],
                     boom_dir_1=315, boom_dir_2=135, sector_width=60)
    sel_avg = np.array(bw.selective_avg(data.Spd1, data.Spd2, data.Dir,
                                        boom_dir_1=315, boom_dir_2=135, sector_width=60))
    assert np_array_equal(sel_avg, result)

    # Test Case 2: Boom 1 is near 0-360 crossover
    result = np.array([1.0, 2.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.0, 2.0, 1.5, 1.5, 2.0, 1.0, 2.0, 2.0, 1.5, np.NaN,
                       1.5, 1.5, 1.5, 1.5, 2.0, 1.0, 1.0])
    sel_avg = np.array(bw.selective_avg(data.Spd1, data.Spd2, data.Dir,
                                        boom_dir_1=20, boom_dir_2=200, sector_width=60))
    assert np_array_equal(sel_avg, result)

    # Test Case 3: Boom 2 is near 0-360 crossover
    result = np.array([2.0, 2.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.5, 1.5, np.NaN,
                       1.5, 1.5, 1.5, 1.5, 2.0, 1.0, 2.0])
    sel_avg = np.array(bw.selective_avg(data.Spd1, data.Spd2, data.Dir,
                                        boom_dir_1=175, boom_dir_2=355, sector_width=60))
    assert np_array_equal(sel_avg, result)

    # Test Case 4: Booms at 90 deg to each other
    result = np.array([1.0, 2.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.0, 1.0, 2.0, 1.5, 1.5, 1.5, 1.0, 2.0, 1.5, 1.5, np.NaN,
                       1.5, 1.5, 1.5, 1.5, 2.0, 1.0, 1.0])
    sel_avg = np.array(bw.selective_avg(data.Spd1, data.Spd2, data.Dir,
                                        boom_dir_1=270, boom_dir_2=180, sector_width=60))
    assert np_array_equal(sel_avg, result)

    # Test Case 5: Sectors overlap error msg
    with pytest.raises(ValueError) as except_info:
        bw.selective_avg(data.Spd1, data.Spd2, data.Dir, boom_dir_1=180, boom_dir_2=185, sector_width=60)
    assert str(except_info.value) == "Sectors overlap! Please check your inputs or reduce the size of " \
                                     "your 'sector_width'."


def test_adjust_slope_offset_single_value():
    assert wndspd_adj == bw.adjust_slope_offset(wndspd, current_slope, current_offset, new_slope, new_offset)


def test_adjust_slope_offset_df():
    assert wndspd_adj_df.equals(bw.adjust_slope_offset(wndspd_df, current_slope, current_offset, new_slope, new_offset))


def test_adjust_slope_offset_series():
    assert wndspd_adj_series.equals(bw.adjust_slope_offset(wndspd_series, current_slope,
                                                           current_offset, new_slope, new_offset))


def test_adjust_slope_offset_arg_str():
    # check error msg if a string is sent as one of the slope or offset arguments
    with pytest.raises(TypeError) as except_info:
        bw.adjust_slope_offset(wndspd, current_slope, current_offset, '0.046', new_offset)
    assert str(except_info.value) == "argument '0.046' is not of data type number"


def test_adjust_slope_offset_arg_wndspd_str():
    # check error msg if a string is sent as the wind speed argument
    with pytest.raises(TypeError) as except_info:
        bw.adjust_slope_offset('8', current_slope, current_offset, new_slope, new_offset)
    assert str(except_info.value) == "wspd argument is not of data type number"


def test_adjust_slope_offset_arg_wndspd_list():
    # check error msg if a list is sent as the wind speed argument
    with pytest.raises(TypeError) as except_info:
        bw.adjust_slope_offset([2, 3, 4, 5], current_slope, current_offset, new_slope, new_offset)
    assert str(except_info.value) == "wspd argument is not of data type number"


def test_adjust_slope_offset_arg_wndspd_df_str():
    # check error msg if a string is an element in the pandas DataFrame
    with pytest.raises(TypeError) as except_info:
        bw.adjust_slope_offset(pd.DataFrame([2, 3, '4', 5]), current_slope, current_offset, new_slope, new_offset)
    assert str(except_info.value) == "some values in the DataFrame are not of data type number"


def test_adjust_slope_offset_arg_wndspd_series_str():
    # check error msg if a string is an element in the pandas DataFrame
    with pytest.raises(TypeError) as except_info:
        bw.adjust_slope_offset(pd.Series([2, 3, '4', 5]), current_slope, current_offset, new_slope, new_offset)
    assert str(except_info.value) == "some values in the Series are not of data type number"


def test_apply_wspd_slope_offset_adj():
    data = bw.apply_wspd_slope_offset_adj(DATA, STATION.measurements)
    assert((DATA_ADJUSTED[WSPD_COLS].fillna(0).round(5) ==
            data[WSPD_COLS].fillna(0).round(5)).all()).all()

    wspd_cols_that_work = ['Spd80mN', 'Spd80mS', 'Spd60mN', 'Spd40mN']
    data2 = bw.apply_wspd_slope_offset_adj(DATA[wspd_cols_that_work], STATION.measurements)
    assert((DATA_ADJUSTED[wspd_cols_that_work].fillna(0).round(5) ==
            data2[wspd_cols_that_work].fillna(0).round(5)).all()).all()

    data1 = bw.apply_wspd_slope_offset_adj(DATA['Spd60mS'], STATION.measurements['Spd60mS'])
    assert (data1.fillna(0).round(10) ==
            DATA_ADJUSTED['Spd60mS'].fillna(0).round(10)).all()


def test_offset_wind_direction_float():
    wdir_offset = float(5)
    assert wdir_offset == bw.offset_wind_direction(float(20), 345)


def test_offset_wind_direction_df():
    wdir_df_offset = pd.DataFrame([355, 15, np.NaN, 25, 335])
    assert wdir_df_offset.equals(bw.offset_wind_direction(pd.DataFrame([10, 30, np.NaN, 40, 350]), 345))


def test_offset_wind_direction_series():
    wdir_series_offset = pd.Series([355, 15, np.NaN, 25, 335])
    assert wdir_series_offset.equals(bw.offset_wind_direction(pd.Series([10, 30, np.NaN, 40, 350]), 345))


def test_apply_wind_vane_dead_band_offset():
    data1 = bw.apply_wind_vane_deadband_offset(DATA['Dir78mS'], STATION.measurements)
    data = bw.apply_wind_vane_deadband_offset(DATA, STATION.measurements)

    assert((DATA_ADJUSTED[WDIR_COLS].fillna(0).round(10) ==
            data[WDIR_COLS].fillna(0).round(10)).all()).all()

    assert (data1.fillna(0).round(10) ==
            data['Dir78mS'].fillna(0).round(10)).all()


def test_freq_str_to_dateoffset():
    # Excluding monthly periods and above as it will depend on which month or year
    periods = ['1S', '1min', '5min', '10min', '15min',
               '1H', '3H', '6H', '1D', '7D',
               '1W', '2W', '1MS', '1M', '3M', '6MS',
               '1AS', '1A', '3A']
    results = [1.0, 60.0, 300.0, 600.0, 900.0,
               3600.0, 10800.0, 21600.0, 86400.0, 604800.0,
               604800.0, 1209600.0, 2678400.0, 2678400.0, 7862400.0, 15724800.0,
               31622400.0, 31622400.0, 94694400.0]

    for idx, period in enumerate(periods):
        if type(bw.transform.transform._freq_str_to_dateoffset(period)) == pd.DateOffset:
            # The data frequency is returned as a DateOffset. The time delta can be in seconds
            # can be derived adding the date offset to an actual date.
            assert (ref_date + bw.transform.transform._freq_str_to_dateoffset(period) - ref_date
                    ).total_seconds() == results[idx]

    # Check that data frequency is returned as a DateOffset.
    assert type(bw.transform.transform._freq_str_to_dateoffset(period)) == pd.DateOffset


def test_round_timestamp_down_to_averaging_prd():
    timestamp = pd.Timestamp('2016-01-09 11:21:11')
    avg_periods = ['10min', '15min', '1H', '3H', '6H', '1D', '7D', '1W', '1MS', '1AS']
    avg_period_start_timestamps = ['2016-1-9 11:20:00', '2016-1-9 11:15:00', '2016-1-9 11:00:00',
                                   '2016-1-9 9:00:00', '2016-1-9 6:00:00', '2016-1-9', '2016-1-9',  '2016-1-9',
                                   '2016-1', '2016']
    for idx, avg_period in enumerate(avg_periods):
        assert avg_period_start_timestamps[idx] == \
               bw.transform.transform._round_timestamp_down_to_averaging_prd(timestamp, avg_period)


def test_get_data_resolution():
    # Dateoffset is used to represent data resolution
    import warnings
    series1 = DATA['Spd80mS'].index
    assert bw.transform.transform._get_data_resolution(series1).kwds == {'minutes': 10}

    series2 = pd.date_range('2010-01-01', periods=150, freq='H')
    assert bw.transform.transform._get_data_resolution(series2).kwds == {'hours': 1}

    series1 = bw.average_data_by_period(DATA['Spd80mN'], period='1M', coverage_threshold=0, return_coverage=False)
    assert bw.transform.transform._get_data_resolution(series1.index).kwds == {'months': 1}

    series1 = bw.average_data_by_period(DATA['Spd80mN'], period='1AS', coverage_threshold=0, return_coverage=False)
    assert bw.transform.transform._get_data_resolution(series1.index).kwds == {'years': 1}

    # hourly series with one instance where difference between adjacent timestamps is 10 min
    series3 = pd.date_range('2010-04-15', '2010-05-01', freq='H').union(pd.date_range('2010-05-01 00:10:00', periods=20,
                                                                                      freq='H'))
    with warnings.catch_warnings(record=True) as w:
        assert bw.transform.transform._get_data_resolution(series3).kwds == {'hours': 1}
        assert len(w) == 1


def test_offset_timestamps():
    series1 = DATA['2016-01-10 00:00:00':]

    # sending index with no start end
    bw.offset_timestamps(series1.index, offset='90min')

    # sending index with start end
    op = bw.offset_timestamps(series1.index, offset='2min', date_from='2016-01-10 00:10:00')
    assert op[0] == pd.to_datetime('2016-01-10 00:00:00')
    assert op[1] == pd.to_datetime('2016-01-10 00:12:00')

    op = bw.offset_timestamps(series1.index, '2min', date_to='2016-01-10 00:30:00')
    assert op[3] == pd.to_datetime('2016-01-10 00:32:00')
    assert op[4] == pd.to_datetime('2016-01-10 00:40:00')

    op = bw.offset_timestamps(series1.index, '3min', date_from='2016-01-10 00:10:00', date_to='2016-01-10 00:30:00')
    assert op[0] == pd.to_datetime('2016-01-10 00:00:00')
    assert op[1] == pd.to_datetime('2016-01-10 00:13:00')
    assert op[5] == pd.to_datetime('2016-01-10 00:50:00')

    op = bw.offset_timestamps(series1.index, '10min', date_from='2016-01-10 00:10:00', date_to='2016-01-10 00:30:00')
    assert op[0] == series1.index[0]
    assert op[1] == series1.index[2]

    # sending DataFrame with datetime index
    op = bw.offset_timestamps(series1, offset='-10min', date_from='2016-01-10 00:20:00')
    assert (op.iloc[1] == series1.iloc[1]).all()
    assert len(op) + 1 == len(series1)
    assert (op.loc['2016-01-10 00:40:00'] == series1.loc['2016-01-10 00:50:00']).all()

    op = bw.offset_timestamps(series1, offset='-10min', date_from='2016-01-10 00:20:00', overwrite=True)
    assert (op.loc['2016-01-10 00:10:00'] == series1.loc['2016-01-10 00:20:00']).all()

    op = bw.offset_timestamps(series1, '10min', date_from='2016-01-10 00:10:00', date_to='2016-01-10 00:30:00')
    assert (op.loc['2016-01-10 00:20:00'] == series1.loc['2016-01-10 00:10:00']).all()
    assert (op.loc['2016-01-10 00:40:00'] == series1.loc['2016-01-10 00:40:00']).all()
    assert len(op) + 1 == len(series1)

    op = bw.offset_timestamps(series1, '10min', date_from='2016-01-10 00:10:00', date_to='2016-01-10 00:30:00',
                              overwrite=True)
    assert (op.loc['2016-01-10 00:40:00'] == series1.loc['2016-01-10 00:30:00']).all()
    assert len(op) + 1 == len(series1)

    # sending Series with datetime index
    op = bw.offset_timestamps(series1.Spd60mN, offset='-10min', date_from='2016-01-10 00:20:00')
    assert (op.iloc[1] == series1.Spd60mN.iloc[1]).all()
    assert len(op) + 1 == len(series1.Spd60mN)
    assert (op.loc['2016-01-10 00:40:00'] == series1.Spd60mN.loc['2016-01-10 00:50:00']).all()

    op = bw.offset_timestamps(series1.Spd60mN, offset='-10min', date_from='2016-01-10 00:20:00', overwrite=True)
    assert (op.loc['2016-01-10 00:10:00'] == series1.Spd60mN.loc['2016-01-10 00:20:00']).all()

    op = bw.offset_timestamps(series1.Spd60mN, '10min', date_from='2016-01-10 00:10:00', date_to='2016-01-10 00:30:00')
    assert (op.loc['2016-01-10 00:20:00'] == series1.Spd60mN.loc['2016-01-10 00:10:00']).all()
    assert (op.loc['2016-01-10 00:40:00'] == series1.Spd60mN.loc['2016-01-10 00:40:00']).all()
    assert len(op) + 1 == len(series1.Spd60mN)

    op = bw.offset_timestamps(series1.Spd60mN, '10min', date_from='2016-01-10 00:10:00', date_to='2016-01-10 00:30:00',
                              overwrite=True)
    assert (op.loc['2016-01-10 00:40:00'] == series1.Spd60mN.loc['2016-01-10 00:30:00']).all()
    assert len(op) + 1 == len(series1.Spd60mN)


def test_average_wdirs():
    wdirs = np.array([350, 10])
    assert bw.average_wdirs(wdirs) == 0.0

    wdirs = np.array([0, 180])
    assert bw.average_wdirs(wdirs) is np.NaN

    wdirs = np.array([90, 270])
    assert bw.average_wdirs(wdirs) is np.NaN

    wdirs = np.array([45, 135])
    assert bw.average_wdirs(wdirs) == 90

    wdirs = np.array([135, 225])
    assert bw.average_wdirs(wdirs) == 180

    wdirs = np.array([45, 315, 225, 135])
    assert bw.average_wdirs(wdirs) is np.NaN

    wdirs = np.array([225, 315])
    assert bw.average_wdirs(wdirs) == 270

    wdirs = np.array([0, 10, 20, 340, 350, 360])
    assert bw.average_wdirs(wdirs) == 0.0

    wdirs_with_nan = [15, np.nan, 25]
    assert round(bw.average_wdirs(wdirs_with_nan), 3) == 20.000

    wspds = [3, 4, 5]
    assert round(bw.average_wdirs(wdirs_with_nan, wspds), 3) == 21.253

    wspds_with_nan = [3, 4, np.nan]
    assert round(bw.average_wdirs(wdirs_with_nan, wspds_with_nan), 3) == 15.0

    wspds_with_nan = [np.nan, np.nan, np.nan]
    assert bw.average_wdirs(wdirs_with_nan, wspds_with_nan) is np.NaN

    wspds_with_nan = [3, 4, np.nan]
    assert round(bw.average_wdirs(pd.Series(wdirs_with_nan), pd.Series(wspds_with_nan)), 3) == 15.0

    wdirs_with_nan = np.array(wdirs_with_nan)
    assert round(bw.average_wdirs(wdirs_with_nan), 3) == 20.000

    wdirs_with_nan = pd.Series(wdirs_with_nan)
    assert round(bw.average_wdirs(wdirs_with_nan), 3) == 20.000

    wdirs_series = pd.Series(wdirs)
    assert bw.average_wdirs(wdirs_series) == 0.0

    wspds = np.array([5, 5, 5, 5, 5, 5])
    assert bw.average_wdirs(wdirs, wspds) == 0.0

    wspds_series = pd.Series(wspds)
    assert bw.average_wdirs(wdirs_series, wspds_series) == 0.0

    wspds = np.array([5, 8.5, 10, 10, 6, 5])
    assert round(bw.average_wdirs(wdirs, wspds), 4) == 0.5774

    wdirs = np.array([[350, 10],
                      [0, 180],
                      [90, 270],
                      [45, 135],
                      [135, 225],
                      [15, np.nan]])
    wdirs_df = pd.DataFrame(wdirs)
    avg_wdirs = np.round(bw.average_wdirs(wdirs_df).values, 3)
    avg_wdirs = np.array([x for x in avg_wdirs if x == x])  # remove nans
    expected_result = [0., np.nan, np.nan, 90., 180., 15.]
    expected_result = np.array([x for x in expected_result if x == x])  # remove nans
    for i, j in zip(avg_wdirs, expected_result):
        assert i == j

    wspds = np.array([[1, 2],
                      [1, 2],
                      [1, 2],
                      [1, 2],
                      [1, 2],
                      [np.nan, 2]])
    wspds_df = pd.DataFrame(wspds)
    avg_wdirs = np.round(bw.average_wdirs(wdirs_df, wspds_df).values, 2)
    avg_wdirs = np.array([x for x in avg_wdirs if x == x])  # remove nans
    expected_result = np.array([3.36, 180., 270., 108.43, 198.43])
    for i, j in zip(avg_wdirs, expected_result):
        assert i == j


def dummy_data_frame(start_date='2016-01-01T00:00:00', end_date='2016-12-31T11:59:59'):
    """
    Returns a DataFrame with wind speed equal to the month of the year, i.e. In January, wind speed = 1 m/s.
    For use in testing.

    :param start_date: Start date Timestamp, i.e. first index in the DataFrame
    :type start_date:  Timestamp as a string in the form YYYY-MM-DDTHH:MM:SS'
    :param end_date: End date Timestamp, i.e. last index in the DataFrame
    :type end_date: Timestamp as a string in the form YYYY-MM-DDTHH:MM:SS'
    :return: pandas.DataFrame
    """

    date_times = {'Timestamp': pd.date_range(start_date, end_date, freq='10T')}

    dummy_wind_speeds = []
    dummy_wdirs = []

    for i, vals in enumerate(date_times['Timestamp']):
        # get list of each month for each date entry as dummy windspeeds
        dummy_wind_speeds.append(vals.month)
        dummy_wdirs.append((vals.month - 1) * 30)

    dummy_wind_speeds_df = pd.DataFrame({'wspd': dummy_wind_speeds, 'wdir': dummy_wdirs}, index=date_times['Timestamp'])
    dummy_wind_speeds_df.index.name = 'Timestamp'

    return dummy_wind_speeds_df


def test_average_data_by_period():
    bw.average_data_by_period(DATA[['Spd80mN']], period='1H')
    # hourly averages
    bw.average_data_by_period(DATA.Spd80mN, period='1H')
    # hourly average with coverage filtering
    bw.average_data_by_period(DATA.Spd80mN, period='1H', coverage_threshold=0.9)
    bw.average_data_by_period(DATA.Spd80mN, period='1H', coverage_threshold=1)
    # return coverage with filtering
    bw.average_data_by_period(DATA.Spd80mN, period='1H', coverage_threshold=0.9,
                              return_coverage=True)
    # return coverage without filtering
    bw.average_data_by_period(DATA.Spd80mN, period='1H', return_coverage=True)

    # monthly averages
    bw.average_data_by_period(DATA.Spd80mN, period='1M')
    # hourly average with coverage filtering
    bw.average_data_by_period(DATA.Spd80mN, period='1M', coverage_threshold=0.9)
    bw.average_data_by_period(DATA.Spd80mN, period='1M', coverage_threshold=1)
    # return coverage with filtering
    bw.average_data_by_period(DATA.Spd80mN, period='1M', coverage_threshold=0.9, return_coverage=True)
    # return coverage without filtering
    bw.average_data_by_period(DATA.Spd80mN, period='1M', return_coverage=True)

    # weekly averages
    bw.average_data_by_period(DATA.Spd80mN, period='2W')
    # hourly average with coverage filtering
    bw.average_data_by_period(DATA.Spd80mN, period='2W', coverage_threshold=0.9)
    bw.average_data_by_period(DATA.Spd80mN, period='2W', coverage_threshold=1)
    # return coverage with filtering
    bw.average_data_by_period(DATA.Spd80mN, period='2W', coverage_threshold=0.9, return_coverage=True)
    # return coverage without filtering
    bw.average_data_by_period(DATA.Spd80mN, period='2W', return_coverage=True)
    # Test error msg when period is less than data resolution
    with pytest.raises(ValueError) as except_info:
        bw.average_data_by_period(DATA.Spd80mN, period='5min')
    assert str(except_info.value) == "The time period specified is less than the temporal resolution of the data. " \
                                     "For example, hourly data should not be averaged to 10 minute data."

    dummy_data = dummy_data_frame()
    average_monthly_speed = bw.average_data_by_period(dummy_data.wspd, period='1M')
    # test average wind speed for each month
    for i in range(0, 11):
        assert average_monthly_speed.iloc[i] == i + 1
    # test average wind speed and direction for each month
    average_monthly_speed = bw.average_data_by_period(dummy_data, period='1M', wdir_column_names='wdir')
    for i in range(0, 11):
        assert average_monthly_speed.iloc[i].wspd == i + 1
        assert round(average_monthly_speed.iloc[i].wdir, 0) == i * 30
    # test when only 1 wdir column is sent
    average_monthly_speed = bw.average_data_by_period(dummy_data['wdir'], period='1M', wdir_column_names='wdir')
    for i in range(0, 11):
        assert round(average_monthly_speed.iloc[i], 0) == i * 30
    # test when the data doesn't actually contain the wdir column name sent
    with pytest.raises(KeyError) as except_info:
        bw.average_data_by_period(dummy_data['wspd'], period='1M', wdir_column_names='wdir')
    assert str(except_info.value) == '"\'wdir\' not in data sent."'
    with pytest.raises(KeyError) as except_info:
        bw.average_data_by_period(dummy_data, period='1M', wdir_column_names='wdirXXXXX')
    assert str(except_info.value) == '"\'wdirXXXXX\' not in data sent."'
    # test when wdir_column_names sent but aggregation_method is not mean
    with pytest.raises(KeyError) as except_info:
        bw.average_data_by_period(dummy_data['wdir'], period='1M', wdir_column_names='wdir', aggregation_method='sum')
    assert str(except_info.value) == '"Vector averaging is only applied when \'aggregation_method\' is \'mean\'. ' \
                                     'Either set \'wdir_column_names\' to None or set \'aggregation_method\'=\'mean\'"'
    # test average wind speed and direction for each month with 99% coverage required
    average_monthly_speed = bw.average_data_by_period(dummy_data, period='1M', wdir_column_names='wdir',
                                                      coverage_threshold=0.99, return_coverage=True)
    assert average_monthly_speed[0].count().wspd == 11  # the returned wspd has only 11 months
    assert average_monthly_speed[1].count().wspd_Coverage == 12  # the returned coverage has 12 months

    # test average annual wind speed
    average_annual_speed = bw.average_data_by_period(dummy_data.wspd, period='1AS')
    assert round(average_annual_speed.iloc[0].item(), 3) == 6.506
    # average DATA to monthly
    data_monthly = bw.average_data_by_period(DATA_CLND[WSPD_COLS + WDIR_COLS], period='1MS',
                                             wdir_column_names=WDIR_COLS,
                                             aggregation_method='mean', coverage_threshold=0.95, return_coverage=True)
    count_months = [('Spd80mN', 19),
                    ('Spd80mS', 17),
                    ('Spd60mN', 19),
                    ('Spd60mS', 19),
                    ('Spd40mN', 19),
                    ('Spd40mS', 19),
                    ('Dir78mS', 16),
                    ('Dir58mS', 8),
                    ('Dir38mS', 19)]
    idx = 0
    for col_name, val in data_monthly[0].count().iteritems():
        assert col_name == count_months[idx][0]
        assert val == count_months[idx][1]
        idx += 1
    count_cov_months = [('Spd80mN_Coverage', 23),
                        ('Spd80mS_Coverage', 23),
                        ('Spd60mN_Coverage', 23),
                        ('Spd60mS_Coverage', 23),
                        ('Spd40mN_Coverage', 23),
                        ('Spd40mS_Coverage', 23),
                        ('Dir78mS_Coverage', 23),
                        ('Dir58mS_Coverage', 23),
                        ('Dir38mS_Coverage', 23)]
    idx = 0
    for col_name, val in data_monthly[1].count().iteritems():
        assert col_name == count_cov_months[idx][0]
        assert val == count_cov_months[idx][1]
        idx += 1

    # test when very low coverage timeseries is used
    data_test = DATA[['Spd80mN', 'Spd80mS', 'Dir78mS']][:'2016-03-31']
    data_test.reset_index(inplace=True)
    drop_indices = np.random.choice(data_test.index, 11800, replace=False)
    data_test = data_test.drop(drop_indices)
    data_test.set_index('Timestamp', inplace=True)
    data_test.sort_index(inplace=True)
    data_monthly, coverage_monthly = bw.average_data_by_period(data_test, period='1M', wdir_column_names='Dir78mS',
                                                               return_coverage=True,
                                                               data_resolution=pd.DateOffset(minutes=10))
    table_count = data_test.resample('1MS', axis=0, closed='left', label='left', base=0,
                                     convention='start', kind='timestamp').count()
    assert (table_count['Dir78mS']['2016-01-01'] / (31 * 24 * 6) - coverage_monthly['Dir78mS_Coverage']['2016-01-01']
            ) < 1e-5
    assert (table_count['Spd80mN']['2016-01-01'] / (31 * 24 * 6) - coverage_monthly['Spd80mN_Coverage']['2016-01-01']
            ) < 1e-5
    # input data_resolution
    data1 = DATA[:'2016-01-10'].copy()
    data1.reset_index(inplace=True)
    drop_indices = np.array([60, 36, 140, 16, 101, 40, 158, 122, 151, 34, 117, 159, 26,
                             169, 132, 124, 98, 141, 127, 100, 115, 119, 59, 17, 166, 61,
                             10, 106, 57, 13, 187, 174, 28, 63, 85, 130, 23, 148, 0,
                             145, 8, 149, 185, 170, 73, 9, 79, 65, 136, 6, 54, 172,
                             108, 29, 107, 102, 123, 168, 89, 182, 173, 167, 125, 33, 114,
                             113, 84, 41, 110, 30, 179, 43, 134, 142, 171, 155, 25, 135,
                             163, 92, 183, 49, 104, 46, 68, 116, 53, 87, 184, 146, 153,
                             1, 77, 164, 161, 165, 94, 4, 58, 103, 19, 2, 48, 88,
                             152, 96, 82, 55, 32, 42, 15, 51, 70, 3, 147, 78, 86,
                             69, 131, 144, 181, 45, 31, 175, 97, 21, 143, 186, 137, 120,
                             176, 80, 156, 14, 105, 47, 67, 22, 12, 128, 71, 5, 139,
                             81, 154, 62, 121, 27, 39, 91, 75, 112, 66, 93, 38, 44,
                             177, 99, 11, 76, 64, 35, 56, 109, 83, 90, 162, 50, 180,
                             52, 72, 129, 111, 157, 150, 20, 24, 118, 178, 160])
    data1 = data1.drop(drop_indices)
    data1 = data1.set_index('Timestamp')
    assert (bw.average_data_by_period(data1.Spd80mS, period='10min', data_resolution=pd.DateOffset(minutes=10)).dropna()
            == data1.Spd80mS).all()
    with pytest.raises(ValueError) as except_info:
        bw.average_data_by_period(data1.Spd80mS, period='10min')
    assert str(except_info.value) == "The time period specified is less than the temporal resolution of the data. " \
                                     "For example, hourly data should not be averaged to 10 minute data."
    with pytest.raises(TypeError) as except_info:
        bw.average_data_by_period(data1.Spd80mS, period='10min', data_resolution=pd.Timedelta('10min'))
    assert str(except_info.value) == "Input data_resolution is Timedelta. A Pandas DateOffset should be used instead."


def test_merge_datasets_by_period():
    mrgd_data = bw.merge_datasets_by_period(DATA_CLND['Spd80mN'], MERRA2['WS50m_m/s'], period='1MS',
                                            wdir_column_names_1=None, wdir_column_names_2=None,
                                            coverage_threshold_1=None, coverage_threshold_2=None,
                                            aggregation_method_1='mean', aggregation_method_2='mean')
    spd80mn_monthly_mean_list = [9.25346307, 8.90438194, 6.43050216, 6.59887454, 8.72965727,
                                 5.10815648, 6.96853427, 7.09395587, 8.18052477, 6.66944556,
                                 6.74182714, 8.90077755, 7.83337582, 9.13450868, 7.48893795,
                                 7.78338958, 6.49058893, 8.52524884, 6.78224843, 6.7158853,
                                 7.08256829, 9.47901579, 7.35934137]
    # data_monthly_index_list = ['2016-01-01', '2016-02-01', '2016-03-01', '2016-04-01',
    #                            '2016-05-01', '2016-06-01', '2016-07-01', '2016-08-01',
    #                            '2016-09-01', '2016-10-01', '2016-11-01', '2016-12-01',
    #                            '2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01',
    #                            '2017-05-01', '2017-06-01', '2017-07-01', '2017-08-01',
    #                            '2017-09-01', '2017-10-01', '2017-11-01']
    spd80mn_monthly_cov_list = [0.71886201, 1., 0.98454301, 1., 0.36536738, 1., 1., 1., 1., 1., 0.93472222, 1.,
                                0.9858871, 1., 1., 1., 1., 1., 1., 1., 1., 0.99283154, 0.74861111]
    m2_monthly_mean_list = [9.62391129, 9.01344253, 6.85649462, 6.66197639, 6.99338038,
                            5.29984306, 6.73991667, 7.11679032, 8.39015556, 6.83381317,
                            6.84408889, 9.0631707, 8.28869355, 9.2853869, 7.62800806,
                            7.73957917, 6.63575403, 7.81355417]

    assert len(mrgd_data) == 18
    for idx, row in enumerate(mrgd_data.iterrows()):
        assert round(spd80mn_monthly_mean_list[idx], 5) == round(row[1]['Spd80mN'], 5)
        assert round(spd80mn_monthly_cov_list[idx], 5) == round(row[1]['Spd80mN_Coverage'], 5)
        assert round(m2_monthly_mean_list[idx], 5) == round(row[1]['WS50m_m/s'], 5)

    mrgd_data = bw.merge_datasets_by_period(DATA_CLND['Spd80mN'], MERRA2['WS50m_m/s'], period='1MS',
                                            wdir_column_names_1=None, wdir_column_names_2=None,
                                            coverage_threshold_1=0.99, coverage_threshold_2=1,
                                            aggregation_method_1='sum', aggregation_method_2='sum')
    assert len(mrgd_data) == 13

    mrgd_data = bw.merge_datasets_by_period(DATA_CLND['Spd80mN'],
                                            MERRA2['WS50m_m/s'][MERRA2.index.month.isin([2, 4, 6, 8, 10, 12])],
                                            period='1MS',
                                            wdir_column_names_1=None, wdir_column_names_2=None,
                                            coverage_threshold_1=0.99, coverage_threshold_2=1,
                                            aggregation_method_1='sum', aggregation_method_2='sum')
    assert len(mrgd_data) == 9

    mrgd_data = bw.merge_datasets_by_period(DATA_CLND['Dir78mS'], MERRA2['WD50m_deg'],
                                            period='1MS',
                                            wdir_column_names_1=['Dir78mS'], wdir_column_names_2='WD50m_deg',
                                            coverage_threshold_1=0.99, coverage_threshold_2=1,
                                            aggregation_method_1='mean', aggregation_method_2='mean')
    assert len(mrgd_data) == 13

    # Test when target data is missing when overlapping timestamp is found.
    data_spd80mn_even_months = DATA_CLND[['Spd80mN', 'Dir78mS']][DATA_CLND.index.month.isin([2, 4, 6, 8, 10, 12])]
    mrgd_data = bw.merge_datasets_by_period(MERRA2['WS50m_m/s']['2016-03-03':],
                                            data_spd80mn_even_months['2016-02-09 00:00:00':],
                                            period='5D',
                                            coverage_threshold_1=1, coverage_threshold_2=1,
                                            aggregation_method_1='mean', aggregation_method_2='mean',
                                            wdir_column_names_2='Dir78mS')
    assert mrgd_data.index[0] == pd.to_datetime('2016-04-02 00:00:00')

    mrgd_data = bw.merge_datasets_by_period(MERRA2['WS50m_m/s']['2016-03-03':],
                                            data_spd80mn_even_months['Spd80mN']['2016-02-09 00:00:00':],
                                            period='5D',
                                            coverage_threshold_1=1, coverage_threshold_2=1,
                                            aggregation_method_1='mean', aggregation_method_2='mean')
    assert mrgd_data.index[0] == pd.to_datetime('2016-04-02 00:00:00')
    # input data_resolution
    data1 = DATA[:'2016-01-10'].copy()
    data1.reset_index(inplace=True)
    drop_indices = np.array([60, 36, 140, 16, 101, 40, 158, 122, 151, 34, 117, 159, 26,
                             169, 132, 124, 98, 141, 127, 100, 115, 119, 59, 17, 166, 61,
                             10, 106, 57, 13, 187, 174, 28, 63, 85, 130, 23, 148, 0,
                             145, 8, 149, 185, 170, 73, 9, 79, 65, 136, 6, 54, 172,
                             108, 29, 107, 102, 123, 168, 89, 182, 173, 167, 125, 33, 114,
                             113, 84, 41, 110, 30, 179, 43, 134, 142, 171, 155, 25, 135,
                             163, 92, 183, 49, 104, 46, 68, 116, 53, 87, 184, 146, 153,
                             1, 77, 164, 161, 165, 94, 4, 58, 103, 19, 2, 48, 88,
                             152, 96, 82, 55, 32, 42, 15, 51, 70, 3, 147, 78, 86,
                             69, 131, 144, 181, 45, 31, 175, 97, 21, 143, 186, 137, 120,
                             176, 80, 156, 14, 105, 47, 67, 22, 12, 128, 71, 5, 139,
                             81, 154, 62, 121, 27, 39, 91, 75, 112, 66, 93, 38, 44,
                             177, 99, 11, 76, 64, 35, 56, 109, 83, 90, 162, 50, 180,
                             52, 72, 129, 111, 157, 150, 20, 24, 118, 178, 160])
    data1 = data1.drop(drop_indices)
    data1 = data1.set_index('Timestamp')
    mrgd_data = bw.merge_datasets_by_period(MERRA2[['WS50m_m/s', 'WD50m_deg']], data1[['Spd80mN', 'Dir78mS']],
                                            period='1MS',
                                            wdir_column_names_1='WD50m_deg', wdir_column_names_2='Dir78mS',
                                            coverage_threshold_1=0, coverage_threshold_2=0,
                                            aggregation_method_1='mean', aggregation_method_2='mean',
                                            data_1_resolution=pd.DateOffset(hours=1),
                                            data_2_resolution=pd.DateOffset(minutes=10))

    assert round(mrgd_data['Spd80mN_Coverage'].values[0], 8) == 0.00179211

