import pytest
import brightwind as bw
import pandas as pd
import numpy as np


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

def test_get_data_resolution():
    import warnings
    series1 = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)['Spd80mS'].index
    assert bw._get_data_resolution(series1).seconds == 600

    series2 = pd.date_range('2010-01-01', periods=150, freq='H')
    assert bw._get_data_resolution(series2).seconds == 3600

    #hourly series with one instance where difference between adjacent timestamps is 10 min
    series3 = pd.date_range('2010-04-15', '2010-05-01', freq='H').union(pd.date_range('2010-05-01 00:10:00', periods=20,
                                                                                    freq='H'))
    with warnings.catch_warnings(record=True) as w:
        assert bw._get_data_resolution(series3).seconds == 3600
        assert len(w) == 1

        
def test_offset_timestamps():
    series1 = bw.load_campbell_scientific(bw.datasets.demo_site_data)#['Spd80mS']

    # sending index with no start end
    bw.offset_timestamps(series1.index, offset='90min')

    # sending index with start end
    op = bw.offset_timestamps(series1.index, offset='2min', date_from='2016-01-01 00:10:00')
    assert op[0] == pd.to_datetime('2016-01-01 00:00:00')
    assert op[1] == pd.to_datetime('2016-01-01 00:12:00')

    op = bw.offset_timestamps(series1.index, '2min', date_to='2016-01-01 00:30:00')
    assert op[3] == pd.to_datetime('2016-01-01 00:32:00')
    assert op[4] == pd.to_datetime('2016-01-01 00:40:00')

    op = bw.offset_timestamps(series1.index, '3min', date_from='2016-01-01 00:10:00', date_to='2016-01-01 00:30:00')
    assert op[0] == pd.to_datetime('2016-01-01 00:00:00')
    assert op[1] == pd.to_datetime('2016-01-01 00:13:00')
    assert op[5] == pd.to_datetime('2016-01-01 00:50:00')

    op = bw.offset_timestamps(series1.index, '10min', date_from='2016-01-01 00:10:00', date_to='2016-01-01 00:30:00')
    assert op[0] == series1.index[0]
    assert op[1] == series1.index[2]

    # sending DataFrame with datetime index
    op = bw.offset_timestamps(series1, offset='-10min', date_from='2016-01-01 00:20:00')
    assert (op.iloc[1] == series1.iloc[1]).all()
    assert len(op)+1 == len(series1)
    assert (op.loc['2016-01-01 00:40:00'] == series1.loc['2016-01-01 00:50:00']).all()

    op = bw.offset_timestamps(series1, offset='-10min', date_from='2016-01-01 00:20:00', overwrite=True)
    assert (op.loc['2016-01-01 00:10:00'] == series1.loc['2016-01-01 00:20:00']).all()

    op = bw.offset_timestamps(series1, '10min', date_from='2016-01-01 00:10:00', date_to='2016-01-01 00:30:00')
    assert (op.loc['2016-01-01 00:20:00'] == series1.loc['2016-01-01 00:10:00']).all()
    assert (op.loc['2016-01-01 00:40:00'] == series1.loc['2016-01-01 00:40:00']).all()
    assert len(op) + 1 == len(series1)

    op = bw.offset_timestamps(series1, '10min', date_from='2016-01-01 00:10:00', date_to='2016-01-01 00:30:00',
                              overwrite=True)
    assert (op.loc['2016-01-01 00:40:00'] == series1.loc['2016-01-01 00:30:00']).all()
    assert len(op) + 1 == len(series1)

    # sending Series with datetime index
    op = bw.offset_timestamps(series1.Spd60mN, offset='-10min', date_from='2016-01-01 00:20:00')
    assert (op.iloc[1] == series1.Spd60mN.iloc[1]).all()
    assert len(op)+1 == len(series1.Spd60mN)
    assert (op.loc['2016-01-01 00:40:00'] == series1.Spd60mN.loc['2016-01-01 00:50:00']).all()

    op = bw.offset_timestamps(series1.Spd60mN, offset='-10min', date_from='2016-01-01 00:20:00', overwrite=True)
    assert (op.loc['2016-01-01 00:10:00'] == series1.Spd60mN.loc['2016-01-01 00:20:00']).all()

    op = bw.offset_timestamps(series1.Spd60mN, '10min', date_from='2016-01-01 00:10:00', date_to='2016-01-01 00:30:00')
    assert (op.loc['2016-01-01 00:20:00'] == series1.Spd60mN.loc['2016-01-01 00:10:00']).all()
    assert (op.loc['2016-01-01 00:40:00'] == series1.Spd60mN.loc['2016-01-01 00:40:00']).all()
    assert len(op) + 1 == len(series1.Spd60mN)

    op = bw.offset_timestamps(series1.Spd60mN, '10min', date_from='2016-01-01 00:10:00', date_to='2016-01-01 00:30:00',
                              overwrite=True)
    assert (op.loc['2016-01-01 00:40:00'] == series1.Spd60mN.loc['2016-01-01 00:30:00']).all()
    assert len(op) + 1 == len(series1.Spd60mN)
=======
def test_average_data_by_period():
    data = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)

    #hourly averages
    bw.average_data_by_period(data.Spd80mN, period='1H')
    #hourly average with coverage filtering
    bw.average_data_by_period(data.Spd80mN, period='1H', coverage_threshold=0.9)
    bw.average_data_by_period(data.Spd80mN, period='1H', coverage_threshold=1)
    #return coverage with filtering
    bw.average_data_by_period(data.Spd80mN, period='1H', coverage_threshold=0.9,
                              return_coverage=True)
    # return coverage without filtering
    bw.average_data_by_period(data.Spd80mN, period='1H', return_coverage=True)

    #monthly averages
    bw.average_data_by_period(data.Spd80mN, period='1M')
    #hourly average with coverage filtering
    bw.average_data_by_period(data.Spd80mN, period='1M', coverage_threshold=0.9)
    bw.average_data_by_period(data.Spd80mN, period='1M', coverage_threshold=1)
    #return coverage with filtering
    bw.average_data_by_period(data.Spd80mN, period='1M', coverage_threshold=0.9, return_coverage=True)
    # return coverage without filtering
    bw.average_data_by_period(data.Spd80mN, period='1M', return_coverage=True)

    #weekly averages
    bw.average_data_by_period(data.Spd80mN, period='2W')
    #hourly average with coverage filtering
    bw.average_data_by_period(data.Spd80mN, period='2W', coverage_threshold=0.9)
    bw.average_data_by_period(data.Spd80mN, period='2W', coverage_threshold=1)
    #return coverage with filtering
    bw.average_data_by_period(data.Spd80mN, period='2W', coverage_threshold=0.9, return_coverage=True)
    # return coverage without filtering
    bw.average_data_by_period(data.Spd80mN, period='2W', return_coverage=True)
