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
    series1 = bw.load_campbell_scientific(bw.datasets.demo_site_data)['Spd80mS'].index
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
    # # sending data-series with datetime index
    # bw.offset_timestamps(series1.Spd80mN, offset='15min')
    #
    # # sending data-series with datetime index with start end
    # bw.offset_timestamps(series1.Spd80mN, offset='-1H', date_from='2016-02-13', date_to='2017-08-14')
    #
    # # sending data-frame with datetime index
    # bw.offset_timestamps(series1, offset='1D')
    #
    # # sending data-frame with datetime index with start end
    # bw.offset_timestamps(series1, offset='-1M', date_from= '2016-08-13', date_to='2017-05-20')
