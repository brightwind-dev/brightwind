#     brightwind is a library that provides wind analysts with easy to use tools for working with meteorological data.
#     Copyright (C) 2018 Stephen Holleran, Inder Preet
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Lesser General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pandas as pd
from brightwind.utils import utils
import warnings

__all__ = ['average_data_by_period',
           'merge_datasets_by_period',
           'average_wdirs',
           'adjust_slope_offset',
           'scale_wind_speed',
           'offset_wind_direction',
           'selective_avg',
           'offset_timestamps']


def _validate_coverage_threshold(coverage_threshold):
    """
    Validate that coverage_threshold is between 0 and 1 and if it is None set to zero.
    :param coverage_threshold: Should be number between or equal to 0 and 1.
    :type coverage_threshold:  float or int
    :return:                   coverage_threshold
    :rtype:                    float or int
    """
    coverage_threshold = 0 if coverage_threshold is None else coverage_threshold
    if coverage_threshold < 0 or coverage_threshold > 1:
        raise TypeError("Invalid coverage_threshold, this should be between or equal to 0 and 1.")
    return coverage_threshold


def _compute_wind_vector(wspd, wdir):
    """
    Returns north and east component of wind-vector
    """
    return wspd*np.cos(wdir), wspd*np.sin(wdir)


def _freq_str_to_timedelta(period):
    """
    Convert a pandas frequency string to a pd.Timedelta.

    This is needed because support for MS and AS was dropped. Pandas frequency strings are available here:
    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

    :param period: Frequency string to be converted to a pd.Timedelta
    :type period:  str
    :return:       A pd.Timedelta
    :rtype:        pd.Timedelta
    """
    if period[-1] == 'M':
        as_timedelta = pd.Timedelta(int(period[:-1]), unit='M')
    elif period[-2:] == 'MS':
        as_timedelta = pd.Timedelta(int(period[:-2]), unit='M')
    elif period[-1] == 'A':
        as_timedelta = pd.Timedelta(365 * int(period[:-1]), unit='D')
    elif period[-2:] == 'AS':
        as_timedelta = pd.Timedelta(365 * int(period[:-2]), unit='D')
    else:
        as_timedelta = pd.Timedelta(period)
    return as_timedelta


def _convert_days_to_hours(prd):
    return str(int(prd[:-1])*24)+'H'


def _convert_weeks_to_hours(prd):
    return str(int(prd[:-1])*24*7)+'H'


def _get_min_overlap_timestamp(df1_timestamps, df2_timestamps):
    """
    Get the minimum overlapping timestamp from two series
    """
    try:
        if df1_timestamps.max() < df2_timestamps.min() or df1_timestamps.min() > df2_timestamps.max():
            raise IndexError("No overlapping data. Dataset ranges are: {0} to {1} and {2} to {3}."
                             .format(df1_timestamps.min(), df1_timestamps.max(),
                                     df2_timestamps.min(), df2_timestamps.max()), )
    except TypeError as type_error:
        if str(type_error) == 'Cannot compare tz-naive and tz-aware timestamps':
            raise TypeError('Cannot compare tz-naive and tz-aware timestamps. One of the input dataset '
                            'is timezone aware, use df.tz_localize(None) to remove the timezone.')
    except Exception as error:
        raise error
    return max(df1_timestamps.min(), df2_timestamps.min())


def _get_data_resolution(data_idx):
    """
    Get the resolution of a timeseries i.e. the most common time interval between timestamps. Also known as the
    averaging period.

    The algorithm finds the most common time difference between consecutive time stamps and returns the
    most common time stamp.

    This function will return a specific Timedelta if a resolution of month or year is identified due to months and
    years having irregular numbers of days. These will be:
    - For monthly data:     pd.Timedelta(1, unit='M')       i.e. 30.436875 days
    - For annual data:      pd.Timedelta(365, unit='D')     i.e. 365 days

    The function also checks the most common time difference against the minimum time difference. If they
    do not match it shows a warning. It is suggested to manually look at the data if such a warning is shown.

    :param data_idx: Timeseries index of a pd.DataFrame or pd.Series.
    :type data_idx:  pd.DataFrame.index or pd.Series.index
    :return:         A time delta object which represents the time difference between consecutive timestamps.
    :rtype:          pd.Timedelta

    **Example usage**
    ::
        import brightwind as bw
        import pandas as pd
        data = bw.load_csv(bw.demo_datasets.demo_data)
        resolution = bw.transform.transform._get_data_resolution(data.Spd80mS.index)

        # To check the number of seconds in resolution
        print(resolution.seconds)

        # To check if the resolution is monthly
        resolution == pd.Timedelta(1, unit='M')


    """
    time_diff_btw_timestamps = data_idx.to_series().diff()
    most_freq_time_diff = time_diff_btw_timestamps.mode().values[0]
    most_freq_time_diff = pd.to_timedelta(most_freq_time_diff)  # convert np.timedelta64 to pd.Timedelta
    minimum_time_diff = time_diff_btw_timestamps.min()

    if most_freq_time_diff.days in [28, 29, 30, 31]:    # check if monthly first
        return pd.Timedelta(1, unit='M')
    elif most_freq_time_diff.days in [365, 366]:        # then if yearly
        return pd.Timedelta(365, unit='D')

    if minimum_time_diff != most_freq_time_diff:
        warnings.warn('\nFrequency of input data may not be determined correctly. Most frequent time '
                      'difference between adjacent timestamps does not match minimum time difference.\n'
                      'Most frequent time difference is:\t{0}\n'
                      'Minimum time difference is:\t\t{1}\n'
                      'Returning most frequent time difference.'.format(most_freq_time_diff, minimum_time_diff))
    return most_freq_time_diff


def _round_down_to_multiple(num, divisor):
    """
    Round the number down to a multiple of the divisor.
    :param num:
    :param divisor:
    :return:
    """
    return num - (num % divisor)


def _round_timestamp_down_to_averaging_prd(timestamp, period):
    """
    Return a timestamp that represents the start of the averaging period based on the timestamp provided.

    :param timestamp: Timestamp to round down from.
    :type timestamp:  pd.Timestamp
    :param period:    Averaging period e.g. '10min', '1H', '3H', '6H', '1D', '7D', '1W', '1MS', '1AS'
    :type period:     str
    :return:          Timestamp to represent the start of an averaging period which covers the timestamp.
    :rtype:           str

    if 10min  it should go to the first whole 10-min period i.e. 00:00, 00:10, 00:20, 00:30, 00:40, 00:50
    if 15min  it should go to the first whole 15-min period i.e. 00:00, 00:15, 00:30, 00:45
    if 1H     it should go to the first whole hour
    if 3H, 4H it should go to the first whole 3 or 4-hour period i.e. 04:00, 08:00, 12:00, 16:00, 20:00
    if 1D     it should go to midnight
    if 7D, 1W it should go to midnight because we don't have a base reference
    if 1M, 1MS it should go to start of month
    if 1A, 1AS it should go to start of year
    """
    if period[-3:] == 'min':
        return '{year}-{month}-{day} {hour}:{minute}:00'.format(year=timestamp.year, month=timestamp.month,
                                                                day=timestamp.day, hour=timestamp.hour,
                                                                minute=_round_down_to_multiple(timestamp.minute,
                                                                                               int(period[:-3])))
    elif period[-1] == 'H':
        return '{year}-{month}-{day} {hour}:00:00'.format(year=timestamp.year, month=timestamp.month, day=timestamp.day,
                                                          hour=_round_down_to_multiple(timestamp.hour,
                                                                                       int(period[:-1])))
    elif period[-1] == 'D' or period[-1] == 'W':
        return '{year}-{month}-{day}'.format(year=timestamp.year, month=timestamp.month, day=timestamp.day,
                                             hour=timestamp.hour)
    elif period[-1] == 'M' or period[-2:] == 'MS':
        return '{year}-{month}'.format(year=timestamp.year, month=timestamp.month)
    elif period[-2:] == 'AS' or period[-1:] == 'A':
        return '{year}'.format(year=timestamp.year)
    else:
        print("Warning: Averaging period not identified returning default timestamps")
        return '{year}-{month}-{day} {hour}:{minute}:{second}'.format(year=timestamp.year, month=timestamp.month,
                                                                      day=timestamp.day, hour=timestamp.hour,
                                                                      minute=timestamp.minute, second=timestamp.second)


def _common_idxs(reference, site):
    """
    Finds overlapping indexes from two DataFrames.
    """
    common = reference.dropna().index.intersection(site.dropna().index)
    return common, len(common)


def _get_overlapping_data(df1, df2, averaging_prd=None):
    """
    Return data from where both datasets overlap onwards. If an averaging period is sent then the datasets
    will start at the first overlapping averaging period.

    :param df1:
    :param df2:
    :param averaging_prd: Averaging period
            - Set period to 10min for 10 minute average, 20min for 20 minute average and so on for 4min, 15min, etc.
            - Set period to 1H for hourly average, 3H for three hourly average and so on for 5H, 6H etc.
            - Set period to 1D for a daily average, 3D for three day average, similarly 5D, 7D, 15D etc.
            - Set period to 1W for a weekly average, 3W for three week average, similarly 2W, 4W etc.
            - Set period to 1M for monthly average
            - Set period to 1AS fo annual average
    :type averaging_prd: str
    :return: df1 and df2 chopped to start at the same timestamp
    """
    if averaging_prd is not None:
        start = _round_timestamp_down_to_averaging_prd(_get_min_overlap_timestamp(df1.index, df2.index), averaging_prd)
    else:
        start = _get_min_overlap_timestamp(df1.index, df2.index)
    return df1[start:], df2[start:]


def _max_coverage_count(data_index, averaged_data_index)->pd.Series:
    """
    For a given resolution of data finds the maximum number of data points in the averaging period
    """
    data_res = _get_data_resolution(data_index)

    max_pts = (averaged_data_index.to_series().diff().shift(-1)) / data_res
    # Fill in the last in the list as it is a NaT
    max_pts[-1] = (((averaged_data_index[-1] + 1 * averaged_data_index[-1].freq) - averaged_data_index[-1]) /
                   data_res)
    if data_res == pd.Timedelta(1, unit='M'):
        # The data resolution is monthly, therefore round the result to 0 decimal to make whole months.
        max_pts = np.round(max_pts, 0)
    return max_pts


def _get_coverage_by_grouper_obj(data, grouper_obj):
    """
    Counts the number of data points in the grouper_obj relative to the maximum possible

    :param data: The data to find the coverage for.
    :type  data: pd.DataFrame or pd.Series
    :param grouper_obj: The object that has grouped the data already. The mean, sum, count, etc. can then be found.
    :type  grouper_obj: pd.DatetimeIndexResampler
    :return:
    """
    coverage = grouper_obj.count().divide(_max_coverage_count(data.index, grouper_obj.mean().index), axis=0)
    return coverage


def average_data_by_period(data, period, wdir_column_names=None, aggregation_method='mean', coverage_threshold=None,
                           return_coverage=False):
    """
    Averages the data by a time period specified.

    Aggregates data by the aggregation_method specified, by default this function averages the data to the period 
    specified.

    A list of wind direction column names can be sent to identify which columns should be vector averaged. This vector
    averaging of wind directions only applies if the aggregation method is 'mean', otherwise it will aggregate based
    on the method sent.

    The data can also by filtered by a coverage threshold to insure a certain amount of data is required in the period.

    This function can be used to find hourly, daily, weekly, etc. averages or sums. Can also return coverage and
    filter the returned data by coverage.

    It will return NaN for intermediary periods where there is no data.

    If you wish to do 'upsampling', i.e. convert from hourly to every 10 minutes, please used the pandas resample
    function directly.

    :param data:               Data to find average or aggregate of
    :type data:                pd.Series or pd.DataFrame
    :param period:             Groups data by the period specified here. The following formats are supported

            - Set period to '10min' for 10 minute average, '30min' for 30 minute average.
            - Set period to '1H' for hourly average, '3H' for three hourly average and so on for '4H', '6H' etc.
            - Set period to '1D' for a daily average, '3D' for three day average, similarly '5D', '7D', '15D' etc.
            - Set period to '1W' for a weekly average, '3W' for three week average, similarly '2W', '4W' etc.
            - Set period to '1M' for monthly average with the timestamp at the start of the month.
            - Set period to '1A' for annual average with the timestamp at the start of the year.

    :type period:              str
    :param wdir_column_names:  List of wind direction column names. These columns, if the aggregation_method is mean,
                               will be vector averaged together instead of a straight mean.
    :type wdir_column_names:   list or str
    :param aggregation_method: Default `mean`, returns the mean of the data for the specified period. Can also
                               use `median`, `prod`, `sum`, `std`,`var`, `max`, `min` which are shorthands for median,
                               product, summation, standard deviation, variance, maximum and minimum respectively.
    :type aggregation_method:  str
    :param coverage_threshold: Coverage is defined as the ratio of number of data points present in the period and the 
                               maximum number of data points that a period should have. Example, for 10 minute data
                               resolution and a period of 1 hour, the maximum number of data points in one period is 6.
                               But if the number if data points available is only 3 for that hour the coverage is
                               3/6=0.5. It should be greater than 0 and less than or equal to 1. It is set to None by
                               default. If it is None or 0, data is not filtered. Otherwise periods where coverage is
                               less than the coverage_threshold are removed.
    :type coverage_threshold:  float
    :param return_coverage:    If True appends and additional column in the DataFrame returned, with coverage calculated
                               for each period. The columns with coverage are named as <column name>_Coverage
    :type return_coverage:     bool
    :returns: A DataFrame with data aggregated with the specified aggregation_method (mean by default). Additionally it
              could be filtered based on coverage and have a coverage column depending on the parameters.
    :rtype:   pd.DataFrame

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)

        # To find hourly averages
        data_hourly = bw.average_data_by_period(data.Spd80mN, period='1H')

        # To find monthly averages
        data_monthly = bw.average_data_by_period(data.Spd80mN, period='1M')

        # To filter months where half of the data is missing
        data_monthly_filtered = bw.average_data_by_period(data.Spd80mN, period='1M', coverage_threshold=0.5)

        # To check the coverage for all months
        data_monthly_filtered = bw.average_data_by_period(data.Spd80mN, period='1M', return_coverage=True)

        # To average wind directions by vector averaging
        data_monthly_wdir_avg = bw.average_data_by_period(data.Dir78mS, period='1M', wdir_column_names='Dir78mS')

    """
    coverage_threshold = _validate_coverage_threshold(coverage_threshold)

    if isinstance(period, str):
        if period[-1] == 'D':
            period = _convert_days_to_hours(period)
        if period[-1] == 'W':
            period = _convert_weeks_to_hours(period)
        if period[-1] == 'M':
            period = period+'S'
        if period[-1] == 'A':
            period = period+'S'
        if period[-1] == 'Y':
            raise TypeError("Please use '1AS' for annual frequency at the start of the year.")

    # Check that the data resolution is not less than the period specified
    if _freq_str_to_timedelta(period) < _get_data_resolution(data.index):
        raise ValueError("The time period specified is less than the temporal resolution of the data. "
                         "For example, hourly data should not be averaged to 10 minute data.")
    data = data.sort_index()
    grouper_obj = data.resample(period, axis=0, closed='left', label='left', base=0,
                                convention='start', kind='timestamp')

    if wdir_column_names is not None and aggregation_method == 'mean':
        # do vector averaging on wdirs if aggregation method of mean is requested
        wdir_column_names = [wdir_column_names] if isinstance(wdir_column_names, str) else wdir_column_names

        # check if wdir_column_names are in dataframe, if not raise error
        if isinstance(data, pd.DataFrame):
            for wdir_col_name in wdir_column_names:
                if wdir_col_name not in data.columns:
                    raise KeyError("'{}' not in data sent.".format(wdir_col_name))
        else:
            for wdir_col_name in wdir_column_names:
                if wdir_col_name != data.name:
                    raise KeyError("'{}' not in data sent.".format(wdir_col_name))

        # separate out wdirs columns for vector averaging. Only need to do if a DataFrame
        non_wdir_col_names = []
        if isinstance(data, pd.DataFrame):
            # if data is a DataFrame then it may have non_wdir columns and more than just a single wdir column
            for col_name in data.columns:
                if col_name not in wdir_column_names:
                    non_wdir_col_names.append(col_name)

        # average wdir data
        # if data is a Series grouper_obj doesn't take columns and averaged data is a Series and not a DataFrame.
        if isinstance(data, pd.DataFrame):
            wdir_grouped_data = grouper_obj[wdir_column_names].agg(average_wdirs)
            wdir_coverage = _get_coverage_by_grouper_obj(data[wdir_column_names], grouper_obj[wdir_column_names])
        else:
            wdir_grouped_data = grouper_obj.agg(average_wdirs)
            wdir_coverage = _get_coverage_by_grouper_obj(data, grouper_obj)

        # average non_wdir data if available
        if non_wdir_col_names:
            non_wdir_grouped_data = grouper_obj[non_wdir_col_names].agg('mean')
            non_wdir_coverage = _get_coverage_by_grouper_obj(data[non_wdir_col_names], grouper_obj[non_wdir_col_names])

            grouped_data = pd.concat([non_wdir_grouped_data, wdir_grouped_data], axis=1)
            coverage = pd.concat([non_wdir_coverage, wdir_coverage], axis=1)
        else:
            grouped_data = wdir_grouped_data
            coverage = wdir_coverage
    elif wdir_column_names is not None and aggregation_method != 'mean':
        raise KeyError("Vector averaging is only applied when 'aggregation_method' is 'mean'. Either set " +
                       "'wdir_column_names' to None or set 'aggregation_method'='mean'")
    else:
        # group data as normal
        grouped_data = grouper_obj.agg(aggregation_method)
        coverage = _get_coverage_by_grouper_obj(data, grouper_obj)

    grouped_data = grouped_data[coverage >= coverage_threshold]

    if return_coverage:
        if isinstance(coverage, pd.DataFrame):
            coverage.columns = [col_name + "_Coverage" for col_name in coverage.columns]
        elif isinstance(coverage, pd.Series):
            coverage = coverage.rename(data.name + '_Coverage')
        else:
            raise TypeError("Coverage not calculated correctly. Coverage", coverage)
        return grouped_data, coverage  # [coverage >= coverage_threshold]
    else:
        return grouped_data


def _vector_avg_of_wdirs_dataframe(wdirs, wspds=None):
    """
    Average wind directions together using vector averaging for a pandas DataFrame. In a DataFrame multiple wind
    direction timeseries are sent. This function will average the wind directions for each *row* of the DataFrame
    returning a pandas Series.

    :param wdirs: Wind directions to calculate the average of
    :type wdirs:  pd.DataFrame
    :param wspds: Wind speeds for the magnitude of the wind direction vector.
                  If not provided the magnitude is assumed to be unity.
                  There must be the same number of columns sent as wind directions and the order of the columns will
                  determine how the wind speeds will match the wind directions.
    :type wspds:  pd.DataFrame
    :return:      Average wind direction for each row of the DataFrame provided.
    :rtype:       pd.Series

    **Example usage**
    ::
        wdirs = np.array([[350, 10],
              [0, 180],
              [90, 270],
              [45, 135],
              [135, 225],
              [15, np.nan]])
        wdirs_df = pd.DataFrame(wdirs)
        _vector_avg_of_wdirs_dataframe(wdirs_df)

        wspds = np.array([[1, 2],
              [1, 2],
              [1, 2],
              [1, 2],
              [1, 2],
              [np.nan, 2]])
        wspds_df = pd.DataFrame(wspds)
        _vector_avg_of_wdirs_dataframe(wdirs_df, wspds_df)

    Note:
    The reason [0, 180] results in 90 and not NaN is because the sin of 180 is not quite zero which results in not
    ending back exactly where you started and so gives 90. Similarly if 10, 190 is sent the mean of the sin is slightly
    negative (-6.9e-17) instead of zero which results in 270 instead of NaN. Similar for cosine when 90, 270 sent.
    Solution is to round both sin and cos to 5 decimal places to make them zero.
    """
    if wspds is None:
        sine = (np.round(np.sin(np.deg2rad(wdirs)), 5)).mean(axis=1)  # sin of each angle, convert to radian first
        cosine = (np.round(np.cos(np.deg2rad(wdirs)), 5)).mean(axis=1)  # cos of each angle, convert to radian first
    else:
        sine = (pd.DataFrame(np.round(np.sin(np.deg2rad(wdirs)), 5).values * wspds.values, index=wdirs.index)).mean(
            axis=1)  # sin of each angle, convert to radian first
        cosine = (pd.DataFrame(np.round(np.cos(np.deg2rad(wdirs)), 5).values * wspds.values, index=wdirs.index)).mean(
            axis=1)  # cos of each angle, convert to radian first

    avg_dir_df = pd.DataFrame({'sine': sine, 'cosine': cosine})

    # If both sine and cosine result in zero then all the directions cancel and you end up where you started which
    # means there is no wind direction => return NaN
    nan_mask = (avg_dir_df['sine'] == 0) & (avg_dir_df['cosine'] == 0)
    avg_dir_df['avg_dir'] = np.rad2deg(np.arctan2(sine, cosine)) % 360
    avg_dir_df['avg_dir'][nan_mask] = np.NaN
    return avg_dir_df['avg_dir']


def _vector_avg_of_wdirs_list(wdirs, wspds=None):
    """
    Average wind directions together using vector averaging for a list, array or pandas Series.

    :param wdirs: Wind directions to calculate the average of
    :type wdirs:  list or array or np.array or pd.Series
    :param wspds: Wind speeds for the magnitude of the wind direction vector.
                  If not provided the magnitude is assumed to be unity.
                  If a list or an array is sent they must be the same length as wdirs.
    :type wspds:  list or array or np.array or pd.Series
    :return:      Average wind direction for the wind directions provided.
    :rtype:       float

    **Example usage**
    ::
        wdirs = np.array([350, 10])
        _vector_avg_of_wdirs_list(wdirs)

        wdirs_series = pd.Series(wdirs)
        _vector_avg_of_wdirs_list(wdirs_series)

        wspds = [5, 6]
        _vector_avg_of_wdirs_list(wdirs, wspds)

    Note:
    The reason [0, 180] results in 90 and not NaN is because the sin of 180 is not quite zero which results in not
    ending back exactly where you started and so gives 90. Similarly if 10, 190 is sent the mean of the sin is slightly
    negative (-6.9e-17) instead of zero which results in 270 instead of NaN. Similar for cosine when 90, 270 sent.
    Solution is to round both sin and cos to 5 decimal places to make them zero.
    """
    # first drop nans, if wind speeds available need to match them first to drop equivalent values
    if wspds is None:
        wdirs = np.array([x for x in wdirs if x == x])
    else:
        a = np.array([wdirs, wspds])
        a = a[:, ~np.isnan(a).any(axis=0)]
        wdirs = a[0]
        wspds = a[1]
    # if the resulting wdir array is empty, return NAN
    if wdirs.size == 0:
        return np.NaN

    if wspds is None:
        sine = np.mean(np.round(np.sin(np.deg2rad(wdirs)), 5))  # sin of each angle, East component
        cosine = np.mean(np.round(np.cos(np.deg2rad(wdirs)), 5))  # cos of each angle, North component
    else:
        sine = np.mean(np.round(np.sin(np.deg2rad(wdirs)), 5) * wspds)  # sin of each angle, East component
        cosine = np.mean(np.round(np.cos(np.deg2rad(wdirs)), 5) * wspds)  # cos of each angle, North component

    # If both sine and cosine result in zero then all the directions cancel and you end up where you started which
    # means there is no wind direction => return NaN
    if sine == 0 and cosine == 0:
        avg_dir = np.NaN
    else:
        avg_dir = np.rad2deg(np.arctan2(sine, cosine)) % 360
        if avg_dir == 360.0:  # preference to have 0 returned instead of 360
            avg_dir = 0.0
    return avg_dir


def average_wdirs(wdirs, wspds=None):
    """
    Average wind directions together using vector averaging. This works for a list, array, np.array, pd.Series or
    pd.DataFrame.

    If a list, array, np.array or pd.Series of wind directions are sent, it will average all the wind speeds in that
    array together returning a single value.
    If a DataFrame with multiple timeseries of wind direction columns is sent, it will average the wind directions for
    each *row* of the DataFrame returning a pandas Series.

    It is also possible to send wind speeds for each wind direction to be used as the magnitude in the vector averaging
    algorithm.

    When a dataset which contains some NANs is sent these are ignored e.g. if 3 wind directions are sent and one of them
    is a NAN, the average of the other two is calculated ignoring the NAN. Similarly, if some wind speeds contain a NAN,
    this and the corresponding wind direction are ignored even if the wind direction is valid as the magnitude of the
    vector is missing.

    :param wdirs: Wind directions to calculate the average of.
    :type wdirs:  list or array or np.array or pd.Series or pd.DataFrame
    :param wspds: Wind speeds for the magnitude of the wind direction vector.
                  If not provided the magnitude is assumed to be unity.
                  If a list or an array is sent they must be the same length as wdirs.
                  If a DataFrame is sent it must have the same number of columns as the wind directions DataFrame and
                  the wind speed column will match it's equivalent wind direction column by the ordering of the columns.
    :type wspds:  list or array or np.array or pd.Series or pd.DataFrame
    :return:      Average wind direction for the wind directions provided.
    :rtype:       float or pd.Series

    **Example usage**
    ::
        import brightwind as bw

        wdirs = np.array([350, 10])
        bw.average_wdirs(wdirs)

        wdirs_series = pd.Series(wdirs)
        bw.average_wdirs(wdirs_series)

        wspds = [5, 6]
        bw.average_wdirs(wdirs, wspds)

        wdirs = np.array([[350, 10],
              [0, 180],
              [90, 270],
              [45, 135],
              [135, 225],
              [15, np.nan]])
        wdirs_df = pd.DataFrame(wdirs)
        bw.average_wdirs(wdirs_df)

        wspds = np.array([[1, 2],
              [1, 2],
              [1, 2],
              [1, 2],
              [1, 2],
              [np.nan, 2]])
        wspds_df = pd.DataFrame(wspds)
        bw.average_wdirs(wdirs_df, wspds_df)

        data = bw.load_csv(bw.demo_datasets.demo_data)
        avg_wdir = bw.average_wdirs(data.Dir78mS)

        wdir_cols = ['Dir78mS', 'Dir58mS', 'Dir38mS']
        avg_wdirs = bw.average_wdirs(data[wdir_cols])

    """
    if type(wdirs) == pd.DataFrame:
        avg_wdir = _vector_avg_of_wdirs_dataframe(wdirs, wspds)
    else:
        avg_wdir = _vector_avg_of_wdirs_list(wdirs, wspds)

    return avg_wdir


def merge_datasets_by_period(data_1, data_2, period,
                             wdir_column_names_1=None, wdir_column_names_2=None,
                             aggregation_method_1='mean', aggregation_method_2='mean',
                             coverage_threshold_1=None, coverage_threshold_2=None,):
    """
    Merge 2 datasets on a time period by aligning the overlapped aggregated data.

    This is done by
    1, First finding the minimum overlapping timestamp and rounding that down the the start timestamp
       of the first averaging period. The datasets are then truncated to start from that timestamp.
    2, The datasets are then aggregated to the time period by the aggregation method specified for each dataset. If
       coverage_threshold is specified, each dataset is filtered by that.
    3, The datasets are then merged and only concurrent timestamps are returned.

    This function utilises the 'average_data_by_period' function.

    :param data_1: First dataset to find average or aggregate of and merge with data_2.
    :type data_1:  pd.DataFrame or pd.Series
    :param data_2: Second dataset to find average or aggregate of and merge with data_1.
    :type data_2:  pd.DataFrame or pd.Series
    :param period: Groups data by the time period specified here. The following formats are supported

            - Set period to '10min' for 10 minute average, '30min' for 30 minute average.
            - Set period to '1H' for hourly average, '3H' for three hourly average and so on for '4H', '6H' etc.
            - Set period to '1D' for a daily average, '3D' for three day average, similarly '5D', '7D', '15D' etc.
            - Set period to '1W' for a weekly average, '3W' for three week average, similarly '2W', '4W' etc.
            - Set period to '1M' for monthly average with the timestamp at the start of the month.
            - Set period to '1A' for annual average with the timestamp at the start of the year.

    :type period:                str
    :param wdir_column_names_1:  List of wind direction column names. These columns, if the aggregation_method is mean,
                                 will be vector averaged together instead of a straight mean.
    :type wdir_column_names_1:   list or str or None
    :param wdir_column_names_2:  List of wind direction column names. These columns, if the aggregation_method is mean,
                                 will be vector averaged together instead of a straight mean.
    :type wdir_column_names_2:   list or str or None
    :param aggregation_method_1: Default `mean`, returns the mean of the data for the specified period. Can also
                                 use `median`, `prod`, `sum`, `std`,`var`, `max`, `min` which are shorthands for median,
                                 product, summation, standard deviation, variance, maximum and minimum respectively.
    :type aggregation_method_1:  str
    :param aggregation_method_2: Default `mean`, returns the mean of the data for the specified period. Can also
                                 use `median`, `prod`, `sum`, `std`,`var`, `max`, `min` which are shorthands for median,
                                 product, summation, standard deviation, variance, maximum and minimum respectively.
    :type aggregation_method_2:  str
    :param coverage_threshold_1: Coverage is defined as the ratio of number of data points present in the period and the
                                 maximum number of data points that a period should have. Example, for 10 minute data
                                 resolution and a period of 1 hour, the maximum number of data points in one period is
                                 6. But if the number if data points available is only 3 for that hour the coverage is
                                 3/6=0.5. It should be greater than 0 and less than or equal to 1. It is set to None by
                                 default. If it is None or 0, data is not filtered. Otherwise periods where coverage is
                                 less than the coverage_threshold are removed.
    :type coverage_threshold_1:  float or None
    :param coverage_threshold_2: Coverage is defined as the ratio of number of data points present in the period and the
                                 maximum number of data points that a period should have. Example, for 10 minute data
                                 resolution and a period of 1 hour, the maximum number of data points in one period is
                                 6. But if the number if data points available is only 3 for that hour the coverage is
                                 3/6=0.5. It should be greater than 0 and less than or equal to 1. It is set to None by
                                 default. If it is None or 0, data is not filtered. Otherwise periods where coverage is
                                 less than the coverage_threshold are removed.
    :type coverage_threshold_2:  float or None
    :return:                     Merged datasets.
    :rtype:                      pd.DataFrame

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)
        m2 = bw.load_csv(bw.demo_datasets.demo_merra2_NE)

        # To find monthly concurrent averages of two datasets
        mrgd_data = bw.merge_datasets_by_period(m2[['WS50m_m/s', 'WD50m_deg']], data[['Spd80mN', 'Dir78mS']],
                                                period='1MS',
                                                wdir_column_names_1='WD50m_deg', wdir_column_names_2='Dir78mS',
                                                coverage_threshold_1=0.95, coverage_threshold_2=0.95,
                                                aggregation_method_1='mean', aggregation_method_2='mean')

    """

    data_1_overlap, data_2_overlap = _get_overlapping_data(data_1.sort_index().dropna(),
                                                           data_2.sort_index().dropna(),
                                                           period)
    coverage_threshold_1 = _validate_coverage_threshold(coverage_threshold_1)
    coverage_threshold_2 = _validate_coverage_threshold(coverage_threshold_2)

    mrgd_data = pd.concat(list(average_data_by_period(data_1_overlap, period=period,
                                                      wdir_column_names=wdir_column_names_1,
                                                      coverage_threshold=coverage_threshold_1,
                                                      aggregation_method=aggregation_method_1,
                                                      return_coverage=True)) +
                          list(average_data_by_period(data_2_overlap, period=period,
                                                      wdir_column_names=wdir_column_names_2,
                                                      coverage_threshold=coverage_threshold_2,
                                                      aggregation_method=aggregation_method_2,
                                                      return_coverage=True)),
                          axis=1)
    return mrgd_data.dropna()


def adjust_slope_offset(wspd, current_slope, current_offset, new_slope, new_offset):
    """
    Adjust a wind speed that already has a slope and offset applied with a new slope and offset.
    Can take either a single wind speed value or a pandas DataFrame/series.

    :param wspd: The wind speed value or series to be adjusted.
    :type wspd: float or pd.DataFrame or pd.Series
    :param current_slope: The current slope that was applied to create the wind speed.
    :type current_slope: float
    :param current_offset: The current offset that was applied to create the wind speed.
    :type current_offset: float
    :param new_slope: The new desired slope to adjust the wind speed by.
    :type new_slope: float
    :param new_offset: The new desired offset to adjust the wind speed by.
    :type new_offset: float
    :return: The adjusted wind speed as a single value or pandas DataFrame.

    The new wind speed is calculated by equating the old and new y=mx+c equations around x and then solving for
    the new wind speed.

    y2 = m2*x + c2   and   y1 = m1*x + c1

    y2 = m2*(y1 - c1)/m1 + c2

    **Example usage**
    ::
        import brightwind as bw
        df = bw.load_campbell_scientific(bw.demo_datasets.demo_campbell_scientific_site_data)
        df['Spd80mS_adj'] = bw.adjust_slope_offset(df.Spd80mS, 0.044, 0.235, 0.04365, 0.236)
        df[['Spd80mS', 'Spd80mS_adj']]

    """
    try:
        return new_slope * ((wspd - current_offset) / current_slope) + new_offset
    except TypeError as type_error:
        for arg_value in [current_slope, current_offset, new_slope, new_offset]:
            if not utils.is_float_or_int(arg_value):
                raise TypeError("argument '" + str(arg_value) + "' is not of data type number")
        if not utils.is_float_or_int(wspd):
            if type(wspd) == pd.DataFrame and (wspd.dtypes == object)[0]:
                raise TypeError('some values in the DataFrame are not of data type number')
            elif type(wspd) == pd.Series and (wspd.dtypes == object):
                raise TypeError('some values in the Series are not of data type number')
            raise TypeError('wspd argument is not of data type number')
        raise type_error
    except Exception as error:
        raise error


def scale_wind_speed(spd, scale_factor: float):
    """
    Scales wind speed by the scale_factor

    :param spd: Series or data frame or a single value of wind speed to scale
    :param scale_factor: Scaling factor in decimal, if scaling factor is 0.8 output would be (1+0.8) times wind speed,
    if it is -0.8 the output would be (1-0.8) times the wind speed
    :return: Series or data frame with scaled wind speeds

    """
    return spd * scale_factor


def offset_wind_direction(wdir, offset: float):
    """
    Add/ subtract offset from wind direction. Keeps the ranges between 0 to 360
    :param wdir: Series or data frame or a single direction to offset
    :param offset: Offset in degrees can be negative or positive
    :return: Series or Dataframe or single value with offsetted directions
    """
    if isinstance(wdir, float):
        return utils._range_0_to_360(wdir + offset)
    elif isinstance(wdir, pd.DataFrame):
        return wdir.add(offset).applymap(utils._range_0_to_360)
    elif isinstance(wdir, pd.Series):
        return wdir.add(offset).apply(utils._range_0_to_360)


def _selective_avg(wspd1, wspd2, wdir, boom_dir1, boom_dir2,
                   inflow_lower1, inflow_higher1, inflow_lower2, inflow_higher2, sector_width):
    # duplicate threshold values into lists which are the same length as other inputs
    inflow_lower1 = [inflow_lower1] * len(wdir)
    inflow_higher1 = [inflow_higher1] * len(wdir)
    inflow_lower2 = [inflow_lower2] * len(wdir)
    inflow_higher2 = [inflow_higher2] * len(wdir)

    # if boom 1 'inflow' sector overlaps with 0/360
    if ((boom_dir1 + 180) % 360) >= (360 - (sector_width/2)) or ((boom_dir1 + 180) % 360) <= (sector_width/2):
        # many nested if statments follow, all within one mapped lambda function
        sel_avg = list(map(lambda spd1,spd2,Dir,inflowlow1,inflowhigh1,inflowlow2,inflowhigh2:
                           # if one value is Nan, use the other one
                           spd2 if (np.isnan(spd1)==True) else (spd1 if np.isnan(spd2)==True
                               # use spd1 if 2 is in mast shadow
                               else (spd1 if Dir >= inflowlow2 and Dir <= inflowhigh2
                                  # use spd2 if 1 is in mast shadow ('left' of 360)
                                  else (spd2 if Dir >= inflowlow1 and Dir <= 360
                                        # use spd2 if 1 is in mast shadow ('right' of 0)
                                        else (spd2 if Dir >= 0 and Dir <= inflowhigh1
                                              # otherwise, selective average
                                              else (spd1 + spd2)/2)))),
                           # end of map function, list input variables
                           wspd1,wspd2,wdir,inflow_lower1,inflow_higher1,inflow_lower2,inflow_higher2))

    # if boom 2 'inflow' sector overlaps with 0/360
    elif ((boom_dir2 + 180) % 360) >= (360 - (sector_width/2)) or ((boom_dir2 + 180) % 360) <= (sector_width/2):
        # many nested if statments follow, all within one mapped lambda function
        sel_avg = list(map(lambda spd1,spd2,Dir,inflowlow1,inflowhigh1,inflowlow2,inflowhigh2:
                           # if one value is Nan, use the other one
                           spd2 if (np.isnan(spd1)==True) else (spd1 if np.isnan(spd2)==True
                               # use spd2 if 1 is in mast shadow
                               else (spd2 if (Dir >= inflowlow1 and Dir <= inflowhigh1)
                                  # use spd1 if 2 is in mast shadow ('left' of 360)
                                  else (spd1 if (Dir >= inflowlow2 and Dir <= 360)
                                        # use spd1 if 2 is in mast shadow ('right' of 0)
                                        else (spd1 if (Dir >= 0 and Dir <= inflowhigh2)
                                              # otherwise, selective average
                                              else (spd1 + spd2)/2)))),
                           # end of map function, list input variables
                           wspd1,wspd2,wdir,inflow_lower1,inflow_higher1,inflow_lower2,inflow_higher2))
    # if neither boom 'inflow' sectors overlap with 0/360 threshold
    else:
        # many nested if statements follow, all within one mapped lambda function
        sel_avg = list(map(lambda spd1, spd2, Dir, inflowlow1, inflowhigh1, inflowlow2, inflowhigh2:
                           # if one value is Nan, use the other one
                           spd2 if np.isnan(spd1)==True else (spd1 if np.isnan(spd2)==True
                              # use spd2 if 1 is in mast shadow
                              else (spd2 if (Dir >= inflowlow1 and Dir <= inflowhigh1)
                                    # use spd1 if 2 is in mast shadow and spd1 is not Nan
                                    else (spd1 if (Dir >= inflowlow2 and Dir <= inflowhigh2)
                                           # otherwise, selective average
                                           else (spd1 + spd2) / 2))),
                            # end of map function, list input variables
                            wspd1,wspd2,wdir,inflow_lower1,inflow_higher1,inflow_lower2,inflow_higher2))
    return sel_avg


def _calc_sector_limits(boom_dir, sector_width):
    inflow_lower = (boom_dir - (sector_width / 2) + 180) % 360
    inflow_higher = (boom_dir + (sector_width / 2) + 180) % 360
    return inflow_lower, inflow_higher


def _sectors_overlap(boom_dir_1, boom_dir_2, sector_width):
    # return True if sectors overlap, False if they don't
    if boom_dir_1 >= boom_dir_2:
        dif = (boom_dir_1 - boom_dir_2) % 360
    else:
        dif = (boom_dir_2 - boom_dir_1) % 360
    if dif > 180:
        dif = 360 - dif
    if dif < sector_width:
        return True
    else:
        return False


def selective_avg(wspd_1, wspd_2, wdir, boom_dir_1, boom_dir_2, sector_width=60):
    """
    Creates a time series of wind speed using data from two anemometers (ideally at the same height) and one wind vane.
    This function either averages the two wind speed values for a given timestamp or only includes the upstream wind
    speed value when the other is in the wake of the mast.

    :param wspd_1: First wind speed time series
    :type wspd_1: pandas.Series
    :param wspd_2: Second wind speed time series
    :type wspd_2: pandas.Series
    :param wdir: Wind direction time series
    :type wdir: pandas.Series
    :param boom_dir_1: Boom direction in degrees of wspd_1
    :type boom_dir_1: float
    :param boom_dir_2: Boom direction in degrees of wspd_2
    :type boom_dir_1: float
    :param sector_width: Angular width of upstream sector within which a boom is deemed to be in the wake of the mast.
    :type sector_width: float
    :return: list of speed values for each timestamp of the input time series
    :rtype: list

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)

        # Normal use: Derive selective average of 80 m anemometer pair
        data['sel_avg_80m'] = bw.selective_avg(data.Spd80mN, data.Spd80mS, wdir=data.Dir78mS,
                                               boom_dir_1=0, boom_dir_2=180, sector_width=60)

        # When boom directions are specified too close to each other, the 'wake' sectors of each boom are found to
        # overlap.
        data['sel_avg_80m'] = bw.selective_avg(data.Spd80mN, data.Spd80mS, wdir=data.Dir78mS,
                                               boom_dir_1=0, boom_dir_2=50, sector_width=60)
        # This will result in the following error:
        # "Sectors overlap! Please check your inputs or reduce the size of your 'sector_width'."


    """
    wspd_1 = utils._convert_df_to_series(wspd_1)
    wspd_2 = utils._convert_df_to_series(wspd_2)
    wdir = utils._convert_df_to_series(wdir)
    if _sectors_overlap(boom_dir_1, boom_dir_2, sector_width):
        raise ValueError("Sectors overlap! Please check your inputs or reduce the size of your 'sector_width'.")
    inflow_lower1, inflow_higher1 = _calc_sector_limits(boom_dir_1, sector_width)
    inflow_lower2, inflow_higher2 = _calc_sector_limits(boom_dir_2, sector_width)
    sel_avg = _selective_avg(wspd_1, wspd_2, wdir, boom_dir_1, boom_dir_2,
                             inflow_lower1, inflow_higher1, inflow_lower2, inflow_higher2, sector_width)
    return sel_avg


def offset_timestamps(data, offset, date_from=None, date_to=None, overwrite=False):
    """
    Offset timestamps by a certain time period

    :param data: DateTimeIndex or Series/DataFrame with DateTimeIndex
    :type data: pandas.DateTimeIndex, pandas.Series, pandas.DataFrame
    :param offset: A string specifying the time to offset the time-series.

            - Set offset to 10min to add 10 minutes to each timestamp, -10min to subtract 10 minutes and so on
                for 4min, 20min, etc.
            - Set offset to 1H to add 1 hour to each timestamp and -1H to subtract and so on for 5H, 6H, etc.
            - Set offset to 1D to add a day and -1D to subtract and so on for 5D, 7D, 15D, etc.
            - Set offset to 1W to add a week and -1W to subtract from each timestamp and so on for 2W, 4W, etc.
            - Set offset to 1M to add a month and -1M to subtract a month from each timestamp and so on for 2M, 3M, etc.
            - Set offset to 1Y to add an year and -1Y to subtract an year from each timestamp and so on for 2Y, 3Y, etc.

    :type offset: str
    :param date_from: (Optional) The timestamp from input data where to start offsetting from.
    :type date_from: str, datetime, dict
    :param date_to: (Optional) The timestamp from input data where to end offsetting.
    :type date_to: str, datetime, dict
    :param overwrite: Change to True to overwrite the unadjusted timestamps if they are same outside of the slice of
        data you want to offset. False by default.
    :type overwrite: bool
    :returns: Offsetted DateTimeIndex/Series/DataFrame, same format is input data

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_campbell_scientific(bw.demo_datasets.demo_site_data)

        #To decrease 10 minutes within a given date range and overwrite the original data
        op1 = bw.offset_timestamps(data, offset='1H', date_from='2016-01-01 00:20:00',
            date_to='2016-01-01 01:40:00', overwrite=True)

        #To decrease 10 minutes within a given date range not overwriting the original data
        op2 = bw.offset_timestamps(data, offset='-10min', date_from='2016-01-01 00:20:00',
            date_to='2016-01-01 01:40:00')

        #Can accept Series or index as input
        op3 = bw.offset_timestamps(data.Spd80mS, offset='1D', date_from='2016-01-01 00:20:00')

        op4 = bw.offset_timestamps(data.index, offset='-10min', date_from='2016-01-01 00:20:00',
            date_from='2016-01-01 01:40:00')

        #Can also except decimal values for offset, like 3.5H for 3 hours and 30 minutes

        op5 = bw.offset_timestamps(data.index, offset='3.5H', date_from='2016-01-01 00:20:00',
            date_from='2016-01-01 01:40:00')

    """
    import datetime
    date_from = pd.to_datetime(date_from)
    date_to = pd.to_datetime(date_to)

    if isinstance(data, pd.Timestamp) or isinstance(data, datetime.date)\
            or isinstance(data, datetime.time)\
            or isinstance(data, datetime.datetime):
        return data + pd.Timedelta(offset)

    if isinstance(data, pd.DatetimeIndex):
        original = pd.to_datetime(data.values)

        if pd.isnull(date_from):
            date_from = data[0]

        if pd.isnull(date_to):
            date_to = data[-1]

        shifted_slice = original[(original >= date_from) & (original <= date_to)] + pd.Timedelta(offset)
        shifted = original[original < date_from].append(shifted_slice)
        shifted = shifted.append(original[original > date_to])
        shifted = shifted.drop_duplicates().sort_values()
        return pd.DatetimeIndex(shifted)

    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):

        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError('Input must have datetime index')
        else:
            original = pd.to_datetime(data.index.values)
            df_copy = data.copy(deep=False)
            if pd.isnull(date_from):
                date_from = data.index[0]

            if pd.isnull(date_to):
                date_to = data.index[-1]

            shifted_slice = original[(original >= date_from) & (original <= date_to)] + pd.Timedelta(offset)
            intersection_front = original[(original < date_from)].intersection(shifted_slice)
            intersection_back = original[(original > date_to)].intersection(shifted_slice)
            if overwrite:
                df_copy = df_copy.drop(intersection_front, axis=0)
                df_copy = df_copy.drop(intersection_back, axis=0)
                sec1 = original[original < date_from].drop(intersection_front)
                sec2 = original[original > date_to].drop(intersection_back)
                shifted = (sec1.append(shifted_slice)).append(sec2)
            else:
                df_copy = df_copy.drop(intersection_front - pd.Timedelta(offset), axis=0)
                df_copy = df_copy.drop(intersection_back - pd.Timedelta(offset), axis=0)
                sec_mid = shifted_slice.drop(intersection_front).drop(intersection_back)
                shifted = (original[(original < date_from)].append(sec_mid)).append(original[(original > date_to)])
            df_copy.index = shifted
            return df_copy.sort_index()
