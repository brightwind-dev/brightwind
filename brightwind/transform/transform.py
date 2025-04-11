import datetime
import numpy as np
import pandas as pd
from brightwind.utils import utils
from brightwind.load.station import _Measurements
from brightwind.load.station import DATE_INSTEAD_OF_NONE
from brightwind.utils.utils import validate_coverage_threshold
import copy
import warnings

__all__ = ['average_data_by_period',
           'merge_datasets_by_period',
           'average_wdirs',
           'adjust_slope_offset',
           'scale_wind_speed',
           'apply_wind_vane_deadband_offset',
           'offset_wind_direction',
           'selective_avg',
           'offset_timestamps',
           'apply_wspd_slope_offset_adj',
           'apply_device_orientation_offset']


def _compute_wind_vector(wspd, wdir):
    """
    Returns north and east component of wind-vector
    """
    return wspd*np.cos(wdir), wspd*np.sin(wdir)


def _freq_str_to_dateoffset(period):
    """
    Convert a pandas frequency string to a pd.DateOffset.

    Pandas frequency strings are available here:
    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

    :param period: Frequency string to be converted to a pd.DateOffset
    :type period:  str
    :return:       A pd.DateOffset
    :rtype:        pd.DateOffset
    """
    if period[-1] == 'M':
        as_dateoffset = pd.DateOffset(months=int(period[:-1]))
    elif period[-2:] == 'MS':
        as_dateoffset = pd.DateOffset(months=int(period[:-2]))
    elif period[-1] == 'A':
        as_dateoffset = pd.DateOffset(years=float(period[:-1]))
    elif period[-2:] == 'AS':
        as_dateoffset = pd.DateOffset(years=float(period[:-2]))
    elif period[-1:] == 'W':
        as_dateoffset = pd.DateOffset(weeks=float(period[:-1]))
    elif period[-1:] == 'D':
        as_dateoffset = pd.DateOffset(days=float(period[:-1]))
    elif period[-1:] == 'H':
        as_dateoffset = pd.DateOffset(hours=float(period[:-1]))
    elif period[-1:] == 'T':
        as_dateoffset = pd.DateOffset(minutes=float(period[:-1]))
    elif period[-3:] == 'min':
        as_dateoffset = pd.DateOffset(minutes=float(period[:-3]))
    elif period[-1:] == 'S':
        as_dateoffset = pd.DateOffset(seconds=float(period[:-1]))
    else:
        raise ValueError('"{}" period not recognized. Only units "M", "MS", "A", "AS", "W", "D", "H", "T", "min", "S" '
                         'are recognized'.format(period))
    return as_dateoffset


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
    most common time frequency. The expected frequency will be one of 'seconds', 'minutes', 'hours', 'days',
    'weeks', 'months', 'years'.

    The function also checks the most common time difference against the minimum time difference. If they
    do not match it shows a warning. It is suggested to manually look at the data if such a warning is shown.

    :param data_idx: Timeseries index of a pd.DataFrame or pd.Series.
    :type data_idx:  pd.DataFrame.index or pd.Series.index
    :return:         A date offset object which represents the time difference between consecutive timestamps.
    :rtype:          pd.DateOffset

    **Example usage**
    ::
        import brightwind as bw
        import pandas as pd
        data = bw.load_csv(bw.demo_datasets.demo_data)
        resolution = bw.transform.transform._get_data_resolution(data.Spd80mS.index)

        # To check the number of seconds in resolution
        print((data.index[0] + resolution - data.index[0]).seconds)

        # To check the unit of resolution
        print(resolution.kwds)

        # To check if the resolution is monthly
        resolution == pd.DateOffset(months=1)

    """
    # ** Strongly suggestion using pandas infer_freq function for this in future revision.
    time_diff_btw_timestamps = data_idx.to_series().diff()
    most_freq_time_diff = time_diff_btw_timestamps.mode().values[0]
    most_freq_time_diff = pd.to_timedelta(most_freq_time_diff)  # convert np.timedelta64 to pd.Timedelta
    minimum_time_diff = time_diff_btw_timestamps.min()

    if most_freq_time_diff.days in [28, 29, 30, 31]:    # check if monthly first
        return pd.DateOffset(months=1)
    elif most_freq_time_diff.days in [365, 366]:        # then if yearly
        return pd.DateOffset(years=1)
    
    if minimum_time_diff != most_freq_time_diff:
        warnings.warn('\nFrequency of input data may not be determined correctly. Most frequent time '
                      'difference between adjacent timestamps does not match minimum time difference.\n'
                      'Most frequent time difference is:\t{0}\n'
                      'Minimum time difference is:\t\t{1}\n'
                      'Returning most frequent time difference.'.format(most_freq_time_diff, minimum_time_diff))

    if most_freq_time_diff.days >= 1 and most_freq_time_diff.days < 28:
        return pd.DateOffset(days=most_freq_time_diff.total_seconds()/(60.*60.*24))
    elif most_freq_time_diff.days < 1 and most_freq_time_diff.total_seconds() >= 60*60:
        return pd.DateOffset(hours=most_freq_time_diff.total_seconds()/(60.*60))
    elif most_freq_time_diff.total_seconds() < (60*60):
        return pd.DateOffset(minutes=most_freq_time_diff.total_seconds()/60.)
    else:
        return pd.DateOffset(seconds=most_freq_time_diff.total_seconds())


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
    # If the start timestamp just happens to be missing from the data, add in a NaN so the
    # averaging will start from this timestamp.
    if not (df2.index == start).any():
        if type(df2) == pd.DataFrame:
            df2 = pd.concat([df2, pd.DataFrame({cols: [np.NaN] for cols in df2.columns},
                                               index=[pd.to_datetime(start)])])
        else:
            df2[pd.to_datetime(start)] = np.NaN
        df2.sort_index(inplace=True)
    if not (df1.index == start).any():
        # df1.loc[pd.to_datetime(start)] = np.NaN
        if type(df1) == pd.DataFrame:
            df1 = pd.concat([df1, pd.DataFrame({cols: [np.NaN] for cols in df1.columns},
                                               index=[pd.to_datetime(start)])])
        else:
            df1[pd.to_datetime(start)] = np.NaN
        df1.sort_index(inplace=True)
    return df1[start:], df2[start:]


def _max_coverage_count(data_index, averaged_data_index, data_resolution=None):
    """
    For a given resolution of data finds the maximum number of data points in the averaging period

    :param data_index:          The index of a Pandas Dataframe or Series to find the maximum number of data points for.
    :type  data_index:          pd.Index
    :param averaged_data_index: The index of the averaged grouped object.
    :type  averaged_data_index: pd.Index
    :param data_resolution:     Data resolution to give as input if the coverage of the data timeseries is extremely low
                                and it is not possible to define the most common time interval between timestamps
    :type data_resolution:      None or pd.DateOffset
    :return max_pts:            The maximum number of data points in the averaging period.
    :rtype:                     np.float64

    """

    if data_resolution is None:
        data_res = _get_data_resolution(data_index)
    elif not isinstance(data_resolution, pd.DateOffset):
        raise TypeError('Input data_resolution is {}. A Pandas DateOffset should be used instead.'.format(
            type(data_resolution).__name__))
    else:
        data_res = data_resolution

    averaged_data_res = averaged_data_index.freq or _get_data_resolution(averaged_data_index)

    time_delta = averaged_data_index.map(lambda x: x + data_res) - averaged_data_index
    max_pts = (averaged_data_index.map(lambda x: x + averaged_data_res) - averaged_data_index)/time_delta

    return max_pts


def _get_coverage_by_grouper_obj(data, grouper_obj, data_resolution=None):
    """
    Counts the number of data points in the grouper_obj relative to the maximum possible

    :param data:            The data to find the coverage for.
    :type  data:            pd.DataFrame or pd.Series
    :param grouper_obj:     The object that has grouped the data already. The mean, sum, count, etc. can then be found.
    :type  grouper_obj:     pd.DatetimeIndexResampler
    :param data_resolution: Data resolution to give as input if the coverage of the data timeseries is extremely low
                            and it is not possible to define the most common time interval between timestamps
    :type data_resolution:  None or pd.DateOffset
    :return:
    """
    coverage = grouper_obj.count().divide(_max_coverage_count(data.index, grouper_obj.mean().index,
                                                              data_resolution=data_resolution), axis=0)
    return coverage


def average_data_by_period(data, period, wdir_column_names=None, aggregation_method='mean', coverage_threshold=None,
                           return_coverage=False, data_resolution=None):
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
    :type coverage_threshold:  float, int or None
    :param return_coverage:    If True appends and additional column in the DataFrame returned, with coverage calculated
                               for each period. The columns with coverage are named as <column name>_Coverage
    :type return_coverage:     bool
    :param data_resolution:    Data resolution to give as input if the coverage of the data timeseries is extremely low
                               and it is not possible to define the most common time interval between timestamps
    :type data_resolution:     None or pd.DateOffset
    :returns:                  A DataFrame with data aggregated with the specified aggregation_method (mean by default).
                               Additionally it could be filtered based on coverage and have a coverage column depending
                               on the parameters.
    :rtype:                    pd.DataFrame

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

        # To check the coverage for all months giving as input the data resolution as 10 min if data coverage is
        # extremely low and it is not possible to define the most common time interval between timestamps
        data_monthly_irregular = bw.average_data_by_period(data.Spd80mN, period='1M', return_coverage=True,
                                                           data_resolution=pd.DateOffset(minutes=10))

    """
    coverage_threshold = validate_coverage_threshold(coverage_threshold)

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
    if data_resolution is None:
        if data.index[0] + _freq_str_to_dateoffset(period) < data.index[0] + _get_data_resolution(data.index):
            raise ValueError("The time period specified is less than the temporal resolution of the data. "
                             "For example, hourly data should not be averaged to 10 minute data.")
    data = data.sort_index()
    grouper_obj = data.resample(period, axis=0, closed='left', label='left',
                                convention='start', kind='timestamp')

    # if period is equal to data resolution then no need to vector average wind direction
    is_period_not_equal_to_resolution = (_freq_str_to_dateoffset(period) != _get_data_resolution(data.index))
    if wdir_column_names is not None and aggregation_method == 'mean' and is_period_not_equal_to_resolution:
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
            wdir_coverage = _get_coverage_by_grouper_obj(data[wdir_column_names], grouper_obj[wdir_column_names],
                                                         data_resolution=data_resolution)
        else:
            wdir_grouped_data = grouper_obj.agg(average_wdirs)
            wdir_coverage = _get_coverage_by_grouper_obj(data, grouper_obj, data_resolution=data_resolution)

        # average non_wdir data if available
        if non_wdir_col_names:
            non_wdir_grouped_data = grouper_obj[non_wdir_col_names].agg('mean')
            non_wdir_coverage = _get_coverage_by_grouper_obj(data[non_wdir_col_names], grouper_obj[non_wdir_col_names],
                                                             data_resolution=data_resolution)

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
        coverage = _get_coverage_by_grouper_obj(data, grouper_obj, data_resolution=data_resolution)
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
                             coverage_threshold_1=None, coverage_threshold_2=None, data_1_resolution=None,
                             data_2_resolution=None):
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
    :param data_1_resolution:    Data resolution of first dataset to give as input if the coverage of the data
                                 timeseries is extremely low and it is not possible to define the most common time
                                 interval between timestamps
    :type data_1_resolution:     None or pd.DateOffset
    :param data_2_resolution:    Data resolution of second dataset to give as input if the coverage of the data
                                 timeseries is extremely low and it is not possible to define the most common
                                 time interval between timestamps
    :type data_2_resolution:     None or pd.DateOffset
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
        # To find monthly concurrent averages of two datasets giving as input the data resolution as 1 hour for data_1
        # and 10 min for data_2 if data coverage is extremely low and it is not possible to define the most common time
        # interval between timestamps for each dataset.
        mrgd_data = bw.merge_datasets_by_period(m2[['WS50m_m/s', 'WD50m_deg']], data[['Spd80mN', 'Dir78mS']],
                                                period='1MS',
                                                wdir_column_names_1='WD50m_deg', wdir_column_names_2='Dir78mS',
                                                coverage_threshold_1=0, coverage_threshold_2=0,
                                                aggregation_method_1='mean', aggregation_method_2='mean',
                                                data_1_resolution=pd.DateOffset(hours=1),
                                                data_2_resolution=pd.DateOffset(minutes=10))

    """
    data_1_overlap, data_2_overlap = _get_overlapping_data(data_1.sort_index().dropna(),
                                                           data_2.sort_index().dropna(),
                                                           period)
    coverage_threshold_1 = validate_coverage_threshold(coverage_threshold_1)
    coverage_threshold_2 = validate_coverage_threshold(coverage_threshold_2)

    mrgd_data = pd.concat(list(average_data_by_period(data_1_overlap, period=period,
                                                      wdir_column_names=wdir_column_names_1,
                                                      coverage_threshold=coverage_threshold_1,
                                                      aggregation_method=aggregation_method_1,
                                                      return_coverage=True, data_resolution=data_1_resolution)) +
                          list(average_data_by_period(data_2_overlap, period=period,
                                                      wdir_column_names=wdir_column_names_2,
                                                      coverage_threshold=coverage_threshold_2,
                                                      aggregation_method=aggregation_method_2,
                                                      return_coverage=True, data_resolution=data_2_resolution)),
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


def _get_consistent_properties_format(measurements, measurement_type_id):
    """
    From the options of:
        mm1.measurements
        mm1.measurements.wdirs
        mm1.measurements['Dir78mS']
    return a consistent list of properties for the measurement_type_id.

    :param measurements:        Measurement information extracted from a WRA Data Model using bw.MeasurementStation
    :type measurements:         list or dict or _Measurements
    :param measurement_type_id: The measurement_type_id to filter for.
    :type measurement_type_id:  str
    :return:                    All the properties as a list
    :rtype:                     list
    """
    property_list = []
    if isinstance(measurements, _Measurements):
        merged_properties = copy.deepcopy(measurements.properties)
        for meas_point in merged_properties:
            meas_type = meas_point.get('measurement_type_id')
            if meas_type is not None and meas_type == measurement_type_id:
                property_list.append(meas_point)
    elif isinstance(measurements, dict):
        for name, properties in measurements.items():
            for prop in properties:
                meas_type = prop.get('measurement_type_id')
                if meas_type is not None and meas_type == measurement_type_id:
                    property_list.append(prop)
    elif isinstance(measurements, list):
        for prop in measurements:
            meas_type = prop.get('measurement_type_id')
            if meas_type is not None and meas_type == measurement_type_id:
                property_list.append(prop)
    return property_list


def apply_wspd_slope_offset_adj(data, measurements, inplace=False):
    """
    Automatically apply wind speed calibration slope and offset adjustments to the timeseries data when they
    differ from the logger programmed slope and offsets. The slope and offset information for each measurement
    and time period is contained in the measurements instance from bw.MeasurementStation.

    This uses the brightwind 'adjust_slope_offset()' function to apply the actual adjustment to the data.

    Note: Be careful not to run this more than once in an assessment, when using inplace=True, as it will
          apply the adjustment again.

    :param data:         Timeseries data.
    :type data:          pd.DataFrame or pd.Series
    :param measurements: Measurement information extracted from a WRA Data Model using bw.MeasurementStation
    :type measurements:  list or dict or _Measurements
    :param inplace:      If 'inplace' is True, the original direction data, contained in 'data', will be
                         modified and replaced with the adjusted direction data. If 'inplace' is False, the
                         original data will not be touched and instead a new DataFrame containing the adjusted
                         direction data is created. To store this adjusted direction data, please ensure it is
                         assigned to a new variable.
    :type inplace:       bool
    :return:             Data with adjusted wind speeds.
    :rtype:              pd.DataFrame or pd.Series

    **Example usage**
    ::
        import brightwind as bw

        mm1 = bw.MeasurementStation(bw.demo_datasets.demo_wra_data_model)
        data = bw.load_csv(bw.demo_datasets.demo_data)

    Adjust wind speeds when the calibration slope and offset differ from the logger
    programmed slope and offset, applying inplace::
        bw.apply_wspd_slope_offset_adj(data, mm1.measurements, inplace=True)

    Adjust wind speeds and assign to new variable::
        data_calib_adj = bw.apply_wspd_slope_offset_adj(data, mm1.measurements)

    Send just the wind speed properties::
        bw.apply_wspd_slope_offset_adj(data, mm1.measurements.wspds, inplace=True)

    Send a specific wind speed property::
        bw.apply_wspd_slope_offset_adj(data, mm1.measurements['Spd60mS'], inplace=True)

    Send a specific wind direction property and data column::
        bw.apply_wspd_slope_offset_adj(data['Spd60mS'], mm1.measurements['Spd60mS'], inplace=True)

    """
    # Depending on what is sent, get wspd properties into a list of properties
    wspd_properties = _get_consistent_properties_format(measurements, 'wind_speed')
    if not wspd_properties:
        raise ValueError('No wind speed measurements found.')

    # copy the data if needed
    data = data.copy(deep=True) if inplace is False else data
    wspd_in_dataset = False
    df = pd.DataFrame(data) if type(data) == pd.Series else data

    # Apply the adjustment
    for wspd_prop in wspd_properties:
        name = wspd_prop['name']
        if name in df.columns:
            wspd_in_dataset = True
            date_to = wspd_prop.get('date_to')
            date_from = wspd_prop.get('date_from')
            if date_to is None or date_to == DATE_INSTEAD_OF_NONE:
                date_to_txt = 'the end of dataset'
            else:
                date_to_txt = date_to

            variables = {
                'slope': 'logger_measurement_config.slope',
                'offset': 'logger_measurement_config.offset',
                'cal_slope': 'calibration.slope',
                'cal_offset': 'calibration.offset'
            }
            none_variables = [v for v in variables.values() if wspd_prop.get(v) is None]

            if none_variables:
                print("{} has {} value set as None. Slope and offset adjustment can't be applied "
                      "from {} to {}.\n".format(utils.bold(name), utils.bold(', '.join(none_variables)),
                                                utils.bold(date_from),
                                                utils.bold(date_to_txt)))
            elif float(wspd_prop[variables['slope']]) != float(wspd_prop[variables['cal_slope']]) or \
                    float(wspd_prop[variables['offset']]) != float(wspd_prop[variables['cal_offset']]):
                try:
                    df.loc[date_from:date_to, name] = \
                        adjust_slope_offset(df[name][date_from:date_to],
                                            current_slope=float(wspd_prop[variables['slope']]),
                                            current_offset=float(wspd_prop[variables['offset']]),
                                            new_slope=float(wspd_prop[variables['cal_slope']]),
                                            new_offset=float(wspd_prop[variables['cal_offset']]))
                    print('{} has slope and offset adjustment applied from {} to {}.\n'
                          .format(utils.bold(name), utils.bold(date_from),
                                  utils.bold(date_to_txt)))
                except TypeError:
                    print('{} has TypeError with logger or calibration slope and offset values. Skipping.\n'
                          .format(utils.bold(name)))
                except Exception as error_msg:
                    print(error_msg)
            else:
                print('{} logger slope and offsets are equal to calibration slope and offsets from '
                      '{} to {}.\n'.format(utils.bold(name),
                                           utils.bold(date_from),
                                           utils.bold(date_to_txt)))
        else:
            print('{} is not found in data.\n'.format(utils.bold(name)))

    if wspd_in_dataset is False:
        print('No wind speed measurement type found in the configurations.')
    # if a Series is sent, send back a Series
    if type(data) == pd.Series:
        df = df[df.columns[0]]
    return df


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
    Add or subtract offset from wind direction. Keeps the ranges between 0 to 360

    :param wdir: Series or data frame or a single direction to offset
    :param offset: Offset in degrees can be negative or positive
    :return: Series or Dataframe or single value with offsetted directions
    """
    if isinstance(wdir, float) or isinstance(wdir, int):
        return utils._range_0_to_360(wdir + offset)
    elif isinstance(wdir, pd.DataFrame):
        return wdir.add(offset).applymap(utils._range_0_to_360)
    elif isinstance(wdir, pd.Series):
        return wdir.add(offset).apply(utils._range_0_to_360)


def apply_wind_vane_deadband_offset(data, measurements, inplace=False):
    """
    Automatically apply deadband offsets of the wind vanes to the timeseries data. The deadband orientation
    information for each wind direction measurement and time period is contained in the measurements
    instance from bw.MeasurementStation.

    This uses the brightwind 'offset_wind_direction()' function to apply the actual adjustment to the data.

    Note: Be careful not to run this more than once in an assessment, when using inplace=True, as it will
          apply an offset again.

    If there is a value in the logger for an offset, then the wind direction data has already been adjusted
    by this amount. This may or may not be equal to the deadband offset. Therefore, the adjustment to be made should
    make up the difference to equal a deadband offset. E.g.

            offset to be applied = deadband - logger offset

    This function accounts for this adjustment.

    :param data:         Timeseries data.
    :type data:          pd.DataFrame or pd.Series
    :param measurements: Measurement information extracted from a WRA Data Model using bw.MeasurementStation
    :type measurements:  list or dict or _Measurements
    :param inplace:      If 'inplace' is True, the original direction data, contained in 'data', will be
                         modified and replaced with the adjusted direction data. If 'inplace' is False, the
                         original data will not be touched and instead a new DataFrame containing the adjusted
                         direction data is created. To store this adjusted direction data, please ensure it is
                         assigned to a new variable.
    :type inplace:       bool
    :return:             Data with adjusted wind direction by the deadband orientation.
    :rtype:              pd.DataFrame or pd.Series

    **Example usage**
    ::
        import brightwind as bw

        mm1 = bw.MeasurementStation(bw.demo_datasets.demo_wra_data_model)
        data = bw.load_csv(bw.demo_datasets.demo_data)

    Adjust wind directions by the deadband orientation, applying inplace::
        bw.apply_wind_vane_deadband_offset(data, mm1.measurements, inplace=True)
        print('\nWind vane deadband offset adjustment is completed.')

    Adjust wind direction by the deadband orientation, and assign to new variable::
        data_deadband_adj = bw.apply_wind_vane_deadband_offset(data, mm1.measurements)

    Send just the wind direction properties::
        bw.apply_wind_vane_deadband_offset(data, mm1.measurements.wdirs, inplace=True)
        print('\nWind vane deadband offset adjustment is completed.')

    Send a specific wind direction property::
        bw.apply_wind_vane_deadband_offset(data, mm1.measurements['Dir78mS'], inplace=True)
        print('\nWind vane deadband offset adjustment is completed.')

    Send a specific wind direction property and data column::
        bw.apply_wind_vane_deadband_offset(data['Dir78mS'], mm1.measurements['Dir78mS'], inplace=True)
        print('\nWind vane deadband offset adjustment is completed.')

    """
    # Depending on what is sent, get wdir properties into a list of properties
    wdirs_properties = _get_consistent_properties_format(measurements, 'wind_direction')
    if not wdirs_properties:
        raise ValueError('No wind direction measurements found.')

    # copy the data if needed
    data = data.copy(deep=True) if inplace is False else data
    wdir_in_dataset = False
    df = pd.DataFrame(data) if type(data) == pd.Series else data

    # Apply the offset
    for wdir_prop in wdirs_properties:
        name = wdir_prop['name']
        if name in df.columns:
            wdir_in_dataset = True
            date_to = wdir_prop.get('date_to')
            if date_to is None or date_to == DATE_INSTEAD_OF_NONE:
                date_to_txt = 'the end of dataset'
            else:
                date_to_txt = date_to

            deadband = wdir_prop.get('vane_dead_band_orientation_deg')
            date_from = wdir_prop['date_from']
            # Account for a logger offset
            logger_offset = wdir_prop.get('logger_measurement_config.offset')
            offset = deadband
            additional_comment_txt = 'to account for deadband'
            if logger_offset is not None and logger_offset != 0 and deadband is not None:
                offset = offset_wind_direction(float(deadband), offset=-float(logger_offset))
                additional_comment_txt = additional_comment_txt + ' and logger offset'

            if offset:
                df[name][date_from:date_to] = \
                    offset_wind_direction(df[name][date_from:date_to],
                                          float(offset))
                print('{0} adjusted by {1} degrees from {2} to {3} {4}.\n'
                      .format(utils.bold(name), utils.bold(str(offset)),
                              utils.bold(date_from), utils.bold(date_to_txt), additional_comment_txt))
            elif offset == 0:
                print('{} has an offset to be applied of 0 from {} to {} {}.\n'
                      .format(utils.bold(name), utils.bold(date_from), utils.bold(date_to_txt),
                              additional_comment_txt))
            else:
                print('{} has dead_band_orientation of None from {} to {}.\n'
                      .format(utils.bold(name), utils.bold(date_from), utils.bold(date_to_txt)))
        else:
            print('{} is not found in data.\n'.format(utils.bold(name)))

    if wdir_in_dataset is False:
        print('No wind direction measurement type found in the data.\n')
    # if a Series is sent, send back a Series
    if type(data) == pd.Series:
        df = df[df.columns[0]]
    return df


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

    :param data:        The timestamp or DateTimeIndex or Dataframe or Series to which apply the time offset.
    :type data:         pandas.DateTimeIndex, pandas.Series, pandas.DataFrame, pandas.Timestamp, datetime.datetime,
                        datetime.date, datetime.time
    :param offset:      A string specifying the time to offset the time-series.

                        - Set offset to 10min to add 10 minutes to each timestamp, -10min to subtract 10 minutes and so
                          on for 4min, 20min, etc.
                        - Set offset to 1H to add 1 hour to each timestamp and -1H to subtract and so on for 5H, 6H,
                          etc.
                        - Set offset to 1D to add a day and -1D to subtract and so on for 5D, 7D, 15D, etc.
                        - Set offset to 1W to add a week and -1W to subtract from each timestamp and so on for 2W,
                          4W, etc.
                        - Set offset to 1M to add a month and -1M to subtract a month from each timestamp and so on
                          for 2M, 3M, etc.
                        - Set offset to 1Y to add an year and -1Y to subtract an year from each timestamp and so on
                          for 2Y, 3Y, etc.

    :type offset:       str
    :param date_from:   (Optional) The timestamp from input data where to start offsetting from. Start date is
                        included in the offsetted data. If format of date_from is YYYY-MM-DD, then the first timestamp
                        of the date is used (e.g if date_from=2023-01-01 then 2023-01-01 00:00 is the first timestamp of
                        when to start offsetting from). If date_from is not given then the offset is applied from the
                        first timestamp of the dataset.
    :type date_from:    str, datetime, dict
    :param date_to:     (Optional) The timestamp from input data where to end offsetting. End date is not included in
                        the offsetted data. If format date_to is YYYY-MM-DD, then the last timestamp of the previous day
                        is used (e.g if date_to=2023-02-01 then 2023-01-31 23:50 is the last timestamp of when to end
                        offsetting to). If date_to is not given then the offset is applied up to the last timestamp of
                        the dataset.
    :type date_to:      str, datetime, dict
    :param overwrite:   Change to True to overwrite the unadjusted timestamps if they are same outside of the slice of
                        data you want to offset. False by default.
    :type overwrite:    bool
    :returns:           Offsetted Timestamp/DateTimeIndex/Series/DataFrame/datetime.datetime/datetime.time,
                        same format is input data

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)

        # To decrease 10 minutes within a given date range and overwrite the original data
        op1 = bw.offset_timestamps(data, offset='1H', date_from='2016-02-01 00:20:00',
            date_to='2016-02-01 01:40:00', overwrite=True)

        # To decrease 10 minutes within a given date range not overwriting the original data
        op2 = bw.offset_timestamps(data, offset='-10min', date_from='2016-02-01 00:20:00',
            date_to='2016-02-01 01:40:00')

        # To decrease 30 minutes within a given date range not overwriting the original data and giving as input dates
        # for date_from and date_to
        op2 = bw.offset_timestamps(DATA, offset='-30min', date_from='2016-01-09', date_to='2016-01-10')

        # Can accept Series or index as input
        op3 = bw.offset_timestamps(data.Spd80mS, offset='1D', date_from='2016-02-01 00:20:00')

        op4 = bw.offset_timestamps(data.index, offset='-10min', date_from='2016-02-01 00:20:00',
            date_to='2016-02-01 01:40:00')

        # Can also except decimal values for offset, like 3.5H for 3 hours and 30 minutes
        op5 = bw.offset_timestamps(data.index, offset='3.5H', date_from='2016-02-01 00:20:00',
            date_to='2016-02-01 01:40:00')

        # Can accept also Timestamp and datetime objects
        bw.offset_timestamps(data.index[0], offset='4H')
        bw.offset_timestamps(datetime.datetime(2016, 2, 1, 0, 20), offset='3.5H')
        bw.offset_timestamps(datetime.date(2016, 2, 1), offset='-5H')
        bw.offset_timestamps(datetime.time(0, 20), offset='30min')

    """

    if pd.isnull(date_from):
        if isinstance(data, pd.DatetimeIndex):
            date_from = data[0]
        elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
            date_from = data.index[0]
    else:
        date_from = pd.to_datetime(date_from)

    if pd.isnull(date_to):
        if isinstance(data, pd.DatetimeIndex):
            date_to = data[-1]
        elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
            date_to = data.index[-1]
    else:
        date_to = pd.to_datetime(date_to)

    if isinstance(data, pd.Timestamp):
        return data + _freq_str_to_dateoffset(offset)

    elif isinstance(data, datetime.date)\
            or isinstance(data, datetime.datetime):
        return (data + _freq_str_to_dateoffset(offset)).to_pydatetime()

    elif isinstance(data, datetime.time):
        return (datetime.datetime.combine(datetime.date.today(), data) + _freq_str_to_dateoffset(offset)).time()

    elif isinstance(data, pd.DatetimeIndex):
        original = pd.to_datetime(data.values)

        shifted_slice = original[(original >= date_from) & (original < date_to)] + _freq_str_to_dateoffset(offset)
        shifted = original[original < date_from].append(shifted_slice)
        shifted = shifted.append(original[original >= date_to])
        shifted = shifted.drop_duplicates().sort_values()
        return pd.DatetimeIndex(shifted)

    elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):

        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError('Input must have datetime index')
        else:
            original = pd.to_datetime(data.index.values)
            df_copy = data.copy(deep=False)

            shifted_slice = original[(original >= date_from) & (original < date_to)] + _freq_str_to_dateoffset(offset)
            intersection_front = original[(original < date_from)].intersection(shifted_slice)
            intersection_back = original[(original >= date_to)].intersection(shifted_slice)
            if overwrite:
                df_copy = df_copy.drop(intersection_front, axis=0)
                df_copy = df_copy.drop(intersection_back, axis=0)
                sec1 = original[original < date_from].drop(intersection_front)
                sec2 = original[original >= date_to].drop(intersection_back)
                shifted = (sec1.append(shifted_slice)).append(sec2)
            else:
                df_copy = df_copy.drop(intersection_front - _freq_str_to_dateoffset(offset), axis=0)
                df_copy = df_copy.drop(intersection_back - _freq_str_to_dateoffset(offset), axis=0)
                sec_mid = shifted_slice.drop(intersection_front).drop(intersection_back)
                shifted = (original[(original < date_from)].append(sec_mid)).append(original[(original >= date_to)])
            df_copy.index = shifted
            return df_copy.sort_index()


def apply_device_orientation_offset(data, measurement_station, wdir_cols=[], inplace=False):
    """
    Applies a device orientation offset to wind direction data from remote sensing devices
    (lidar, sodar, or floating lidar) to align measurements with north.

    The device's as-installed orientation is specified in
    `vertical_profiler_properties.device_orientation_deg` according to the IEA Wind Task 43
    WRA Data Model. This data model is represented using the `bw.MeasurementStation()` object.

    If `wdir_cols` is empty, the offset is applied to all wind direction columns reported in
    the measurement station. Otherwise, only the specified columns are adjusted.

    This function uses the brightwind 'offset_wind_direction()' function to apply the adjustment to the data.

    **Note:** When using `inplace=True`, be careful not to apply this function multiple times within
              the same assessment, as the offset will be applied again.

    If a logger offset is defined in `logger_measurement_config.logger_offset`, this indicates
    that the wind direction data has already been adjusted by this amount. This value may differ
    from `device_orientation_deg`, so the function calculates the difference to ensure the total
    adjustment equals the intended device orientation:

        offset_to_apply = device_orientation_deg - logger_offset

    If configuration 'date_from' is equal to previous configuration 'date_to' then date ranges are considered as
    [from, to) where 'from' is inclusive and 'to' is exclusive.

    Overlapping periods in 'vertical_profiler_properties' with non-zero device orientation values are not supported
    and will raise an error.

    :param data:                        Timeseries data.
    :type data:                         pd.DataFrame or pd.Series
    :param measurement_station:         A simplified object to represent the IEA WRA Task 43 data model.
    :type measurement_station:          brightwind.load.station.MeasurementStation
    :param wdir_cols:                   Wind direction column names to apply the offset to. If empty, all wind direction 
                                        columns in the data are used. Default is an empty list.
    :type wdir_cols:                    list
    :param inplace:                     If 'inplace' is True, the original direction data, contained in 'data', will be
                                        modified and replaced with the adjusted direction data. If 'inplace' is False, the
                                        original data will not be touched and instead a new DataFrame containing the adjusted
                                        direction data is created. To store this adjusted direction data, please ensure it is
                                        assigned to a new variable. Default is False
    :type inplace:                      bool, optional
    :return:                            Data with adjusted wind direction by the offset derived accounting for the
                                        orientation of the device relative to north.
    :rtype:                             pd.DataFrame or pd.Series
    
    **Example usage**
    ::
        fl1 = bw.MeasurementStation(bw.demo_datasets.floating_lidar_demo_iea43_wra_data_model_v1_3)
        data = bw.load_csv(bw.demo_datasets.demo_floating_lidar_data)

    Adjust only some wind directions by the device orientation, and assign to new variable:
        data_dev_orient_adj = bw.apply_device_orientation_offset(data, fl1, wdir_cols=['Dir_250m', 'Dir_200m'])

    ::
    Adjust all wind directions by the device orientation, applying inplace:
        bw.apply_device_orientation_offset(data, fl1, inplace=True)
        print('Wind direction device orientation offset adjustment is completed.')
    
    """
    
    if measurement_station.type not in ['lidar', 'floating_lidar', 'sodar']:
        raise ValueError(f"Device type: {measurement_station.type} is not supported for this function.")
    
    measurements = measurement_station.measurements
    wdirs_properties = _get_consistent_properties_format(measurements, 'wind_direction')
    # copy the data if needed
    data = data.copy(deep=True) if inplace is False else data
    wdir_not_in_dataset = False
    col_not_in_data = []
    col_not_in_datamodel = []
    df = pd.DataFrame(data) if isinstance(data, pd.Series) else data

    if wdir_cols:
        wdirs_properties_temp = []
        for col in wdir_cols:
            if col not in df.columns:
                wdir_not_in_dataset = True
                col_not_in_data.append(col)
            if not any(col == prop['name'] for prop in wdirs_properties):
                col_not_in_datamodel.append(col)
            else:
                # keep wind direction properties only for input wdir_cols
                wdirs_properties_temp.extend([prop for prop in wdirs_properties if col == prop['name']])
        wdirs_properties = wdirs_properties_temp

    _check_vertical_profiler_properties_overlap(measurement_station, df)

    # Apply the offset
    for wdir_prop in wdirs_properties:
        name = wdir_prop['name']
        if name in df.columns:
            date_to = wdir_prop.get('date_to')
            date_from = wdir_prop.get('date_from')
            date_from = (df.index[0].strftime('%Y-%m-%dT%H:%M:%S') 
                         if date_from is None or date_from == DATE_INSTEAD_OF_NONE else date_from)
            logger_offset = wdir_prop.get('logger_measurement_config.offset')
            for device_properties in measurement_station:
                meas_station_data_model_from = device_properties.get('date_from')
                meas_station_data_model_from = (df.index[0].strftime('%Y-%m-%dT%H:%M:%S') if
                                            meas_station_data_model_from is None or meas_station_data_model_from ==
                                            DATE_INSTEAD_OF_NONE else meas_station_data_model_from)
                meas_station_data_model_to = device_properties.get('date_to')
                
                if date_to is None or date_to == DATE_INSTEAD_OF_NONE:
                    date_to_tmp = meas_station_data_model_to
                else:
                    idx_pos = df.index.get_indexer([pd.Timestamp(date_to)], method='nearest')[0]
                    date_to_tmp = df.index[idx_pos + 1].strftime('%Y-%m-%dT%H:%M:%S') if idx_pos + 1 < len(df.index) else date_to

                if (((meas_station_data_model_to is None) or (date_from <= meas_station_data_model_to)) and 
                    ((date_to_tmp is None) or (meas_station_data_model_from is None) or 
                     (date_to_tmp >= meas_station_data_model_from))):
                    device_orientation_deg = device_properties.get('device_orientation_deg')
                    apply_offset_from = (date_from if date_from > meas_station_data_model_from 
                                         else meas_station_data_model_from)
                    if date_to_tmp is None or meas_station_data_model_to is None:
                        apply_offset_to = date_to_tmp if date_to_tmp is not None else meas_station_data_model_to
                    else:
                        apply_offset_to = min(date_to_tmp, meas_station_data_model_to)

                    df[name] = _apply_dir_offset_target_orientation(
                        df[name], logger_offset, device_orientation_deg, apply_offset_from, apply_offset_to,
                        target_orientation_name='device orientation')
        else:
            wdir_not_in_dataset = True
            col_not_in_data.append(name)
    
    if wdir_not_in_dataset:
        indexes = np.unique(col_not_in_data, return_index=True)[1]
        col_not_in_data = [col_not_in_data[index] for index in sorted(indexes)]
        print_text = 'Following wind direction measurement(s) not found in the data'
        if wdir_cols:
            print(print_text + ' for the requested `wdir_cols`: {}.'.format(utils.bold(str(col_not_in_data))))
        else:
            print(print_text + ': {}.'.format(utils.bold(str(col_not_in_data))))
    if col_not_in_datamodel:
        print('No device orientation offset applied to following requested measurement(s) as no wind direction '
              'measurement type found in `meas_station_data_models` for these: {}.'
              .format(utils.bold(str(col_not_in_datamodel))))
    # if a Series is sent, send back a Series
    if isinstance(data, pd.Series):
        df = df[df.columns[0]]
        data.update(df)
    return df


def _check_vertical_profiler_properties_overlap(measurement_station, df):
    """
    Checks if in vertical_profiler_properties there are any overlapping 
    date ranges with device orientation values.
    
    Date ranges are considered as [from, to) where 'from' is inclusive and 'to' is exclusive.
    Overlapping date ranges with device orientation values are not supported 
    by apply_device_orientation_offset function and will raise an error.
    
    :param measurement_station:         A simplified object to represent the IEA WRA Task 43 data model.
    :type measurement_station:          brightwind.load.station.MeasurementStation
    :param df:                          DataFrame with the time series data
    :type df:                           pd.DataFrame
    :raises ValueError:                 If overlapping date ranges with device orientation values are detected
    """
    date_ranges = []
    
    for device_properties in measurement_station:
        date_from = device_properties.get('date_from')
        if date_from is None or date_from == DATE_INSTEAD_OF_NONE:
            date_from = df.index[0].strftime('%Y-%m-%dT%H:%M:%S')
        date_to = device_properties.get('date_to')
        if date_to is None or date_to == DATE_INSTEAD_OF_NONE:
            date_to = df.index[-1].strftime('%Y-%m-%dT%H:%M:%S')
        device_orientation_deg = device_properties.get('device_orientation_deg', None)
        
        date_ranges.append({
            'from': pd.to_datetime(date_from),
            'to': pd.to_datetime(date_to),
            'device_orientation_deg': device_orientation_deg,
            'model': device_properties
        })
    
    for i, range1 in enumerate(date_ranges):
        for j, range2 in enumerate(date_ranges):
            if i >= j:
                continue
            # Check if date ranges overlap
            overlap = (range1['from'] < range2['to'] and range2['from'] < range1['to'])
            
            # If ranges overlap and at least one device_orientation_deg has a value different than None, raise error
            if overlap and (range1['device_orientation_deg'] is not None or
                            range2['device_orientation_deg'] is not None):
                raise ValueError(
                    f"Overlapping periods detected on vertical_profiler_properties with at least one " +
                    "device_orientation_deg value different than None: \n"
                    f"{range1['from']} to {range1['to']} (device_orientation_deg: {range1['device_orientation_deg']}°) "
                    f"and "
                    f"{range2['from']} to {range2['to']} (device_orientation_deg: {range2['device_orientation_deg']}°)"
                    f"\nthis is currently unsupported."
                )
    return False


def _apply_dir_offset_target_orientation(wdir_data, logger_offset, target_orientation, apply_offset_from,
                                         apply_offset_to, target_orientation_name):
    """
    Function to apply the required offset to the wind direction data based on the logger offset and a target
    orientation.
    Note that if `wdir_data` is a DataFrame, the adjustment derived from `logger_offset` and `target_orientation` 
    is applied to all columns.
    
    This function uses the brightwind 'offset_wind_direction()' function to apply the actual adjustment to 
    the wind direction data.

    If there is a value in the logger measurement config for an offset, then the wind direction data has already 
    been adjusted by this amount. This may or may not be equal to the target orientation. Therefore, 
    the adjustment to be made should make up the difference to equal a target orientation. E.g.

            offset to be applied = target_orientation - logger_offset

    This function accounts for this adjustment.

    Date ranges are considered as [from, to) where 'from' is inclusive and 'to' is exclusive.

    :param wdir_data:               The wind direction data time series.
    :type wdir_data:                pd.Series or pd.DataFrame
    :param logger_offset:           The logger offset value in degrees for the input wind direction data.
    :type logger_offset:            float
    :param target_orientation:      The target orientation value in degrees.
    :type target_orientation:       float
    :param apply_offset_from:       The date to apply the offset from.
    :type apply_offset_from:        str | datetime.datetime | pd.Timestamp
    :param apply_offset_to:         The date to apply the offset to, treated in and exclusive manner.
    :type apply_offset_to:          str | datetime.datetime | pd.Timestamp
    :param target_orientation_name: The target orientation name to use for the print statements. 
                                    e.g 'device orientation' or 'deadband orientation'
    :type target_orientation_name:  str
    """

    offset = target_orientation
    wdir_names = list(wdir_data.columns) if isinstance(wdir_data, pd.DataFrame) else wdir_data.name
    additional_comment_txt = 'to account for {}'.format(target_orientation_name)

    if apply_offset_to is None:
        to_text = "end of dataframe"
        mask = (wdir_data.index >= pd.Timestamp(apply_offset_from))
    else:
        mask = (wdir_data.index >= pd.Timestamp(apply_offset_from)) & (wdir_data.index < pd.Timestamp(apply_offset_to))
        idx_pos = wdir_data.index.get_indexer([pd.Timestamp(apply_offset_to)], method='nearest')[0]
        apply_offset_to_inclusive = wdir_data.index[idx_pos - 1].strftime('%Y-%m-%dT%H:%M:%S')
        to_text = f"{apply_offset_to}, exclusive but inclusive of {apply_offset_to_inclusive}"    

    if logger_offset is not None and logger_offset != 0 and target_orientation is not None:
        offset = offset_wind_direction(float(target_orientation), offset=-float(logger_offset))
        additional_comment_txt = additional_comment_txt + ' and logger offset'
    if offset:            
        # Apply offset only to the masked data
        wdir_data.loc[mask] = offset_wind_direction(wdir_data.loc[mask], float(offset))
        
        print('{0} adjusted by {1} degrees from {2} to {3} {4}.\n'
              .format(utils.bold(str(wdir_names)), utils.bold(str(offset)),
                      utils.bold(str(apply_offset_from)), utils.bold(to_text),
                      additional_comment_txt))
    elif offset == 0:            
        print('{0} has an offset to be applied of 0 degrees from {1} to {2} {3}.\n'
              .format(utils.bold(str(wdir_names)), utils.bold(str(apply_offset_from)),
                      utils.bold(to_text),
                      additional_comment_txt))
    else:            
        print('{0} has {1} as None from {2} to {3}.\n'
              .format(utils.bold(str(wdir_names)), target_orientation_name,
                      utils.bold(str(apply_offset_from)), utils.bold(to_text)))
    
    return wdir_data
