import copy  
import datetime  
import warnings  
from typing import Union  

import numpy as np  
import pandas as pd

import brightwind.transform.scale
from brightwind.utils import utils
from brightwind.load.station import _Measurements
from brightwind.load.station import MeasurementStation
from brightwind.load.station import DATE_INSTEAD_OF_NONE
from brightwind.utils.utils import validate_coverage_threshold

__all__ = ['average_data_by_period',
           'merge_datasets_by_period',
           'average_wdirs',
           'adjust_slope_offset',
           'scale_wind_speed',
           'apply_wind_vane_deadband_offset',
           'offset_wind_direction',
           'selective_avg',
           'offset_timestamps',
           'get_logger_utc_offsets',
           'convert_timestamps_to_utc',
           'apply_wspd_slope_offset_adj',
           'apply_device_orientation_offset']

_warned_a = False  # warning for pandas 'A' frequency string so only shows once
_warned_as = False  # warning for pandas 'AS' frequency string so only shows once
_warned_h = False  # warning for pandas 'H' frequency string so only shows once
_warned_t = False  # warning for pandas 'T' frequency string so only shows once
_warned_s = False  # warning for pandas 'S' frequency string so only shows once


def _compute_wind_vector(wspd, wdir):
    """
    Returns north and east component of wind-vector
    """
    return wspd*np.cos(wdir), wspd*np.sin(wdir)


def dataframe_map(df, func, **kwargs):
    """
    Apply a function element-wise to a DataFrame.
    
    Compatibility wrapper for DataFrame.map() (pandas >=2.1) and 
    DataFrame.applymap() (pandas <2.1).

    :param df:              DataFrame to apply function to
    :type df:               pd.DataFrame
    :param func:            Function to apply element-wise
    :type func:             Callable
    :return:                Transformed DataFrame
    :rtype:                 pd.DataFrame
    """
    if hasattr(df, 'map'):
        # pandas >= 2.1
        return df.map(func, **kwargs)
    else:
        # pandas < 2.1
        warnings.warn(
            "Support for pandas versions <2.1 is ending in a future brightwind version.",
            DeprecationWarning,
            stacklevel=2
        )
        return df.applymap(func, **kwargs)


def _normalize_freq_string(period):
    """
    Convert a deprecated pandas frequency string to its modern equivalent.

    Pandas frequency strings are available here:
    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

    :param period: Frequency string that may or may not be deprecated
    :type period:  str
    :return:       Frequency string in it's modern equivalent form where necessary.
    :rtype:        str
    """
    global _warned_a, _warned_as, _warned_h, _warned_t, _warned_s
    pandas_version = tuple(map(int, pd.__version__.split('.')[:2]))
    
    # Handle deprecated 'A' -> 'Y' (but 'YS' is preferred for year-start)
    if period.endswith('A') and not period.endswith('BA'):
        if not _warned_a:
            warnings.warn(
                "'A' frequency string is deprecated by Pandas v2.2.0 and will be removed in a future "
                "brightwind version.",
                DeprecationWarning,
                stacklevel=3
            )
            _warned_a = True
        if pandas_version < (2, 2):
            return period[:-1] + 'YS'
    
    # Handle deprecated 'AS' -> 'YS'
    if period.endswith('AS'):
        if not _warned_as:
            warnings.warn(
                "'AS' frequency string is deprecated by Pandas v2.2.0 and will be removed in a future "
                "brightwind version. "
                "Please use 'YS' instead.",
                DeprecationWarning,
                stacklevel=3
            )
            _warned_as = True
        if pandas_version < (2, 2):
            return period[:-2] + 'YS'
    
    # Handle deprecated 'H' -> 'h'
    if period.endswith('H'):
        if not _warned_h:
            warnings.warn(
                "'H' frequency string is deprecated by Pandas v2.2.0 and will be removed in a future "
                "brightwind version. "
                "Please use 'h' instead.",
                DeprecationWarning,
                stacklevel=3
            )
            _warned_h = True
        if pandas_version < (2, 2):
            return period[:-1] + 'h'
    
    # Handle deprecated 'T' -> 'min'
    if period.endswith('T'):
        if not _warned_t:
            warnings.warn(
                "'T' frequency string is deprecated by Pandas v2.2.0 and will be removed in a future "
                "brightwind version. "
                "Please use 'min' instead.",
                DeprecationWarning,
                stacklevel=3
            )
            _warned_t = True
        if pandas_version < (2, 2):
            return period[:-1] + 'min'
    
    # Handle deprecated 'S' -> 's' (but not 'MS', 'YS', 'AS', etc.)
    if period.endswith('S') and not any(period.endswith(x) for x in ['MS', 'YS', 'AS', 'NS']):
        if not _warned_s:
            warnings.warn(
                "'S' frequency string is deprecated by Pandas v2.2.0 and will be removed in a future "
                "brightwind version. "
                "Please use 's' instead.",
                DeprecationWarning,
                stacklevel=3
            )
            _warned_s = True
        if pandas_version < (2, 2):
            return period[:-1] + 's'
    
    return period


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
    global _warned_a
    global _warned_as
    global _warned_h
    global _warned_t
    global _warned_s

    if period[-1] == 'M':
        as_dateoffset = pd.DateOffset(months=int(period[:-1]))
    elif period[-2:] == 'MS':
        as_dateoffset = pd.DateOffset(months=int(period[:-2]))
    elif period[-1] == 'A':
        if not _warned_a:
            warnings.warn(
                "'A' frequency string is deprecated by Pandas v2.2.0 and will be removed in a future "
                "brightwind version.",
                DeprecationWarning,
                stacklevel=3
            )
            _warned_a = True
        as_dateoffset = pd.DateOffset(years=float(period[:-1]))
    elif period[-2:] == 'AS':
        if not _warned_as:
            warnings.warn(
                "'AS' frequency string is deprecated by Pandas v2.2.0 and will be removed in a future "
                "brightwind version. "
                "Please use 'YS' instead.",
                DeprecationWarning,
                stacklevel=3
            )
            _warned_as = True
        as_dateoffset = pd.DateOffset(years=float(period[:-2]))
    elif period[-2:] == 'YS':
        as_dateoffset = pd.DateOffset(years=float(period[:-2]))
    elif period[-1:] == 'W':
        as_dateoffset = pd.DateOffset(weeks=float(period[:-1]))
    elif period[-1:] == 'D':
        as_dateoffset = pd.DateOffset(days=float(period[:-1]))
    elif period[-1:] == 'H':
        if not _warned_h:
            warnings.warn(
                "'H' frequency string is deprecated by Pandas v2.2.0 and will be removed in a future "
                "brightwind version. "
                "Please use 'h' instead.",
                DeprecationWarning,
                stacklevel=3
            )
            _warned_h = True
        as_dateoffset = pd.DateOffset(hours=float(period[:-1]))
    elif period[-1:] == 'h':
        as_dateoffset = pd.DateOffset(hours=float(period[:-1]))
    elif period[-1:] == 'T':
        if not _warned_t:
            warnings.warn(
                "'T' frequency string is deprecated by Pandas v2.2.0 and will be removed in a future "
                "brightwind version. "
                "Please use 'min' instead.",
                DeprecationWarning,
                stacklevel=3
            )
            _warned_t = True
        as_dateoffset = pd.DateOffset(minutes=float(period[:-1]))
    elif period[-3:] == 'min':
        as_dateoffset = pd.DateOffset(minutes=float(period[:-3]))
    elif period[-1:] == 'S':
        if not _warned_s:
            warnings.warn(
                "'S' frequency string is deprecated by Pandas v2.2.0 and will be removed in a future "
                "brightwind version. "
                "Please use 's' instead.",
                DeprecationWarning,
                stacklevel=3
            )
            _warned_s = True
        as_dateoffset = pd.DateOffset(seconds=float(period[:-1]))
    elif period[-1:] == 's':
        as_dateoffset = pd.DateOffset(seconds=float(period[:-1]))
    else:
        raise ValueError('"{}" period not recognized. Only units "M", "MS", "YS", "W", "D", "h", "min", "s" '
                         'are recognized'.format(period))


    return as_dateoffset


def _convert_days_to_hours(prd):
    return str(int(prd[:-1])*24)+'h'


def _convert_weeks_to_hours(prd):
    return str(int(prd[:-1])*24*7)+'h'


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
    :param period:    Averaging period e.g. '10min', '1h', '3h', '6h', '1D', '7D', '1W', '1MS', '1YS'
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
    period = _normalize_freq_string(period)
    if period[-3:] == 'min':
        return '{year}-{month}-{day} {hour}:{minute}:00'.format(year=timestamp.year, month=timestamp.month,
                                                                day=timestamp.day, hour=timestamp.hour,
                                                                minute=_round_down_to_multiple(timestamp.minute,
                                                                                               int(period[:-3])))
    elif period[-1] == 'h' or period[-1] == 'H':
        return '{year}-{month}-{day} {hour}:00:00'.format(year=timestamp.year, month=timestamp.month, day=timestamp.day,
                                                          hour=_round_down_to_multiple(timestamp.hour,
                                                                                       int(period[:-1])))
    elif period[-1] == 'D' or period[-1] == 'W':
        return '{year}-{month}-{day}'.format(year=timestamp.year, month=timestamp.month, day=timestamp.day,
                                             hour=timestamp.hour)
    elif period[-1] == 'M' or period[-2:] == 'MS':
        return '{year}-{month}'.format(year=timestamp.year, month=timestamp.month)
    elif period[-2:] == 'YS' or period[-1:] == 'A' or period[-2:] == 'AS':
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
            df2 = pd.concat([df2, pd.DataFrame({cols: [np.nan] for cols in df2.columns},
                                               index=[pd.to_datetime(start)])])
        else:
            df2[pd.to_datetime(start)] = np.nan
        df2.sort_index(inplace=True)
    if not (df1.index == start).any():
        # df1.loc[pd.to_datetime(start)] = np.nan
        if type(df1) == pd.DataFrame:
            df1 = pd.concat([df1, pd.DataFrame({cols: [np.nan] for cols in df1.columns},
                                               index=[pd.to_datetime(start)])])
        else:
            df1[pd.to_datetime(start)] = np.nan
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

    time_delta = (averaged_data_index + data_res) - averaged_data_index
    max_pts = ((averaged_data_index + averaged_data_res) - averaged_data_index)/time_delta

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


def _wdir_sine_cosine(wdirs):
    """
    Sine (East) and cosine (North) components of each wind direction, rounded to 5 decimal places.

    The rounding is needed so that opposite directions cancel out. For example the sine of 180 is not quite zero, so
    without rounding 0 and 180 would average to 90 instead of NaN. Similarly 10 and 190 would give 270 and 90 and 270
    would give 0 or 180.

    :param wdirs: Wind directions in degrees.
    :type wdirs:  list or np.array or pd.Series or pd.DataFrame
    :return:      Rounded sine and cosine of each wind direction.
    :rtype:       tuple
    """
    wdirs_rad = np.deg2rad(wdirs)
    return np.round(np.sin(wdirs_rad), 5), np.round(np.cos(wdirs_rad), 5)


def _sine_cosine_to_wdir(sine, cosine):
    """
    Convert mean sine and cosine components into a wind direction between 0 and 360 degrees.

    If both sine and cosine are zero the directions cancel out and there is no wind direction so NaN is returned. 0 is
    returned instead of 360.

    :param sine:   Mean sine (East) component.
    :type sine:    float or pd.Series or pd.DataFrame
    :param cosine: Mean cosine (North) component.
    :type cosine:  float or pd.Series or pd.DataFrame
    :return:       Wind direction.
    :rtype:        float or pd.Series or pd.DataFrame
    """
    avg_dir = np.rad2deg(np.arctan2(sine, cosine)) % 360
    if isinstance(avg_dir, (pd.Series, pd.DataFrame)):
        avg_dir = avg_dir.mask((sine == 0) & (cosine == 0))
        return avg_dir.mask(avg_dir == 360.0, 0.0)
    if sine == 0 and cosine == 0:
        return np.nan
    return 0.0 if avg_dir == 360.0 else avg_dir


def _vector_avg_of_wdirs_by_period(wdirs, period):
    """
    Vector average wind directions for each averaging period. This gives the same result as applying
    _vector_avg_of_wdirs_list() to each period but derives the sine and cosine components for all timestamps at once.

    :param wdirs:   Wind directions to average, with a DatetimeIndex sorted in ascending order.
    :type wdirs:    pd.Series or pd.DataFrame
    :param period:  Period to average by, as used by pd.DataFrame.resample().
    :type period:   str or pd.DateOffset
    :return:        Average wind direction for each averaging period.
    :rtype:         pd.Series or pd.DataFrame
    """
    sine, cosine = _wdir_sine_cosine(wdirs)
    sine = sine.resample(period, closed='left', label='left').mean()
    cosine = cosine.resample(period, closed='left', label='left').mean()
    return _sine_cosine_to_wdir(sine, cosine)


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
            - Set period to '1h' for hourly average, '3h' for three hourly average and so on for '4h', '6h' etc.
            - Set period to '1D' for a daily average, '3D' for three day average, similarly '5D', '7D', '15D' etc.
            - Set period to '1W' for a weekly average, '3W' for three week average, similarly '2W', '4W' etc.
            - Set period to '1M' for monthly average with the timestamp at the start of the month.
            - Set period to '1YS' for annual average with the timestamp at the start of the year.

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
        data_hourly = bw.average_data_by_period(data.Spd80mN, period='1h')

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
    period = _normalize_freq_string(period)

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
            raise TypeError("Please use '1YS' for annual frequency at the start of the year.")
    
    # Check that the data resolution is not less than the period specified
    if data_resolution is None:
        if data.index[0] + _freq_str_to_dateoffset(period) < data.index[0] + _get_data_resolution(data.index):
            raise ValueError("The time period specified is less than the temporal resolution of the data. "
                             "For example, hourly data should not be averaged to 10 minute data.")
    data = data.sort_index()
    grouper_obj = data.resample(period, closed='left', label='left')

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
            wdir_grouped_data = _vector_avg_of_wdirs_by_period(data[wdir_column_names], period)
            wdir_coverage = _get_coverage_by_grouper_obj(data[wdir_column_names], grouper_obj[wdir_column_names],
                                                         data_resolution=data_resolution)
        else:
            wdir_grouped_data = _vector_avg_of_wdirs_by_period(data, period)
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
    See _wdir_sine_cosine() for why sine and cosine are rounded.
    """
    sine, cosine = _wdir_sine_cosine(wdirs)
    if wspds is None:
        sine = sine.mean(axis=1)
        cosine = cosine.mean(axis=1)
    else:
        sine = pd.DataFrame(sine.values * wspds.values, index=wdirs.index).mean(axis=1)
        cosine = pd.DataFrame(cosine.values * wspds.values, index=wdirs.index).mean(axis=1)

    return _sine_cosine_to_wdir(sine, cosine).rename('avg_dir')


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
    See _wdir_sine_cosine() for why sine and cosine are rounded.
    """
    # first drop nans, if wind speeds available need to match them first to drop equivalent values
    if wspds is None:
        wdirs = np.asarray(wdirs, dtype=float)
        wdirs = wdirs[~np.isnan(wdirs)]
    else:
        a = np.array([wdirs, wspds])
        a = a[:, ~np.isnan(a).any(axis=0)]
        wdirs = a[0]
        wspds = a[1]
    # if the resulting wdir array is empty, return NAN
    if wdirs.size == 0:
        return np.nan

    sine, cosine = _wdir_sine_cosine(wdirs)
    if wspds is not None:
        sine = sine * wspds
        cosine = cosine * wspds

    return _sine_cosine_to_wdir(np.mean(sine), np.mean(cosine))


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
            - Set period to '1h' for hourly average, '3h' for three hourly average and so on for '4h', '6h' etc.
            - Set period to '1D' for a daily average, '3D' for three day average, similarly '5D', '7D', '15D' etc.
            - Set period to '1W' for a weekly average, '3W' for three week average, similarly '2W', '4W' etc.
            - Set period to '1M' for monthly average with the timestamp at the start of the month.
            - Set period to '1YS' for annual average with the timestamp at the start of the year.

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


def apply_wspd_slope_offset_adj(data, measurements, inplace=False, apply_to_related_statistics=False):
    """
    Automatically apply wind speed calibration slope and offset adjustments to the timeseries data when they
    differ from the logger programmed slope and offsets. The slope and offset information for each measurement
    and time period is contained in the measurements instance from bw.MeasurementStation.

    This uses the brightwind 'adjust_slope_offset()' function to apply the actual adjustment to the data.

    Note: Be careful not to run this more than once in an assessment, when using inplace=True, as it will
          apply the adjustment again.

    :param data:                        Timeseries data.
    :type data:                         pd.DataFrame or pd.Series
    :param measurements:                Measurement information extracted from a WRA Data Model using 
                                        bw.MeasurementStation
    :type measurements:                 list or dict or _Measurements
    :param inplace:                     If True, the original data is modified in place. If False, the original data
                                        will not be touched and instead a new DataFrame containing the adjusted data
                                        is returned. To store this, please ensure it is assigned to a new variable.
                                        Note: 'inplace=True' does not work when a subset of columns is passed as
                                        'data' (e.g. data[['Spd80mS', 'Spd60mS']]) as pandas returns a copy in that
                                        case — use 'inplace=False' and assign the result instead.
    :type inplace:                      bool
    :param apply_to_related_statistics: If True, apply the adjustment to related statistics (e.g. if Spd60mS is
                                        adjusted, also adjust Spd60mS_max, Spd60mS_min, Spd60mS_gust, Spd60mS_sd,
                                        Spd60mS_median, Spd60mS_mode and Spd60mS_range). Statistics such as
                                        Spd60mS_ti, Spd60mS_ti30sec and Spd60mS_sum are not adjusted.
                                        If False, only apply the adjustment to the specific wind speed properties.
                                        If True then the function expects the column name convention where the average
                                        has nothing appended, max is appended with '_max', min is appended with '_min',
                                        gust is appended with '_gust', sd is appended with '_sd', median is appended
                                        with '_median', mode is appended with '_mode' and range is appended with
                                        '_range'.
                                        If the column name convention is different, set this parameter to False and the
                                        adjustment will only be applied to the specific wind speed properties or rename
                                        your data columns. Defaults to False.
    :type apply_to_related_statistics:  bool
    :return:                            Data with adjusted wind speeds.
    :rtype:                             pd.DataFrame or pd.Series

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

    Send a specific wind speed property and data column::
        bw.apply_wspd_slope_offset_adj(data['Spd60mS'], mm1.measurements['Spd60mS'], inplace=True)

    Send a specific wind speed property and data column and adjust related statistics::
        data_edited = data.copy(deep=True)
        data_edited = data_edited.rename(columns={'Spd80mSMax': 'Spd80mS_max', 'Spd80mSStd': 'Spd80mS_sd'})
        data_calib_adj = bw.apply_wspd_slope_offset_adj(data_edited[['Spd80mS', 'Spd80mS_max', 'Spd80mS_sd']], 
                                                        mm1.measurements['Spd80mS'], inplace=False, 
                                                        apply_to_related_statistics=True)

    """
    # Depending on what is sent, get wspd properties into a list of properties
    wspd_properties = _get_consistent_properties_format(measurements, 'wind_speed')
    if not wspd_properties:
        raise ValueError("No wind speed measurements found in the 'measurements' input. "
                         "No slope and offset adjustments can be applied.")

    # copy the data if needed
    data = data.copy(deep=True) if inplace is False else data
    wspd_in_dataset = False
    df = pd.DataFrame(data) if type(data) == pd.Series else data

    col_not_in_data = []
    # Apply the adjustment
    for wspd_prop in wspd_properties:
        name = wspd_prop['name']
        if not apply_to_related_statistics:
            associated_statistics = [""]
            uncorrected_associated_statistics = []
        else:
            associated_statistics = [
                "" if prop["statistic_type_id"] == "avg" else prop["statistic_type_id"]
                for prop in wspd_prop["logger_measurement_config.column_name"]
                if prop["statistic_type_id"] in ["avg", "max", "min", "gust", "sd", "median", "mode", "range"]
            ]
        uncorrected_var_names = [
            f"{name}_{stat}" for stat in ["ti", "ti30sec", "sum"]
            if f"{name}_{stat}" in df.columns
        ]
        if uncorrected_var_names:
            print(
                f"Found {', '.join(uncorrected_var_names)} in data, these will not be corrected for slope and "
                "offset adjustments."
                )

        date_to = wspd_prop.get('date_to')
        date_from = wspd_prop.get('date_from')
        date_to_txt = 'the end of dataset' if (date_to is None or date_to == DATE_INSTEAD_OF_NONE) else date_to

        variables = {
            'slope': 'logger_measurement_config.slope',
            'offset': 'logger_measurement_config.offset',
            'cal_slope': 'calibration.slope',
            'cal_offset': 'calibration.offset'
        }
        none_variables = [v for v in variables.values() if wspd_prop.get(v) is None]

        # Split stats into those found / not found in df
        stats_in_data = []
        for stat in associated_statistics:
            # This assumed variable naming is based on what BrightHub uses
            var_name = f"{name}_{stat}" if stat else name
            if var_name in df.columns:
                stats_in_data.append((stat, var_name))
            else:
                col_not_in_data.append(var_name)

        if not stats_in_data:
            continue

        wspd_in_dataset = True
        if none_variables:
            all_var_names = ', '.join(var_name for _, var_name in stats_in_data)
            print("{} has {} value set as None. Slope and offset adjustment can't be applied "
                "from {} to {}.\n".format(utils.bold(all_var_names), utils.bold(', '.join(none_variables)),
                                          utils.bold(date_from), utils.bold(date_to_txt)))
        elif float(wspd_prop[variables['slope']]) != float(wspd_prop[variables['cal_slope']]) or \
                float(wspd_prop[variables['offset']]) != float(wspd_prop[variables['cal_offset']]):
            adjusted_slope_offset = []
            adjusted_slope_only = []
            for stat, var_name in stats_in_data:
                try:
                    current_offset = 0 if stat in ["sd", "range"] else float(wspd_prop[variables['offset']])
                    new_offset = 0 if stat in ["sd", "range"] else float(wspd_prop[variables['cal_offset']])
                    df.loc[date_from:date_to, var_name] = adjust_slope_offset(
                        df[var_name][date_from:date_to],
                        current_slope=float(wspd_prop[variables['slope']]),
                        current_offset=current_offset,
                        new_slope=float(wspd_prop[variables['cal_slope']]),
                        new_offset=new_offset
                        )
                    (adjusted_slope_only if stat in ["sd", "range"] else adjusted_slope_offset).append(var_name)
                    
                except TypeError:
                    print('{} has TypeError with logger or calibration slope and offset values. Skipping.\n'
                        .format(utils.bold(var_name)))
                except Exception as error_msg:
                    print(error_msg)
            if adjusted_slope_offset and adjusted_slope_only:
                print('{} has slope and offset adjustment applied and {} has slope adjustment applied '
                    'from {} to {}.\n'
                    .format(utils.bold(', '.join(adjusted_slope_offset)),
                            utils.bold(', '.join(adjusted_slope_only)),
                            utils.bold(date_from), utils.bold(date_to_txt)))
            elif adjusted_slope_offset:
                print('{} has slope and offset adjustment applied from {} to {}.\n'
                    .format(utils.bold(', '.join(adjusted_slope_offset)),
                            utils.bold(date_from), utils.bold(date_to_txt)))
            elif adjusted_slope_only:
                print('{} has slope adjustment applied from {} to {}.\n'
                    .format(utils.bold(', '.join(adjusted_slope_only)),
                            utils.bold(date_from), utils.bold(date_to_txt)))
        else:
            all_var_names = ', '.join(var_name for _, var_name in stats_in_data)
            print(
                f'{utils.bold(all_var_names)} logger slope and offsets are equal to calibration slope and offsets from '
                f'{utils.bold(date_from)} to {utils.bold(date_to_txt)}. No adjustment applied.\n'
                )

    if wspd_in_dataset is False:
        print(
            'None of the wind speed measurements reported in the "measurements" input is found in the data. '
            'No slope and offset adjustments can be applied.\n'
            )
    if col_not_in_data and wspd_in_dataset:
        print(
            f"Following wind speed measurement(s) reported in the 'measurements' input not found in the data: "
            f"{utils.bold(str(col_not_in_data))}."
            )

    # if a Series is sent, write back inplace and send back a Series
    if type(data) == pd.Series:
        if inplace:
            data[:] = df[df.columns[0]]
        df = df[df.columns[0]]
    return df


def scale_wind_speed(spd: Union[pd.Series, pd.DataFrame, float, int],
                     scale_factor: Union[int, float]
                     ) -> Union[pd.Series, pd.DataFrame, float, int]:
    """
    Scales wind speed by the scale_factor

    :param spd:             Wind speed value(s) to scale.
    :type spd:              pandas.Series or pandas.DataFrame or float or int
    :param scale_factor:    Scaling factor to use for scaling wind speed.
                            If scaling factor is 0.8, output would be 0.8 times wind speed.
    :type scale_factor:     int or float
    :return:                Value(s) of scaled wind speed. Output type depends on type(spd).
    :rtype:                 pandas.Series or pandas.DataFrame or float or int

        **Example usage**
    ::
    import brightwind as bw
    import pandas as pd
    import numpy as np

    data = bw.load_campbell_scientific(bw.demo_datasets.demo_campbell_scientific_data)

    # scale float by scale_factor of 0.5
    bw.scale_wind_speed(3, 0.5)
    # 1.5

    # scale np.array by scale_factor of 0.5
    bw.scale_wind_speed(np.array([0, 1, 2]), 0.5)
    # array([0. , 0.5, 1. ])

    # scale pd.Series by scale_factor of 0.5
    bw.scale_wind_speed(data['Spd40mN'].tail(3), 0.5)
    # Timestamp
    # 2017-11-23 10:30:00    4.0150
    # 2017-11-23 10:40:00    3.4055
    # 2017-11-23 10:50:00    2.9325
    # Name: Spd40mN, dtype: float64

    # scale pd.DataFrame by scale factor of 2
    bw.scale_wind_speed(data.tail(3)[['Spd60mS', 'Spd40mS']], 2)
    # 	          Spd60mS	    Spd40mS
    # Timestamp		
    # 2017-11-23 10:30:00	16.900	15.750
    # 2017-11-23 10:40:00	14.318	13.336
    # 2017-11-23 10:50:00	12.808	11.498
    """
    return brightwind.transform.scale.apply_scale_factor(spd, scale_factor)


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
        return dataframe_map(wdir.add(offset), utils._range_0_to_360)
    elif isinstance(wdir, pd.Series):
        return wdir.add(offset).apply(utils._range_0_to_360)


def apply_wind_vane_deadband_offset(
        data, measurements, inplace=False, return_results_table=False, apply_to_related_statistics=False
        ):
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

    :param data:                        Timeseries data.
    :type data:                         pd.DataFrame or pd.Series
    :param measurements:                Measurement information extracted from a WRA Data Model using 
                                        bw.MeasurementStation
    :type measurements:                 list or dict or _Measurements
    :param inplace:                     If True, the original data is modified in place. If False, the original data
                                        will not be touched and instead a new DataFrame containing the adjusted data
                                        is returned. To store this, please ensure it is assigned to a new variable.
                                        Note: 'inplace=True' does not work when a subset of columns is passed as
                                        'data' (e.g. data[['Dir78mS', 'Dir60mS']]) as pandas returns a copy in that
                                        case — use 'inplace=False' and assign the result instead.
    :type inplace:                      bool
    :param return_results_table:        Optional key to return a dataframe containing deadband offset, logger offset 
                                        and applied offset for each directional sensor and the time period it is 
                                        relevant for.
    :type return_results_table:         pd.DataFrame
    :param apply_to_related_statistics: If True, apply the adjustment to related statistics (e.g. if Dir60mS is 
                                        adjusted, also adjust Dir60mS_max, Dir60mS_min and Dir60mS_gust).
                                        If False, only apply the adjustment to the specific wind direction properties.
                                        If True then the function expects the column name convention where the average
                                        has nothing appended, max is appended with '_max', min is appended with '_min'
                                        and gust is appended with '_gust'.
                                        If the column name convention is different, set this parameter to False and the 
                                        adjustment will only be applied to the specific wind direction properties or 
                                        rename your data columns. Defaults to False.
    :type apply_to_related_statistics:  bool
    :return:                            Data with adjusted wind direction by the deadband orientation, or where 
                                        return_results_table is specified, a tuple of the data and a DataFrame of 
                                        deadband offsets.
    :rtype:                             pd.DataFrame | pd.Series | Tuple[pd.DataFrame | pd.Series, pd.DataFrame]

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

    Specifying that the deadband offsets should be returned::
        data_adj, results_table = bw.apply_wind_vane_deadband_offset(
            data['Dir78mS'], mm1.measurements['Dir78mS'], inplace=True, return_results_table=True
            )
        results_table

    """
    # Depending on what is sent, get wdir properties into a list of properties
    wdirs_properties = _get_consistent_properties_format(measurements, 'wind_direction')
    if not wdirs_properties:
        raise ValueError("No wind direction measurements found in the 'measurements' input. "
                         "No deadband offset adjustments can be applied.")

    # copy the data if needed
    data = data.copy(deep=True) if inplace is False else data
    wdir_in_dataset = False
    df = pd.DataFrame(data) if type(data) == pd.Series else data

    # Apply the offset
    rows = []
    col_not_in_data = []
    for i, wdir_prop in enumerate(wdirs_properties):
        name = wdir_prop['name']
        if not apply_to_related_statistics:
            associated_statistics = [name]
        else:
            # This assumed variable naming is based on what BrightHub uses
            associated_statistics = [
                name if prop["statistic_type_id"] == "avg" else f"{name}_{prop['statistic_type_id']}"
                for prop in wdir_prop["logger_measurement_config.column_name"]
                if prop["statistic_type_id"] in ["avg", "max", "min", "gust"]
            ]
        for var_name in associated_statistics:
            if var_name in df.columns:
                date_from, date_to = _resolve_period_boundaries(df, wdirs_properties, i, name)
                deadband = wdir_prop.get('vane_dead_band_orientation_deg')
                logger_offset = wdir_prop.get('logger_measurement_config.offset')
                height = wdir_prop.get('height_m')
                wdir_in_dataset = True
                df[var_name], applied_results = _apply_dir_offset_target_orientation(
                    df[var_name], 
                    logger_offset, 
                    deadband, 
                    date_from, 
                    date_to, 
                    target_orientation_name='dead band orientation', 
                    heights=height, 
                    target_orientation_table_name="Vane Dead Band Orientation [deg]"
                    )
                rows.append(applied_results)
            else:
                col_not_in_data.append(var_name)

    if wdir_in_dataset is False:
        print('None of the wind direction measurements reported in the "measurements" input is found in the data. '
              'No deadband offset adjustments can be applied.\n')
    if col_not_in_data and wdir_in_dataset:
        print(
            f"Following wind direction measurement(s) reported in the 'measurements' input not found in the data: "
            f"{utils.bold(str(col_not_in_data))}."
            )
    # if a Series is sent, send back a Series
    if isinstance(data, pd.Series):
        df = df[df.columns[0]]
    if return_results_table:
        if rows:
            return df, _aggregate_consecutive_periods(
                rows, target_orientation_table_name="Vane Dead Band Orientation [deg]"
                )
        else:
            return df, None
    return df


def _aggregate_consecutive_periods(offset_applied_tables, target_orientation_table_name):
    """
    Function creates a dataframe from list of tables containing offset applied for each measurement. For each table 
    neighbouring rows are agreegated into one row if they are consecutive periods that contain the same information 
    for the following columns: "Name", "Height [m], "Logger Offset", "Offset Applied [deg]" 
    and target_orientation_table_name.

    The list of offset_applied_tables can be created using _apply_dir_offset_target_orientation function.

    :param offset_applied_tables:           List of tables containing offset applied for each measurement.
    :type offset_applied_tables:            List[pd.DataFrame]
    :param target_orientation_table_name:   Name of the column on tables representing the target orientation
                                            e.g. "Vane Dead Band Orientation [deg]".
    :type target_orientation_table_name:    str
    :return:                                DataFrame of offset applied for each measurement, with consecutive periods 
                                            merged when appropriate.
    :rtype:                                 pd.DataFrame
    """
    results_df = pd.concat(offset_applied_tables).sort_values(
        by=["Height [m]", "Date From"], ascending=[False, True]
    )
    results_df['consecutive_group'] = (
        (results_df['Name'] != results_df['Name'].shift()) |
        (results_df['Height [m]'] != results_df['Height [m]'].shift()) |
        (results_df[target_orientation_table_name] != results_df[target_orientation_table_name].shift()) |
        (results_df['Logger Offset'] != results_df['Logger Offset'].shift()) |
        (results_df['Offset Applied [deg]'] != results_df['Offset Applied [deg]'].shift())
    ).cumsum()
    
    # Group and aggregate consecutive periods
    results_table = results_df.groupby([
        'Name',
        'consecutive_group',
        'Height [m]',
        target_orientation_table_name,
        'Logger Offset',
        'Offset Applied [deg]'
    ], dropna=False).agg({
        'Date From': 'first',
        'Date To': lambda x: None if any(d is None for d in x) else max(x)
    }).reset_index(drop=False).drop(columns=['consecutive_group']).sort_values(
        by=["Height [m]", "Date From", "Name"], ascending=[False, True, True]
        ).set_index("Name")
    results_table = results_table.fillna({
        'Offset Applied [deg]': 0
        })
    return results_table


def _selective_avg(wspd1, wspd2, wdir, boom_dir1, boom_dir2,
                   inflow_lower1, inflow_higher1, inflow_lower2, inflow_higher2, sector_width):
    # duplicate threshold values into lists which are the same length as other inputs
    inflow_lower1 = [inflow_lower1] * len(wdir)
    inflow_higher1 = [inflow_higher1] * len(wdir)
    inflow_lower2 = [inflow_lower2] * len(wdir)
    inflow_higher2 = [inflow_higher2] * len(wdir)

    # if boom 1 'inflow' sector overlaps with 0/360
    if ((boom_dir1 + 180) % 360) >= (360 - (sector_width/2)) or ((boom_dir1 + 180) % 360) <= (sector_width/2):
        # many nested if statements follow, all within one mapped lambda function
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
                        - Set offset to 1h to add 1 hour to each timestamp and -1h to subtract and so on for 5h, 6h,
                          etc.
                        - Set offset to 1D to add a day and -1D to subtract and so on for 5D, 7D, 15D, etc.
                        - Set offset to 1W to add a week and -1W to subtract from each timestamp and so on for 2W,
                          4W, etc.
                        - Set offset to 1M to add a month and -1M to subtract a month from each timestamp and so on
                          for 2M, 3M, etc.
                        - Set offset to 1Y to add an year and -1Y to subtract a year from each timestamp and so on
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
        op1 = bw.offset_timestamps(data, offset='1h', date_from='2016-02-01 00:20:00',
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

        # Can also except decimal values for offset, like 3.5h for 3 hours and 30 minutes
        op5 = bw.offset_timestamps(data.index, offset='3.5h', date_from='2016-02-01 00:20:00',
            date_to='2016-02-01 01:40:00')

        # Can accept also Timestamp and datetime objects
        bw.offset_timestamps(data.index[0], offset='4h')
        bw.offset_timestamps(datetime.datetime(2016, 2, 1, 0, 20), offset='3.5h')
        bw.offset_timestamps(datetime.date(2016, 2, 1), offset='-5h')
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
            date_to = data[-1] + pd.DateOffset(seconds=1)
        elif isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
            date_to = data.index[-1] + pd.DateOffset(seconds=1)
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
        return pd.DatetimeIndex(shifted, name=data.name)

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
            df_copy.index = shifted.rename(data.index.name)
            return df_copy.sort_index()


def _date_from_sort_key(period):
    """Sort key for configuration periods by `date_from`, with no `date_from` sorted first."""
    date_from = period.get('date_from')
    return pd.Timestamp.min if date_from is None else pd.to_datetime(date_from)


def _format_utc_offset(offset_hrs):
    """Format an offset from UTC in hours as e.g. 'UTC+01:00', 'UTC-05:00' or 'UTC+05:30'."""
    sign = '-' if offset_hrs < 0 else '+'
    hours, minutes = divmod(int(round(abs(offset_hrs) * 60)), 60)
    return 'UTC{}{:02d}:{:02d}'.format(sign, hours, minutes)


def _format_timestamps(timestamps, max_shown=10):
    """List timestamps as text, showing at most `max_shown` of them."""
    text = ', '.join(str(timestamp) for timestamp in timestamps[:max_shown])
    if len(timestamps) > max_shown:
        text += ' and {} more'.format(len(timestamps) - max_shown)
    return text


def get_logger_utc_offsets(measurement_station, logger_serial_numbers=None):
    """
    Get the logger offset from UTC (`offset_from_utc_hrs`) over time from the logger main configs of a measurement
    station, with the period each offset applies to and the logger it comes from.

    The logger main configs of each logger (identified by `logger_serial_number`) are sorted by `date_from` (a
    config with no `date_from` is treated as the earliest). Within each logger, a config's `date_to` that differs
    from the next config's `date_from` (e.g. a `date_to` given as the last timestamp of the config, i.e.
    inclusive) is set to the next config's `date_from`, so the returned `date_to` is exclusive.

    If more than one logger has configs covering the same period, e.g. concurrent loggers or a replaced logger
    whose configs overlap with the new logger's, the offset of the logger with the highest priority is used for
    that period. A lower priority logger is used for the periods the higher priority loggers do not cover. The
    priority is the order of `logger_serial_numbers`. If `logger_serial_numbers` is not given, all the loggers are
    used, ordered by the total time their configs cover (longest first), then by their earliest `date_from`, and a
    warning is raised for any period where loggers with different offsets from UTC overlap.

    Consecutive periods of the same logger and offset are merged. Configs with no offset on record are returned
    with an offset of `None`, so that periods with an unknown offset are not merged into the periods either side
    of them. Periods not covered by any logger are not returned.

    :param measurement_station:     A simplified object to represent the IEA Wind Task 43 WRA Data Model.
    :type measurement_station:      bw.MeasurementStation
    :param logger_serial_numbers:   (Optional) The serial numbers of the loggers to use, highest priority first.
                                    Only these loggers are used. Default is all the loggers of the station.
    :type logger_serial_numbers:    list[str]
    :return:                        One dict per period, sorted earliest to latest, with keys
                                    'logger_serial_number', 'date_from', 'date_to' (either date can be `None`, i.e.
                                    open ended; `date_to` is exclusive except for the last config of a logger, which
                                    is as given in the data model) and 'offset_from_utc_hrs' (a float, or `None` if
                                    no offset is on record).
    :rtype:                         list[dict]
    :raises TypeError:              If `measurement_station` is not a `bw.MeasurementStation`.
    :raises ValueError:             If a serial number in `logger_serial_numbers` is not one of the station's
                                    loggers.

    **Example usage**
    ::
        import brightwind as bw

        measurement_station = bw.MeasurementStation(bw.demo_datasets.demo_wra_data_model)
        bw.get_logger_utc_offsets(measurement_station)

        # Use only the logger with serial number '1234567'
        bw.get_logger_utc_offsets(measurement_station, logger_serial_numbers=['1234567'])

    """
    if not isinstance(measurement_station, MeasurementStation):
        raise TypeError("'measurement_station' must be a bw.MeasurementStation, not {}."
                        .format(type(measurement_station).__name__))

    logger_main_configs = measurement_station.logger_main_configs.properties
    if isinstance(logger_main_configs, dict):
        logger_main_configs = [logger_main_configs]

    loggers = {}
    for logger_main_config in logger_main_configs or []:
        offset_from_utc_hrs = logger_main_config.get('offset_from_utc_hrs')
        config = {'date_from': logger_main_config.get('date_from'), 'date_to': logger_main_config.get('date_to'),
                  'offset_from_utc_hrs': None if offset_from_utc_hrs is None else float(offset_from_utc_hrs)}
        configs = loggers.setdefault(logger_main_config.get('logger_serial_number'), [])
        # A logger main config can be repeated, e.g. once for each lidar config
        if config not in configs:
            configs.append(config)

    if logger_serial_numbers is not None:
        if isinstance(logger_serial_numbers, str):
            logger_serial_numbers = [logger_serial_numbers]
        unknown = [serial_number for serial_number in logger_serial_numbers if serial_number not in loggers]
        if unknown:
            raise ValueError("Logger serial number(s) {} not found in the logger main configs. The loggers of the "
                             "measurement station are: {}.".format(unknown, list(loggers)))
        priority = list(logger_serial_numbers)
    else:
        priority = None

    # Make each logger's configs end where its next config starts, so date_to is exclusive
    for serial_number, configs in loggers.items():
        configs.sort(key=_date_from_sort_key)
        for i, config in enumerate(configs):
            next_config = configs[i + 1] if i < len(configs) - 1 else None
            config['date_to'] = _resolve_period_date_to(config['date_to'], pd.Timestamp.max, next_config)
        loggers[serial_number] = [config for config in configs
                                  if config['date_to'] is None or config['date_from'] is None
                                  or pd.to_datetime(config['date_to']) > pd.to_datetime(config['date_from'])]

    if priority is None:
        priority = _logger_priority_by_coverage(loggers)
    return _logger_utc_offset_timeline(loggers, priority, warn_overlaps=logger_serial_numbers is None)


def _logger_priority_by_coverage(loggers):
    """
    Order the loggers by the total time their configs cover (longest first), then by their earliest `date_from`.
    An open ended config is taken to run from the earliest, or to the latest, date of all the configs.

    :param loggers: The configs of each logger, by serial number, each a dict with 'date_from', 'date_to' and
                    'offset_from_utc_hrs'.
    :type loggers:  dict
    :return:        The serial numbers, highest priority first.
    :rtype:         list
    """
    dates = [pd.to_datetime(config[key]) for configs in loggers.values() for config in configs
             for key in ('date_from', 'date_to') if config[key] is not None]
    earliest = min(dates) if dates else pd.Timestamp(0)
    latest = max(dates) if dates else pd.Timestamp(0)

    def sort_key(serial_number):
        configs = loggers[serial_number]
        starts = [earliest if config['date_from'] is None else pd.to_datetime(config['date_from'])
                  for config in configs]
        ends = [latest if config['date_to'] is None else pd.to_datetime(config['date_to']) for config in configs]
        covered = sum((max(end - start, pd.Timedelta(0)) for start, end in zip(starts, ends)), pd.Timedelta(0))
        return -covered, min(starts), str(serial_number)

    return sorted(loggers, key=sort_key)


def _logger_utc_offset_timeline(loggers, priority, warn_overlaps):
    """
    Combine the configs of several loggers into one timeline of offsets from UTC. The timeline is cut wherever a
    config starts or ends, and each piece takes the offset of the first logger in `priority` with a config covering
    it. Consecutive pieces of the same logger and offset are merged, and pieces no logger covers are dropped.

    :param loggers:         The configs of each logger, by serial number, each a dict with 'date_from', 'date_to'
                            (exclusive) and 'offset_from_utc_hrs'.
    :type loggers:          dict
    :param priority:        The serial numbers of the loggers to use, highest priority first.
    :type priority:         list
    :param warn_overlaps:   If True, warn about pieces where loggers with different offsets from UTC overlap.
    :type warn_overlaps:    bool
    :return:                The timeline, see `get_logger_utc_offsets()`.
    :rtype:                 list[dict]
    """
    candidates = [
        {'rank': rank, 'logger_serial_number': serial_number,
         'start': pd.Timestamp.min if config['date_from'] is None else pd.to_datetime(config['date_from']),
         'end': pd.Timestamp.max if config['date_to'] is None else pd.to_datetime(config['date_to']),
         'offset_from_utc_hrs': config['offset_from_utc_hrs']}
        for rank, serial_number in enumerate(priority) for config in loggers[serial_number]]
    cuts = sorted({moment for candidate in candidates for moment in (candidate['start'], candidate['end'])})

    timeline, overlaps = [], []
    for start, end in zip(cuts, cuts[1:]):
        covering = [candidate for candidate in candidates if candidate['start'] <= start and candidate['end'] >= end]
        if not covering:
            continue
        active = min(covering, key=lambda candidate: (candidate['rank'], -candidate['start'].value))
        if len({candidate['offset_from_utc_hrs'] for candidate in covering}) > 1:
            overlaps.append((start, end, active, covering))
        key = (active['logger_serial_number'], active['offset_from_utc_hrs'])
        if timeline and timeline[-1]['_end'] == start and timeline[-1]['_key'] == key:
            timeline[-1]['_end'] = end
        else:
            timeline.append({'_key': key, '_start': start, '_end': end})

    if warn_overlaps and overlaps:
        details = []
        for start, end, active, covering in overlaps[:5]:
            details.append('from {} to {} (exclusive): {}; logger {} used'.format(
                'the start' if start == pd.Timestamp.min else start,
                'the end' if end == pd.Timestamp.max else end,
                ', '.join('logger {} at {}'.format(
                    candidate['logger_serial_number'],
                    'an unknown offset' if candidate['offset_from_utc_hrs'] is None
                    else _format_utc_offset(candidate['offset_from_utc_hrs'])) for candidate in covering),
                active['logger_serial_number']))
        warnings.warn('Loggers with different offsets from UTC have logger main configs covering the same period. '
                      'The logger with the longest coverage is used. Set `logger_serial_numbers` to choose the '
                      'loggers and their priority. {}{}.'.format('; '.join(details),
                                                                 ' and {} more'.format(len(overlaps) - 5)
                                                                 if len(overlaps) > 5 else ''))

    date_format = '%Y-%m-%dT%H:%M:%S'
    return [{'logger_serial_number': piece['_key'][0],
             'date_from': None if piece['_start'] == pd.Timestamp.min else piece['_start'].strftime(date_format),
             'date_to': None if piece['_end'] == pd.Timestamp.max else piece['_end'].strftime(date_format),
             'offset_from_utc_hrs': piece['_key'][1]}
            for piece in timeline]


def _get_utc_offset_periods(utc_offsets, logger_serial_numbers=None):
    """
    Return the periods of logger offset from UTC given to `convert_timestamps_to_utc()`, validated and sorted by
    `date_from`.
    """
    if isinstance(utc_offsets, MeasurementStation):
        utc_offsets = get_logger_utc_offsets(utc_offsets, logger_serial_numbers=logger_serial_numbers)
    elif logger_serial_numbers is not None:
        raise TypeError("'logger_serial_numbers' can only be used when 'utc_offsets' is a bw.MeasurementStation.")
    elif isinstance(utc_offsets, dict):
        utc_offsets = [utc_offsets]
    if not isinstance(utc_offsets, list) or not all(isinstance(period, dict) for period in utc_offsets):
        raise TypeError("'utc_offsets' must be a bw.MeasurementStation or a list of dicts with keys 'date_from', "
                        "'date_to' and 'offset_from_utc_hrs'.")

    periods = []
    for period in utc_offsets:
        if 'offset_from_utc_hrs' not in period:
            raise ValueError("Each period in 'utc_offsets' must have an 'offset_from_utc_hrs' key: {}."
                             .format(period))
        offset_from_utc_hrs = period['offset_from_utc_hrs']
        if offset_from_utc_hrs is not None:
            if isinstance(offset_from_utc_hrs, bool) or not isinstance(offset_from_utc_hrs, (int, float, np.number)):
                raise TypeError("'offset_from_utc_hrs' must be a number of hours or None, not {}."
                                .format(repr(offset_from_utc_hrs)))
            offset_from_utc_hrs = float(offset_from_utc_hrs)
            if abs(offset_from_utc_hrs) > 14:
                raise ValueError("An offset from UTC of {:+g} hours is outside the valid range of -14 to +14 hours."
                                 .format(offset_from_utc_hrs))
        periods.append({'logger_serial_number': period.get('logger_serial_number'),
                        'date_from': period.get('date_from'), 'date_to': period.get('date_to'),
                        'offset_from_utc_hrs': offset_from_utc_hrs})
    return sorted(periods, key=_date_from_sort_key)


def convert_timestamps_to_utc(data, utc_offsets, logger_serial_numbers=None, inplace=False):
    """
    Convert the timestamps of logger data from logger time to UTC, using the logger offset from UTC of each period,
    e.g. `offset_from_utc_hrs` in the logger main configs of a measurement station.

    If the measurement station has more than one logger with configs covering the same period, the offset of the
    logger with the highest priority is used, see `get_logger_utc_offsets()`. Use `logger_serial_numbers` to choose
    the loggers and their priority.

    The timestamps of each period are shifted by minus that period's offset, e.g. data logged at UTC+01:00 are
    shifted by -1 hour and data logged at UTC-05:00 are shifted by +5 hours. The shift of each timestamp is worked
    out from the original timestamps before any are moved, so data shifted out of one period are never shifted
    again by the next period.

    Each change of logger time must be given as a separate period, including any change between summer time and
    winter time if the logger clock followed local summer time. Data logged during a change of logger time that is
    not given as a period will be converted incorrectly, and this can't be detected by this function.

    Periods are treated as [date_from, date_to), i.e. `date_from` inclusive and `date_to` exclusive. A `date_to`
    given as the last timestamp of the period (inclusive) is also accepted: if a period's `date_to` differs from
    the next period's `date_from`, the period is taken to end where the next one starts, so any gap between the two
    periods is converted with the earlier period's offset. If the last period's `date_to` is at or after the last
    timestamp of the data, the period runs to the end of the data. This is the same as for
    `apply_wind_vane_deadband_offset()` and `apply_device_orientation_offset()`.

    Where the offset from UTC changes between periods, the converted timestamps either side of the change are
    checked and one of the following is reported:

    - A gap, e.g. where the logger clock was moved back by an hour (end of summer time) and the logger overwrote
      the repeated hour, so that hour of data is missing.
    - An overlap, where converted timestamps from the earlier period run past the start of the later period. Any
      duplicate timestamps this creates are removed, keeping the data from the earlier period.
    - Continuous timestamps, e.g. where the logger clock was moved forward by an hour (start of summer time) and
      the hour skipped in logger time is closed by the conversion.

    Timestamps not covered by any period, or in a period with no offset on record, are left unchanged and a warning
    is raised.

    To then convert the data from UTC to local standard time (i.e. without summer time), shift the full record by
    the local offset from UTC with `offset_timestamps()`, e.g. `bw.offset_timestamps(data_utc, offset='1h')`.

    **Note:** When using `inplace=True`, be careful not to apply this function more than once to the same data, as
    the timestamps will be shifted again.

    :param data:            Timeseries data with timestamps in logger time.
    :type data:             pd.DataFrame or pd.Series
    :param utc_offsets:     The logger offset from UTC over time, either a measurement station whose logger main
                            configs hold the offset (`offset_from_utc_hrs`), or a list of dicts with keys
                            'date_from', 'date_to' and 'offset_from_utc_hrs' (a number of hours, e.g. 1, -5 or 5.5,
                            or `None` if not known), as returned by `get_logger_utc_offsets()`. `None` dates are
                            open ended.
    :type utc_offsets:      bw.MeasurementStation or list[dict]
    :param logger_serial_numbers: (Optional) Only when `utc_offsets` is a measurement station. The serial numbers
                            of the loggers to use, highest priority first. Only these loggers are used. Default is
                            all the loggers of the station, ordered as described in `get_logger_utc_offsets()`.
    :type logger_serial_numbers: list[str]
    :param inplace:         If True, the timestamps of `data` are converted in place. If False, `data` is not
                            changed and a new object with converted timestamps is returned. To store this, please
                            ensure it is assigned to a new variable. Default is False.
    :type inplace:          bool
    :return:                Data with timestamps in UTC.
    :rtype:                 pd.DataFrame or pd.Series
    :raises TypeError:      If `data` is not a pd.DataFrame or pd.Series with a timezone naive pd.DatetimeIndex,
                            `utc_offsets` is not in one of the formats above, or `logger_serial_numbers` is given
                            without a measurement station.
    :raises ValueError:     If `data` has duplicate timestamps, as these can't be assigned to a period (e.g. an hour
                            recorded twice when the logger clock was moved back; resolve them before converting),
                            an offset is missing or outside -14 to +14 hours, or a logger serial number is not one of
                            the station's loggers.

    **Example usage**
    ::
        import brightwind as bw

        data = bw.load_csv(bw.demo_datasets.demo_data)

        # Logger clock on UTC+01:00 until the end of summer time at 2016-10-30 02:00 logger time, then on UTC+00:00
        utc_offsets = [{'date_from': None, 'date_to': '2016-10-30T02:00:00', 'offset_from_utc_hrs': 1},
                       {'date_from': '2016-10-30T02:00:00', 'date_to': None, 'offset_from_utc_hrs': 0}]

        # Convert timestamps to UTC, and assign to a new variable
        data_utc = bw.convert_timestamps_to_utc(data, utc_offsets)

        # Convert timestamps from UTC to local standard time UTC+01:00 for the full record
        data_local = bw.offset_timestamps(data_utc, offset='1h')

        # Use the logger offsets from UTC in the logger main configs of a measurement station (the demo station
        # has no offset on record, so a warning is raised and the timestamps are left unchanged)
        measurement_station = bw.MeasurementStation(bw.demo_datasets.demo_wra_data_model)
        data_utc = bw.convert_timestamps_to_utc(data, measurement_station)

        # Use only the logger with serial number '1234567' (in order of priority if more than one is given)
        data_utc = bw.convert_timestamps_to_utc(data, measurement_station, logger_serial_numbers=['1234567'])

        # Convert timestamps to UTC, applying in place
        bw.convert_timestamps_to_utc(data, utc_offsets, inplace=True)

    """
    if not isinstance(data, (pd.Series, pd.DataFrame)) or not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("'data' must be a pd.DataFrame or pd.Series with a pd.DatetimeIndex.")
    if data.index.tz is not None:
        raise TypeError("'data' already has timezone aware timestamps ({}). Use pandas tz_convert('UTC') instead."
                        .format(data.index.tz))
    if data.index.has_duplicates:
        duplicates = data.index[data.index.duplicated()].unique()
        raise ValueError(
            "'data' has {} duplicate timestamp(s), which can't be assigned to a period: {}. This can happen when the "
            "logger clock was moved back (e.g. at the end of summer time) and the repeated hour was recorded twice. "
            "Resolve the duplicate timestamps before converting to UTC.".format(len(duplicates),
                                                                                 _format_timestamps(duplicates)))
    periods = _get_utc_offset_periods(utc_offsets, logger_serial_numbers)

    data = data if inplace else data.copy(deep=True)
    if data.empty:
        return data
    data.sort_index(inplace=True)
    timestamps = data.index
    shift_hrs = np.full(len(timestamps), np.nan)

    periods_with_data = []
    for i, period in enumerate(periods):
        date_from = period['date_from']
        next_period = periods[i + 1] if i < len(periods) - 1 else None
        date_to = _resolve_period_date_to(period['date_to'], timestamps[-1], next_period)

        in_period = np.isnan(shift_hrs)
        if date_from is not None:
            in_period &= np.asarray(timestamps >= pd.to_datetime(date_from))
        if date_to is not None:
            in_period &= np.asarray(timestamps < pd.to_datetime(date_to))
        if not in_period.any():
            continue

        period_txt = 'from {} to {}'.format(date_from or 'the start of the data',
                                            '{} (exclusive)'.format(date_to) if date_to else 'the end of the data')
        if period['logger_serial_number'] is not None:
            period_txt += ' (logger {})'.format(period['logger_serial_number'])
        offset_from_utc_hrs = period['offset_from_utc_hrs']
        if offset_from_utc_hrs is None:
            shift_hrs[in_period] = 0
            warnings.warn('No logger offset from UTC is on record {}. The {} timestamp(s) in this period are left '
                          'unchanged.'.format(period_txt, in_period.sum()))
            continue

        shift_hrs[in_period] = -offset_from_utc_hrs
        periods_with_data.append((offset_from_utc_hrs, in_period))
        if offset_from_utc_hrs == 0:
            print('Timestamps {} are already in UTC (logger time is UTC+00:00). No change made.'.format(period_txt))
        else:
            print('Timestamps {} converted from logger time ({}) to UTC by shifting them by {} hours.'
                  .format(period_txt, utils.bold(_format_utc_offset(offset_from_utc_hrs)),
                          utils.bold('{:+g}'.format(-offset_from_utc_hrs))))

    not_covered = np.isnan(shift_hrs)
    if not_covered.any():
        shift_hrs[not_covered] = 0
        warnings.warn('{} timestamp(s) are not covered by any period with a logger offset from UTC and are left '
                      'unchanged: {}.'.format(not_covered.sum(), _format_timestamps(timestamps[not_covered])))

    converted = (timestamps + pd.to_timedelta(np.round(shift_hrs * 60), unit='min')).rename(timestamps.name)
    _report_utc_offset_changes(timestamps, converted, periods_with_data)

    duplicated = converted.duplicated(keep='first')
    if duplicated.any():
        warnings.warn('{} duplicate timestamp(s) were created by the conversion to UTC and have been removed, keeping '
                      'the data from the earlier period: {}.'.format(duplicated.sum(),
                                                                     _format_timestamps(converted[duplicated])))
        data.drop(index=timestamps[duplicated], inplace=True)
        converted = converted[~duplicated]

    data.index = converted
    data.sort_index(inplace=True)
    return data


def _report_utc_offset_changes(timestamps, converted, periods_with_data):
    """
    Report any gap or overlap in the converted timestamps where the logger offset from UTC changes between
    consecutive periods. See `convert_timestamps_to_utc()`.

    :param timestamps:          The original timestamps, in logger time.
    :type timestamps:           pd.DatetimeIndex
    :param converted:           The converted timestamps, in UTC, in the same order as `timestamps`.
    :type converted:            pd.DatetimeIndex
    :param periods_with_data:   The offset from UTC and the mask of the timestamps of each period with data and an
                                offset on record, in time order.
    :type periods_with_data:    list[tuple[float, np.ndarray]]
    """
    data_resolution = timestamps.to_series().diff().median() if len(timestamps) > 1 else pd.Timedelta(0)
    for (offset_before, in_period_before), (offset_after, in_period_after) in zip(periods_with_data,
                                                                                  periods_with_data[1:]):
        if offset_before == offset_after:
            continue
        last_before = converted[in_period_before].max()
        first_after = converted[in_period_after].min()
        change_txt = 'where the logger offset from UTC changes from {} to {}'.format(
            _format_utc_offset(offset_before), _format_utc_offset(offset_after))
        if first_after <= last_before:
            warnings.warn(
                'The converted timestamps overlap from {} to {} (inclusive, UTC) {}, as the converted timestamps of '
                'the earlier period run past the start of the later period. Check the dates of this change of logger '
                'time.'.format(first_after, last_before, change_txt))
        elif first_after - last_before > data_resolution:
            gap_from = last_before + data_resolution
            warnings.warn(
                'There is a gap of {:g} hours in the converted timestamps from {} to {} (exclusive, UTC) {}. If the '
                'logger clock was moved back at this point (e.g. at the end of summer time), the logger may have '
                'overwritten the repeated data, or there may be a gap in the logger data.'
                .format((first_after - gap_from) / pd.Timedelta(hours=1), gap_from, first_after, change_txt))
        else:
            print('The converted timestamps are continuous at {} (UTC) {}.'.format(utils.bold(str(first_after)),
                                                                                  change_txt))


def apply_device_orientation_offset(
        data, measurement_station, wdir_cols=[], inplace=False, return_results_table=False, 
        apply_to_related_statistics=False
        ):
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

    If a logger offset is defined in `logger_measurement_config.offset`, this indicates
    that the wind direction data has already been adjusted by this amount. This value may differ
    from `device_orientation_deg`, so the function calculates the difference to ensure the total
    adjustment equals the intended device orientation:

        offset_to_apply = device_orientation_deg - logger_measurement_config.offset

    Configuration date ranges are treated as half-open intervals: [date_from, date_to),
    meaning 'date_from' is inclusive and 'date_to' is exclusive.

    Overlapping periods with non-zero `device_orientation_deg` values in `vertical_profiler_properties`
    are not supported and will raise an error.

    :param data:                                Timeseries data.
    :type data:                                 pd.DataFrame or pd.Series
    :param measurement_station:                 A simplified object to represent the IEA Wind Task 43 WRA Data Model.
    :type measurement_station:                  bw.MeasurementStation
    :param wdir_cols:                           Wind direction column names to apply the offset to. If empty, all wind 
                                                direction columns in the data are used. Default is an empty list.
    :type wdir_cols:                            list
    :param inplace:                             If True, the original data is modified in place. If False, the
                                                original data will not be touched and instead a new DataFrame
                                                containing the adjusted data is returned. To store this, please
                                                ensure it is assigned to a new variable.
                                                Note: 'inplace=True' does not work when a subset of columns is
                                                passed as 'data' (e.g. data[['col1', 'col2']]) as pandas returns
                                                a copy in that case — use 'inplace=False' and assign the result
                                                instead. Default is False.
    :type inplace:                              bool, optional
    :param return_results_table:                If True, returns a DataFrame containing the device orientation, 
                                                logger orientation and offset applied for each relevant time period.
    :type return_results_table:                 bool, optional
    :param apply_to_related_statistics:         If True, apply the adjustment to related statistics (e.g. if Dir60mS is 
                                                adjusted, also adjust Dir60mS_max, Dir60mS_min and Dir60mS_gust).
                                                If False, only apply the adjustment to the specific wind direction
                                                properties.
                                                If True then the function expects the column name convention where the
                                                average has nothing appended, max is appended with '_max', min is
                                                appended with '_min' and gust is appended with '_gust'.
                                                If the column name convention is different, set this
                                                parameter to False and the adjustment will only be applied to the 
                                                specific wind direction properties or rename your data columns. 
                                                Defaults to False.
    :type apply_to_related_statistics:          bool
    :return:                                    Data with wind direction adjusted by the orientation offset.
    :rtype:                                     pd.DataFrame | pd.Series | Tuple[pd.DataFrame | pd.Series, pd.DataFrame]
    
    **Example usage**
    ::
        import brightwind as bw

        fl1 = bw.MeasurementStation(bw.demo_datasets.floating_lidar_demo_iea43_wra_data_model_v1_3)
        data = bw.load_csv(bw.demo_datasets.demo_floating_lidar_data)

        # Adjust only specific wind direction columns:
        data_dev_orient_adj = bw.apply_device_orientation_offset(data, fl1, wdir_cols=['Dir_40m', 'Dir_50m'])

    Adjust all wind directions in-place:
    ::
        bw.apply_device_orientation_offset(data, fl1, inplace=True)
        print('Wind direction device orientation offset adjustment is completed.')

    Return the orientation offset results table along with adjusted timeseries data::
    ::
        # Adjust only specific wind direction columns:
        data_dev_orient_adj, results_table = bw.apply_device_orientation_offset(
            data, fl1, wdir_cols=['Dir_40m', 'Dir_50m'], return_results_table=True
            )
        results_table
    
    """
    
    if measurement_station.type not in ['lidar', 'floating_lidar', 'sodar']:
        raise ValueError(f"Device type: {measurement_station.type} is not supported for this function.")
    
    measurements = measurement_station.measurements
    wdirs_properties = _get_consistent_properties_format(measurements, 'wind_direction')
    if not wdirs_properties:
        raise ValueError("No wind direction measurements found in the 'measurement_station' input. "
                         "No device orientation offset adjustments can be applied.")
    
    measurement_station_items = list(measurement_station)
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

    rows = []
    # Apply the offset
    for i, wdir_prop in enumerate(wdirs_properties):
        name = wdir_prop['name']
        if not apply_to_related_statistics:
            associated_statistics = [name]
        else:
            # This assumed variable naming is based on what BrightHub uses
            associated_statistics = [
                name if prop["statistic_type_id"] == "avg" else f"{name}_{prop['statistic_type_id']}"
                for prop in wdir_prop["logger_measurement_config.column_name"]
                if prop["statistic_type_id"] in ["avg", "max", "min", "gust"]
            ]
        for var_name in associated_statistics:
            if var_name in df.columns:
                date_from, date_to = _resolve_period_boundaries(df, wdirs_properties, i, name)
                
                logger_offset = wdir_prop.get('logger_measurement_config.offset')
                for j, device_properties in enumerate(measurement_station):
                    meas_station_data_model_from = device_properties.get('date_from')
                    meas_station_data_model_from = (df.index[0].strftime('%Y-%m-%dT%H:%M:%S') if
                                                    meas_station_data_model_from is None or meas_station_data_model_from ==
                                                    DATE_INSTEAD_OF_NONE else meas_station_data_model_from)
                    next_device_properties = (measurement_station_items[j + 1]
                                              if j < len(measurement_station_items) - 1 else None)
                    meas_station_data_model_to = _resolve_period_date_to(
                        device_properties.get('date_to'), df.index[-1], next_device_properties)

                    if date_to is None or date_to == DATE_INSTEAD_OF_NONE:
                        date_to_tmp = meas_station_data_model_to
                    else:
                        date_to_tmp = date_to
                    
                    date_range_overlaps = False
                    if meas_station_data_model_to is None and date_to_tmp is None:
                        date_range_overlaps = True
                    elif meas_station_data_model_to is None:
                        date_range_overlaps = date_to_tmp is None or date_to_tmp >= meas_station_data_model_from
                    elif date_to_tmp is None:
                        date_range_overlaps = date_from <= meas_station_data_model_to
                    else:
                        date_range_overlaps = (
                            date_from <= meas_station_data_model_to and
                            date_to_tmp >= meas_station_data_model_from
                        )

                    if date_range_overlaps:
                        device_orientation_deg = device_properties.get('device_orientation_deg')
                        apply_offset_from = (date_from if date_from > meas_station_data_model_from
                                             else meas_station_data_model_from)
                        if date_to_tmp is None or meas_station_data_model_to is None:
                            apply_offset_to = date_to_tmp if date_to_tmp is not None else meas_station_data_model_to
                        else:
                            apply_offset_to = min(date_to_tmp, meas_station_data_model_to)
                        height = wdir_prop.get('height_m')

                        df[var_name], applied_results = _apply_dir_offset_target_orientation(
                            df[var_name], logger_offset, device_orientation_deg, apply_offset_from, apply_offset_to,
                            target_orientation_name='device orientation', heights=height, 
                            target_orientation_table_name="Device Orientation [deg]"
                            )
                                
                        rows.append(applied_results)
            else:
                wdir_not_in_dataset = True
                col_not_in_data.append(var_name)
    
    if wdir_not_in_dataset:
        indexes = np.unique(col_not_in_data, return_index=True)[1]
        col_not_in_data = [col_not_in_data[index] for index in sorted(indexes)]
        print_text = "Following wind direction measurement(s) reported in the 'measurement_station' input " \
                     "not found in the data"
        if wdir_cols:
            print(print_text + ' for the requested `wdir_cols`: {}.'.format(utils.bold(str(col_not_in_data))))
        else:
            print(print_text + ': {}.'.format(utils.bold(str(col_not_in_data))))
    if col_not_in_datamodel:
        print('No device orientation offset applied to following `wdir_cols` requested measurement(s) '
              'as no wind direction measurement type found in `measurement_station` input for these: {}.'
              .format(utils.bold(str(col_not_in_datamodel))))
    # if a Series is sent, send back a Series
    if isinstance(data, pd.Series):
        df = df[df.columns[0]]
        data.update(df)

    if return_results_table:
        if rows:
            return df, _aggregate_consecutive_periods(rows, target_orientation_table_name="Device Orientation [deg]")
        else:
            return df, None
    return df


def _resolve_period_boundaries(df, wdirs_properties, current_index, name):
    """
    Normalize logger period time boundaries to enforce [date_from, date_to) half-open interval logic.
    
    This ensures data points are not corrected twice by making the earlier logger property period
    end exactly when the next entry starts.

    :param df:                  Wind measurement dataframe with datetime index
    :type df:                   pd.DataFrame
    :param wdirs_properties:    Consistent list of properties for the wind_direction
    :type wdirs_properties:     list
    :param current_index:       Index of wind direction property currently being processed
    :type current_index:        int
    :param name:                Wind direction name
    :type name:                 str
    :return:                    Tuple of (date_from, date_to) as ISO format strings. date_to may be None to indicate
                                the period extends to the end of the dataset. 
    :rtype:                     Tuple[str, str]
    """
    next_period = None
    if current_index < len(wdirs_properties) - 1:
        if wdirs_properties[current_index + 1].get('name') == name:
            next_period = wdirs_properties[current_index + 1]
    date_to = _resolve_period_date_to(wdirs_properties[current_index].get('date_to'), df.index[-1], next_period)

    date_from = wdirs_properties[current_index].get('date_from')
    date_from = (df.index[0].strftime('%Y-%m-%dT%H:%M:%S')
                    if date_from is None or date_from == DATE_INSTEAD_OF_NONE else date_from)

    return date_from, date_to


def _resolve_period_date_to(date_to, last_timestamp, next_period=None):
    """
    Return the end of a configuration period to use with the [date_from, date_to) convention, i.e. `date_to`
    exclusive, accepting a `date_to` that was given either as exclusive (equal to the next period's `date_from`) or
    as inclusive (the last timestamp of the period). This ensures a timestamp is never in two periods, and the last
    timestamp of a period is not missed.

    - If `date_to` is at or after the last timestamp of the data, `None` is returned (the period runs to the end of
      the data), so the last timestamp is not missed.
    - If there is a next period and its `date_from` differs from `date_to`, the next period's `date_from` is
      returned, so that the period ends exactly where the next one starts. Note that any gap between the two
      periods is therefore included in this period.

    :param date_to:         The `date_to` of the period, as given in the data model. `None` if open ended.
    :type date_to:          str or None
    :param last_timestamp:  The last timestamp of the data.
    :type last_timestamp:   pd.Timestamp
    :param next_period:     The next period of the same item, sorted by `date_from`, or `None` if this is the last
                            period.
    :type next_period:      dict or None
    :return:                The exclusive end of the period, or `None` if the period runs to the end of the data.
    :rtype:                 str or None
    """
    # If date_to has been explicitly set as the last timestamp of the data (or later), set it to None. This avoids
    # missing the last timestamp due to the [date_from, date_to) logic
    if date_to is not None and pd.to_datetime(date_to) >= last_timestamp:
        date_to = None

    # If the [date_from, date_to) convention has not been used, force it by setting date_to to the date_from of the
    # next period
    if next_period is not None and next_period.get('date_from') != date_to:
        date_to = next_period.get('date_from')

    return date_to


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


def _apply_dir_offset_target_orientation(
        wdir_data, logger_offset, target_orientation, apply_offset_from, apply_offset_to, target_orientation_name,
        heights, target_orientation_table_name
        ):
    """
    Function to apply the required offset to the wind direction data based on the logger offset and a target
    orientation.
    Note that if `wdir_data` is a DataFrame, the adjustment derived from `logger_offset` and `target_orientation` 
    is applied to all columns.

    Function returns also a DataFrame containing the target orientation, logger orientation and offset applied
    for each relevant time period.
    
    This function uses the brightwind 'offset_wind_direction()' function to apply the actual adjustment to 
    the wind direction data.

    If there is a value in the logger measurement config for an offset, then the wind direction data has already 
    been adjusted by this amount. This may or may not be equal to the target orientation. Therefore, 
    the adjustment to be made should make up the difference to equal a target orientation. E.g.

            offset to be applied = target_orientation - logger_offset

    This function accounts for this adjustment.

    Date ranges are considered as [from, to) where 'from' is inclusive and 'to' is exclusive.

    :param wdir_data:                       The wind direction data time series.
    :type wdir_data:                        pd.Series or pd.DataFrame
    :param logger_offset:                   The logger offset value in degrees for the input wind direction data.
    :type logger_offset:                    float
    :param target_orientation:              The target orientation value in degrees.
    :type target_orientation:               float
    :param apply_offset_from:               The date to apply the offset from.
    :type apply_offset_from:                str | datetime.datetime | pd.Timestamp
    :param apply_offset_to:                 The date to apply the offset to, treated in and exclusive manner.
    :type apply_offset_to:                  str | datetime.datetime | pd.Timestamp
    :param target_orientation_name:         The target orientation name to use for the print statements. 
                                            e.g 'device orientation' or 'deadband orientation'
    :type target_orientation_name:          str
    :param heights:                         The height(s) corresponding to the measurement(s) in
                                            wdir_data .
    :type heights:                          list | float
    :param target_orientation_table_name:   The target orientation name to use for the column of the returned results
                                            DataFrame.      
    :type target_orientation_table_name:    str 
    :return:                                A tuple of the data and a DataFrame with the appropriate offset applied
                                            in addition to a table of the offset information.
    :rtype:                                 Tuple[pd.DataFrame, pd.DataFrame]    
    """

    offset = target_orientation
    wdir_names = list(wdir_data.columns) if isinstance(wdir_data, pd.DataFrame) else [wdir_data.name]
    additional_comment_txt = 'to account for {}'.format(target_orientation_name)

    if apply_offset_to is None or apply_offset_to == DATE_INSTEAD_OF_NONE:
        to_text = "end of data"
        mask = (wdir_data.index >= pd.Timestamp(apply_offset_from))
    else:
        mask = (wdir_data.index >= pd.Timestamp(apply_offset_from)) & (wdir_data.index < pd.Timestamp(apply_offset_to))
        if pd.Timestamp(apply_offset_to) not in wdir_data.index:
            valid_timestamps = wdir_data.index[wdir_data.index <= pd.Timestamp(apply_offset_to)]
            apply_offset_to_inclusive = valid_timestamps[-1].strftime('%Y-%m-%dT%H:%M:%S')
        else:
            idx_pos = wdir_data.index.get_indexer([pd.Timestamp(apply_offset_to)], method='nearest')[0]
            apply_offset_to_inclusive = wdir_data.index[idx_pos - 1].strftime('%Y-%m-%dT%H:%M:%S')
        to_text = f"{apply_offset_to} (exclusive)"
        if apply_offset_from > apply_offset_to_inclusive:
            return wdir_data, pd.DataFrame([])   

    if logger_offset is not None and logger_offset != 0 and target_orientation is not None:
        offset = offset_wind_direction(float(target_orientation), offset=-float(logger_offset))
        additional_comment_txt = additional_comment_txt + ' and logger offset'
    if offset:    
        # Apply offset only to the masked data
        wdir_data.loc[mask] = offset_wind_direction(wdir_data.loc[mask], float(offset))
        print_statement = '{0} adjusted by {1} degrees from {2} to {3} {4}.\n'
        if logger_offset is None:       
            print_statement = print_statement.strip("\n") + ' The logger offset value is set as None.\n'

        print(print_statement.format(utils.bold(", ".join(wdir_names)), utils.bold(str(offset)),
                        utils.bold(str(apply_offset_from)), utils.bold(to_text),
                        additional_comment_txt))
    elif offset == 0:  
        print_statement = '{0} has an offset to be applied of 0 degrees from {1} to {2} {3}.\n'
        if logger_offset is None:  
            print_statement = print_statement.strip("\n") + ' The logger offset value is set as None.\n'     
        print(print_statement.format(utils.bold(str(", ".join(wdir_names))), utils.bold(str(apply_offset_from)),
                    utils.bold(to_text),
                    additional_comment_txt))
        
    else:
        if logger_offset is None:
            print('{0} has {1} as None and logger offset as None from {2} to {3}. No adjustment applied.\n'
                .format(utils.bold(str(", ".join(wdir_names))), target_orientation_name,
                        utils.bold(str(apply_offset_from)), utils.bold(to_text)))
        else:
            print('{0} has {1} as None from {2} to {3}. No adjustment applied.\n'
                .format(utils.bold(str(", ".join(wdir_names))), target_orientation_name,
                        utils.bold(str(apply_offset_from)), utils.bold(to_text)))
        
    rows = []
    if not isinstance(heights, (list, tuple, np.ndarray)):
        heights = [heights] * len(wdir_names)
    for (wdir_name, height) in zip(wdir_names, heights):
        rows.append({
            "Name": wdir_name,
            "Height [m]": height,
            target_orientation_table_name: target_orientation,
            "Logger Offset": logger_offset,
            "Offset Applied [deg]": offset,
            "Date From": apply_offset_from,
            "Date To": apply_offset_to
            })
    return wdir_data, pd.DataFrame(rows)
