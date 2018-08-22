from analyse.reanalysis import mean_of_monthly_means
import numpy as np
import pandas as pd

def _convert_days_to_hours(prd):
    return str(int(prd[:-1])*24)+'H'


def _convert_weeks_to_hours(prd):
    return str(int(prd[:-1])*24*7)+'H'


def _get_min_overlap_timestamp( df1_timestamps, df2_timestamps):
    """Get the minimum overlapping timestamp from two series"""
    if df1_timestamps.max()<df2_timestamps.min() or df1_timestamps.min()>df2_timestamps.max():
        raise IndexError("No overlapping data. Ranges: {0} to {1}  and {2} to {3}"\
                         .format(df1_timestamps.min(),df1_timestamps.max(),df2_timestamps.min(),df2_timestamps.max()),)
    return max(df1_timestamps.min(), df2_timestamps.min())


def _get_data_resolution(data_idx):
    """Get the frequency of data i.e. the most common time interval between timestamps. Returns a timedelta object"""
    import warnings
    time_diff_btw_timestamps = data_idx.to_series().diff()
    most_freq_time_diff = time_diff_btw_timestamps.mode().values[0]
    minimum_time_diff = time_diff_btw_timestamps.min()
    if minimum_time_diff != most_freq_time_diff:
        warnings.warn("Frequency of input "
                      "data might not be determined correctly (mode does not "
                      "match minimum time difference) mode: {0}  minimum time difference {1}".format(most_freq_time_diff,minimum_time_diff))
    return pd.to_timedelta(most_freq_time_diff, unit='s')


def _round_timestamp_down_to_averaging_prd(timestamp, period):
    if period[-3:] == 'min':
        return '{year}-{month}-{day} {hour}:00:00'.format(year=timestamp.year, month=timestamp.month,
                                                             day=timestamp.day, hour=timestamp.hour)
    elif period[-1] == 'H' or period[-1]=='D' or period[-1]=='W':
        return '{year}-{month}-{day}'.format(year=timestamp.year,month=timestamp.month, day=timestamp.day,
                                             hour=timestamp.hour)
    elif period[-1] == 'M' or period[-2:]=='MS':
        return '{year}-{month}'.format(year=timestamp.year, month=timestamp.month)
    elif period[-2:] == 'AS' or period[-1:] == 'A':
        return '{year}'.format(year=timestamp.year)
    else:
        print("Warning: Averaging period not identified returning default timestamps")
        return '{year}-{month}-{day} {hour}:{minute}:{second}'.format(year=timestamp.year, month=timestamp.month,
                                                                      day=timestamp.day, hour=timestamp.hour,
                                                                      minute=timestamp.minute,second=timestamp.second)


def _common_idxs(reference, site):
    """Finds overlapping indexes from two dataframes.
    Coverage is 1 if whole site data is covered. Also returns the number data points
    """
    common = reference.index.intersection(site.index)
    return common, len(common)


def calc_target_value_by_linear_model(ref_value: float, slope: float, offset: float)-> np.float64:
    return (ref_value*slope) + offset


def calc_lt_ref_speed(data: pd.DataFrame, date_from: str='', date_to: str=''):
    """Calculates and returns long term reference speed. Accepts a dataframe
    with timestamps as index column and another column with wind-speed. You can also specify
    date_from and date_to to calculate the long term reference speed for only that period.
    :param: data: Pandas dataframe with timestamp as index and a column with wind-speed
    :param: date_from: Start date as string in format YYYY-MM-DD
    :param: date_to: End date as string in format YYYY-MM-DD
    :returns: Long term reference speed
    """
    import datetime
    if (isinstance(date_from, datetime.date) or isinstance(date_from, datetime.datetime))\
        and (isinstance(date_to,datetime.date) or isinstance(date_to, datetime.datetime)):
        data = data[date_from:date_to]
    elif date_from and date_to:
        import datetime as dt
        date_from = dt.datetime.strptime(date_from[:10], "%Y-%m-%d")
        date_to = dt.datetime.strptime(date_to[:10], "%Y-%m-%d")
        data = data[date_from:date_to]
    return mean_of_monthly_means(data).get_value(index=0, col='MOMM')


def _get_overlapping_data(df1, df2, period):
    start = _round_timestamp_down_to_averaging_prd(_get_min_overlap_timestamp(df1.index, df2.index), period)
    return df1[start:],df2[start:]


def _max_coverage_count(data_index, averaged_data_index)->pd.Series:
    """For a given resolution of data finds the maximum number of data points in the averaging period"""
    max_pts = (averaged_data_index.to_series().diff().shift(-1)) / _get_data_resolution(data_index)
    max_pts[-1] = ((averaged_data_index[-1] + 1) - averaged_data_index[-1]) / _get_data_resolution(data_index)
    max_pts.name = 'Max_pts'
    return max_pts


def _filter_by_coverage_threshold(data, data_averaged, coverage_threshold, filter=True):
    """Remove data with coverage less than coverage_threshold"""
    data_averaged['Coverage'] = data_averaged['Count'] / _max_coverage_count(data.index, data_averaged.index)
    if filter:
        return data_averaged[data_averaged["Coverage"] >= coverage_threshold]
    else:
        return data_averaged


def _average_data_by_period(data: pd.Series, period: str, aggregation_method='mean', drop_count=False) -> pd.DataFrame:
    """Averages the data by the time period specified by period.
    Set period to 1D for a daily average, 3D for three hourly average, similarly 5D, 7D, 15D etc.
    Set period to 1H for hourly average, 3H for three hourly average and so on for 5H, 6H etc.
    Set period to 1M for monthly average
    Set period to 1AS for annual taking start of the year as the date
    For minutes use 10min, 20 min, etc.
    """
    data = data.sort_index()
    if period[-1] == 'D':
        period = _convert_days_to_hours(period)
    if period[-1] == 'W':
        period = _convert_weeks_to_hours(period)
    if period[-1] == 'M':
        period = period+'S'
    grouper_obj = data.resample(period, axis=0, closed='left', label='left',base=0,
                                convention='start', kind='timestamp')
    if aggregation_method == 'mean':
        grouped_data = grouper_obj.mean()
    if aggregation_method == 'sum':
        grouped_data = grouper_obj.sum()
    if not drop_count:
        num_data_points = grouper_obj.count()
        num_data_points.name = 'Count'
        grouped_data = pd.concat([grouped_data, num_data_points], axis=1)
    return grouped_data


def get_coverage(data: pd.Series, period: str='1M'):
    return _filter_by_coverage_threshold(data, _average_data_by_period(data, period),
                                  coverage_threshold=0, filter=False)
