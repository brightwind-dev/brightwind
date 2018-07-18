import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List, Dict
from analyse.reanalysis import mean_of_monthly_means
import numpy as np
import matplotlib.pyplot as plt

def _convert_days_to_hours(prd):
    return str(int(prd[:-1])*24)+'H'


def _convert_weeks_to_hours(prd):
    return str(int(prd[:-1])*24*7)+'H'


def _get_min_overlap_timestamp( df1_timestamps, df2_timestamps):
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
    if period[-3:]=='min':
        return '{year}-{month}-{day} {hour}:00:00'.format(year=timestamp.year, month=timestamp.month,
                                                             day = timestamp.day, hour=timestamp.hour)
    elif period[-1]=='H' or period[-1]=='D' or period[-1]=='W':
        return '{year}-{month}-{day}'.format(year=timestamp.year,
                                                    month=timestamp.month, day=timestamp.day, hour=timestamp.hour)
    elif period[-1]=='M' or period[-2:]=='MS':
        return '{year}-{month}'.format(year=timestamp.year, month=timestamp.month)
    elif period[-2:]=='AS' or period[-1:]=='A':
        return '{year}'.format(year=timestamp.year)
    else:
        print("Warning: Averaging period not identified returning default timestamps")
        return '{year}-{month}-{day} {hour}:{minute}:{second}'.format(year=timestamp.year, month=timestamp.month,
                                                             day = timestamp.day, hour=timestamp.hour,
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
    #print("Start ", start)
    return df1[start:],df2[start:]


def _max_coverage_count(data_index, averaged_data_index)->pd.Series:
    max_pts = (averaged_data_index.to_series().diff().shift(-1)) / _get_data_resolution(data_index)
    max_pts[-1] = ((averaged_data_index[-1] + 1) - averaged_data_index[-1]) / _get_data_resolution(data_index)
    max_pts.name = 'Max_pts'
    return max_pts


def _filter_by_coverage_threshold(data, data_averaged, coverage_threshold):
    data_averaged['Coverage'] = data_averaged['Count'] / _max_coverage_count(data.index, data_averaged.index)
    return data_averaged[data_averaged["Coverage"]>=coverage_threshold]


def _average_data_by_period(data: pd.Series, period: str) -> pd.DataFrame:
    """Averages the data by the time period specified by period.
    Set period to 1D for a daily average, 3D for three hourly average, similarly 5D, 7D, 15D etc.
    Set period to 1H for hourly average, 3H for three hourly average and so on for 5H, 6H
     etc.
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
    #print("period",period)
    #print("Before resampling:", data)
    grouper_obj = data.resample(period, axis=0, closed='left', label='left',base=0,
                                convention='start', kind='timestamp')
    #print(grouper_obj)
    num_data_points = grouper_obj.count()
    num_data_points.name = 'Count'
    grouped_means = grouper_obj.mean()
    grouped_data = pd.concat([grouped_means, num_data_points], axis=1)
    #print("After resampling", grouped_data)

    #grouped_data = grouped_data[grouped_data['Count'] >= min_pts_in_period].dropna()
    #grouped_data.drop(['Count'], axis=1)
    return grouped_data


def _preprocess_data_for_correlations(ref: pd.DataFrame, target:pd.DataFrame, averaging_prd, coverage_threshold):
    ref_overlap, target_overlap = _get_overlapping_data(ref, target, averaging_prd)
    #print("ref_overlap:", ref_overlap.index)
    #print("target_overlap:", target_overlap.index)
    ref_overlap_avgd = _average_data_by_period(ref_overlap, averaging_prd)
    target_overlap_avgd = _average_data_by_period(target_overlap, averaging_prd)

    ref_filtered_for_coverage = _filter_by_coverage_threshold(ref, ref_overlap_avgd, coverage_threshold)
    target_filtered_for_coverage = _filter_by_coverage_threshold(target, target_overlap_avgd, coverage_threshold)
    #print(pd.concat([ref_filtered_for_coverage, target_filtered_for_coverage], axis=1, join='inner'))

    return ref_filtered_for_coverage.drop(['Count','Coverage'], axis=1), \
           target_filtered_for_coverage.drop(['Count','Coverage'], axis=1)


def _scatter_plot(x, y, x_label, y_label):
    fig2 = plt.figure(111)
    scat = fig2.add_subplot(111)
    scat.set_xlabel(x_label)
    scat.set_ylabel(y_label)
    scat.scatter(x, y)
    fig2.set_figwidth(10)
    fig2.set_figheight(10)
    plt.show()



def linear_regression(ref: pd.Series, target: pd.Series, averaging_prd: str, coverage_threshold: float, plot:bool=False):
    """Accepts two dataframes with timestamps as indexes and averaging period.
    :param: ref_speed : Dataframe containing reference speed as a column, timestamp as the index.
    :param: target_speed: Dataframe containing target speed as a column, timestamp as the index.
    :averaging_prd: Groups data by the period specified by period.
        2T, 2 min for minutely average
        Set period to 1D for a daily average, 3D for three hourly average, similarly 5D, 7D, 15D etc.
        Set period to 1H for hourly average, 3H for three hourly average and so on for 5H, 6H etc.
        week
        Set period to 1MS for monthly average
        Set period to 1AS fo annual average
    :return :A dictionary containing the following keys, r2, slope, offset and num_data_points
    """
    ref = ref.sort_index().dropna()
    target = target.sort_index().dropna()

    ref_processed, target_processed = _preprocess_data_for_correlations(ref, target, averaging_prd, coverage_threshold)
    #print("ref_processed:",ref_processed.columns, ref_processed)
    #print("target_processed:", target_processed.columns, target_processed)

    common_idxs, data_pts = _common_idxs(ref_processed, target_processed)

    if plot == True:
        _scatter_plot(ref_processed.loc[common_idxs].values,target_processed.loc[common_idxs].values, "Reference Data", "Target Data  ")
    # Linear Regression
    model = LinearRegression()
    model.fit(ref_processed.loc[common_idxs].values,
              target_processed.loc[common_idxs].values)
    r2 = model.score(ref_processed.loc[common_idxs].values,
                     target_processed.loc[common_idxs].values)
    prediction = model.predict(ref_processed.loc[common_idxs].values)
    slope = model.coef_
    offset = model.intercept_
    rmse = mean_squared_error(prediction,target_processed.loc[common_idxs].values) **0.5
    mae = mean_absolute_error(prediction,target_processed.loc[common_idxs].values)
    # lt_ref_speed = mean_of_monthly_means(ref_speed).mean()
    # predicted_lt_speed  = calc_target_value_by_linear_model(lt_ref_speed, slope[0], offset[0])
    return {'num_data_points': data_pts, 'slope': slope, 'offset': offset, 'r2': r2, 'RMSE':rmse, 'MAE':mae}