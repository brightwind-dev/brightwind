import numpy as np
import pandas as pd
import math
from utils import utils


def _compute_wind_vector(wspd, wdir):
    """Returns north and east component of wind-vector"""
    return wspd*np.cos(wdir), wspd*np.sin(wdir)


def _convert_days_to_hours(prd):
    return str(int(prd[:-1])*24)+'H'


def _convert_weeks_to_hours(prd):
    return str(int(prd[:-1])*24*7)+'H'


def _get_min_overlap_timestamp( df1_timestamps, df2_timestamps):
    """Get the minimum overlapping timestamp from two series"""
    if df1_timestamps.max()\
            <df2_timestamps.min() or df1_timestamps.min()>df2_timestamps.max():
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


def _get_overlapping_data(df1, df2, period):
    start = _round_timestamp_down_to_averaging_prd(_get_min_overlap_timestamp(df1.index, df2.index), period)
    return df1[start:],df2[start:]


def _max_coverage_count(data_index, averaged_data_index)->pd.Series:
    """For a given resolution of data finds the maximum number of data points in the averaging period"""
    max_pts = (averaged_data_index.to_series().diff().shift(-1)) / _get_data_resolution(data_index)
    max_pts[-1] = ((averaged_data_index[-1] + 1) - averaged_data_index[-1]) / _get_data_resolution(data_index)
    return max_pts


def _get_coverage_series(data, grouper_obj):
    coverage = grouper_obj.count().divide(_max_coverage_count(data.index, grouper_obj.mean().index), axis=0)
    return coverage


def average_data_by_period(data: pd.Series, period, aggregation_method='mean', filter=False, coverage_threshold=1, return_coverage=False) -> pd.DataFrame:
    """Averages the data by the time period specified by period.
    Set period to 1D for a daily average, 3D for three hourly average, similarly 5D, 7D, 15D etc.
    Set period to 1H for hourly average, 3H for three hourly average and so on for 5H, 6H etc.
    Set period to 1M for monthly average
    Set period to 1AS for annual taking start of the year as the date
    For minutes use 10min, 20 min, etc.
    Can be a DateOffset object too
    """
    data = data.sort_index()
    if isinstance(period, str):
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

    coverage = _get_coverage_series(data, grouper_obj)

    if filter:
        grouped_data = grouped_data.loc[(coverage >= coverage_threshold)]

    if return_coverage:
        return grouped_data, coverage.rename('Coverage')
    else:
        return grouped_data


def scale_wind_speed(spd: pd.Series, scale_factor: float) ->pd.Series:
    """
    Scales wind speed by the scale_factor
    :param spd: Series or data frame or a single value of wind speed to scale
    :param scale_factor: Scaling factor in decimal, if scaling factor is 0.8 output would be (1+0.8) times wind speed,
    if it is -0.8 the output would be (1-0.8) times the wind speed
    :return: Series or data frame with scaled wind speeds
    """
    return spd*(1+scale_factor)


def offset_wind_direction(dir: pd.Series, offset: float) -> pd.Series:
    """
    Add/ subtract offset from wind direction. Keeps the ranges between 0 to 360
    :param dir: Series or data frame or a single direction to offset
    :param offset: Offset in degrees can be negative or positive
    :return: Series or data frame with offsetted directions
    """
    if isinstance(dir, float):
        return utils._range_0_to_360(dir + offset)
    else:
        return (dir + offset).apply(utils._range_0_to_360)


def _preprocess_data_for_correlations(ref: pd.DataFrame, target: pd.DataFrame, averaging_prd, coverage_threshold,
                                      aggregation_method_ref='mean', aggregation_method_target='mean'):
    ref_overlap, target_overlap = _get_overlapping_data(ref.sort_index().dropna(), target.sort_index().dropna(), averaging_prd)
    from pandas.tseries.frequencies import to_offset
    ref_resolution = _get_data_resolution(ref_overlap.index)
    target_resolution = _get_data_resolution(target_overlap.index)
    if ref_resolution > target_resolution and (to_offset(ref_resolution)!= to_offset(averaging_prd)):
        target_overlap = average_data_by_period(target_overlap, to_offset(ref_resolution), filter=True,
                                                   coverage_threshold=1, aggregation_method=aggregation_method_target)
    if ref_resolution < target_resolution and (to_offset(target_resolution)!= to_offset(averaging_prd)):
        ref_overlap = average_data_by_period(ref_overlap, to_offset(target_resolution), filter=True,
                                                coverage_threshold=1, aggregation_method=aggregation_method_ref)
    common_idxs, data_pts = _common_idxs(ref_overlap, target_overlap)
    ref_concurrent = ref_overlap.loc[common_idxs]
    target_concurrent = target_overlap.loc[common_idxs]
    return average_data_by_period(ref_concurrent, averaging_prd, filter=True, coverage_threshold=coverage_threshold, aggregation_method=aggregation_method_ref), \
           average_data_by_period(target_concurrent, averaging_prd, filter=True, coverage_threshold=coverage_threshold, aggregation_method=aggregation_method_target)


def _preprocess_dir_data_for_correlations(ref_spd: pd.DataFrame, ref_dir: pd.DataFrame, target_spd:pd.DataFrame,
                                          target_dir: pd.DataFrame, averaging_prd, coverage_threshold):
    ref_N, ref_E= _compute_wind_vector(ref_spd.sort_index().dropna(), ref_dir.sort_index().dropna().map(math.radians))
    target_N, target_E = _compute_wind_vector(target_spd.sort_index().dropna(), target_dir.sort_index().dropna().map(math.radians))
    ref_N_avgd, target_N_avgd = _preprocess_data_for_correlations(ref_N, target_N, averaging_prd=averaging_prd,
                                                                  coverage_threshold=coverage_threshold)
    ref_E_avgd, target_E_avgd = _preprocess_data_for_correlations(ref_E, target_E, averaging_prd=averaging_prd,
                                                                  coverage_threshold=coverage_threshold)
    ref_dir_avgd = np.arctan2(ref_E_avgd, ref_N_avgd).map(math.degrees).map(utils._range_0_to_360)
    target_dir_avgd = np.arctan2(target_E_avgd, target_N_avgd).map(math.degrees).map(utils._range_0_to_360)

    return round(ref_dir_avgd.loc[:]), round(target_dir_avgd.loc[:])


# def _dir_averager(spd_overlap, dir, averaging_prd, coverage_threshold):
#     vec = pd.concat([spd_overlap, dir.apply(degree_to_radian)], axis=1, join='inner')
#     vec.columns = ['spd', 'dir']
#     vec['N'], vec['E'] = _compute_wind_vector(vec['spd'], vec['dir'])
#     vec_N_avgd = average_data_by_period(vec['N'], averaging_prd, filter=False, return_coverage=False)
#     vec_E_avgd = average_data_by_period(vec['E'], averaging_prd, filter=False,return_coverage=False)
#     vec_dir_avgd = np.arctan2(vec_E_avgd.loc[:,vec_E_avgd.columns], vec_N_avgd.loc[:,vec_N_avgd.columns]).applymap(radian_to_degree).applymap(utils._range_0_to_360)
#     vec_dir_avgd.loc[:] = round(vec_dir_avgd.loc[:])
#     vec_dir_avgd = pd.concat([vec_dir_avgd,vec_E_avgd['Count']], axis=1, join='inner')
#     return vec_dir_avgd


# def _preprocess_data_for_correlations(ref: pd.DataFrame, target: pd.DataFrame, averaging_prd, coverage_threshold):
#     """A wrapper function that calls other functions necessary for pre-processing the data"""
#     ref = ref.sort_index().dropna()
#     target = target.sort_index().dropna()
#     ref_overlap, target_overlap = tf._get_overlapping_data(ref, target, averaging_prd)
#     ref_overlap_avgd = tf.average_data_by_period(ref_overlap, averaging_prd)
#     target_overlap_avgd = tf.average_data_by_period(target_overlap, averaging_prd)
#     ref_filtered_for_coverage = tf._filter_by_coverage_threshold(ref, ref_overlap_avgd, coverage_threshold)
#     target_filtered_for_coverage = tf._filter_by_coverage_threshold(target, target_overlap_avgd, coverage_threshold)
#     common_idxs, data_pts = tf._common_idxs(ref_filtered_for_coverage, target_filtered_for_coverage)
#     return ref_filtered_for_coverage.drop(['Count', 'Coverage'], axis=1).loc[common_idxs], \
#                     target_filtered_for_coverage.drop(['Count', 'Coverage'], axis=1).loc[common_idxs]
