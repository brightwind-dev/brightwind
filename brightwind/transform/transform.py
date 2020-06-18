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
import math
from brightwind.utils import utils

__all__ = ['average_data_by_period',
           'adjust_slope_offset',
           'scale_wind_speed',
           'offset_wind_direction',
           'selective_avg',
           'offset_timestamps']


def _compute_wind_vector(wspd, wdir):
    """
    Returns north and east component of wind-vector
    """
    return wspd*np.cos(wdir), wspd*np.sin(wdir)


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
            raise IndexError("No overlapping data. Ranges: {0} to {1}  and {2} to {3}"
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
    Get the frequency of data i.e. the most common time interval between timestamps.

    The algorithm finds the most common time difference between consecutive time stamps and returns the
    most common time stamp. Also checks the most common time difference and the minimum time difference. If they
    do not match it shows a warning. It is suggested to manually look at the data if such a warning is shown.

    :param data_idx: Indexes of the DataFrame or series
    :type data_idx: pandas.DataFrame.index or pandas.Series.index
    :return: A time delta object which represents the time difference between consecutive timestamps.
    :rtype: pandas.Timedelta

    **Example usage**
    ::
        import brightwind as bw
        df = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)
        resolution = bw._get_data_resolution(df.Spd80mS.index)
        #To check the number of seconds in resolution
        print(resolution.seconds)


    """

    import warnings
    time_diff_btw_timestamps = data_idx.to_series().diff()
    most_freq_time_diff = time_diff_btw_timestamps.mode().values[0]
    minimum_time_diff = time_diff_btw_timestamps.min()
    if minimum_time_diff != most_freq_time_diff:
        warnings.warn("Frequency of input data might not be determined correctly (most frequent time "
                      "difference between adjacent timestamps"
                      " does not match minimum time difference) most frequent time difference: {0}  "
                      "minimum time difference {1}. Using most frequent time difference as resolution"
                      .format(pd.to_timedelta(most_freq_time_diff, unit='s'), minimum_time_diff))
    return pd.to_timedelta(most_freq_time_diff, unit='s')


def _round_timestamp_down_to_averaging_prd(timestamp, period):
    if period[-3:] == 'min':
        return '{year}-{month}-{day} {hour}:00:00'.format(year=timestamp.year, month=timestamp.month,
                                                          day=timestamp.day, hour=timestamp.hour)
    elif period[-1] == 'H' or period[-1] == 'D' or period[-1] == 'W':
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
    Coverage is 1 if whole site data is covered. Also returns the number data points
    """
    common = reference.index.intersection(site.index)
    return common, len(common)


def _get_overlapping_data(df1, df2, period):
    if isinstance(period, str):
        start = _round_timestamp_down_to_averaging_prd(_get_min_overlap_timestamp(df1.index, df2.index), period)
    else:
        start = _get_min_overlap_timestamp(df1.index, df2.index)
    return df1[start:], df2[start:]


def _max_coverage_count(data_index, averaged_data_index)->pd.Series:
    """
    For a given resolution of data finds the maximum number of data points in the averaging period
    """
    max_pts = (averaged_data_index.to_series().diff().shift(-1)) / _get_data_resolution(data_index)
    max_pts[-1] = (((averaged_data_index[-1] + 1*averaged_data_index[-1].freq) - averaged_data_index[-1]) /
                   _get_data_resolution(data_index))
    return max_pts


def _get_coverage_series(data, grouper_obj):
    coverage = grouper_obj.count().divide(_max_coverage_count(data.index, grouper_obj.mean().index), axis=0)
    return coverage


def average_data_by_period(data, period, aggregation_method='mean', coverage_threshold=None,
                           return_coverage=False):
    """
    Averages the data by the time period specified by period.

    Aggregates data by the aggregation_method specified, by default this function averages the data to the period 
    specified. Can be used to find hourly, daily, weekly, etc. averages or sums. Can also return coverage and 
    filter the returned data by coverage.

    :param data: Data to find average or aggregate of
    :type data: pandas.Series
    :param period: Groups data by the period specified here. The following formats are supported

            - Set period to 10min for 10 minute average, 20min for 20 minute average and so on for 4min, 15min, etc.
            - Set period to 1H for hourly average, 3H for three hourly average and so on for 5H, 6H etc.
            - Set period to 1D for a daily average, 3D for three day average, similarly 5D, 7D, 15D etc.
            - Set period to 1W for a weekly average, 3W for three week average, similarly 2W, 4W etc.
            - Set period to 1M for monthly average
            - Set period to 1AS fo annual average
            - Can be a DateOffset object too

    :type period: str or pandas.DateOffset
    :param aggregation_method: Default `mean`, returns the mean of the data for the specified period. Can also use
        `median`, `prod`, `sum`, `std`,`var`, `max`, `min` which are shorthands for median, product, summation,
        standard deviation, variance, maximum and minimum respectively.
    :type aggregation_method: str
    :param coverage_threshold: Coverage is defined as the ratio of number of data points present in the period and the 
        maximum number of data points that a period should have. Example, for 10 minute data resolution and a period of 
        1 hour, the maximum number of data points in one period is 6. But if the number if data points available is only
        3 for that hour the coverage is 3/6=0.5. It should be greater than 0 and less than or equal to 1. It is set to 
        None by default. If it is None or 0, data is not filtered. Otherwise periods are removed where coverage is less 
        than the coverage_threshold are removed.
    :type coverage_threshold: float
    :param return_coverage: If True appends and additional column in the DataFrame returned, with coverage calculated
        for each period. The columns with coverage are named as <column name>_Coverage
    :type return_coverage: bool
    :returns: A DataFrame with data aggregated with the specified aggregation_method (mean by default). Additionally it
        could be filtered based on coverage and have a coverage column depending on the parameters.
    :rtype: DataFrame

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)

        #To find hourly averages
        data_hourly = bw.average_data_by_period(data.Spd80mN, period='1H')

        #To find monthly averages
        data_monthly = bw.average_data_by_period(data.Spd80mN, period='1M')

        #To filter months where half of the data is missing
        data_monthly_filtered = bw.average_data_by_period(data.Spd80mN, period='1M', coverage_threshold=0.5)

        #To check the coverage for all months
        data_monthly_filtered = bw.average_data_by_period(data.Spd80mN, period='1M', return_coverage=True)


    """
    if coverage_threshold is None:
        coverage_threshold = 0

    if coverage_threshold < 0 or coverage_threshold > 1:
        raise TypeError("Invalid coverage_threshold, should be between 0 and 1, both ends inclusive")

    data = data.sort_index()
    if isinstance(period, str):
        if period[-1] == 'D':
            period = _convert_days_to_hours(period)
        if period[-1] == 'W':
            period = _convert_weeks_to_hours(period)
        if period[-1] == 'M':
            period = period+'S'
        if period[-1] == 'Y':
            raise TypeError("Please use '1AS' for annual frequency at the start of the year.")
    grouper_obj = data.resample(period, axis=0, closed='left', label='left', base=0,
                                convention='start', kind='timestamp')

    grouped_data = grouper_obj.agg(aggregation_method)
    coverage = _get_coverage_series(data, grouper_obj)

    grouped_data = grouped_data[coverage >= coverage_threshold]

    if return_coverage:
        if isinstance(coverage, pd.DataFrame):
            coverage.columns = [col_name+"_Coverage" for col_name in coverage.columns]
        elif isinstance(coverage, pd.Series):
            coverage = coverage.rename(grouped_data.name+'_Coverage')
        else:
            raise TypeError("Coverage not calculated correctly. Coverage", coverage)
        return grouped_data, coverage[coverage >= coverage_threshold]
    else:
        return grouped_data


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
        df = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)
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
        data = bw.load_csv(bw.datasets.demo_data)

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


def _preprocess_data_for_correlations(ref: pd.DataFrame, target: pd.DataFrame, averaging_prd, coverage_threshold,
                                      aggregation_method_ref='mean', aggregation_method_target='mean',
                                      get_coverage=False):
    ref_overlap, target_overlap = _get_overlapping_data(ref.sort_index().dropna(), target.sort_index().dropna(),
                                                        averaging_prd)
    from pandas.tseries.frequencies import to_offset
    ref_resolution = _get_data_resolution(ref_overlap.index)
    target_resolution = _get_data_resolution(target_overlap.index)
    if (to_offset(ref_resolution) != to_offset(averaging_prd)) and \
            (to_offset(target_resolution) != to_offset(averaging_prd)):
        if ref_resolution > target_resolution:
            target_overlap = average_data_by_period(target_overlap, to_offset(ref_resolution),
                                                    coverage_threshold=1,
                                                    aggregation_method=aggregation_method_target)
        if ref_resolution < target_resolution:
            ref_overlap = average_data_by_period(ref_overlap, to_offset(target_resolution),
                                                 coverage_threshold=1,
                                                 aggregation_method=aggregation_method_ref)
        common_idxs, data_pts = _common_idxs(ref_overlap, target_overlap)
        ref_overlap = ref_overlap.loc[common_idxs]
        target_overlap = target_overlap.loc[common_idxs]

    if get_coverage:
        return pd.concat([average_data_by_period(ref_overlap, averaging_prd,
                                                 coverage_threshold=0, aggregation_method=aggregation_method_ref)] +
                         list(average_data_by_period(target_overlap, averaging_prd,
                                                     coverage_threshold=0, aggregation_method=aggregation_method_target,
                                                     return_coverage=True)),
                         axis=1)
    else:
        ref_processed, target_processed = average_data_by_period(ref_overlap, averaging_prd,
                                                                 coverage_threshold=coverage_threshold,
                                                                 aggregation_method=aggregation_method_ref), \
                                          average_data_by_period(target_overlap, averaging_prd,
                                                                 coverage_threshold=coverage_threshold,
                                                                 aggregation_method=aggregation_method_target)
        concurrent_idxs, data_pts = _common_idxs(ref_processed, target_processed)
        return ref_processed.loc[concurrent_idxs], target_processed.loc[concurrent_idxs]


def _preprocess_dir_data_for_correlations(ref_spd: pd.DataFrame, ref_dir: pd.DataFrame, target_spd: pd.DataFrame,
                                          target_dir: pd.DataFrame, averaging_prd, coverage_threshold):
    ref_N, ref_E= _compute_wind_vector(ref_spd.sort_index().dropna(), ref_dir.sort_index().dropna().map(math.radians))
    target_N, target_E = _compute_wind_vector(target_spd.sort_index().dropna(),
                                              target_dir.sort_index().dropna().map(math.radians))
    ref_N_avgd, target_N_avgd = _preprocess_data_for_correlations(ref_N, target_N, averaging_prd=averaging_prd,
                                                                  coverage_threshold=coverage_threshold)
    ref_E_avgd, target_E_avgd = _preprocess_data_for_correlations(ref_E, target_E, averaging_prd=averaging_prd,
                                                                  coverage_threshold=coverage_threshold)
    ref_dir_avgd = np.arctan2(ref_E_avgd, ref_N_avgd).map(math.degrees).map(utils._range_0_to_360)
    target_dir_avgd = np.arctan2(target_E_avgd, target_N_avgd).map(math.degrees).map(utils._range_0_to_360)
    return round(ref_dir_avgd.loc[:]), round(target_dir_avgd.loc[:])


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
        data = bw.load_campbell_scientific(bw.datasets.demo_site_data)

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
