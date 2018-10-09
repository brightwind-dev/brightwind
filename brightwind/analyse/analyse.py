import pandas as pd
import numpy as np
from transform import transform as tf
from utils import utils


def get_concurrent_coverage(ref, target, averaging_prd, aggregation_method_ref='mean', aggregation_method_target='mean'):
    """
    Accepts ref and target data and returns the coverage of concurrent data.
    :param ref: Reference data
    :type ref: pandas.Series
    :param target: Target data
    :type target: pandas.Series
    :param averaging_prd: Groups data by the period specified by period.

            * 2T, 2 min for minutely average
            * Set period to 1D for a daily average, 3D for three hourly average, similarly 5D, 7D, 15D etc.
            * Set period to 1H for hourly average, 3H for three hourly average and so on for 5H, 6H etc.
            * Set period to 1M for monthly average
            * Set period to 1AS fo annual average

    :type averaging_prd: str
    :param aggregation_method_ref: (Optional) Calculates mean of the data for the given averaging_prd by default. Can be
            changed to 'sum', 'std', 'max', 'min', etc. or a user defined function
    :param aggregation_method_target: (Optional) Calculates mean of the data for the given averaging_prd by default.
            Can be changed to 'sum', 'std', 'max', 'min', etc. or a user defined function
    :return: A dataframe with concurrent coverage and resolution of the new data. The columns with coverage are named as
            <column name>_Coverage
    """
    coverage_df = tf._preprocess_data_for_correlations(ref=ref, target=target, averaging_prd=averaging_prd, coverage_threshold=0,
                                                aggregation_method_ref = aggregation_method_ref,
                                                aggregation_method_target = aggregation_method_target,
                                                get_coverage = True)
    coverage_df.columns = ["Coverage" if "_Coverage" in col else col for col in coverage_df.columns ]
    return coverage_df


def mean_of_monthly_means(df: pd.DataFrame) -> pd.DataFrame:

    """ Return series of mean of monthly means for each column in the dataframe with timestamp as the index.
        Calculate the monthly mean for each calendar month and then average the resulting 12 months.
    """
    monthly_df: pd.DataFrame = df.groupby(df.index.month).mean()
    momm_series: pd.Series = monthly_df.mean()
    momm_df: pd.DataFrame = pd.DataFrame([momm_series], columns=['MOMM'])
    return momm_df


def calc_target_value_by_linear_model(ref_value: float, slope: float, offset: float):
    """
    :rtype: np.float64
    """
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
        data = data.loc[date_from:date_to, :]
    elif date_from and date_to:
        import datetime as dt
        date_from = dt.datetime.strptime(date_from[:10], "%Y-%m-%d")
        date_to = dt.datetime.strptime(date_to[:10], "%Y-%m-%d")
        data = data.loc[date_from:date_to, :]
    return mean_of_monthly_means(data).get_value(index=0, col='MOMM')


def get_sector_ratio(wdspd_1, wdspd_2, direction, sectors=12):
    """Accepts two speed series and one direction series and returns the speed ratio by sector
    in a table
    :param wdspd_1: First wind speed series. This is divisor series.
    :type: wdspd_1: pandas.Series
    :param wdpsd_2: Second wind speed series
    :type: wdspd_1: pandas.Series
    :param direction: Series of wind directions
    :type direction: pandas.Series
    :returns Table of speed ratio by sector
    """
    sec_rat = pd.concat([wdspd_1[wdspd_1 >3].rename('speed_1'), wdspd_2[wdspd_2>3].rename('speed_2'), direction.rename('dir')], axis=1,
                        join='inner')
    sec_rat['dir'] = sec_rat['dir'].mod(360)
    sector_ratio = get_distribution_by_wind_sector(sec_rat['speed_2']/sec_rat['speed_1'], sec_rat['dir'],
                                                   sectors= sectors, aggregation_method='mean', direction_bin_array=None,
                                                   direction_bin_labels=None)
    return sector_ratio


def get_distribution(var1_series, var2_series, var2_bin_array=np.arange(-0.5, 41, 1), var2_bin_labels=None,
                     aggregation_method='%frequency'):
    """Accepts 2 series of same/different variables and computes the distribution of first variable with respect to
    the bins of another variable.
    :param var1_series: Series of the variable whose distribution we need to find
    :param var2_series: Series of the variable which we want to bin
    :param var2_bin_array: Array of numbers where adjacent elements of array form a bin
    :param var2_bin_labels: Labels of bins to be used, uses (bin-start, bin-end] format by default
    :param aggregation_method: Statistical method used to find distribution it can be mean, max, min, std, count,
    describe, a custom function, etc,computes frequency in percentages by default
    :returns A DataFrame/Series with bins as row indexes and columns with statistics chosen by aggregation_method"""
    if isinstance(var1_series, pd.DataFrame) and var1_series.shape[1]==1:
        var1_series = var1_series.iloc[:,0]
    if isinstance(var2_series, pd.DataFrame) and var2_series.shape[1]==1:
        var2_series = var2_series.iloc[:,0]
    var1_series = var1_series.dropna()
    var2_series = var2_series.dropna()
    var2_binned_series = pd.cut(var2_series, var2_bin_array, right=False, labels=var2_bin_labels).rename('variable_bin')
    data = pd.concat([var1_series.rename('data'), var2_binned_series], join='inner', axis=1)
    if aggregation_method == '%frequency':
        return data.groupby(['variable_bin'])['data'].count().rename('%frequency')/len(data) * 100.0
    else:
        return data.groupby(['variable_bin'])['data'].agg(aggregation_method)


def _get_direction_bin_labels(sectors, direction_bins, zero_centred=True):
    mapper = dict()
    for i, lower_bound in enumerate(direction_bins[:sectors]):
        if i == 0 and zero_centred:
            mapper[i+1] = '{0}-{1}'.format(direction_bins[-2], direction_bins[1])
        else:
            mapper[i+1] = '{0}-{1}'.format(lower_bound, direction_bins[i+1])
    return mapper.values()


def map_direction_bin(wdir, bins, sectors):
    kwargs = {}
    if wdir == max(bins):
        kwargs['right'] = True
    else:
        kwargs['right'] = False
    bin_num = np.digitize([wdir], bins, **kwargs)[0]
    if bin_num == sectors+1:
        bin_num = 1
    return bin_num


def get_binned_direction_series(direction_series, sectors, direction_bin_array=None):
    """ Accepts a series with wind directions and number of sectors  you want to divide.
    :param  direction_series: Series of directions to bin
    :param  sectors: number of direction sectors
    :param direction_bin_array: An optional parameter, if you want custom direction bins pass an array
                        of the bins. If nto specified direction_bins will be centered around 0
    :returns  A series with direction-bins, bins centered around 0 degree by default if direction_bin_array
    is not specified
    """
    if direction_bin_array is None:
        direction_bin_array = utils.get_direction_bin_array(sectors)
    return direction_series.apply(map_direction_bin, bins=direction_bin_array, sectors=sectors)


def get_distribution_by_wind_sector(var_series, direction_series, sectors=12, aggregation_method='%frequency',
                                    direction_bin_array=None, direction_bin_labels=None):
    """Accepts a series of a variable and  wind direction. Computes the distribution of first variable with respect to
    wind direction sectors
    :param var_series: Series of the variable whose distribution we need to find
    :param direction_series: Series of wind directions between [0-360]
    :param sectors: Number of sectors to bin direction to. The first sector is centered at 0 by default. To change that
            behaviour specify direction_bin_array
    :param aggregation_method: Statistical method used to find distribution it can be mean, max, min, std, count,
            describe, a custom function, etc. Computes frequency in percentages by default
    :param direction_bin_array: Optional, to change default behaviour of first sector centered at 0 assign an array of
            bins to this
    :param direction_bin_labels: Optional, you can specify an array of labels to be used for the bins. uses string
            labels of the format '30-90' by default
    :returns A dataframe/series with wind direction sector as row indexes and columns with statistics chosen by
            aggregation_method
    """
    var_series = var_series.dropna()
    direction_series = direction_series.dropna()
    if direction_bin_array is None:
        direction_bin_array = utils.get_direction_bin_array(sectors)
        zero_centered = True
    else:
        sectors = len(direction_bin_array)-1
        zero_centered = False
    if direction_bin_labels is None:
        direction_bin_labels = _get_direction_bin_labels(sectors, direction_bin_array, zero_centered)
    direction_binned_series = get_binned_direction_series(direction_series, sectors, direction_bin_array)\
        .rename('direction_bin')
    data = pd.concat([var_series.rename('data'), direction_binned_series], join='inner', axis=1)
    if aggregation_method == '%frequency':
        result = data.groupby(['direction_bin'])['data'].count().rename('%frequency')/len(data) * 100.0
    else:
        result = data.groupby(['direction_bin'])['data'].agg(aggregation_method)
    result.index = direction_bin_labels
    return result


def get_freq_table(var_series, direction_series, var_bin_array=np.arange(-0.5, 41, 1), sectors=12, var_bin_labels=None,
                   direction_bin_array=None, direction_bin_labels=None, freq_as_percentage=True):
    """Accepts a variable series and direction series and computes a frequency table of percentages. Both variable and
    direction are binned
    :param var_series: Series of variable to be binned
    :param direction_series: Series of wind directions between [0-360]
    :param var_bin_array: Array of numbers where adjacent elements of array form a bin
    :param sectors: Number of sectors to bin direction to. The first sector is centered at 0 by default. To change that
            behaviour specify direction_bin_array
    :param var_bin_labels: Optional, an array of labels to use for variable bins
    :param direction_bin_array: Optional, to change default behaviour of first sector centered at 0 assign an array of
    bins to this
    :param direction_bin_labels: Optional, you can specify an array of labels to be used for the bins. uses string
    labels of the format '30-90' by default
    :param freq_as_percentage: Optional, True by default. Returns the frequency as percentages. To return just the count
    change it to False
    :returns A DataFrame with row indexes as variable bins and columns as wind direction bins.
    """
    var_series = var_series.dropna()
    direction_series = direction_series.dropna()
    if direction_bin_array is None:
        direction_bin_array = utils.get_direction_bin_array(sectors)
        zero_centered = True
    else:
        sectors = len(direction_bin_array)-1
        zero_centered = False
    if direction_bin_labels is None:
        direction_bin_labels = _get_direction_bin_labels(sectors, direction_bin_array, zero_centered)
    var_binned_series = pd.cut(var_series, var_bin_array, right=False, labels=var_bin_labels).rename('variable_bin')
    direction_binned_series = get_binned_direction_series(direction_series, sectors, direction_bin_array).rename(
        'direction_bin')
    data = pd.concat([var_series.rename('var_data'), var_binned_series, direction_binned_series],axis=1).dropna()
    if freq_as_percentage:
        result = pd.crosstab(data.loc[:,'variable_bin'],data.loc[:,'direction_bin']) / len(data) *100.0
    else:
        result = pd.crosstab(data.loc[:, 'variable_bin'], data.loc[:, 'direction_bin'])
    result.columns = direction_bin_labels
    return result.sort_index()


def get_time_continuity_gaps(data):
    """
    Returns the gaps in timestamps for the data, that means that data isn't available for that period.

    :param data: Data for checking continuity, timestamp must be the index
    :type data: pandas.Series or pandas.DataFrame
    :return: A dataframe with days lost and the start and end date between them
    :rtype : pandas.DataFrame

    """
    indexes = data.dropna(how='all').index
    continuity = pd.DataFrame({'Date To': indexes.values.flatten()[1:], 'Date From': indexes.values.flatten()[:-1]})
    continuity['Days Lost'] = (continuity['Date To'] - continuity['Date From']) / pd.Timedelta('1 days')
    return continuity[continuity['Days Lost'] != (tf._get_data_resolution(indexes) / pd.Timedelta('1 days'))]


def get_coverage(data, period='1M', aggregation_method='mean'):
    """
    Get the data coverage over the period specified

    :param data: Data to check the coverage of
    :type data: pandas.Series or pandas.DataFrame
    :param period: Groups data by the period specified by period.

            - 2T, 2 min for minutely average
            - Set period to 1D for a daily average, 3D for three hourly average, similarly 5D, 7D, 15D etc.
            - Set period to 1H for hourly average, 3H for three hourly average and so on for 5H, 6H etc.
            - Set period to 1MS for monthly average
            - Set period to 1AS fo annual average

    :type period: string or pandas.DateOffset
    :param aggregation_method: (Optional) Calculates mean of the data for the given averaging_prd by default. Can be
            changed to 'sum', 'std', 'max', 'min', etc. or a user defined function
    :type aggregation_method: str
    :return: A dataframe with coverage and resolution of the new data. The columns with coverage are named as
            <column name>_Coverage
    """

    return pd.concat(list(tf.average_data_by_period(data, period=period, aggregation_method=aggregation_method,
                                                    filter=False, return_coverage=True)),axis=1, join='outer')


def basic_stats(data):
    """
    Gives basic statistics like mean, standard deviation, count, etc. of data,  excluding NaN values

    :param data: Meteorological data
    :param data type: pandas.Series or pandas.DataFrame
    :rtype: A dataframe or series containing statistics
    """
    if isinstance(data, pd.DataFrame):
        return data.describe(percentiles=[0.5]).T.drop(['50%'], axis=1)
    else:
        return data.to_frame().describe(percentiles=[0.5]).T.drop(['50%'], axis=1)


def get_TI_by_speed(wdspd, wdspd_std, speed_bin_array=np.arange(-0.5, 41, 1), speed_bin_labels=range(0, 41),
                    percentile=90):
    """
    Accepts a wind speed series and its standard deviation, calculates turbulence intensity (TI) and returns the
    distribution by of TI by speed bins

    :param wdspd: Wind speed data series
    :type wdspd: pandas.Series
    :param wdspd_std: Wind speed standard deviation data series
    :type wdspd_std: pandas.Series
    :param speed_bin_array: (Optional) Array of wind speeds where adjacent elements of array form a bin
    :type speed_bin_array: List or array
    :param speed_bin_labels: (Optional) Labels to use for speed bins, 0, 1, 2, 3 .. and so on by default
    :type speed_bin_labels: List, range or array
    :param percentile: The percentile representative of TI (see return for more information)
    :type percentile: float, int
    :return: TI distribution with columns names as:

            * Mean_TI (average TI for a speed bin),
            * TI_Count ( number of data points in the bin),
            * Rep_TI (Representative TI set at 90 percentile by default,
            * TI_2Sigma (2 sigma TI),
            * Char_TI (characteristic TI)
    :rtype: pandas.DataFrame

    """

    if isinstance(wdspd, pd.DataFrame) and wdspd.shape[1]==1:
        wdspd = wdspd.iloc[:,0]
    if isinstance(wdspd_std, pd.DataFrame) and wdspd_std.shape[1]==1:
        wdspd_std = wdspd_std.iloc[:,0]
    ti = pd.concat([wdspd.rename('wdspd'), wdspd_std.rename('wdspd_std')], axis=1, join='inner')
    ti['Turbulence_Intensity'] = ti['wdspd_std'] / ti['wdspd']
    ti_dist = pd.concat([get_distribution(var1_series=ti['Turbulence_Intensity'],
                                             var2_series=ti['wdspd'],
                                             var2_bin_array=speed_bin_array,
                                             var2_bin_labels=speed_bin_labels,
                                             aggregation_method='mean').rename("Mean_TI"),
                         get_distribution(var1_series=ti['Turbulence_Intensity'],
                                             var2_series=ti['wdspd'],
                                             var2_bin_array=speed_bin_array,
                                             var2_bin_labels=speed_bin_labels,
                                             aggregation_method='count').rename("TI_Count"),
                         get_distribution(var1_series=ti['Turbulence_Intensity'],
                                             var2_series=ti['wdspd'],
                                             var2_bin_array=speed_bin_array,
                                             var2_bin_labels=speed_bin_labels,
                                             aggregation_method=lambda x: np.percentile(x, q=percentile)).rename("Rep_TI"),
                         get_distribution(var1_series=ti['Turbulence_Intensity'],
                                             var2_series=ti['wdspd'],
                                             var2_bin_array=speed_bin_array,
                                             var2_bin_labels=speed_bin_labels,
                                             aggregation_method='std').rename("TI_2Sigma")], axis=1, join='inner')
    ti_dist.loc[:,'Char_TI'] = ti_dist.loc[:,'Mean_TI'] + (ti_dist.loc[:,'TI_2Sigma'] / ti_dist.index)
    ti_dist.loc[0, 'Char_TI'] = 0
    ti_dist.index.rename('Speed Bin', inplace=True)
    return ti_dist.dropna(how='any')


def get_TI_by_sector(wdspd, wdspd_std, wddir, min_speed=0, sectors=12, direction_bin_array=None, direction_bin_labels=None):
    """
    Accepts a wind speed series, its standard deviation and a direction series, calculates turbulence intensity (TI)
    and returns the distribution by of TI by sector

    :param wdspd: Wind speed data series
    :type wdspd: pandas.Series
    :param wdspd_std: Wind speed standard deviation data series
    :type wdspd_std: pandas.Series
    :param wddir: Wind direction series
    :type wddir: pandas.Series
    :param direction_bin_array: (Optional) Array of wind speeds where adjacent elements of array form a bin
    :param direction_bin_array: (Optional) To change default behaviour of first sector centered at 0 assign an array of
            bins to this
    :param direction_bin_labels: (Optional) you can specify an array of labels to be used for the bins. uses string
            labels of the format '30-90' by default
    :return: TI distribution with columns names as:

            * Mean_TI (average TI for a speed bin),
            * TI_Count ( number of data points in the bin),

    :rtype: pandas.DataFrame

    """
    if isinstance(wdspd, pd.DataFrame) and wdspd.shape[1]==1:
        wdspd = wdspd.iloc[:,0]
    if isinstance(wdspd_std, pd.DataFrame) and wdspd_std.shape[1]==1:
        wdspd_std = wdspd_std.iloc[:,0]
    if isinstance(wddir, pd.DataFrame) and wddir.shape[1]==1:
        wddir = wddir.iloc[:,0]
    ti = pd.concat([wdspd.rename('wdspd'), wdspd_std.rename('wdspd_std'), wddir.rename('wddir')], axis=1, join='inner')
    ti = ti[ti['wdspd']>=min_speed]
    ti['Turbulence_Intensity'] = ti['wdspd_std'] / ti['wdspd']
    ti_dist = pd.concat([get_distribution_by_wind_sector(var_series=ti['Turbulence_Intensity'],
                                             direction_series=ti['wddir'],
                                             sectors=sectors, direction_bin_array=direction_bin_array,
                                             direction_bin_labels=direction_bin_labels,
                                             aggregation_method='mean').rename("Mean_TI"),
                         get_distribution_by_wind_sector(var_series=ti['Turbulence_Intensity'],
                                             direction_series=ti['wddir'],
                                             sectors=sectors, direction_bin_array=direction_bin_array,
                                                            direction_bin_labels=direction_bin_labels,
                                             aggregation_method='count').rename("TI_Count")], axis=1, join='outer')

    ti_dist.index.rename('Direction Bin', inplace=True)
    return ti_dist.dropna(how='all')


def get_12x24(var_series,aggregation_method='mean'):
    """
    Accepts a variable series and returns 12x24 (months x hours) table for the variable.
    :param var_series:
    :param aggregation_method: 'mean' by default calculates mean of the variable passed. Can change it to
            'sum', 'std', 'min', 'max', 'percentile' for sum, standard deviation, minimum, maximum, percentile
             of the variable respectively. Can also pass a function.
    :type aggregation_method: str or function
    :return: A dataframe with hours as row labels and months as column labels.
    """
    table_12x24 = pd.concat([var_series.rename('Variable'), var_series.index.to_series().dt.month.rename('Month'),
                             var_series.index.to_series().dt.hour.rename('Hour')], axis=1,join='inner')
    return table_12x24.pivot_table(index='Hour', columns='Month', values='Variable',aggfunc=aggregation_method)