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


import pandas as pd
import numpy as np
from brightwind.transform import transform as tf
from brightwind.utils import utils
from brightwind.analyse import plot as plt
import matplotlib


__all__ = ['concurrent_coverage', 'monthly_means', 'momm', 'dist', 'dist_of_wind_speed',
           'distribution_by_dir_sector', 'freq_table', 'freq_distribution', 'time_continuity_gaps', 'coverage',
           'dist_12x24', 'TI', 'wspd_ratio_by_dir_sector', 'Shear', 'calc_air_density']


def concurrent_coverage(ref, target, averaging_prd, aggregation_method_target='mean'):
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
    :param aggregation_method_target: (Optional) Calculates mean of the data for the given averaging_prd by default.
            Can be changed to 'sum', 'std', 'max', 'min', etc. or a user defined function
    :return: A DataFrame with concurrent coverage and resolution of the new data. The columns with coverage are named as
            <column name>_Coverage

    """
    coverage_df = tf._preprocess_data_for_correlations(ref=ref, target=target, averaging_prd=averaging_prd,
                                                       coverage_threshold=0,
                                                       aggregation_method_target=aggregation_method_target,
                                                       get_coverage=True)
    coverage_df.columns = ["Coverage" if "_Coverage" in col else col for col in coverage_df.columns]
    return coverage_df


def calc_target_value_by_linear_model(ref_value: float, slope: float, offset: float):
    """
    :rtype: np.float64
    """
    return (ref_value*slope) + offset


def monthly_means(data, return_data=False, return_coverage=False, ylabel='Wind speed [m/s]'):
    """
    Plots means for calendar months in a timeseries plot. Input can be a series or a DataFrame. Can
    also return data of monthly means with a plot.

    :param data: A timeseries to find monthly means of. Can have multiple columns
    :type data: Series or DataFrame
    :param return_data: To return data of monthly means along with the plot.
    :type return_data: bool
    :param return_coverage: To return monthly coverage along with the data and plot. Also plots the coverage on the
        same graph if only a single series was passed to data.
    :type return_coverage: bool
    :param ylabel: Label for the y-axis, Wind speed [m/s] by default
    :type   ylabel: str
    :return: A plot of monthly means for the input data. If return data is true it returns a tuple where
        the first element is plot and second is data pertaining to monthly means.

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.shell_flats_80m_csv)

        monthly_means_plot, monthly_means = bw.monthly_means(data, return_data=True)
        print("Monthly means data for all the columns:")
        print(monthly_means)
        print("Monthly means plot for all the columns:")
        monthly_means_plot

        # For a single column only
        bw.monthly_means(data.WS80mWS425NW_Avg)

        # Return coverage
        monthly_means_plot, monthly_means = bw.monthly_means(data.WS80mWS425NW_Avg, return_coverage=True)
        monthly_means_plot

    """

    df, covrg = tf.average_data_by_period(data, period='1MS', return_coverage=True)
    if return_data and not return_coverage:
        return plt.plot_monthly_means(df, ylbl=ylabel), df
    if return_coverage:
        return plt.plot_monthly_means(df, covrg, ylbl=ylabel),  pd.concat([df, covrg], axis=1)
    return plt.plot_monthly_means(df, ylbl=ylabel)


def _mean_of_monthly_means_basic_method(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame of mean of monthly means for each column in the DataFrame with timestamp as the index.
    Calculate the monthly mean for each calendar month and then average the resulting 12 months.
    """
    monthly_df: pd.DataFrame = df.groupby(df.index.month).mean().mean().to_frame()
    monthly_df.columns = ['MOMM']
    return monthly_df


def momm(data: pd.DataFrame, date_from: str = '', date_to: str = ''):
    """
    Calculates and returns long term reference speed. Accepts a DataFrame
    with timestamps as index column and another column with wind-speed. You can also specify
    date_from and date_to to calculate the long term reference speed for only that period.

    :param data: Pandas DataFrame with timestamp as index and a column with wind-speed
    :param date_from: Start date as string in format YYYY-MM-DD
    :param date_to: End date as string in format YYYY-MM-DD
    :returns: Long term reference speed

    """
    if isinstance(data, pd.Series):
        momm_data = data.to_frame()
    else:
        momm_data = data.copy()
    sliced_data = utils.slice_data(momm_data, date_from, date_to)
    output = _mean_of_monthly_means_basic_method(sliced_data)
    if output.shape == (1, 1):
        return output.values[0][0]
    return output


def _get_direction_bin_labels(sectors, direction_bins, zero_centred=True):
    mapper = dict()
    for i, lower_bound in enumerate(direction_bins[:sectors]):
        if i == 0 and zero_centred:
            mapper[i+1] = '{0}-{1}'.format(direction_bins[-2], direction_bins[1])
        else:
            mapper[i+1] = '{0}-{1}'.format(lower_bound, direction_bins[i+1])
    return mapper.values()


def _map_direction_bin(wdir, bins, sectors):
    kwargs = {}
    if wdir == max(bins):
        kwargs['right'] = True
    else:
        kwargs['right'] = False
    bin_num = np.digitize([wdir], bins, **kwargs)[0]
    if bin_num == sectors+1:
        bin_num = 1
    return bin_num


def _convert_df_to_series(df):
    """
    Convert a pd.DataFrame to a pd.Series.
    If more than 1 column is in the DataFrame then it will raise a TypeError.
    If the sent argument is not a DataFrame it will return itself.
    :param df:
    :return:
    """
    if isinstance(df, pd.DataFrame) and df.shape[1] == 1:
        return df.iloc[:, 0]
    elif isinstance(df, pd.DataFrame) and df.shape[1] > 1:
        raise TypeError('DataFrame cannot be converted to a Series as it contains more than 1 column.')
    return df


def dist(var_series, var_to_bin_against=None, bins=None, bin_labels=None,
         max_y_value=None, aggregation_method='%frequency', return_data=False):
    """
    Calculates the distribution of a variable against itself as per the bins specified. Can
    also pass another variable for finding distribution with respect to another variable.

    :param var_series: Time-series of the variable whose distribution we need to find
    :type var_series: pandas.Series
    :param var_to_bin_against: (optional) Times-series of the variable which we want to bin against if required
           to bin against another variable.
    :type var_to_bin_against: pandas.Series, None
    :param bins: Array of numbers where adjacent elements of array form a bin. If set to None if derives
                 the min and max from the var_to_bin_against series and creates array in steps of 1.
    :type bins: list, array, None
    :param bin_labels: Labels of bins to be used, uses (bin-start, bin-end] format by default
    :type bin_labels: list, array, None
    :param max_y_value: Max value for the y-axis of the plot to be set. Default will be relative to max calculated
                        data value.
    :type max_y_value: float, int
    :param aggregation_method: Statistical method used to find distribution. It can be mean, max, min, std, count,
           %frequency or a custom function. Computes frequency in percentages by default.
    :type aggregation_method: str or function
    :param return_data: Set to True if you want the data returned.
    :type return_data: bool
    :returns: A distribution plot and, if requested, a pandas.Series with bins as row indexes and column with
              statistics chosen by aggregation_method.

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.datasets.demo_data)

        #For distribution of %frequency of wind speeds
        dist = bw.dist(data.Spd40mN, bins=[0, 8, 12, 21], bin_labels=['normal', 'gale', 'storm'])

        #For distribution of temperature
        temp_dist = bw.dist(data.T2m)

        #For distribution of temperature with set bin array
        temp_dist = bw.dist(data.T2m, bins=[-10, 4, 12, 18, 30])

        #For custom aggregation function
        def custom_agg(x):
            return x.mean()+(2*x.std())
        temp_dist = bw.dist(data.T2m, bins=[-10, 4, 12, 18, 30], aggregation_method=custom_agg)

        #For distribution of mean wind speeds with respect to temperature
        spd_dist = bw.dist(data.Spd40mN, var_to_bin_against=data.T2m,
                           bins=[-10, 4, 12, 18, 30],
                           bin_labels=['freezing', 'cold', 'mild', 'hot'], aggregation_method='mean')

    """
    if var_to_bin_against is None:
        var_to_bin_against = var_series.copy(deep=False)
    var_series = _convert_df_to_series(var_series)
    var_to_bin_against = _convert_df_to_series(var_to_bin_against)
    var_series = var_series.dropna()
    var_to_bin_against = var_to_bin_against.dropna()
    if bins is None:
        bins = np.arange(round(var_to_bin_against.min()-0.5)-0.5, var_to_bin_against.max()+0.5, 1)
    var_binned_series = pd.cut(var_to_bin_against, bins, right=False).rename('variable_bin')
    data = pd.concat([var_series.rename('data'), var_binned_series], join='inner', axis=1)
    if aggregation_method == '%frequency':
        distribution = data.groupby(['variable_bin'])['data'].count().rename('%frequency')/len(data) * 100.0
    else:
        distribution = data.groupby(['variable_bin'])['data'].agg(aggregation_method)

    if not isinstance(aggregation_method, str):
        aggregation_method = aggregation_method.__name__
    graph = plt.plot_freq_distribution(distribution.replace([np.inf, -np.inf], np.NAN).dropna(), max_y_value=max_y_value,
                                       labels=bin_labels, y_label=aggregation_method)
    if bin_labels is not None:
        distribution.index = bin_labels
    if return_data:
        return graph, distribution
    return graph


def dist_of_wind_speed(wspd, max_speed=30, max_y_value=None, return_data=False):
    """
    Accepts a wind speed time series and computes it's frequency distribution. That is, how often does the wind
    blow within each wind speed bin.

    :param wspd: Time series of the wind speed variable whose distribution we need to find.
    :type wspd: pd.Series
    :param max_speed: Max wind speed to consider, default is 30 m/s.
    :type max_speed: int
    :param max_y_value: Max value for the y-axis of the plot to be set. Default will be relative to max calculated
                        data value.
    :type max_y_value: float, int
    :param return_data: Set to True if you want the data returned.
    :type return_data: bool

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.datasets.demo_data)

        freq_dist_plot, freq_dist = bw.dist_of_wind_speed(data.Spd80mN, return_data=True)

    """
    freq_dist = dist(wspd, var_to_bin_against=None, bins=np.arange(-0.5, max_speed+1, 1), bin_labels=None,
                             max_y_value=max_y_value, aggregation_method='%frequency', return_data=True)
    if return_data:
        return freq_dist[0], freq_dist[1]
    return freq_dist[0]


def freq_distribution(wspd, max_speed=30, max_y_value=None, return_data=False):
    """
    This is a wrapper function around `dist_of_wind_speed()`. Please see that function's documentation.
    :param wspd:
    :param max_speed:
    :param max_y_value:
    :param return_data:
    :return:
    """
    return dist_of_wind_speed(wspd, max_speed=max_speed, max_y_value=max_y_value, return_data=return_data)


def _binned_direction_series(direction_series, sectors, direction_bin_array=None):
    """
    Accepts a series with wind directions and number of sectors  you want to divide.

    :param  direction_series: Series of directions to bin
    :param  sectors: number of direction sectors
    :param direction_bin_array: An optional parameter, if you want custom direction bins pass an array
                        of the bins. If nto specified direction_bins will be centered around 0
    :returns: A series with direction-bins, bins centered around 0 degree by default if direction_bin_array
    is not specified

    """
    if direction_bin_array is None:
        direction_bin_array = utils.get_direction_bin_array(sectors)
    return direction_series.dropna().apply(_map_direction_bin, bins=direction_bin_array, sectors=sectors)


def _get_direction_binned_series(sectors, direction_series, direction_bin_array=None, direction_bin_labels=None):
    if direction_bin_array is None:
        direction_bin_array = utils.get_direction_bin_array(sectors)
        zero_centered = True
    else:
        sectors = len(direction_bin_array)-1
        zero_centered = False
    if direction_bin_labels is None:
        direction_bin_labels = _get_direction_bin_labels(sectors, direction_bin_array, zero_centered)
    direction_binned_series = _binned_direction_series(direction_series, sectors, direction_bin_array)\
        .rename('direction_bin')
    return direction_binned_series, direction_bin_labels, sectors, direction_bin_array, zero_centered


def distribution_by_dir_sector(var_series, direction_series, sectors=12, aggregation_method='%frequency',
                               direction_bin_array=None, direction_bin_labels=None, return_data=False):
    """
    Derive the distribution of a time series variable with respect to wind direction sectors. For example, if time
    series of wind speeds is sent, it produces a wind rose.

    :param var_series: Time series of the variable whose distribution we need to find.
    :type var_series:  pd.Series
    :param direction_series: Time series of wind directions between [0-360].
    :type direction_series:  pd.Series
    :param sectors: Number of direction sectors to bin in to. The first sector is centered at 0 by default. To change
                    that behaviour specify direction_bin_array, which overwrites sectors.
    :type sectors: int
    :param aggregation_method: Statistical method used to find distribution it can be mean, max, min, std, count,
            %frequency or a custom function. Computes frequency in percentages by default.
    :type aggregation_method: str
    :param direction_bin_array: Optional, to change default behaviour of first sector centered at 0 assign an array of
            bins to this.
    :type direction_bin_array: list, array, None
    :param direction_bin_labels: Optional, you can specify an array of labels to be used for the bins. Uses string
            labels of the format '30-90' by default. Overwrites sectors.
    :type direction_bin_labels: list, array, None
    :param return_data: Set to True if you want the data returned.
    :type return_data: bool
    :returns: A plot of a rose and a DataFrame/Series with wind direction sector as row indexes and columns with
                statistics chosen by aggregation_method.

    **Example usage**
    ::
        import brightwind as bw
        df = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)

        rose, distribution = bw.distribution_by_dir_sector(df.Spd40mN, df.Dir38mS, return_data=True)

        #For using custom bins
        rose, distribution = bw.distribution_by_dir_sector(df.Spd40mN, df.Dir38mS,
                                direction_bin_array=[0,90,130,200,360],
                                direction_bin_labels=['northerly','easterly','southerly','westerly'],
                                return_data=True)

        #For measuring standard deviation in a sector rather than frequency in percentage (default)
        rose, distribution = bw.distribution_by_dir_sector(df.Spd40mN, df.Dir38mS, aggregation_method='std',
            return_data=True)

    """
    var_series = _convert_df_to_series(var_series)
    direction_series = _convert_df_to_series(direction_series)
    var_series = var_series.dropna()
    direction_series = direction_series.dropna()
    direction_binned_series, direction_bin_labels, sectors, direction_bin_array, zero_centered = \
        _get_direction_binned_series(sectors, direction_series, direction_bin_array, direction_bin_labels)
    data = pd.concat([var_series.rename('data'), direction_binned_series], join='inner', axis=1)
    if aggregation_method == '%frequency':
        result = data.groupby(['direction_bin'])['data'].count().rename('%frequency')/len(data) * 100.0
    else:
        result = data.groupby(['direction_bin'])['data'].agg(aggregation_method)

    for i in range(1, sectors+1):
        if not (i in result.index):
            result[i] = 0.0
    result = result.sort_index()
    result.index = direction_bin_labels
    if return_data:
        return plt.plot_rose(result), result
    else:
        return plt.plot_rose(result)


def freq_table(var_series, direction_series, var_bin_array=np.arange(-0.5, 41, 1), var_bin_labels=None, sectors=12,
               direction_bin_array=None, direction_bin_labels=None, freq_as_percentage=True,
               plot_bins=None, plot_labels=None, return_data=False):
    """
    Accepts a variable series and direction series and computes a frequency table of percentages. Both variable and
    direction are binned

    :param var_series: Series of variable to be binned
    :type var_series: pandas.Series
    :param direction_series: Series of wind directions between [0-360]
    :type direction_series: pandas.Series
    :param var_bin_array: List of numbers where adjacent elements of array form a bin. For instance, for bins
        [0,3),[3,8),[8,10) the list will be [0, 3, 8, 10]
    :type var_bin_array: list
    :param var_bin_labels: Optional, an array of labels to use for variable bins
    :type var_bin_labels: list
    :param sectors: Number of sectors to bin direction to. The first sector is centered at 0 by default. To change that
            behaviour specify direction_bin_array, it overwrites sectors
    :type sectors: int
    :param direction_bin_array: To add custom bins for direction sectors, overwrites sectors. For instance,
        for direction bins [0,120), [120, 215), [215, 360) the list would be [0, 120, 215, 360]
    :type direction_bin_array: list
    :param direction_bin_labels: Optional, you can specify an array of labels to be used for the bins. uses string
        labels of the format '30-90' by default
    :type direction_bin_labels: list(float), list(str)
    :param freq_as_percentage: Optional, True by default. Returns the frequency as percentages. To return just the
        count, set to False
    :type freq_as_percentage: bool
    :param return_data:  Set to True if you want to return the frequency table too.
    :type return_data: bool
    :param plot_bins: (Optional) Bins to use for gradient in the rose. Different bins will be plotted with different
        color. Chooses six bins to plot by default.
    :type plot_bins: list
    :param plot_labels: (Optional) Labels to use for different colors in the rose. By default chooses the end points of
        bin
    :type plot_labels: list(str), list(float)
    :returns: A wind rose plot with gradients in the rose. Also returns a frequency table if return_data is True
    :rtype: plot or tuple(plot, pandas.DataFrame)

    **Example usage**
    ::
        import brightwind as bw
        df = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)

        #Simple use
        rose, freq_table = bw.freq_table(df.Spd40mN, df.Dir38mS, return_data=True)

        #To use 3 bins for wind speed [0,8), [8, 14), [14, 41) and label them as ['low', 'mid', 'high']. Can be used for
        #variabes other than wind speed too
        rose, freq_table = bw.freq_table(df.Spd40mN, df.Dir38mS, var_bin_array=[0,8,14,41],
            var_bin_labels=['low', 'mid', 'high'], return_data=True)


        #Use custom direction bins
        rose, freq_table = bw.freq_table(df.Spd40mN, df.Dir38mS, direction_bin_array=[0,90,130,200,360],
                           direction_bin_labels=['northerly','easterly','southerly','westerly'], return_data=True)


        #Can also combine custom direction and variable_bins
        rose, tab = bw.freq_table(df.Spd40mN, df.Dir38mS, direction_bin_array=[0,90,130,200,360],
                           direction_bin_labels=['northerly','easterly','southerly','westerly'], plot_bins=None,
                           plot_labels=None, return_data=True)

        #For classic wind rose plot use the following plot_bins and plot_labels
        tab = bw.freq_table(df.Spd40mN, df.Dir38mS, plot_bins=[0,3,6,9,12,15,41],
            plot_labels=['0-3 m/s', '4-6 m/s', '7-9 m/s', '10-12 m/s', '13-15 m/s', '15+ m/s'], plot_bins=None,
            plot_labels=None, return_data=True)

    """
    var_series = _convert_df_to_series(var_series)
    direction_series = _convert_df_to_series(direction_series)
    var_series = var_series.dropna()
    direction_series = direction_series.dropna()
    direction_binned_series, direction_bin_labels, sectors, direction_bin_array, zero_centered = \
        _get_direction_binned_series(sectors, direction_series, direction_bin_array, direction_bin_labels)

    var_binned_series = pd.cut(var_series, var_bin_array, right=False).rename('variable_bin')
    data = pd.concat([var_series.rename('var_data'), var_binned_series, direction_binned_series], axis=1).dropna()
    if freq_as_percentage:
        result = pd.crosstab(data.loc[:, 'variable_bin'], data.loc[:, 'direction_bin']) / len(data) * 100.0
    else:
        result = pd.crosstab(data.loc[:, 'variable_bin'], data.loc[:, 'direction_bin'])
    for i in range(1, sectors+1):
        if not (i in result.columns):
            result.insert(i-1, i, 0.0)

    result.columns = _get_direction_bin_labels(sectors, direction_bin_array, zero_centered)
    result = result.sort_index()

    # Creating a graph before renaming the direction labels, to help identify sectors while plotting
    graph = plt.plot_rose_with_gradient(result, plot_bins=plot_bins, plot_labels=plot_labels,
                                        percent_symbol=freq_as_percentage)

    if direction_bin_labels is not None:
        result.columns = direction_bin_labels
    if var_bin_labels is not None:
        result.index = var_bin_labels

    if return_data:
        return graph, result
    else:
        return graph


def time_continuity_gaps(data):
    """
    Returns the start and end timestamps of missing data periods. Also days lost.

    A missing data period is one where data is not available for some consecutive timestamps. This breaks
    time continuity of the data. The function calculates the sampling period (resolution) of the data by
    finding the most common time difference between consecutive timestamps. Then it searches where the time
    difference between consecutive timestamps does not match the sampling period, this is the missing data period.
    It returns a DataFrame where the first column is the starting timestamp of the missing period and the second
    column is the end date of the missing period. An additional column also shows how many days of data were lost
    in a missing period.


    :param data: Data for checking continuity, timestamp must be the index
    :type data: pandas.Series or pandas.DataFrame
    :return: A DataFrame with the start and end timestamps of missing gaps in the data along with the size of the gap
        in days lost.
    :rtype: pandas.DataFrame

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.shell_flats_80m_csv)
        bw.time_continuity_gaps(data['WS70mA100NW_Avg'])

    """
    indexes = data.dropna(how='all').index
    continuity = pd.DataFrame({'Date From': indexes.values.flatten()[:-1], 'Date To': indexes.values.flatten()[1:]})
    continuity['Days Lost'] = (continuity['Date To'] - continuity['Date From']) / pd.Timedelta('1 days')
    # Remove indexes where no days are lost before returning
    return continuity[continuity['Days Lost'] != (tf._get_data_resolution(indexes) / pd.Timedelta('1 days'))]


def coverage(data, period='1M', aggregation_method='mean'):
    """
    Get the data coverage over the period specified.

    Coverage is defined as the ratio of number of data points present in the period and the maximum number of
    data points that a period should have. Example, for 10 minute data resolution and a period of 1 hour the
    maximum number of data points in one period is 6. But if the number if data points available is only 3 for that
    hour the coverage is 3/6=0.5 . For more details see average_data_by_period as this function is a wrapper around it.

    :param data: Data to find average or aggregate of
    :type data: pandas.Series or pandas.DataFrame
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
    :return: A DataFrame with data aggregated with the specified aggregation_method (mean by default) and coverage.
            The columns with coverage are named as <column name>_Coverage

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)

        #To find hourly coverage
        data_hourly = bw.coverage(data.Spd80mN, period='1H')

        #To find hourly coverage for multiple columns
        data_hourly_multiple = bw.coverage(data[['Spd80mS','Spd60mN']], period='1H')

        #To find monthly_coverage
        data_monthly = bw.coverage(data.Spd80mN, period='1M')

        #To find monthly_coverage of variance
        data_monthly_var = bw.coverage(data.Spd80mN, period='1M', aggregation_method='var')


    See Also
    --------
    bw.average_data_by_period
    """

    return tf.average_data_by_period(data, period=period, aggregation_method=aggregation_method,
                                     return_coverage=True)[1]


def basic_stats(data):
    """
    Gives basic statistical measures of the data, the DataFrame returned includes the following columns

    - count, number of data points available for each column of the data
    - mean, mean of each column of data
    - std, standard deviation of each column of data
    - min, minimum value of each column of data
    - max, maximum value of each column of data

    :param data: It can be a DataFrame containing meteorological data or a series of some variable like wind speed,
        direction, temperature, etc.
    :type data: pandas.Series or pandas.DataFrame
    :rtype: A DataFrame with columns count, mean, std, min amd max.

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)
        bw.basic_stats(data)
        bw.basic_stats(data['Gust_Max_1'])

    """
    if isinstance(data, pd.DataFrame):
        return data.describe(percentiles=[0.5]).T.drop(['50%'], axis=1)
    else:
        return data.to_frame().describe(percentiles=[0.5]).T.drop(['50%'], axis=1)


def dist_12x24(var_series, aggregation_method='mean', var_name_label=None, return_data=False):
    """
    Accepts a variable series and returns a plot of 12x24 (12 months x 24 hours) for the 'mean' of the variable with
    the table of data as an optional return. The aggregation_method 'mean' can be can be changed as outlined below.
    :param var_series: Variable to compute 12x24 for
    :type var_series: pandas.Series
    :param aggregation_method: 'mean' by default, calculates mean of the variable passed. Can change it to
            'sum', 'std', 'min', 'max', for sum, standard deviation, minimum, maximum. Can also pass a function.
    :type aggregation_method: str or function
    :param var_name_label: (Optional) Label to appear on the plot, can be name and unit of the variable
    :type var_name_label: str
    :param return_data: Set to True if you want the data returned.
    :type return_data: bool
    :return: A plot with gradients showing , also a 12x24 table with hours as row labels and months as column labels
        when return_data is True

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.datasets.demo_data)

        # For 12x24 table of means
        graph, table12x24 = bw.dist_12x24(data.Spd40mN, var_name_label='wind speed [m/s]', return_data=True)

        # For 12x24 table of sums
        graph, table12x24 = bw.dist_12x24(data.PrcpTot, aggregation_method='sum')

        #For a custom aggregation_method
        def custom_agg(x):
            return x.mean()+(2*x.std())

        graph, table12x24 = bw.dist_12x24(data.PrcpTot, aggregation_method=custom_agg, return_data=True)

    """
    if isinstance(var_series, pd.DataFrame):
        var_series = var_series[var_series.columns[0]]
    if isinstance(var_series, pd.Series) and var_name_label is None:
        var_name_label = var_series.name
    table_12x24 = pd.concat([var_series.rename('Variable'), var_series.index.to_series().dt.month.rename('Month'),
                             var_series.index.to_series().dt.hour.rename('Hour')], axis=1, join='inner')
    pvt_tbl = table_12x24.pivot_table(index='Hour', columns='Month', values='Variable', aggfunc=aggregation_method)
    if not isinstance(aggregation_method, str):
        aggregation_method = aggregation_method.__name__
    if return_data:
        return plt.plot_12x24_contours(pvt_tbl, label=(var_name_label, aggregation_method)),\
               pvt_tbl
    return plt.plot_12x24_contours(pvt_tbl, label=(var_name_label, aggregation_method))


class TI:

    @staticmethod
    def calc(wspd, wspd_std):
        ti = pd.concat([wspd[wspd > 3].rename('wspd'), wspd_std.rename('wspd_std')], axis=1, join='inner')
        return ti['wspd_std'] / ti['wspd']

    @staticmethod
    def by_speed(wspd, wspd_std, speed_bin_array=np.arange(-0.5, 41, 1), speed_bin_labels=range(0, 41),
                 percentile=90, IEC_class=None, return_data=False):
        """
        Accepts a wind speed series and its standard deviation, calculates turbulence intensity (TI) and returns the
        distribution by of TI by speed bins

        :param wspd: Wind speed data series
        :type wspd: pandas.Series
        :param wspd_std: Wind speed standard deviation data series
        :type wspd_std: pandas.Series
        :param speed_bin_array: (Optional) Array of wind speeds where adjacent elements of array form a bin
        :type speed_bin_array: List or array
        :param speed_bin_labels: (Optional) Labels to use for speed bins, 0, 1, 2, 3 .. and so on by default
        :type speed_bin_labels: List, range or array
        :param percentile: The percentile representative of TI (see return for more information)
        :type percentile: float, int
        :param IEC_class: By default IEC class 2005 is used for custom class pass a DataFrame. Note we have removed
                option to include IEC Class 1999 as no longer appropriate.
                This may need to be placed in a separate function when updated IEC standard is released
        :param return_data: Set to True if you want the data returned.
        :type return_data: bool
        :return: TI distribution with columns names as:

                * Mean_TI (average TI for a speed bin),
                * TI_Count ( number of data points in the bin),
                * Rep_TI (Representative TI set at 90 percentile by default,
                * TI_2Sigma (2 sigma TI),
                * Char_TI (characteristic TI)

        :rtype: pandas.DataFrame

        """
        ti = pd.concat([wspd.rename('wspd'), wspd_std.rename('wspd_std')], axis=1, join='inner')
        ti['Turbulence_Intensity'] = TI.calc(ti['wspd'], ti['wspd_std'])
        ti_dist = pd.concat([
            dist(var_series=ti['Turbulence_Intensity'], var_to_bin_against=ti['wspd'],
                 bins=speed_bin_array, bin_labels=speed_bin_labels,
                 aggregation_method='mean', return_data=True)[-1].rename("Mean_TI"),
            dist(var_series=ti['Turbulence_Intensity'],
                 var_to_bin_against=ti['wspd'],
                 bins=speed_bin_array,
                 bin_labels=speed_bin_labels,
                 aggregation_method='count', return_data=True)[-1].rename("TI_Count"),
            dist(var_series=ti['Turbulence_Intensity'],
                 var_to_bin_against=ti['wspd'],
                 bins=speed_bin_array,
                 bin_labels=speed_bin_labels,
                 aggregation_method=lambda x: np.percentile(x, q=percentile),
                 return_data=True)[-1].rename("Rep_TI"),
            dist(var_series=ti['Turbulence_Intensity'],
                 var_to_bin_against=ti['wspd'],
                 bins=speed_bin_array,
                 bin_labels=speed_bin_labels,
                 aggregation_method='std', return_data=True)[-1].rename("TI_2Sigma")], axis=1, join='inner')
        categ_index = dist(var_series=ti['Turbulence_Intensity'], var_to_bin_against=ti['wspd'],
                           bins=speed_bin_array, aggregation_method='mean', return_data=True)[-1].index
        num_index = [i.mid for i in categ_index]
        ti_dist.loc[:, 'Char_TI'] = ti_dist.loc[:, 'Mean_TI'] + (ti_dist.loc[:, 'TI_2Sigma'] / num_index)
        # ti_dist.loc[0, 'Char_TI'] = 0
        ti_dist.index.rename('Speed Bin', inplace=True)
        # return ti_dist
        if return_data:
            return plt.plot_TI_by_speed(wspd, wspd_std, ti_dist, IEC_class=IEC_class), ti_dist.dropna(how='any')
        return plt.plot_TI_by_speed(wspd, wspd_std, ti_dist, IEC_class=IEC_class)

    @staticmethod
    def by_sector(wspd, wspd_std, wdir, min_speed=0, sectors=12, direction_bin_array=None,
                  direction_bin_labels=None, return_data=False):
        """
        Accepts a wind speed series, its standard deviation and a direction series, calculates turbulence intensity (TI)
        and returns the distribution by of TI by sector

        :param wspd: Wind speed data series
        :type wspd: pandas.Series
        :param wspd_std: Wind speed standard deviation data series
        :type wspd_std: pandas.Series
        :param wdir: Wind direction series
        :type wdir: pandas.Series
        :param min_speed: Set the minimum wind speed.
        :type min_speed: float
        :param sectors: Set the number of direction sectors. Usually 12, 16, 24, 36 or 72.
        :type sectors: int
        :param direction_bin_array: (Optional) Array of wind speeds where adjacent elements of array form a bin
        :param direction_bin_array: (Optional) To change default behaviour of first sector centered at 0 assign an
            array of bins to this
        :param direction_bin_labels: (Optional) you can specify an array of labels to be used for the bins. uses string
                labels of the format '30-90' by default
        :param return_data: Set to True if you want the data returned.
        :type return_data: bool
        :return: TI distribution with columns names as:

                * Mean_TI (average TI for a speed bin),
                * TI_Count ( number of data points in the bin)

        :rtype: pandas.DataFrame

        """
        ti = pd.concat([wspd.rename('wspd'), wspd_std.rename('wspd_std'), wdir.rename('wdir')], axis=1,
                       join='inner')
        ti = ti[ti['wspd'] >= min_speed]
        ti['Turbulence_Intensity'] = TI.calc(ti['wspd'], ti['wspd_std'])
        ti_dist = pd.concat([
            distribution_by_dir_sector(var_series=ti['Turbulence_Intensity'],
                                       direction_series=ti['wdir'],
                                       sectors=sectors, direction_bin_array=direction_bin_array,
                                       direction_bin_labels=direction_bin_labels,
                                       aggregation_method='mean', return_data=True)[-1].rename("Mean_TI"),
            distribution_by_dir_sector(var_series=ti['Turbulence_Intensity'],
                                       direction_series=ti['wdir'],
                                       sectors=sectors, direction_bin_array=direction_bin_array,
                                       direction_bin_labels=direction_bin_labels,
                                       aggregation_method='count', return_data=True)[-1].rename("TI_Count")], axis=1,
            join='outer')

        ti_dist.index.rename('Direction Bin', inplace=True)
        if return_data:
            return plt.plot_TI_by_sector(ti['Turbulence_Intensity'], ti['wdir'], ti_dist), ti_dist.dropna(how='all')
        else:
            return plt.plot_TI_by_sector(ti['Turbulence_Intensity'], ti['wdir'], ti_dist)

    @staticmethod
    def twelve_by_24(wspd, wspd_std, return_data=False, var_name_label='Turbulence Intensity'):
        tab_12x24, graph = dist_12x24(TI.calc(wspd, wspd_std), return_data=True, var_name_label=var_name_label)
        if return_data:
            return tab_12x24, graph
        return graph


def _calc_ratio(var_1, var_2, min_var=3, max_var=50):

    var_1_bounded = var_1[(var_1 >= min_var) & (var_1 < max_var)]
    var_2_bounded = var_2[(var_2 >= min_var) & (var_2 < max_var)]
    ratio = pd.concat([var_1_bounded.rename('var_1'), var_2_bounded.rename('var_2')], axis=1, join='inner')

    return ratio['var_2'] / ratio['var_1']


def wspd_ratio_by_dir_sector(wspd_1, wspd_2, wdir, sectors=72, min_wspd=3, direction_bin_array=None, boom_dir_1=-1,
                             boom_dir_2=-1, return_data=False):
    """
    Calculates the wind speed ratio of two wind speed time series and plots this ratio, averaged by direction sector,
    in a polar plot using a wind direction time series. The averaged ratio by sector can be optionally returned
    in a pd.DataFrame.
    
    If boom directions are specified, these will be overlayed on the plot. A boom direction of '-1' assumes top
    mounted and so doesn't plot.

    :param wspd_1: First wind speed time series. This is divisor.
    :type: wspd_1: pandas.Series
    :param wspd_2: Second wind speed time series, dividend.
    :type: wspd_2: pandas.Series
    :param wdir: Series of wind directions
    :type wdir: pandas.Series
    :param sectors: Set the number of direction sectors. Usually 12, 16, 24, 36 or 72.
    :type sectors: int
    :param min_wspd: Minimum wind speed to be used.
    :type: min_wpd: float
    :param direction_bin_array: (Optional) Array of numbers where adjacent elements of array form a bin. This
                                 overwrites the sectors.
    :param boom_dir_1: Boom direction in degrees of wspd_1. If top mounted leave default as -1.
    :type boom_dir_1: float
    :param boom_dir_2: Boom direction in degrees of wspd_2. If top mounted leave default as -1.
    :type boom_dir_2: float
    :param return_data:  Set to True if you want the data returned.
    :type return_data: bool
    :returns: A wind speed ratio plot showing the average ratio by sector and scatter of individual data points.
    :rtype: plot, pandas.DataFrame

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.datasets.demo_data)

        #For plotting both booms
        bw.wspd_ratio_by_dir_sector(data.Spd80mN, data.Spd80mS, wdir=data.Dir78mS, boom_dir_1=0, boom_dir_2=180)

        #For plotting no booms
        bw.wspd_ratio_by_dir_sector(data.Spd80mN, data.Spd80mS, wdir=data.Dir78mS)

        #If one boom is top mounted, say Spd80mS
        bw.wspd_ratio_by_dir_sector(data.Spd80mN, data.Spd80mS, wdir=data.Dir78mS, boom_dir_2=180)

        #To use your custom direction bins, for example (0-45), (45-135), (135-180), (180-220), (220-360)
        bw.wspd_ratio_by_dir_sector(data.Spd80mN, data.Spd80mS, wdir=data.Dir78mS,
                                    direction_bin_array=[0, 45, 135, 180, 220, 360], boom_dir_1=0, boom_dir_2=180)

    """
    sec_rat = _calc_ratio(wspd_1, wspd_2, min_wspd)
    common_idxs = sec_rat.index.intersection(wdir.index)
    sec_rat_plot, sec_rat_dist = distribution_by_dir_sector(sec_rat.loc[common_idxs], wdir.loc[common_idxs], sectors=sectors,
                                              aggregation_method='mean', direction_bin_array=direction_bin_array,
                                              direction_bin_labels=None,return_data=True)

    matplotlib.pyplot.close()
    sec_rat_dist = sec_rat_dist.rename('Mean_Sector_Ratio').to_frame()
    if return_data:
        return plt.plot_sector_ratio(sec_rat.loc[common_idxs], wdir.loc[common_idxs],
                                     sec_rat_dist, [wspd_1.name, wspd_2.name],
                                     boom_dir_1=boom_dir_1, boom_dir_2=boom_dir_2), sec_rat_dist
    return plt.plot_sector_ratio(sec_rat.loc[common_idxs], wdir.loc[common_idxs], sec_rat_dist,
                                 [wspd_1.name, wspd_2.name],
                                 boom_dir_1=boom_dir_1, boom_dir_2=boom_dir_2)


class Shear:
    def power_law(wspds, heights, min_speed=3, return_alpha=False):
        wspds = wspds.dropna()
        mean_wspds = wspds[(wspds > min_speed).all(axis=1)].mean()
        alpha, c = _calc_shear(mean_wspds.values, heights, return_coeff=True)
        if return_alpha:
            return plt.plot_shear(alpha, c, mean_wspds.values, heights), alpha
        return plt.plot_shear(alpha, c, mean_wspds.values, heights)

    def by_sector(wspds, heights, wdir, sectors=12, min_speed=3, direction_bin_array=None, direction_bin_labels=None,
                  return_data=False):
        common_idxs = wspds.index.intersection(wdir.index)
        shear = wspds[(wspds > min_speed).all(axis=1)].apply(_calc_shear, heights=heights, axis=1).loc[common_idxs]
        shear_dist = pd.concat([
            distribution_by_dir_sector(var_series=shear,
                                       direction_series=wdir.loc[common_idxs],
                                       sectors=sectors, direction_bin_array=direction_bin_array,
                                       direction_bin_labels=direction_bin_labels,
                                       aggregation_method='mean', return_data=True)[-1].rename("Mean_Shear"),
            distribution_by_dir_sector(var_series=shear,
                                       direction_series=wdir.loc[common_idxs],
                                       sectors=sectors, direction_bin_array=direction_bin_array,
                                       direction_bin_labels=direction_bin_labels,
                                       aggregation_method='count', return_data=True)[-1].rename("Shear_Count")], axis=1, join='outer')
        shear_dist.index.rename('Direction Bin', inplace=True)
        if return_data:
            return plt.plot_shear_by_sector(shear, wdir.loc[shear.index.intersection(wdir.index)], shear_dist), \
                   shear_dist
        else:
            return plt.plot_shear_by_sector(shear, wdir.loc[shear.index.intersection(wdir.index)], shear_dist)

    def twelve_by_24(wspds, heights, min_speed=3, return_data=False, var_name_label='Shear'):
        tab_12x24 = dist_12x24(wspds[(wspds > min_speed).all(axis=1)].apply(_calc_shear, heights=heights, axis=1),
                               return_data=True)[1]
        if return_data:
            return plt.plot_12x24_contours(tab_12x24, var_name_label=var_name_label), tab_12x24
        return plt.plot_12x24_contours(tab_12x24, var_name_label=var_name_label)

    def scale(alpha, wspd, height, height_to_scale_to):
        scale_factor = (height_to_scale_to / height)**alpha
        return wspd*scale_factor


def _calc_shear(wspds, heights, return_coeff=False) -> (np.array, float):
    """
    Derive the best fit power law exponent (as 1/alpha) from a given time-step of speed data at 2 or more elevations

    :param wspds: List of wind speeds [m/s]
    :param heights: List of heights [m above ground]. The position of the height in the list must be the same position
        in the list as its
    corresponding wind speed value.
    :return: The shear value (alpha), as the inverse exponent of the best fit power law, based on the form:
        (v1/v2) = (z1/z2)^(1/alpha)

    METHODOLOGY:
        Derive natural log of elevation and speed data sets
        Derive coefficients of linear best fit along log-log distribution
        Characterise new distribution of speed values based on linear best fit
        Derive 'alpha' based on gradient of first and last best fit points (function works for 2 or more points)
        Return alpha value

    """

    logheights = np.log(heights)  # take log of elevations
    logwspds = np.log(wspds)  # take log of speeds
    coeffs = np.polyfit(logheights, logwspds, deg=1)  # get coefficients of linear best fit to log distribution
    if return_coeff:
        return coeffs[0], np.exp(coeffs[1])
    return coeffs[0]


def calc_air_density(temperature, pressure, elevation_ref=None, elevation_site=None, lapse_rate=-0.113,
                     specific_gas_constant=286.9):
    """
    Calculates air density for a given temperature and pressure and extrapolates that to the site if both reference
    and site elevations are given.

    :param temperature: Temperature values in degree Celsius
    :type temperature: float or pandas.Series or pandas.DataFrame
    :param pressure: Pressure values in hectopascal, hPa, (1,013.25 hPa = 101,325 Pa = 101.325 kPa =
                    1 atm = 1013.25 mbar)
    :type pressure: float or pandas.Series or pandas.DataFrame
    :param elevation_ref: Elevation, in meters, of the reference temperature and pressure location.
    :type elevation_ref: Floating point value (decimal number)
    :param elevation_site: Elevation, in meters, of the site location to calculate air density for.
    :type elevation_site: Floating point values (decimal number)
    :param lapse_rate: Air density lapse rate kg/m^3/km, default is -0.113
    :type lapse_rate: Floating point value (decimal number)
    :param specific_gas_constant: Specific gas constant, R, for humid air J/(kg.K), default is 286.9
    :type specific_gas_constant:  Floating point value (decimal number)
    :return: Air density in kg/m^3
    :rtype: float or pandas.Series depending on the input

        **Example usage**
    ::
        import brightwind as bw

        #For a series of air densities
        data = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)
        air_density = bw.calc_air_density(data.T2m, data.P2m)

        #For a single value
        bw.calc_air_density(15, 1013)

        #For a single value with ref and site elevation
        bw.calc_air_density(15, 1013, elevation_ref=0, elevation_site=200)



    """

    temp = temperature
    temp_kelvin = temp + 273.15     # to convert deg C to Kelvin.
    pressure = pressure * 100       # to convert hPa to Pa
    ref_air_density = pressure / (specific_gas_constant * temp_kelvin)

    if elevation_ref is not None and elevation_site is not None:
        site_air_density = round(ref_air_density + (((elevation_site - elevation_ref) / 1000) * lapse_rate), 3)
        return site_air_density
    elif elevation_site is None and elevation_ref is not None:
        raise TypeError('elevation_site should be a number')
    elif elevation_site is not None and elevation_ref is None:
        raise TypeError('elevation_ref should be a number')
    else:
        return ref_air_density
