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

__all__ = ['concurrent_coverage', 'monthly_means', 'momm', 'distribution', 'distribution_by_wind_speed',
           'distribution_by_dir_sector', 'freq_table', 'time_continuity_gaps', 'coverage', 'basic_stats',
           'twelve_by_24', 'TI', 'SectorRatio', 'Shear']


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


def momm(data: pd.DataFrame, date_from: str='', date_to: str=''):
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
    sliced_data = utils._slice_data(momm_data, date_from, date_to)
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


def distribution(var1_series, var2_series, var2_bin_array=np.arange(-0.5, 41, 1), var2_bin_labels=None,
                 aggregation_method='%frequency'):
    """
    Accepts 2 series of same/different variables and computes the distribution of first variable with respect to
    the bins of another variable.

    :param var1_series: Series of the variable whose distribution we need to find
    :param var2_series: Series of the variable which we want to bin
    :param var2_bin_array: Array of numbers where adjacent elements of array form a bin
    :param var2_bin_labels: Labels of bins to be used, uses (bin-start, bin-end] format by default
    :param aggregation_method: Statistical method used to find distribution it can be mean, max, min, std, count,
        describe, a custom function, etc,computes frequency in percentages by default
    :returns: A DataFrame/Series with bins as row indexes and columns with statistics chosen by aggregation_method

    """
    if isinstance(var1_series, pd.DataFrame) and var1_series.shape[1] == 1:
        var1_series = var1_series.iloc[:, 0]
    if isinstance(var2_series, pd.DataFrame) and var2_series.shape[1] == 1:
        var2_series = var2_series.iloc[:, 0]
    var1_series = var1_series.dropna()
    var2_series = var2_series.dropna()
    var2_binned_series = pd.cut(var2_series, var2_bin_array, right=False, labels=var2_bin_labels).rename('variable_bin')
    data = pd.concat([var1_series.rename('data'), var2_binned_series], join='inner', axis=1)
    if aggregation_method == '%frequency':
        return data.groupby(['variable_bin'])['data'].count().rename('%frequency')/len(data) * 100.0
    else:
        return data.groupby(['variable_bin'])['data'].agg(aggregation_method)


def distribution_by_wind_speed(wspd, return_data=False):
    """
    Accepts 2 series of same/different variables and computes the distribution of first variable with respect to
    the bins of another variable.

    :param wspd: Series of the variable whose distribution we need to find
    :param return_data: Set to True if you want the data returned.
    :type return_data: bool

    """
    freq_dist = distribution(wspd, wspd, var2_bin_array=np.arange(-0.5, 41, 1), var2_bin_labels=None,
                             aggregation_method='%frequency')
    if return_data:
        return plt.plot_freq_distribution(freq_dist), freq_dist
    return plt.plot_freq_distribution(freq_dist)


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


def _get_direction_binned_series(sectors, direction_series, direction_bin_array, direction_bin_labels):
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
                               direction_bin_array=None, direction_bin_labels=None):
    """
    Accepts a series of a variable and  wind direction. Computes the distribution of first variable with respect to
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
    :returns: A DataFrame/series with wind direction sector as row indexes and columns with statistics chosen by
            aggregation_method

    """
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
    return result


def freq_table(var_series, direction_series, var_bin_array=np.arange(-0.5, 41, 1), var_bin_labels=None, sectors=12,
               direction_bin_array=None, direction_bin_labels=None, freq_as_percentage=True,
               plot_bins=[0, 3, 6, 9, 12, 15, 41],
               plot_labels=['0-3 m/s', '4-6 m/s', '7-9 m/s', '10-12 m/s', '13-15 m/s', '15+ m/s'], return_data=False):
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
                           direction_bin_labels=['northerly','easterly','southerly','westerly'], return_data=True)

        #For classic wind rose plot use the following plot_bins and plot_labels
        tab = bw.freq_table(df.Spd40mN, df.Dir38mS, plot_bins=[0,3,6,9,12,15,41],
            plot_labels=['0-3 m/s', '4-6 m/s', '7-9 m/s', '10-12 m/s', '13-15 m/s', '15+ m/s'], return_data=True)

    """
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

    #Creating a graph before renaming the direction labels, to help identify sectors while plotting
    graph = plt.plot_rose_with_gradient(result, plot_bins=plot_bins, plot_labels=plot_labels,
                                             percent_symbol=freq_as_percentage)

    if direction_bin_labels is not None:
        result.columns = direction_bin_labels
    if var_bin_labels is not None:
        result.index = var_bin_labels

    # return result
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
    #Remove indexes where no days are lost before returning
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


def twelve_by_24(var_series, aggregation_method='mean', return_data=False, var_name=''):
    """
    Accepts a variable series and returns 12x24 (months x hours) table for the variable.
    :param var_series:
    :param aggregation_method: 'mean' by default calculates mean of the variable passed. Can change it to
            'sum', 'std', 'min', 'max', 'percentile' for sum, standard deviation, minimum, maximum, percentile
             of the variable respectively. Can also pass a function.
    :type aggregation_method: str or function
    :param return_data: Set to True if you want the data returned.
    :type return_data: bool
    :param var_name: Name and units of the variable to appear on the plot
    :type var_name: str
    :return: A DataFrame with hours as row labels and months as column labels.

    """
    table_12x24 = pd.concat([var_series.rename('Variable'), var_series.index.to_series().dt.month.rename('Month'),
                             var_series.index.to_series().dt.hour.rename('Hour')], axis=1, join='inner')

    pvt_tbl = table_12x24.pivot_table(index='Hour', columns='Month', values='Variable', aggfunc=aggregation_method)
    if return_data:
        return plt.plot_12x24_contours(pvt_tbl, title=var_name),\
               pvt_tbl
    return plt.plot_12x24_contours(pvt_tbl, title=var_name)


class TI:

    def calc(wspd, wspd_std):
        ti = pd.concat([wspd[wspd > 3].rename('wdspd'), wspd_std.rename('wspd_std')], axis=1, join='inner')
        return ti['wspd_std'] / ti['wdspd']

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
        ti = pd.concat([wspd.rename('wdspd'), wspd_std.rename('wdspd_std')], axis=1, join='inner')
        ti['Turbulence_Intensity'] = TI.calc(ti['wdspd'], ti['wdspd_std'])
        ti_dist = pd.concat([
            distribution(var1_series=ti['Turbulence_Intensity'], var2_series=ti['wspd'],
                         var2_bin_array=speed_bin_array, var2_bin_labels=speed_bin_labels,
                         aggregation_method='mean').rename("Mean_TI"),
            distribution(var1_series=ti['Turbulence_Intensity'],
                         var2_series=ti['wspd'],
                         var2_bin_array=speed_bin_array,
                         var2_bin_labels=speed_bin_labels,
                         aggregation_method='count').rename("TI_Count"),
            distribution(var1_series=ti['Turbulence_Intensity'],
                         var2_series=ti['wspd'],
                         var2_bin_array=speed_bin_array,
                         var2_bin_labels=speed_bin_labels,
                         aggregation_method=lambda x: np.percentile(x, q=percentile)).rename("Rep_TI"),
            distribution(var1_series=ti['Turbulence_Intensity'],
                         var2_series=ti['wspd'],
                         var2_bin_array=speed_bin_array,
                         var2_bin_labels=speed_bin_labels,
                         aggregation_method='std').rename("TI_2Sigma")], axis=1, join='inner')
        ti_dist.loc[:, 'Char_TI'] = ti_dist.loc[:, 'Mean_TI'] + (ti_dist.loc[:, 'TI_2Sigma'] / ti_dist.index)
        ti_dist.loc[0, 'Char_TI'] = 0
        ti_dist.index.rename('Speed Bin', inplace=True)
        if return_data:
            return plt.plot_TI_by_speed(wspd, wspd_std, ti_dist, IEC_class=IEC_class), ti_dist.dropna(how='any')
        return plt.plot_TI_by_speed(wspd, wspd_std, ti_dist, IEC_class=IEC_class)

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
        ti = pd.concat([wspd.rename('wdspd'), wspd_std.rename('wdspd_std'), wdir.rename('wddir')], axis=1,
                       join='inner')
        ti = ti[ti['wspd'] >= min_speed]
        ti['Turbulence_Intensity'] = TI.calc(ti['wspd'], ti['wspd_std'])
        ti_dist = pd.concat([
            distribution_by_dir_sector(var_series=ti['Turbulence_Intensity'],
                                       direction_series=ti['wdir'],
                                       sectors=sectors, direction_bin_array=direction_bin_array,
                                       direction_bin_labels=direction_bin_labels,
                                       aggregation_method='mean').rename("Mean_TI"),
            distribution_by_dir_sector(var_series=ti['Turbulence_Intensity'],
                                       direction_series=ti['wdir'],
                                       sectors=sectors, direction_bin_array=direction_bin_array,
                                       direction_bin_labels=direction_bin_labels,
                                       aggregation_method='count').rename("TI_Count")], axis=1, join='outer')

        ti_dist.index.rename('Direction Bin', inplace=True)
        if return_data:
            return plt.plot_TI_by_sector(ti['Turbulence_Intensity'], ti['wdir'], ti_dist), ti_dist.dropna(how='all')
        else:
            return plt.plot_TI_by_sector(ti['Turbulence_Intensity'], ti['wdir'], ti_dist)

    def twelve_by_24(wspd, wspd_std, return_data=False, var_name='Turbulence Intensity'):
        tab_12x24, graph = twelve_by_24(TI.calc(wspd, wspd_std), return_data=True, var_name=var_name)
        if return_data:
            return tab_12x24, graph
        return graph


class SectorRatio:

    def calc(wspd_1, wspd_2):
        sec_rat = pd.concat([wspd_1[wspd_1 > 3].rename('speed_1'), wspd_2[wspd_2 > 3].rename('speed_2')],
                            axis=1, join='inner')
        return sec_rat['speed_2']/sec_rat['speed_1']

    def by_sector(wspd_1, wspd_2, wdir, sectors=72, direction_bin_array=None,
                  boom_dir_1=-1, boom_dir_2=-1, return_data=False):
        """
        Accepts two speed series and one direction series and returns the speed ratio by sector
        in a table

        :param wspd_1: First wind speed series. This is divisor series.
        :type: wspd_1: pandas.Series
        :param wspd_2: Second wind speed series
        :type: wspd_2: pandas.Series
        :param wdir: Series of wind directions
        :type wdir: pandas.Series
        :param sectors: Set the number of direction sectors. Usually 12, 16, 24, 36 or 72.
        :type sectors: int
        :param direction_bin_array:
        :param boom_dir_1: Boom direction in degrees of speed_col_name_1.
        :param boom_dir_2: Boom direction in degrees of speed_col_name_2.
        :param return_data:  Set to True if you want the data returned.
        :type return_data: bool
        :returns: A speed ratio plot showing average speed ratio by sector and scatter of individual datapoints.

        """
        sec_rat = SectorRatio.calc(wspd_1, wspd_2)
        common_idxs = sec_rat.index.intersection(wdir.index)
        sec_rat_dist = distribution_by_dir_sector(sec_rat.loc[common_idxs], wdir.loc[common_idxs], sectors=sectors,
                                                  aggregation_method='mean', direction_bin_array=direction_bin_array,
                                                  direction_bin_labels=None).rename('Mean_Sector_Ratio').to_frame()

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
                                       aggregation_method='mean').rename("Mean_Shear"),
            distribution_by_dir_sector(var_series=shear,
                                       direction_series=wdir.loc[common_idxs],
                                       sectors=sectors, direction_bin_array=direction_bin_array,
                                       direction_bin_labels=direction_bin_labels,
                                       aggregation_method='count').rename("Shear_Count")], axis=1, join='outer')
        shear_dist.index.rename('Direction Bin', inplace=True)
        if return_data:
            return plt.plot_shear_by_sector(shear, wdir.loc[shear.index.intersection(wdir.index)], shear_dist), \
                   shear_dist
        else:
            return plt.plot_shear_by_sector(shear, wdir.loc[shear.index.intersection(wdir.index)], shear_dist)

    def twelve_by_24(wspds, heights, min_speed=3, return_data=False, var_name='Shear'):
        tab_12x24 = twelve_by_24(wspds[(wspds > min_speed).all(axis=1)].apply(_calc_shear, heights=heights, axis=1),
                                 return_data=True)[1]
        if return_data:
            return plt.plot_12x24_contours(tab_12x24, title=var_name), tab_12x24
        return plt.plot_12x24_contours(tab_12x24, title=var_name)

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
