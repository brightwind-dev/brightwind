import pandas as pd
import numpy as np
import datetime
from datetime import timedelta, date
import math
import calendar
from math import e
from brightwind.analyse import plot as plt
from brightwind.analyse.analyse import distribution_by_dir_sector, dist_12x24, coverage, _convert_df_to_series
import re
import warnings
import matplotlib.dates as mdates
pd.options.mode.chained_assignment = None

__all__ = ['Average', 'BySector', 'TimeOfDay']



class TimeOfDay:

    def __init__(self, wspds, heights, calc_method='power_law', min_speed=3, by_month=True, day_start_time='07', daily_sectors = 2):

        """
        Calculates shear based on power law

        :param wspds: DataFrame or list of wind speeds for calculating shear
        :type wspds: list of pandas.DataFrame
        :param heights: List of anemometer heights
        :type heights: list
        :param min_speed: Only speeds higher than this would be considered for calculating shear, default is 3
        :type min_speed: float
        :return:  Shear object containing shear exponents, a plot and other data.

        **Example usage**
        ::
            # Load anemometer data to calculate exponents
            data = bw.load_csv(bw.datasets.demo_data)
            anemometers = data[['Spd80mS', 'Spd60mS','Spd40mS']]
            heights = [80, 60, 40]

            # Using with a DataFrame of wind speeds
            shear_power_law = bw.Shear.Average(anemometers, heights , return_object=True)

            # View attributes of Shear objects
            # View exponents calculated
            shear_object_power_law.alpha

            # View plot
            shear_object_power_law.plot

            # View input data
            shear_object_power_law.wspds

            # View other information
            shear_object_power_law.info

        """

        start_times = pd.Series([])
        time_wspds = pd.Series([])
        mean_time_wspds = pd.Series([])
        alpha = pd.Series([])
        c = pd.Series([])
        alpha_monthly = pd.DataFrame([])

        if not isinstance(wspds, pd.DataFrame):
            wspds = pd.DataFrame(wspds).T
        wspds = wspds.dropna()
        cvg = coverage(wspds[wspds > min_speed].dropna(), period='1AS').sum()[1]

        interval = 24 / daily_sectors
        start_times[0] = datetime.datetime.strptime(day_start_time, '%H')
        dt = datetime.timedelta(hours=interval)

        for i in range(1, daily_sectors):
            start_times[i] = start_times[i - 1] + dt

        for j in range(0, 12):

            anemometers_monthly = wspds[wspds.index.month == j + 1]
            for i in range(0, daily_sectors):

                if i == daily_sectors - 1:
                    start_times[i] = start_times[i].strftime("%H:%M:%S")
                    start = str(start_times[i].time())
                    end = str(start_times[0].time())
                    time_wspds[i] = pd.DataFrame(anemometers_monthly).between_time(start, end, include_end=False)
                    mean_time_wspds[i] = time_wspds[i][(time_wspds[i] > min_speed).all(axis=1)].mean().dropna()
                else:
                    start_times[i] = start_times[i].strftime("%H:%M:%S")
                    start = str(start_times[i].time())
                    end = str(start_times[i + 1].time())
                    time_wspds[i] = pd.DataFrame(anemometers_monthly).between_time(start, end, include_end=False)
                    mean_time_wspds[i] = time_wspds[i][(time_wspds[i] > min_speed).all(axis=1)].mean().dropna()

            for i in range(0, len(mean_time_wspds)):
                alpha[i], c[i] = _calc_power_law(mean_time_wspds[i].values, heights, return_coeff=True)

            alpha_monthly = pd.concat([alpha_monthly, alpha], axis=1)

        if mean_time_wspds.shape[0] == 0:
            raise ValueError('None of the input wind speeds are greater than the min_speed, cannot calculate shear')

        #alpha_monthly = pd.concat([(alpha_monthly[start_times[0].hour:]), (alpha_monthly[:start_times[0].hour])],
                              #  axis=0)
        alpha_monthly.index = start_times
        alpha_monthly.index = alpha_monthly.index.time
        if by_month is True:
            alpha_monthly.columns = calendar.month_abbr[1:13]

        else:
            alpha_monthly = pd.DataFrame(alpha_monthly.mean(axis=1))
            alpha_monthly.columns = ['12 Month Average']

        info = {}
        input_wind_speeds = {'heights(m)': [heights], 'column_names': [list(wspds.columns.values)],
                             'min_spd(m/s)': str(min_speed)}
        info['input_wind_speeds'] = input_wind_speeds
        info['concurrent_period(years)'] = str("{:.3f}".format(cvg))

        self.wspds = wspds
        self. plot = plt.plot_shear_time_of_day(alpha_monthly)
        self.origin = 'TimeOfDay'
        self.info = info
        self.calc_method = calc_method
        self.alpha = pd.DataFrame(alpha_monthly)

class Average:

    def __init__(self, wspds, heights, calc_method='power_law', min_speed=3):

        """
        Calculates shear based on power law

        :param wspds: DataFrame or list of wind speeds for calculating shear
        :type wspds: list of pandas.DataFrame
        :param heights: List of anemometer heights
        :type heights: list
        :param min_speed: Only speeds higher than this would be considered for calculating shear, default is 3
        :type min_speed: float
        :return:  Shear object containing shear exponents, a plot and other data.

        **Example usage**
        ::
            # Load anemometer data to calculate exponents
            data = bw.load_csv(bw.datasets.demo_data)
            anemometers = data[['Spd80mS', 'Spd60mS','Spd40mS']]
            heights = [80, 60, 40]

            # Using with a DataFrame of wind speeds
            shear_power_law = bw.Shear.Average(anemometers, heights , return_object=True)

            # View attributes of Shear objects
            # View exponents calculated
            shear_object_power_law.alpha

            # View plot
            shear_object_power_law.plot

            # View input data
            shear_object_power_law.wspds

            # View other information
            shear_object_power_law.info

        """
        if not isinstance(wspds, pd.DataFrame):
            wspds = pd.DataFrame(wspds).T
        wspds = wspds.dropna()
        cvg = coverage(wspds[wspds > min_speed].dropna(), period='1AS').sum()[1]
        mean_wspds = wspds[(wspds > min_speed).all(axis=1)].mean().dropna()
        if mean_wspds.shape[0] == 0:
            raise ValueError('None of the input wind speeds are greater than the min_speed, cannot calculate shear')

        if calc_method == 'power_law':
            alpha, c = _calc_power_law(mean_wspds.values, heights, return_coeff=True)
            self.alpha = alpha
            self.plot = plt.plot_power_law(alpha, c, mean_wspds.values, heights)

        elif calc_method == 'log_law':
            slope, c = _calc_log_law(mean_wspds.values, heights, return_coeff=True)
            roughness_coefficient = _calc_roughness_coeff(mean_wspds, heights)
            self.roughness_coefficient = roughness_coefficient
            self.plot = plt.plot_log_law(slope, c, mean_wspds.values, heights)
            self.slope =slope
            self.intercept = c

        else:
            raise ValueError('Please enter a valid calculation method, "power_law or "log_law"')

        info = {}
        input_wind_speeds = {'heights(m)': [heights], 'column_names': [list(wspds.columns.values)],
                             'min_spd(m/s)': str(min_speed)}
        info['input_wind_speeds'] = input_wind_speeds
        info['concurrent_period(years)'] = str("{:.3f}".format(cvg))

        self.wspds = wspds
        self.origin = 'Average'
        self.info = info
        self.calc_method = calc_method

    def apply_alpha(self, wspds, height, height_to_scale_to):
        """"
        Applies shear exponent calculated with the power law to a wind speed and scales wind speed from one height to
        another

       :param self: Shear object to use when applying shear to the data
       :type self: Shear object
       :param wspds: Wind speed time series to apply shear to
       :type wspds: Pandas Series
       :param height: height of above wspds
       :type height: float
       :param height_to_scale_to: height to which wspds should be scaled to
       :type height_to_scale_to: float
       :return: a DataFrame showing original wind speed, scaled wind speed, wind direction (if applicable)
        and the shear exponent used.

        **Example Usage**
        ::
            # Load anemometer data to calculate exponents
            data = bw.load_csv(bw.datasets.demo_data)
            anemometers = data[['Spd80mS', 'Spd60mS','Spd40mS']]
            heights = [80, 60, 40]

            # Get power law object
            shear_power_law = bw.Shear.Average(anemometers, heights , return_object=True)

           # Scale wind speeds using exponents
           # Specify wind speeds to be scaled
           windseries = data['Spd40mN']
           height = 50
           height_to_scale_to =70
           shear_object_power_law.apply_alpha(windseries, height, height_to_scale_to)

           """
        return _apply(self, wspds, height, height_to_scale_to)


class BySector:

    def __init__(self, wspds, heights, wdir, calc_method = 'power_law', sectors=12, min_speed=3, direction_bin_array=None,
                 direction_bin_labels=None):

        """
        Calculates the shear exponent for each directional bin

        :param wspds: Wind speed measurements for calculating shear
        :type wspds:  pandas DataFrame or Series
        :param heights: List of anemometer heights
        :type heights: list
        :param wdir: Wind direction measurements
        :type wdir:  pandas DataFrame or Series
        :param sectors: number of sectors for the shear to be calculated for
        :type sectors: int
        :param:min_speed:  Only speeds higher than this would be considered for calculating shear, default is 3
        :type: min_speed: float
        :param direction_bin_array: specific array of directional bins to be used. Default is that bins are calculated
        by 360/sectors
        :type direction_bin_array: array
        :param: direction_bin_labels: labels to be given to the above direction_bin array
        :type direction_bin_labels: array
        :return: returns a shear object containing a plot, all inputted data and a series of calculated shear
        exponents.

         **Example usage**
        ::
            # Load anemometer data to calculate exponents
            data = bw.load_csv(bw.datasets.demo_data)
            anemometers = data[['Spd80mS', 'Spd60mS','Spd40mS']]
            heights = [80, 60, 40]
            directions = data['Dir78mS']

            # Calculate shear exponents using default bins ([345,15,45,75,105,135,165,195,225,255,285,315,345])
            shear_by_sector= bw.Shear.BySector(anemometers, heights, directions)

            # Calculate shear exponents using custom bins
            custom_bins = [0,30,60,90,120,150,180,210,240,270,300,330,360]
            shear_by_sector_custom_bins = bw.Shear.BySector(anemometers,heights,data['Dir78mS'],
             direction_bin_array=custom_bins)

            # View attributes of Shear objects
            # View exponents calculated
            shear_object_power_law.alpha

            # View plot
            shear_object_power_law.plot

            # View input data
            shear_object_power_law.wspds

            # View other information
            shear_object_power_law.info
        """

        if direction_bin_array is not None:
            sectors = len(direction_bin_array) - 1

        common_idxs = wspds.index.intersection(wdir.index)
        wdir = _convert_df_to_series(wdir)

        if calc_method == 'power_law':
            shear = wspds[(wspds > min_speed).all(axis=1)].apply(_calc_power_law, heights=heights, axis=1)
        elif calc_method == 'log_law':
            shear = wspds[(wspds > min_speed).all(axis=1)].apply(_calc_log_law, heights=heights, axis=1)
        else:
            raise ValueError('Please enter a valid calculation method, "power_law or "log_law"')

        shear = shear.loc[shear.index.intersection(common_idxs)]
        cvg = coverage(wspds[wspds > min_speed].dropna(), period='1AS').sum()[1]
        info = {}
        input_wind_speeds = {'heights(m)': [heights], 'column_names': [list(wspds.columns.values)], 'min_spd(m/s)': [3]}
        input_wind_dir = {'heights(m)': [re.findall(r'\d+', str(wdir.name))],
                          'column_names': [list(wspds.columns.values)]}
        info['input_wind_speeds'] = input_wind_speeds
        info['input_wind_dir'] = input_wind_dir
        info['concurrent_period(years)'] = str("{:.3f}".format(cvg))
        shear_dist = pd.concat([
            distribution_by_dir_sector(var_series=shear,
                                       direction_series=wdir.loc[common_idxs],
                                       sectors=sectors, direction_bin_array=direction_bin_array,
                                       direction_bin_labels=direction_bin_labels,
                                       aggregation_method='mean',
                                       return_data=True)[1].rename("Mean_Shear"),
            distribution_by_dir_sector(var_series=shear,
                                       direction_series=wdir.loc[common_idxs],
                                       sectors=sectors, direction_bin_array=direction_bin_array,
                                       direction_bin_labels=direction_bin_labels,
                                       aggregation_method='count',
                                       return_data=True)[1].rename("Shear_Count")], axis=1, join='outer')
        shear_dist.index.rename('Direction Bin', inplace=True)

        self.plot = plt.plot_shear_by_sector(shear, wdir.loc[shear.index.intersection(wdir.index)], shear_dist)
        self.wspds = wspds
        self.wdir = wdir
        self.sectors = sectors
        self.origin = 'by_sector'
        self.calc_method = calc_method
        self.alpha = shear_dist['Mean_Shear']
        self.info = info

    def apply_alpha(self, wspds, wdir, height, height_to_scale_to):
        """"
        Applies the corresponding shear exponent calculated for each directional bin to a wind speed series and scales
        wind speed from one height to another

        :param self: Shear object to use when applying shear to the data
        :type self: Shear object
        :param wspds: Wind speed time series to apply shear to
        :type wspds: Pandas Series
        :param wdir: wind direction measurements of wspds, only required if shear is to be applied by direction sector.
        :type wdir: Pandas Series
        :param height: height of above wspds
        :type height: float
        :param height_to_scale_to: height to which wspds should be scaled to
        :type height_to_scale_to: float

        :return: a DataFrame showing original wind speed, scaled wind speed, wind direction (if applicable)
         and the shear exponent used.

         **Example Usage**
         ::
            # Load anemometer data to calculate exponents
            data = bw.load_csv(bw.datasets.demo_data)
            anemometers = data[['Spd80mS', 'Spd60mS']]
            heights = [80, 60]
            directions = data[['Dir78mS']]
            sectors = 12

            # Calculate shear exponents
            shear_object_by_sector = bw.Shear.BySector(anemometers, heights, directions)

            # Calculate shear exponents using default bins ([345,15,45,75,105,135,165,195,225,255,285,315,345])
            shear_by_sector= bw.Shear.BySector(anemometers, heights, directions)

            # Scale wind speeds using exponents
            # Specify wind speeds to be scaled
            windseries = data['Spd40mN']
            height = 40
            height_to_scale_to =80
            wdir = data['DIr48mS']
            shear_by_sector.apply_alpha(windseries, wdir, height, height_to_scale_to)

            """

        return _apply(self, wspds, height, height_to_scale_to, wdir=wdir, calc_method='power_law')


def test_scale(wspds, height, height_to_scale_to, calc_method, slope, intercept):

    graph_speed = slope*np.log(height) + intercept
    error = -(graph_speed-wspds)/graph_speed

    scaled_graph_speed = slope*np.log(height_to_scale_to) + intercept
    corrected_speed = scaled_graph_speed + error*scaled_graph_speed

    return corrected_speed


def _calc_roughness_coeff(wspds, heights):

    wspds = wspds.sort_values()
    heights = pd.Series(heights).sort_values()
    roughness_coefficient = pd.Series([])
    for i in range(0, len(wspds)-1):
        roughness_coefficient[i] = e**(((wspds.iloc[i] * np.log(heights.iloc[i+1])) -
                                        (wspds.iloc[i+1] * np.log(heights.iloc[i])))
                                       / (wspds.iloc[i] - wspds.iloc[i+1]))

    return roughness_coefficient.mean()

def _calc_log_law(wspds, heights, return_coeff=False) -> (np.array, float):

    logheights = np.log(heights)
    coeffs = np.polyfit(logheights, wspds, deg=1)
    if return_coeff:
        return coeffs[0], coeffs[1]
    return coeffs[0]

def _calc_power_law(wspds, heights, return_coeff=False) -> (np.array, float):
    """
    Derive the best fit power law exponent (as 1/alpha) from a given time-step of speed data at 2 or more elevations

    :param wspds: List of wind speeds [m/s]
    :param heights: List of heights [m above ground]. The position of the height in the list must be the same
        position in the list as its corresponding wind speed value.
    :return: The shear value (alpha), as the inverse exponent of the best fit power law, based on the form:
        $(v1/v2) = (z1/z2)^(1/alpha)$

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


def _by_12x24(wspds, heights, min_speed=3, return_data=False, var_name='Shear'):
    tab_12x24 = dist_12x24(wspds[(wspds > min_speed).all(axis=1)].apply(_calc_power_law, heights=heights,
                                                                        axis=1), return_data=True)[1]
    if return_data:
        return plt.plot_12x24_contours(tab_12x24, label=(var_name, 'mean')), tab_12x24
    return plt.plot_12x24_contours(tab_12x24,  label=(var_name, 'mean'))


def scale(wspds, height, height_to_scale_to, calc_method, alpha=None, roughness_coefficient=None):
    """"
    Scales wind speeds from one height to another given a value of shear exponent

   :param alpha: Shear exponent to be used when scaling wind speeds
   :type alpha: Float
   :param wspds: Wind speed time series to apply shear to
   :type wspds: Pandas Series
   :param height: height of above wspds
   :type height: float
   :param height_to_scale_to: height to which wspds should be scaled to
   :type height_to_scale_to: float
   :return: a DataFrame showing original wind speed, scaled wind speed
    and the shear exponent used.

    **Example Usage**
    ::

       # Scale wind speeds using exponents
       # Specify wind speeds to be scaled
       windseries = data['Spd40mN']

       # Specify alpha to use
       alpha = .2

       height = 40
       height_to_scale_to =80
       wdir = data['DIr48mS']
       Shear.scale(alpha, windseries, height, height_to_scale_to)

    """
    if calc_method == 'power_law':
        scale_factor = (height_to_scale_to / height) ** alpha
        scale_string = 'Shear_Exponent'
        scale_variable = alpha

    elif calc_method =='log_law':
        scale_factor = np.log(height_to_scale_to/roughness_coefficient)/(np.log(height/roughness_coefficient))
        print(scale_factor)
        scale_string = 'Roughness_Coefficient'
        scale_variable = roughness_coefficient

    scale_variable = pd.DataFrame([scale_variable] * len(wspds))
    scale_variable.index = wspds.index
    result = pd.concat([wspds, wspds * scale_factor, scale_variable], axis=1)
    result.columns = ['Unscaled_Wind_Speeds' + '(' + str(height) + 'm)',
                      'Scaled_Wind_Speeds' + '(' + str(height_to_scale_to) + 'm)', scale_string]
    return result


def _apply(self, wspds, height, height_to_scale_to, wdir=None):

    scaled_wspds = pd.Series([])
    result = pd.Series([])

    if self.origin == 'by_sector':

        if wdir is None:
            raise ValueError('A wind direction series, wdir, is required for scaling wind speeds by '
                             'direction sector. Check origin of Shear object using ".origin"')
        # initilise series for later use
        alpha_bounds = pd.Series([])
        by_sector = pd.Series([])
        # join wind speeds and directions together in DataFrame

        df = pd.concat([wspds, wdir], axis=1)
        df.columns = ['Unscaled_Wind_Speeds', 'Wind_Direction']

        # get directional bin edges from Shear.by_sector output
        for i in range(self.sectors):
            alpha_bounds[i] = float(re.findall(r"[-+]?\d*\.\d+|\d+", self.alpha.index[i])[0])
            if i == self.sectors - 1:
                alpha_bounds[i + 1] = -float(re.findall(r"[-+]?\d*\.\d+|\d+", self.alpha.index[i])[1])

        #
        for i in range(0, self.sectors):
            if alpha_bounds[i] > alpha_bounds[i+1]:
                by_sector[i] = df[
                    (df['Wind_Direction'] >= alpha_bounds[i]) | (df['Wind_Direction'] < alpha_bounds[i + 1])]

            elif alpha_bounds[i + 1] == 360:
                by_sector[i] = df[(df['Wind_Direction'] >= alpha_bounds[i])]

            else:
                by_sector[i] = df[
                    (df['Wind_Direction'] >= alpha_bounds[i]) & (df['Wind_Direction'] < alpha_bounds[i + 1])]

            by_sector[i].columns = ['Unscaled_Wind_Speeds', 'Wind_Direction']

            scaled_wspds[i] = scale(by_sector[i]['Unscaled_Wind_Speeds'], height,
                                    height_to_scale_to, self.calc_method, alpha=self.alpha[i]).iloc[:, 1]
            by_sector[i]['Scaled_Wind_Speeds'] = scaled_wspds[i]
            by_sector[i]['Shear_Exponent'] = self.alpha[i]
            by_sector[i] = by_sector[i][
                ['Wind_Direction', 'Unscaled_Wind_Speeds', 'Scaled_Wind_Speeds', 'Shear_Exponent']]

            if i == 0:
                result = by_sector[i]
            else:
                result = pd.concat([result, by_sector[i]], axis=0)

        result.columns = ['Wind Direction', 'Unscaled_Wind_Speeds' + '(' + str(height) + 'm)',
                          'Scaled_Wind_Speeds' + '(' + str(height_to_scale_to) + 'm)', 'Shear_Exponent']
        result.sort_index(axis='index', inplace=True)

    if self.origin == 'Average':

        if wdir is not None:
            warnings.warn('Warning: Wind direction will not be accounted for when calculating scaled wind speeds.'
                          ' The shear exponents for this object were not calculated by sector. '
                          'Check the origin of the object using ".origin". ')

        if self.calc_method == 'power_law':
            result = scale(wspds, height, height_to_scale_to, self.calc_method, alpha=self.alpha)
        elif self.calc_method == 'log_law':
            result = scale(wspds, height, height_to_scale_to, self.calc_method, roughness_coefficient=self.roughness_coefficient)

    return result


if __name__ == '__main__':

    import brightwind as bw

    data = bw.load_csv(r'C:\Users\lukec\demo_data.csv')
    anemometers = data[['Spd80mN', 'Spd60mN', 'Spd40mN']]
    heights = [80, 60, 40]
    test = bw.Shear.TimeOfDay(anemometers, heights, daily_sectors=24, day_start_time='12', by_month=True)

