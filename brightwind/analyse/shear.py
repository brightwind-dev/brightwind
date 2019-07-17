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
import datetime
import calendar
from math import e
from brightwind.analyse import plot as plt
# noinspection PyProtectedMember
from brightwind.analyse.analyse import distribution_by_dir_sector, dist_12x24, coverage, _convert_df_to_series
from ipywidgets import FloatProgress
from IPython.display import display
from IPython.display import clear_output
import re
import warnings

pd.options.mode.chained_assignment = None

__all__ = ['']


def _unequal_wspd_heights_error_msg(wspds, heights):
    raise ValueError('An equal number of wind speed data series and heights is required. ' +
                     str(len(wspds.columns)) + ' wind speed(s) and ' +
                     str(len(heights)) + ' height(s) were given.')


class TimeSeries:

    def __init__(self, wspds, heights, min_speed=3, calc_method='power_law', max_plot_height=None, maximise_data=False):
        """
        Calculates alpha, using the power law, or the roughness coefficient, using the log law, for each timestamp of a
        wind series.

       :param wspds: pandas DataFrame, list of pandas.Series or list of wind speeds to be used for calculating shear.
       :type wspds:  pandas.DataFrame, list of pandas.Series or list.
       :param heights: List of anemometer heights.
       :type heights: list
       :param min_speed: Only speeds higher than this would be considered for calculating shear, default is 3.
       :type min_speed: float
       :param calc_method: method to use for calculation, either 'power_law' (returns alpha) or 'log_law'
                          (returns the roughness coefficient).
       :type calc_method: str
       :param max_plot_height: height to which the wind profile plot is extended.
       :type max_plot_height: float
       :param maximise_data: If maximise_data is True, calculations will be carried out on all data where two or
                             more anemometers readings exist for a timestamp. If False, calculations will only be
                             carried out on timestamps where readings exist for all anemometers.
       :type maximise_data: Boolean
       :return TimeSeries object containing calculated alpha/roughness coefficient values, a plot
               and other data.
       :rtype TimeSeries object

       **Example usage**
       ::
            import brightwind as bw
            import pprint

            # Load anemometer data to calculate exponents
            data = bw.load_csv(C:\\Users\\Stephen\\Documents\\Analysis\\demo_data)
            anemometers = data[['Spd80mS', 'Spd60mS','Spd40mS']]
            heights = [80, 60, 40]

            # Using with a DataFrame of wind speeds
            timeseries_power_law = bw.Shear.TimeSeries(anemometers, heights, maximise_data=True)
            timeseries_log_law = bw.Shear.TimeSeries(anemometers, heights, calc_method='log_law', max_plot_height=120)

            # Get the alpha or roughness values calculated
            timeseries_power_law.alpha
            timeseries_log_law.roughness

            # View plot
            timeseries_power_law.plot
            timeseries_log_law.plot

            # View input anemometer data
            timeseries_power_law.wspds
            timeseries_log_law.wspds

            # View other information
            pprint.pprint(timeseries_power_law.info)
            pprint.pprint(timeseries_log_law.info)

       """

        if not isinstance(wspds, pd.DataFrame):
            wspds = pd.DataFrame(wspds).T

        if len(wspds.columns) != len(heights):
            _unequal_wspd_heights_error_msg(wspds, heights)

        info = {}
        input_data = {}
        output_data = {}
        self.origin = 'TimeSeries'
        self.calc_method = calc_method
        wspds = wspds.dropna()

        print('This may take a while...')

        if isinstance(wspds.index, pd.DatetimeIndex):
            if maximise_data is False:
                cvg = coverage(wspds[wspds > min_speed].dropna(), period='1AS').sum()[1]
            else:
                _wspds = wspds[wspds > min_speed]
                count = _wspds.count(axis=1)
                count = count[count >= 2]
                count.rename('count', inplace=True)
                cvg = coverage(count, period='1AS').sum()

            output_data['concurrent_period_in_years'] = float("{:.3f}".format(cvg))

        if calc_method == 'power_law':
            alpha_c = (wspds[(wspds > min_speed).all(axis=1)].apply(_calc_power_law, heights=heights, return_coeff=True,
                                                                    maximise_data=maximise_data, axis=1))
            self._alpha = alpha_c.iloc[:, 0]
            output_data['alpha'] = self._alpha

        elif calc_method == 'log_law':
            slope_intercept = (wspds[(wspds > min_speed).all(axis=1)].apply(_calc_log_law, heights=heights,
                                                                            return_coeff=True,
                                                                            maximise_data=maximise_data, axis=1))
            slope = slope_intercept.iloc[:, 0]
            intercept = slope_intercept.iloc[:, 1]
            self._roughness = _calc_roughness(slope=slope, intercept=intercept)
            output_data['roughness'] = self._roughness

        input_wind_speeds = {'heights(m)': heights, 'column_names': list(wspds.columns.values),
                             'min_spd(m/s)': min_speed}
        input_data['input_wind_speeds'] = input_wind_speeds
        input_data['calculation_method'] = calc_method
        info['input data'] = input_data
        info['output data'] = output_data
        self.wspds = wspds
        self.info = info
        self.calc_method = calc_method
        clear_output()
        avg_plot = Average(wspds=wspds, heights=heights, calc_method=calc_method, max_plot_height=max_plot_height)
        self.plot = avg_plot.plot

    @property
    def alpha(self):
        return self._alpha

    @property
    def roughness(self):
        return self._roughness

    def apply(self, wspds, height, shear_to):
        """"
        Applies shear calculated to a wind speed time series and scales wind speed from one height to
        another for each matching timestamp.

       :param self: TimeSeries object to use when applying shear to the data.
       :type self: TimeSeries object
       :param wspds: Wind speed time series to apply shear to.
       :type wspds: pandas.Series
       :param height: height of above wspds.
       :type height: float
       :param shear_to: height to which wspds should be scaled to.
       :type shear_to: float
       :return: a pandas.Series of the scaled wind speeds.
       :rtype: pandas.Series 

        **Example Usage**
        ::
            import brightwind as bw

            # Load anemometer data to calculate exponents
            data = bw.load_csv(C:\\Users\\Stephen\\Documents\\Analysis\\demo_data)
            anemometers = data[['Spd80mS', 'Spd60mS','Spd40mS']]
            heights = [80, 60, 40]

            # Get power law object
            timeseries_power_law = bw.Shear.TimeSeries(anemometers, heights)
            timeseries_log_law = bw.Shear.TimeSeries(anemometers, heights, calc_method='log_law')

            # Scale wind speeds using calculated exponents
            timeseries_power_law.apply(data['Spd40mN'], height=40, shear_to=70)
            timeseries_log_law.apply(data['Spd40mN'], height=40, shear_to=70)
           """

        return _apply(self, wspds, height, shear_to)


class TimeOfDay:

    def __init__(self, wspds, heights, min_speed=3, calc_method='power_law', by_month=True, segment_start_time=7,
                 segments_per_day=2, plot_type='step'):
        """
        Calculates alpha, using the power law, or the roughness coefficient, using the log law, for a wind series binned
        by time of the day and (optionally by) month, depending on the user's inputs. The alpha/roughness coefficient
        values are calculated based on the average wind speeds at each measurement height in each bin.

        :param wspds: pandas.DataFrame, list of pandas.Series or list of wind speeds to be used for calculating shear.
        :type wspds:  pandas.DataFrame, list of pandas.Series or list.
        :param heights: List of anemometer heights..
        :type heights: list
        :param min_speed: Only speeds higher than this would be considered for calculating shear, default is 3
        :type min_speed: float
        :param calc_method: method to use for calculation, either 'power_law' (returns alpha) or 'log_law'
                            (returns the roughness coefficient).
        :type calc_method: str
        :param by_month: If True, calculate alpha or roughness coefficient values for each daily segment and month.
                         If False, average alpha or roughness coefficient values are calculated for each daily segment
                         across all months.
        :type by_month: Boolean
        :param segment_start_time: Starting time for first segment.
        :type segment_start_time: int
        :param segments_per_day: Number of segments into which each 24 period is split. Must be a divisor of 24.
        :type segments_per_day: int
        :param plot_type: Type of plot to be generated. Options include 'line', 'step' and '12x24'.
        :type plot_type: str
        :return: TimeOfDay object containing calculated alpha/roughness coefficient values, a plot
                 and other data.
        :rtype: TimeOfDay object

        **Example usage**
        ::
            import brightwind as bw
            import pprint

            # Load anemometer data to calculate exponents
            data = bw.load_csv(C:\\Users\\Stephen\\Documents\\Analysis\\demo_data)
            anemometers = data[['Spd80mS', 'Spd60mS','Spd40mS']]
            heights = [80, 60, 40]

            # Using with a DataFrame of wind speeds
            timeofday_power_law = bw.Shear.TimeOfDay(anemometers, heights, daily_segments=2, segment_start_time=7)
            timeofday_log_law = bw.Shear.TimeOfDay(anemometers, heights, calc_method='log_law', by_month=False)

            # Get alpha or roughness values calculated
            timeofday_power_law.alpha
            timeofday_log_law.roughness

            # View plot
            timeofday_power_law.plot
            timeofday_log_law.plot

            # View input data
            timeofday_power_law.wspds
            timeofday_log_law.wspds

            # View other information
            pprint.pprint(timeofday_power_law.info)
            pprint.pprint(timeofday_log_law.info)

        """
        # initialise empty series for later use
        start_times = pd.Series([])
        time_wspds = pd.Series([])
        mean_time_wspds = pd.Series([])
        c = pd.Series([])
        slope = pd.Series([])
        intercept = pd.Series([])
        alpha = pd.Series([])
        roughness_coefficient = pd.Series([])
        slope_df = pd.DataFrame([])
        intercept_df = pd.DataFrame([])
        roughness_coefficient_df = pd.DataFrame([])
        alpha_df = pd.DataFrame([])

        info = {}
        input_data = {}
        output_data = {}

        if not isinstance(wspds, pd.DataFrame):
            wspds = pd.DataFrame(wspds).T
        wspds = wspds.dropna()

        if len(wspds.columns) != len(heights):
            _unequal_wspd_heights_error_msg(wspds, heights)

        cvg = coverage(wspds[wspds > min_speed].dropna(), period='1AS').sum()[1]

        # time of day shear calculations
        interval = int(24 / segments_per_day)

        if by_month is False and plot_type == '12x24':
            raise ValueError("12x24 plot is only possible when 'by_month=True'")

        if not int(segment_start_time) % 1 == 0:
            raise ValueError("'segment_start_time' must be an integer between 0 and 24'")

        if not (24 % segments_per_day == 0) | (segments_per_day == 1):
            raise ValueError("'segments_per_day' must be a divisor of 24'")

        segment_start_time = str(segment_start_time)
        start_times[0] = datetime.datetime.strptime(segment_start_time, '%H')
        dt = datetime.timedelta(hours=interval)

        # extract wind speeds for each daily segment
        for i in range(1, segments_per_day):
            start_times[i] = start_times[i - 1] + dt

        # extract wind speeds for each month
        for j in range(0, 12):

            anemometers_df = wspds[wspds.index.month == j + 1]
            for i in range(0, segments_per_day):

                if segments_per_day == 1:
                    mean_time_wspds[i] = anemometers_df.mean().dropna()

                elif i == segments_per_day - 1:
                    start_times[i] = start_times[i].strftime("%H:%M:%S")
                    start = str(start_times[i].time())
                    end = str(start_times[0].time())
                    time_wspds[i] = pd.DataFrame(anemometers_df).between_time(start, end, include_end=False)
                    mean_time_wspds[i] = time_wspds[i][(time_wspds[i] > min_speed).all(axis=1)].mean().dropna()

                else:
                    start_times[i] = start_times[i].strftime("%H:%M:%S")
                    start = str(start_times[i].time())
                    end = str(start_times[i + 1].time())
                    time_wspds[i] = pd.DataFrame(anemometers_df).between_time(start, end, include_end=False)
                    mean_time_wspds[i] = time_wspds[i][(time_wspds[i] > min_speed).all(axis=1)].mean().dropna()

            # calculate shear
            if calc_method == 'power_law':
                for i in range(0, len(mean_time_wspds)):
                    alpha[i], c[i] = _calc_power_law(mean_time_wspds[i].values, heights, return_coeff=True)
                alpha_df = pd.concat([alpha_df, alpha], axis=1)

            if calc_method == 'log_law':
                for i in range(0, len(mean_time_wspds)):
                    slope[i], intercept[i] = _calc_log_law(mean_time_wspds[i].values, heights, return_coeff=True)
                    roughness_coefficient[i] = _calc_roughness(slope=slope[i], intercept=intercept[i])
                roughness_coefficient_df = pd.concat([roughness_coefficient_df, roughness_coefficient], axis=1)
                slope_df = pd.concat([slope_df, slope], axis=1)
                intercept_df = pd.concat([intercept_df, intercept], axis=1)

        # error check
        if mean_time_wspds.shape[0] == 0:
            raise ValueError('None of the input wind speeds are greater than the min_speed, cannot calculate shear')

        if calc_method == 'power_law':
            alpha_df.index = start_times
            alpha_df.index = alpha_df.index.time
            alpha_df.sort_index(inplace=True)

            if by_month is True:
                alpha_df.columns = calendar.month_abbr[1:13]
                self.plot = plt.plot_shear_time_of_day(_fill_df_12x24(alpha_df), calc_method=calc_method,
                                                       plot_type=plot_type)

            else:
                alpha_df = pd.DataFrame(alpha_df.mean(axis=1))
                alpha_df.columns = ['12 Month Average']
                self.plot = plt.plot_shear_time_of_day(pd.DataFrame((_fill_df_12x24(alpha_df)).iloc[:, 0]),
                                                       calc_method=calc_method, plot_type=plot_type)

            output_data['alpha'] = alpha_df
            self._alpha = alpha_df

        if calc_method == 'log_law':
            roughness_coefficient_df.index = slope_df.index = intercept_df.index = start_times
            roughness_coefficient_df.index = slope_df.index = intercept_df.index = roughness_coefficient_df.index.time
            roughness_coefficient_df.sort_index(inplace=True)

            slope_df.sort_index(inplace=True)
            intercept_df.sort_index(inplace=True)

            if by_month is True:
                roughness_coefficient_df.columns = slope_df.columns = intercept_df.columns = calendar.month_abbr[1:13]
                self.plot = plt.plot_shear_time_of_day(_fill_df_12x24(roughness_coefficient_df),
                                                       calc_method=calc_method, plot_type=plot_type)
            else:
                slope_df = pd.DataFrame(slope_df.mean(axis=1))
                intercept_df = pd.DataFrame(intercept_df.mean(axis=1))
                roughness_coefficient_df = pd.DataFrame(roughness_coefficient_df.mean(axis=1))
                roughness_coefficient_df.columns = slope_df.columns = intercept_df.columns = ['12_month_average']
                self.plot = plt.plot_shear_time_of_day(
                    pd.DataFrame(_fill_df_12x24(roughness_coefficient_df).iloc[:, 0]),
                    calc_method=calc_method, plot_type=plot_type)

            roughness_coefficient_df.index.name = 'segment_start_time'
            output_data['roughness'] = roughness_coefficient_df
            self._roughness = roughness_coefficient_df

        input_wind_speeds = {'heights(m)': heights, 'column_names': list(wspds.columns.values),
                             'min_spd(m/s)': min_speed}
        input_data['input_wind_speeds'] = input_wind_speeds
        input_data['segments_per_day'] = segments_per_day
        input_data['segment_start_time'] = int(segment_start_time)
        input_data['calculation_method'] = calc_method
        output_data['concurrent_period_in_years'] = float("{:.3f}".format(cvg))
        info['input data'] = input_data
        info['output data'] = output_data

        self.wspds = wspds
        self.origin = 'TimeOfDay'
        self.info = info
        self.calc_method = calc_method

    @property
    def alpha(self):
        return self._alpha

    @property
    def roughness(self):
        return self._roughness

    def apply(self, wspds, height, shear_to):
        """
        Applies shear calculated to a wind speed time series by time of day (and optionally by month) to scale
        wind speed from one height to another.

       :param self: TimeOfDay object to use when applying shear to the data.
       :type self: TimeOfDay object
       :param wspds: Wind speed time series to apply shear to.
       :type wspds: pandas.Series
       :param height: height of above wspds.
       :type height: float
       :param shear_to: height to which wspds should be scaled.
       :type shear_to: float
       :return: a pandas.Series of the scaled wind speeds.
       :rtype: pandas.Series

        **Example Usage**
        ::
            import brightwind as bw

            # Load anemometer data to calculate exponents
            data = bw.load_csv(C:\\Users\\Stephen\\Documents\\Analysis\\demo_data)
            anemometers = data[['Spd80mS', 'Spd60mS','Spd40mS']]
            heights = [80, 60, 40]

            # Get power law object
            timeofday_power_law = bw.Shear.TimeOfDay(anemometers, heights)
            timeofday_log_law = bw.Shear.TimeOfDay(anemometers, heights, calc_method='log_law')

            # Scale wind speeds using calculated exponents
            timeofday_power_law.apply(data['Spd40mN'], height=40, shear_to=70)
            timeofday_log_law.apply(data['Spd40mN'], height=40, shear_to=70)

        """

        return _apply(self, wspds, height, shear_to)


class Average:

    def __init__(self, wspds, heights, min_speed=3, calc_method='power_law', plot_both=False, max_plot_height=None):
        """
         Calculates alpha, using the power law, or the roughness coefficient, using the log law, based on the average
         wind speeds of each supplied time series.

        :param wspds: pandas.DataFrame, list of pandas.Series or list of wind speeds to be used for calculating shear.
        :type wspds:  pandas.DataFrame, list of pandas.Series or list.
        :param heights: List of anemometer heights
        :type heights: list
        :param min_speed: Only speeds higher than this would be considered for calculating shear, default is 3.
        :type min_speed: float
        :param calc_method: method to use for calculation, either 'power_law' (returns alpha) or 'log_law'
                                  (returns the roughness coefficient).
        :type calc_method: str
        :param max_plot_height: height to which the wind profile plot is extended.
        :type max_plot_height: float
        :return:  Average object containing calculated alpha/roughness coefficient values, a plot and other data.
        :rtype: Average object

        **Example usage**
        ::
            import brightwind as bw
            import pprint

            # Load anemometer data to calculate exponents
            data = bw.load_csv(C:\\Users\\Stephen\\Documents\\Analysis\\demo_data)
            anemometers = data[['Spd80mS', 'Spd60mS','Spd40mS']]
            heights = [80, 60, 40]

            # Using with a DataFrame of wind speeds
            average_power_law = bw.Shear.Average(anemometers, heights)
            average_log_law = bw.Shear.Average(anemometers, heights, calc_method='log_law', max_plot_height=120)

            # Get the alpha or roughness values calculated
            average_power_law.alpha
            average_log_law.roughness

            # View plot
            average_power_law.plot
            average_log_law.plot

            # View input data
            average_power_law.wspds
            average_log_law.wspds

            # View other information
            pprint.pprint(average_power_law.info)
            pprint.pprint(average_log_law.info)

        """
        if not isinstance(wspds, pd.DataFrame):
            wspds = pd.DataFrame(wspds).T

        wspds = wspds.dropna()

        if len(wspds.columns) != len(heights):
            _unequal_wspd_heights_error_msg(wspds, heights)

        info = {}
        input_data = {}
        output_data = {}

        if isinstance(wspds.index, pd.DatetimeIndex):
            cvg = coverage(wspds[wspds > min_speed].dropna(), period='1AS').sum()[1]
            output_data['concurrent_period_in_years'] = float("{:.3f}".format(cvg))
        mean_wspds = wspds[(wspds > min_speed).all(axis=1)].mean().dropna()

        if mean_wspds.shape[0] == 0:
            raise ValueError('None of the input wind speeds are greater than the min_speed, cannot calculate shear')

        if calc_method == 'power_law':
            alpha, c = _calc_power_law(mean_wspds.values, heights, return_coeff=True)
            if plot_both is True:
                slope, intercept = _calc_log_law(mean_wspds.values, heights, return_coeff=True)
                self.plot = plt.plot_power_law(plot_both=True, avg_alpha=alpha, avg_c=c, avg_slope=slope,
                                               avg_intercept=intercept,
                                               wspds=mean_wspds.values, heights=heights,
                                               max_plot_height=max_plot_height)
            else:
                self.plot = plt.plot_power_law(alpha, c, mean_wspds.values, heights, max_plot_height=max_plot_height)
            self._alpha = alpha
            output_data['alpha'] = alpha

        elif calc_method == 'log_law':
            slope, intercept = _calc_log_law(mean_wspds.values, heights, return_coeff=True)
            roughness = _calc_roughness(slope=slope, intercept=intercept)
            self._roughness = roughness
            if plot_both is True:
                alpha, c = _calc_power_law(mean_wspds.values, heights, return_coeff=True)
                self.plot = plt.plot_power_law(avg_alpha=alpha, avg_c=c, avg_slope=slope, avg_intercept=intercept,
                                               wspds=mean_wspds.values, heights=heights,
                                               max_plot_height=max_plot_height)
            else:
                self.plot = plt.plot_log_law(slope, intercept, mean_wspds.values, heights,
                                             max_plot_height=max_plot_height)

            output_data['roughness'] = roughness

        else:
            raise ValueError("Please enter a valid calculation method, either 'power_law' or 'log_law'.")

        input_wind_speeds = {'heights(m)': heights, 'column_names': list(wspds.columns.values),
                             'min_spd(m/s)': min_speed}
        input_data['input_wind_speeds'] = input_wind_speeds
        input_data['calculation_method'] = calc_method
        info['input data'] = input_data
        info['output data'] = output_data

        self.wspds = wspds
        self.origin = 'Average'
        self.info = info
        self.calc_method = calc_method

    @property
    def alpha(self):
        return self._alpha

    @property
    def roughness(self):
        return self._roughness

    def apply(self, wspds, height, shear_to):
        """
        Applies average shear calculated to a wind speed time series to scale wind speed from one height to another.

       :param self: Average object to use when applying shear to the data.
       :type self: Average object
       :param wspds: Wind speed time series to apply shear to.
       :type wspds: pandas.Series
       :param height: height of above wspds.
       :type height: float
       :param shear_to: height to which wspds should be scaled to.
       :type shear_to: float
       :return: a pandas.Series of the scaled wind speeds.
       :rtype: pandas.Series

        **Example Usage**
        ::
            import brightwind as bw

            # Load anemometer data to calculate exponents
            data = bw.load_csv(C:\\Users\\Stephen\\Documents\\Analysis\\demo_data)
            anemometers = data[['Spd80mS', 'Spd60mS','Spd40mS']]
            heights = [80, 60, 40]

            # Get power law object
            average_power_law = bw.Shear.Average(anemometers, heights)
a           average_log_law = bw.Shear.Average(anemometers, heights, calc_method='log_law', max_plot_height=120)

           # Scale wind speeds using exponents
           average_power_law.apply(data['Spd40mN'], height=40, shear_to=70)
           average_log_law.apply(data['Spd40mN'], height=40, shear_to=70)

           """
        return _apply(self, wspds, height, shear_to)


class BySector:

    def __init__(self, wspds, heights, wdir, min_speed=3, calc_method='power_law', sectors=12,
                 direction_bin_array=None, direction_bin_labels=None):
        """
        Calculates alpha, using the power law, or the roughness coefficient, using the log law, for a wind series binned
         by direction. The alpha/roughness coefficient values are calculated based on the average wind speeds at each
        measurement height in each bin.


        :param wspds: pandas.DataFrame, list of pandas.Series or list of wind speeds to be used for calculating shear.
        :type wspds: pandas.DataFrame, list of pandas.Series or list.
        :param heights: List of anemometer heights
        :type heights: list
        :param wdir: Wind direction measurements
        :type wdir: pandas.DataFrame or Series
        :param: min_speed: Only speeds higher than this would be considered for calculating shear, default is 3.
        :type: min_speed: float
        :param calc_method: Method to use for calculation, either 'power_law' (returns alpha) or 'log_law'
                            (returns the roughness coefficient).
        :type calc_method: str
        :param sectors: Number of sectors for the shear to be calculated for.
        :type sectors: int
        :param direction_bin_array: Specific array of directional bins to be used. If None, bins are calculated
                                    by 360/sectors.
        :type direction_bin_array: array
        :param direction_bin_labels: Labels to be given to the above direction_bin array.
        :type direction_bin_labels: array
        :return: BySector object containing calculated alpha/roughness coefficient values, a plot and other data.
        :rtype: BySector object

         **Example usage**
        ::
            import brightwind as bw
            import pprint

            # Load anemometer data to calculate exponents
            data = bw.load_csv(C:\\Users\\Stephen\\Documents\\Analysis\\demo_data)
            anemometers = data[['Spd80mS', 'Spd60mS','Spd40mS']]
            heights = [80, 60, 40]
            directions = data['Dir78mS']

            # Calculate shear exponents using default bins ([345,15,45,75,105,135,165,195,225,255,285,315,345])
            by_sector_power_law= bw.Shear.BySector(anemometers, heights, directions)
            by_sector_log_law= bw.Shear.BySector(anemometers, heights, directions, calc_method='log_law')

            # Calculate shear exponents using custom bins
            custom_bins = [0,30,60,90,120,150,180,210,240,270,300,330,360]
            by_sector_power_law_custom_bins = bw.Shear.BySector(anemometers, heights, directions,
            direction_bin_array=custom_bins)

            # Get alpha or roughness values calculated
            by_sector_power_law.alpha
            by_sector_log_law.roughness

            # View plot
            by_sector_power_law.plot
            by_sector_log_law.plot

            # View input data
            by_sector_power_law.wspds
            by_sector_log_law.wspds

            # View other information
            pprint.pprint(by_sector_power_law.info)
            pprint.pprint(by_sector_log_law.info)

        """

        if not isinstance(wspds, pd.DataFrame):
            wspds = pd.DataFrame(wspds).T

        wspds = wspds.dropna()

        if len(wspds.columns) != len(heights):
            _unequal_wspd_heights_error_msg(wspds, heights)

        if direction_bin_array is not None:
            sectors = len(direction_bin_array) - 1
        wspds = wspds[wspds > 3].dropna()
        wdir = _convert_df_to_series(wdir)
        mean_wspds = pd.Series([])
        mean_wspds_df = pd.DataFrame([])
        count_df = pd.DataFrame([])
        count = pd.Series([])

        cvg = coverage(wspds[wspds > min_speed].dropna(), period='1AS').sum()[1]
        info = {}
        output_data = {}
        input_data = {}

        print('This may take a while...')

        for i in range(len(wspds.columns)):

            w = wspds.iloc[:, i]
            plot, mean_wspds[i] = distribution_by_dir_sector(w, wdir, direction_bin_array=direction_bin_array,
                                                             aggregation_method='mean', return_data=True)

            plot, count[i] = distribution_by_dir_sector(w, wdir, direction_bin_array=direction_bin_array,
                                                        aggregation_method='count', return_data=True)

            if i == 0:
                mean_wspds_df = mean_wspds[i].copy()
                count_df = count[i].copy()
            else:
                mean_wspds_df = pd.concat([mean_wspds_df, mean_wspds[i]], axis=1)
                count_df = pd.concat([count_df, count[i]], axis=1)

        count_df = count_df.mean(axis=1)
        wind_rose_plot, wind_rose_dist = distribution_by_dir_sector(wspds.iloc[:, 0], wdir,
                                                                    direction_bin_array=direction_bin_array,
                                                                    direction_bin_labels=direction_bin_labels,
                                                                    return_data=True)
        if calc_method == 'power_law':

            alpha = mean_wspds_df.apply(_calc_power_law, heights=heights, return_coeff=False, axis=1)

            wind_rose_plot, wind_rose_dist = distribution_by_dir_sector(wspds.iloc[:, 0], wdir,
                                                                        direction_bin_array=direction_bin_array,
                                                                        direction_bin_labels=direction_bin_labels,
                                                                        return_data=True)

            output_data['alpha'] = alpha
            output_data['alpha_count'] = count_df
            self.alpha_count = count_df
            self._alpha = alpha
            clear_output()
            self.plot = plt.plot_shear_by_sector(scale_variable=alpha, wind_rose_data=wind_rose_dist,
                                                 calc_method=calc_method)

        elif calc_method == 'log_law':

            slope_intercept = mean_wspds_df.apply(_calc_log_law, heights=heights, return_coeff=True, axis=1)

            slope = slope_intercept.iloc[:, 0]
            intercept = slope_intercept.iloc[:, 1]
            roughness = _calc_roughness(slope=slope, intercept=intercept)
            output_data['roughness'] = roughness
            output_data['roughness_count'] = count_df
            self.roughness_count = count_df
            self._roughness = roughness
            self.plot = plt.plot_shear_by_sector(scale_variable=roughness, wind_rose_data=wind_rose_dist,
                                                 calc_method=calc_method)

        else:
            raise ValueError("Please enter a valid calculation method, either 'power_law' or 'log_law'.")

        input_wind_speeds = {'heights(m)': heights, 'column_names': list(wspds.columns.values),
                             'min_spd(m/s)': min_speed}
        input_wind_dir = {'heights(m)': float((re.findall(r'\d+', str(wdir.name))[0])),
                          'column_names': str(wdir.name)}
        input_data['input_wind_speeds'] = input_wind_speeds
        input_data['input_wind_dir'] = input_wind_dir
        input_data['sectors'] = sectors
        input_data['direction_bins'] = direction_bin_array
        input_data['calculation_method'] = calc_method
        output_data['concurrent_period_in_years'] = float("{:.3f}".format(cvg))
        info['input data'] = input_data
        info['output data'] = output_data
        self.wspds = wspds
        self.wdir = wdir
        self.origin = 'BySector'
        self.sectors = sectors
        self.calc_method = calc_method
        self.info = info

    @property
    def alpha(self):
        return self._alpha

    @property
    def roughness(self):
        return self._roughness

    def apply(self, wspds, wdir, height, shear_to):
        """
        Applies shear calculated to a wind speed time series by wind direction to scale
        wind speed from one height to another.

        :param self: BySector object to use when applying shear to the data
        :type self: BySector object
        :param wspds: Wind speed time series to apply shear to.
        :type wspds: pandas.Series
        :param wdir: Wind direction measurements of wspds, only required if shear is to be applied by direction sector.
        :type wdir: pandas.Series
        :param height: Height of wspds.
        :type height: float
        :param shear_to: Height to which wspds should be scaled to.
        :type shear_to: float
        :return: A pandas.Series of the scaled wind speeds.
        :rtype: pandas.Series

         **Example Usage**
         ::
            import brightwind as bw

            # Load anemometer data to calculate exponents
            data = bw.load_csv(C:\\Users\\Stephen\\Documents\\Analysis\\demo_data)
            anemometers = data[['Spd80mS', 'Spd60mS']]
            heights = [80, 60]
            directions = data[['Dir78mS']]

            # Calculate shear exponents
            by_sector_power_law = bw.Shear.BySector(anemometers, heights, directions)
            by_sector_log_law = bw.Shear.BySector(anemometers, heights, directions, calc_method='log_law')

            # Calculate shear exponents using default bins ([345,15,45,75,105,135,165,195,225,255,285,315,345])
            by_sector_power_law= bw.Shear.BySector(anemometers, heights, directions)

            # Scale wind speeds using exponents
            by_sector_power_law.apply(data['Spd40mN'], data['Dir38mS'], height=40, shear_to=70)
            by_sector_log_law.apply(data['Spd40mN'], data['Dir38mS'], height=40, shear_to=70)

        """

        return _apply(self, wspds=wspds, height=height, shear_to=shear_to, wdir=wdir)


def _log_roughness_scale(wspds, height, shear_to, roughness_coefficient):
    """
    Scale wind speeds using the logarithmic wind shear law.

    :param wspds: wind speeds at height z1, U1
    :param height: z1
    :param shear_to: z2
    :param roughness_coefficient: z0
    :return: Scaled wind speeds, U2
    :rtype: pandas.Series or float

    METHODOLOGY:

                                        U2 = (ln(z2/z0)/ln(z1/z0))U1
        Where:
                - U2 is the wind speed at height z2
                - U1 is the wind speed at height z1
                - z1 is the lower height
                - z2 is the upper height
                - zo is the roughness coefficient
    """
    scale_factor = np.log(shear_to / roughness_coefficient) / (np.log(height / roughness_coefficient))
    scaled_wspds = wspds * scale_factor

    return scaled_wspds


def _calc_log_law(wspds, heights, return_coeff=False, maximise_data=False) -> (np.array, float):
    """
    Derive the best fit logarithmic law line from a given time-step of speed data at 2 or more elevations

    :param wspds: List of wind speeds [m/s]
    :param heights: List of heights [m above ground]. The position of the height in the list must be the same
        position in the list as its corresponding wind speed value.
    :return: The slope and intercept of the best fit line, as defined above
    :rtype: pandas.Series and float

    METHODOLOGY:
        Derive natural log of elevation data sets
        Derive coefficients of linear best fit along ln(heights)- wspds distribution
        Characterise new distribution of speed values based on linear best fit
        Return the slope and the intercept of this linear best fit.
        The slope and intercept can then be used to find the corresponding roughness coefficient, using the
        equivilant laws:

                                            1)  $U(z) = (v/k)*ln(z/zo)$
        which can be rewritten as:
                                            $U(z) = (v/k)*ln(z) - (v/k)ln(zo)$
                                            where zo = e ** (-c/m) of this line

        Where:

            - U(z) is the wind speed at height z
            - v is the friction velocity at the location
            - k is the Von Karmen constant, taken aa .4
            - z is the height
            - zo is the roughness coefficient

                                            2)  $U2 = (ln(z2/z0)/ln(z1/z0))U1$

        """

    if maximise_data:
        log_heights = np.log(
            pd.Series(heights).drop(wspds[wspds == 0].index.values.astype(int)))  # take log of elevations
        wspds = wspds.drop(wspds[wspds == 0].index.values.astype(int))

    else:
        log_heights = np.log(heights)  # take log of elevations

    coeffs = np.polyfit(log_heights, wspds, deg=1)
    if return_coeff:
        return pd.Series([coeffs[0], coeffs[1]])
    return coeffs[0]


def _calc_power_law(wspds, heights, return_coeff=False, maximise_data=False) -> (np.array, float):
    """
    Derive the best fit power law exponent (as 1/alpha) from a given time-step of speed data at 2 or more elevations

    :param wspds: pandas.Series or list of wind speeds [m/s]
    :param heights: List of heights [m above ground]. The position of the height in the list must be the same
        position in the list as its corresponding wind speed value.
    :return: The shear value (alpha), as the inverse exponent of the best fit power law, based on the form:
        $(v1/v2) = (z1/z2)^(1/alpha)$
    :rtype: pandas.Series and float

    METHODOLOGY:
        Derive natural log of elevation and speed data sets
        Derive coefficients of linear best fit along log-log distribution
        Characterise new distribution of speed values based on linear best fit
        Derive 'alpha' based on gradient of first and last best fit points (function works for 2 or more points)
        Return alpha value

    """
    if maximise_data:
        log_heights = np.log(pd.Series(heights).drop(wspds[wspds == 0].index.values.astype(int)))
        log_wspds = np.log(wspds.drop(wspds[wspds == 0].index.values.astype(int)))

    else:
        log_heights = np.log(heights)  # take log of elevations
        log_wspds = np.log(wspds)  # take log of speeds

    coeffs = np.polyfit(log_heights, log_wspds, deg=1)  # get coefficients of linear best fit to log distribution
    if return_coeff:
        return pd.Series([coeffs[0], np.exp(coeffs[1])])
    return coeffs[0]


def _calc_roughness(slope, intercept):
    return e**(-intercept/slope)


def _by_12x24(wspds, heights, min_speed=3, return_data=False, var_name='Shear'):
    tab_12x24 = dist_12x24(wspds[(wspds > min_speed).all(axis=1)].apply(_calc_power_law, heights=heights,
                                                                        axis=1), return_data=True)[1]
    if return_data:
        return plt.plot_12x24_contours(tab_12x24, label=(var_name, 'mean')), tab_12x24
    return plt.plot_12x24_contours(tab_12x24, label=(var_name, 'mean'))


def scale(wspd,  height, shear_to, alpha=None, roughness=None, calc_method='power_law'):
    """
    Scales wind speeds from one height to another given a value of alpha or roughness coefficient (zo)

    :param wspd: Wind speed time series to apply shear to.
    :type wspd: pandas.Series
    :param height: height of above wspds.
    :type height: float
    :param shear_to: Height to which wspd should be scaled to.
    :type shear_to: float
    :param alpha: Shear exponent to be used when scaling wind speeds.
    :type alpha: Float
    :param roughness: Roughness coefficient to be used when scaling wind speeds.
    :type roughness: float
    :param calc_method: calculation method used to scale the wind speed.
            Using either:   1) 'power_law' :    $(v1/v2) = (z1/z2)^(1/alpha)$
                            2) 'log_law':       $v2 = (ln(z2/z0)/ln(z1/z0))v1$
    :type calc_method: string
    :return: a pandas series of the scaled wind speed
    :return: pandas.Series or float

    **Example Usage**
    ::

       # Scale wind speeds using exponents

       # Specify alpha to use
       alpha_value = .2

       # Specify roughness coefficient to use
       zo = .03

       height = 40
       shear_to = 80

       scaled_by_power_law = bw.Shear.scale(data['Spd40mN'], height, shear_to, alpha=alpha_value)
       scaled_by_log_law = bw.Shear.scale(data['Spd40mN'], height, shear_to, roughness=zo, calc_method='log_law')

    """
    return _scale(wspds=wspd, height=height, shear_to=shear_to, calc_method=calc_method,
                  alpha=alpha, roughness_coefficient=roughness)


def _scale(wspds, height, shear_to, calc_method='power_law', alpha=None, roughness_coefficient=None, origin=None):
    """
    Private function for execution of scale()
    """
    if not isinstance(wspds, pd.Series):
        wspds = pd.Series(wspds)

    if calc_method == 'power_law':
        scale_factor = (shear_to / height) ** alpha
        scaled_wspds = wspds * scale_factor

    elif calc_method == 'log_law':
        if origin == 'TimeSeries':
            scaled_wspds = _log_roughness_scale(wspds=wspds, height=height,
                                                shear_to=shear_to, roughness_coefficient=roughness_coefficient)
        else:
            scaled_wspds = wspds.apply(_log_roughness_scale, args=(height, shear_to, roughness_coefficient))
    else:
        raise ValueError("Please enter a valid calculation method, either 'power_law' or 'log_law'.")

    return scaled_wspds


def _apply(self, wspds, height, shear_to, wdir=None):
    scaled_wspds = pd.Series([])
    result = pd.Series([])

    if self.origin == 'TimeSeries':

        if self.calc_method == 'power_law':
            df = pd.concat([wspds, self.alpha], axis=1).dropna()
            scaled_wspds = _scale(wspds=df.iloc[:, 0], height=height, shear_to=shear_to,
                                  calc_method='power_law', alpha=df.iloc[:, 1])

        else:
            df = pd.concat([wspds, self.roughness], axis=1).dropna()
            scaled_wspds = _scale(wspds=df.iloc[:, 0], height=height, shear_to=shear_to, calc_method=self.calc_method,
                                  roughness_coefficient=self._roughness, origin=self.origin)

        result = scaled_wspds.dropna()

    if self.origin == 'TimeOfDay':

        if self.calc_method == 'power_law':
            filled_alpha = _fill_df_12x24(self.alpha)

        else:
            filled_roughness_coefficient = _fill_df_12x24(self._roughness)
            filled_alpha = filled_roughness_coefficient

        df_wspds = [[None for y in range(12)] for x in range(24)]
        f = FloatProgress(min=0, max=24 * 12, description='Calculating', bar_style='success')
        display(f)

        for i in range(0, 24):

            for j in range(0, 12):

                if i == 23:
                    df_wspds[i][j] = wspds[
                        (wspds.index.time >= filled_alpha.index[i]) & (wspds.index.month == j + 1)]
                else:
                    df_wspds[i][j] = wspds[
                        (wspds.index.time >= filled_alpha.index[i]) & (wspds.index.time < filled_alpha.index[i + 1]) & (
                                wspds.index.month == j + 1)]

                if self.calc_method == 'power_law':
                    df_wspds[i][j] = _scale(df_wspds[i][j], shear_to=shear_to, height=height,
                                            alpha=filled_alpha.iloc[i, j], calc_method=self.calc_method)

                else:
                    df_wspds[i][j] = _scale(df_wspds[i][j], shear_to=shear_to, height=height,
                                            roughness_coefficient=filled_roughness_coefficient.iloc[i, j],
                                            calc_method=self.calc_method)

                scaled_wspds = pd.concat([scaled_wspds, df_wspds[i][j]], axis=0)
                f.value += 1

        result = scaled_wspds.sort_index()
        f.close()

    if self.origin == 'BySector':

        # initilise series for later use
        bin_edges = pd.Series([])
        by_sector = pd.Series([])

        if self.calc_method == 'power_law':
            direction_bins = self.alpha
        else:
            direction_bins = self._roughness

        # join wind speeds and directions together in DataFrame
        df = pd.concat([wspds, wdir], axis=1)
        df.columns = ['Unscaled_Wind_Speeds', 'Wind_Direction']

        # get directional bin edges from Shear.by_sector output
        for i in range(self.sectors):
            bin_edges[i] = float(re.findall(r"[-+]?\d*\.\d+|\d+", direction_bins.index[i])[0])
            if i == self.sectors - 1:
                bin_edges[i + 1] = -float(re.findall(r"[-+]?\d*\.\d+|\d+", direction_bins.index[i])[1])

        for i in range(0, self.sectors):
            if bin_edges[i] > bin_edges[i + 1]:
                by_sector[i] = df[
                    (df['Wind_Direction'] >= bin_edges[i]) | (df['Wind_Direction'] < bin_edges[i + 1])]

            elif bin_edges[i + 1] == 360:
                by_sector[i] = df[(df['Wind_Direction'] >= bin_edges[i])]

            else:
                by_sector[i] = df[
                    (df['Wind_Direction'] >= bin_edges[i]) & (df['Wind_Direction'] < bin_edges[i + 1])]

            by_sector[i].columns = ['Unscaled_Wind_Speeds', 'Wind_Direction']

            if self.calc_method == 'power_law':
                scaled_wspds[i] = _scale(wspds=by_sector[i]['Unscaled_Wind_Speeds'], height=height,
                                         shear_to=shear_to,
                                         calc_method=self.calc_method, alpha=self.alpha[i])

            elif self.calc_method == 'log_law':
                scaled_wspds[i] = _scale(wspds=by_sector[i]['Unscaled_Wind_Speeds'], height=height,
                                         shear_to=shear_to,
                                         calc_method=self.calc_method,
                                         roughness_coefficient=self._roughness[i])

            if i == 0:
                result = scaled_wspds[i]
            else:
                result = pd.concat([result, scaled_wspds[i]], axis=0)

        result.sort_index(axis='index', inplace=True)

    if self.origin == 'Average':

        if wdir is not None:
            warnings.warn('Warning: Wind direction will not be accounted for when calculating scaled wind speeds.'
                          ' The shear exponents for this object were not calculated by sector. '
                          'Check the origin of the object using ".origin". ')

        if self.calc_method == 'power_law':
            result = _scale(wspds=wspds, height=height, shear_to=shear_to,
                            calc_method=self.calc_method, alpha=self.alpha)

        elif self.calc_method == 'log_law':
            result = _scale(wspds=wspds, height=height, shear_to=shear_to,
                            calc_method=self.calc_method, roughness_coefficient=self._roughness)

    new_name = wspds.name + '_scaled_to_' + str(shear_to) + 'm'
    result.rename(new_name, inplace=True)
    return result


def _fill_df_12x24(data):
    """
    Fills a pandas.DataFrame or Series to be a 12 month x 24 hour pandas.DataFrame by duplicating entries.
    Used for plotting TimeOfDay shear.

    :param data: pandas.DataFrame or Series to be turned into a 12x24 dataframe
    :type data: pandas.Series or pandas.DataFrame.
    :return: 12x24 pandas.DataFrame
    
    """
    # create copy for later use
    df_copy = data.copy()
    interval = int(24 / len(data))
    # set index for new data frame to deal with less than 24 sectors
    idx = pd.date_range('2017-01-01 00:00', '2017-01-01 23:00', freq='1H')

    # create new dataframe with 24 rows only interval number of unique values
    df = pd.DataFrame(index=pd.DatetimeIndex(idx).time, columns=df_copy.columns)
    df = pd.concat(
        [(df[df_copy.index[0].hour:]), (df[:df_copy.index[0].hour])],
        axis=0)

    for i in range(0, len(df_copy)):
        df.iloc[i * interval:(i + 1) * interval, :] = pd.DataFrame(df_copy.iloc[i, :]).values.T

    df.sort_index(inplace=True)
    df_copy.sort_index(inplace=True)

    if len(df.columns) == 1:
        df_12x24 = pd.DataFrame([[None for y in range(12)] for x in range(24)])
        df_12x24.index = df.index
        for i in range(12):
            df_12x24.iloc[:, i] = df.iloc[:, 0]
        df = df_12x24

    return df
