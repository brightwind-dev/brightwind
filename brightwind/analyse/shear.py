import pandas as pd
import numpy as np
from brightwind.transform import transform as tf
from brightwind.utils import utils
from brightwind.analyse import plot as plt
from brightwind.analyse.analyse import distribution_by_dir_sector, dist_12x24, coverage, _convert_df_to_series
import re
import warnings

__all__ = ['PowerLaw',
           'BySector',
           'by_sector',
           'power_law'
           ]


class PowerLaw:

    def __init__(self, plot=None, wspds=None, alpha=None, origin=None,
                 info=None):

        self.plot = plot
        self.wspds = wspds
        self.origin = origin
        self.alpha = alpha
        self.info = info

    def apply(self, wspds, height, height_to_scale_to):
        """"
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
                   anemometers = data[['Spd80mS', 'Spd60mS']]
                   heights = [80, 60]

                   # Calculate shear exponents
                   shear_object_power_law = bw.Shear.power_law(anemometers, heights, return_object=True)

                   # View attributes of Shear objects
                   # View exponents calculated
                     shear_object_power_law.alpha

                   # View plots
                   shear_object_power_law.plot

                   # View input data
                   shear_object_power_law.wspds

                   # View other information
                   shear_object_power_law.info

                   # Scale wind speeds using exponents
                   # Specify wind speeds to be scaled
                   wspds = bw.load_csv(r'mywindspeedtimeseries.csv')
                   height = 50
                   height_to_scale_to =70
                   shear_object_power_law=bw.Shear.apply( wspds, height, height_to_scale_to)

                   """
        return _apply(self, wspds, height, height_to_scale_to)


class BySector:

    def __init__(self, plot=None, wspds=None, wdir=None, sectors=None, alpha=None, origin=None,
                 info=None):

        self.plot = plot
        self.wspds = wspds
        self.wdir = wdir
        self.sectors = sectors
        self.origin = origin
        self.alpha = alpha
        self.info = info

    def apply(self, wspds, height, height_to_scale_to, wdir):
        """"
                :param self: Shear object to use when applying shear to the data
                :type self: Shear object
                :param wspds: Wind speed time series to apply shear to
                :type wspds: Pandas Series
                :param height: height of above wspds
                :type height: float
                :param height_to_scale_to: height to which wspds should be scaled to
                :type height_to_scale_to: float
                :param wdir: wind direction measurements of wspds, only required if shear is to be applied by direction sector.
                :type wdir: Pandas Series
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
                    shear_object_by_sector = bw.Shear.by_sector(anemometers, heights, directions, return_object=True)

                    # View attributes of Shear objects
                    # View exponents calculated
                    shear_object_by_sector.alpha

                    # View plots
                    shear_object_by_sector.plot

                    # View input data
                    shear_object_by_sector.wspds
                    shear_object_by_sector.wdir

                    # View other information
                    shear_object_by_sector.info

                    # Scale wind speeds using exponents
                    # Specify wind speeds to be scaled
                    wspds = bw.load_csv(r'mywindspeedtimeseries.csv')
                    wdir = bw.load_csv(r'mywinddirectiontimeseries.csv')
                    height = 50
                    height_to_scale_to =70
                    """

        return _apply(self, wspds, height, height_to_scale_to, wdir=wdir)


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


def power_law(wspds, heights, min_speed=3, return_object=False):
    """
    Calculates shear based on power law

    :param wspds: DataFrame or list of wind speeds for calculating shear
    :type wspds: list of pandas.DataFrame
    :param heights: List of anemometer heights
    :type heights: list
    :param min_speed: Only speeds higher than this would be considered for calculating shear, default is 3
    :type min_speed: float
    :param return_object: Return a Shear object containing shear exponents calculated from supplied data, input data and plit
    :type return_object: bool
    :return: A plot with shear plotted along with the means. Also returns a Shear object containing shear exponents and other data
    hen return_object is True.

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.datasets.demo_data)

        #Using with a DataFrame of wind speeds
        graph, alpha = bw.Shear.power_law(data[['Spd80mN', 'Spd60mN', 'Spd40mN']], heights = [80, 60, 40], return_alpha=True)

        #List of wind speeds
        graph, alpha = bw.Shear.power_law([1, 8, 4], heights = [80, 60, 40], return_alpha=True)

        #To change minimum wind speed to filter and not return alpha
        pow_law = bw.Shear.power_law([1, 8, 4], heights = [80, 60, 40], min_speed=5)

    """
    if not isinstance(wspds, pd.DataFrame):
        wspds = pd.DataFrame(wspds).T
    wspds = wspds.dropna()
    cvg = coverage(wspds[wspds > min_speed].dropna(), period='1AS').sum()[1]
    mean_wspds = wspds[(wspds > min_speed).all(axis=1)].mean().dropna()
    if mean_wspds.shape[0] == 0:
        raise ValueError('None of the input wind speeds are greater than the min_speed, cannot calculate shear')
    alpha, c = _calc_power_law(mean_wspds.values, heights, return_coeff=True)
    info = {}
    input_wind_speeds = {'heights(m)': [heights], 'column_names': [list(wspds.columns.values)],
                         'min_spd(m/s)': str(min_speed)}
    info['input_wind_speeds'] = input_wind_speeds
    info['concurrent_period(years)'] = str("{:.3f}".format(cvg))

    if return_object:
        shear_object = PowerLaw(plot=(plt.plot_shear(alpha, c, mean_wspds.values, heights)), wspds=wspds,
                                 origin='by_power_law', alpha=alpha, info=info)
        return shear_object
    return plt.plot_shear(alpha, c, mean_wspds.values, heights)


def by_sector(wspds, heights, wdir, sectors=12, min_speed=3, direction_bin_array=None, direction_bin_labels=None,
              return_object=False):
    """
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
    :param return_object: if True returns a Shear Object, if False returns a plot
    :type return_object: boolean
    :return: returns a shear object containing a plot, all inputted data and a series of calculated shear
    exponents if True, returns a plot if False
    """
    if direction_bin_array is not None:
        sectors = len(direction_bin_array)-1

    common_idxs = wspds.index.intersection(wdir.index)
    wdir = _convert_df_to_series(wdir)
    shear = wspds[(wspds > min_speed).all(axis=1)].apply(_calc_power_law, heights=heights, axis=1)
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

    if return_object:
        shear_object = BySector(plot=(plt.plot_shear_by_sector(shear, wdir.loc[shear.index.intersection(wdir.index)],
                                                            shear_dist)), wspds=wspds, wdir=wdir,
                             sectors=sectors, alpha=shear_dist['Mean_Shear'], origin='by_sector', info=info)
        return shear_object
    else:
        return plt.plot_shear_by_sector(shear, wdir.loc[shear.index.intersection(wdir.index)], shear_dist)


def _by_12x24(wspds, heights, min_speed=3, return_data=False, var_name='Shear'):
    tab_12x24 = dist_12x24(wspds[(wspds > min_speed).all(axis=1)].apply(_calc_power_law, heights=heights,
                                                                        axis=1), return_data=True)[1]
    if return_data:
        return plt.plot_12x24_contours(tab_12x24, label=(var_name, 'mean')), tab_12x24
    return plt.plot_12x24_contours(tab_12x24,  label=(var_name, 'mean'))


def _scale(alpha, wspd, height, height_to_scale_to):
    scale_factor = (height_to_scale_to / height) ** alpha
    return wspd * scale_factor


def _apply(self, wspds, height, height_to_scale_to, wdir=None):

        """"
        :param self: Shear object to use when applying shear to the data
        :type self: Shear object
        :param wspds: Wind speed time series to apply shear to
        :type wspds: Pandas Series
        :param height: height of above wspds
        :type height: float
        :param height_to_scale_to: height to which wspds should be scaled to
        :type height_to_scale_to: float
        :param wdir: wind direction measurements of wspds, only required if shear is to be applied by direction sector.
        :type wdir: Pandas Series
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
            shear_object_power_law = bw.Shear.power_law(anemometers, heights, return_object=True)
            shear_object_by_sector = bw.Shear.by_sector(anemometers, heights, directions, return_object=True)

            # View attributes of Shear objects
            # View exponents calculated
              shear_object_by_sector.alpha
              shear_object_power_law.alpha

            # View plots
            shear_object_by_sector.plot
            shear_object_power_law.plot

            # View input data
            shear_object_by_sector.wspds
            shear_object_by_sector.wdir
            shear_object_power_law.wspds

            # View other information
            shear_object_by_sector.info
            shear_object_power_law.info

            # Scale wind speeds using exponents
            # Specify wind speeds to be scaled
            wspds = bw.load_csv(r'mywindspeedtimeseries.csv')
            wdir = bw.load_csv(r'mywinddirectiontimeseries.csv')
            height = 50
            height_to_scale_to =70
            shear_object_by_sector.apply(wspds, height, height_to_scale_to, wdir)
            shear_object_power_law=bw.Shear.apply( wspds, height, height_to_scale_to)

            """
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
                alpha_bounds[i] = int(re.findall(r'\d+', self.alpha.index[i])[0])
                if i == self.sectors - 1:
                    alpha_bounds[i + 1] = int(re.findall(r'\d+', self.alpha.index[i])[1])

            #
            for i in range(0, self.sectors):
                if i == 0:
                    by_sector[i] = df[
                        (df['Wind_Direction'] >= alpha_bounds[i]) | (df['Wind_Direction'] < alpha_bounds[i + 1])]

                else:
                    by_sector[i] = df[
                        (df['Wind_Direction'] >= alpha_bounds[i]) & (df['Wind_Direction'] < alpha_bounds[i + 1])]

                by_sector[i].columns = ['Unscaled_Wind_Speeds', 'Wind_Direction']
                scaled_wspds[i] = _scale(self.alpha[i], by_sector[i]['Unscaled_Wind_Speeds'], height,
                                              height_to_scale_to)
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

        if self.origin == 'by_power_law':

            if wdir is not None:
                warnings.warn('Warning: Wind direction will not be accounted for when calculating scaled wind speeds.'
                              ' The shear exponents for this object were not calculated by sector. '
                              'Check the origin of the object using ".origin". ')

            scaled_wspds = _scale(self.alpha, wspds, height, height_to_scale_to)
            alpha = pd.DataFrame([self.alpha]*len(wspds))
            alpha.index = wspds.index
            result = pd.concat([wspds, scaled_wspds, alpha], axis=1)
            result.columns = ['Unscaled_Wind_Speeds' + '(' + str(height) + 'm)',
                              'Scaled_Wind_Speeds' + '(' + str(height_to_scale_to) + 'm)', 'Shear_Exponent']

            result['Unscaled_Wind_Speeds' + '(' + str(height) + 'm)'] = result['Unscaled_Wind_Speeds' + '(' +
                                                                               str(height) + 'm)'].map('{:,.3f}'.format)
            result['Scaled_Wind_Speeds' + '(' + str(height_to_scale_to) + 'm)'] = \
                result['Scaled_Wind_Speeds' + '(' + str(height_to_scale_to) + 'm)'].map('{:,.3f}'.format)
            result['Shear_Exponent'] = result['Shear_Exponent'].map('{:,.3f}'.format)

            result.sort_index(axis='index', inplace=True)

        return result

if __name__ == '__main__':
    import brightwind as bw
    data = bw.load_csv(bw.datasets.demo_data)
    anemometers = data[['Spd80mS', 'Spd60mS']]
    heights = [80, 60]
    directions = data[['Dir78mS']]
    sectors = 12
    bin_array = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 330]
    object_shear = bw.Shear.Shear.by_sector(anemometers, heights, directions,
                                            return_object=True)
    object_shear.apply(data['Spd80mS'], 60, 80, directions)