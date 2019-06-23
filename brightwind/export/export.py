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

import os
import pandas as pd
import datetime
import re

__all__ = ['export_tab_file', 'export_csv']


def _calc_mean_speed_of_freq_tab(freq_tab):
    """
    Calculate the mean wind speed derived from a frequency table.

    :param freq_tab: Frequency distribution by wind speed and direction sector. Data output from the bw.freq_table().
    :return:
    """
    local_freq_tab = freq_tab.copy()
    local_freq_tab.index = {(interval.right + interval.left)/2 for interval in local_freq_tab.index}
    sum_winds_for_all_sectors = local_freq_tab.sum(axis=1, skipna=True)
    mid_bin_wind_speed = local_freq_tab.index.to_series()
    mean_wind_speed = (sum(sum_winds_for_all_sectors * mid_bin_wind_speed)) / sum(sum_winds_for_all_sectors)
    return mean_wind_speed


def export_tab_file(freq_tab, height, lat, long, file_name=None, folder_path=None, dir_offset=0.0):
    """
    Export a WaSP tab file using the output from the freq_table() function.

    :param freq_tab: Frequency distribution by wind speed and direction sector. Data output from the bw.freq_table().
    :param height: Height that the freq table represents in meters.
    :type height: float
    :param lat: Latitude of the measurement location.
    :type lat: float
    :param long: Longitude of the measurement location.
    :type long: float
    :param file_name: The file name under which the tab file will be saved, or use the default,
           i.e '2019-06-07_brightwind_tab_export.tab'
    :type file_name: str
    :param folder_path: The directory where the tab file will be saved, default is the working directory.
    :type folder_path: str
    :param dir_offset: Direction offset, default 0.0.
    :type dir_offset: float
    :return: Creates a WAsP frequency tab file which can be used in the WAsP, WindFarmer and openWind software.

    **Example Usage**
    ::
        import brightwind as bw
        df = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)
        wind_rose, freq_tab = bw.freq_table(df.Spd80mN, df.Dir78mS, return_data=True)
        bw.export_tab_file(freq_tab, 80, 54.2, -7.6, file_name='campbell_tab_file', folder_path=r'C:\\some\\folder\\')

    """

    if file_name is None:
        file_name = str(datetime.datetime.now().strftime("%Y-%m-%d")) + '_brightwind_tab_export.tab'

    if folder_path is None:
        folder_path = os.getcwd()

    if '.tab' not in file_name:
        file_name = file_name + '.tab'

    file_name_print = os.path.splitext(file_name)[0]
    file_path = os.path.join(folder_path, file_name)

    lat = float(lat)
    long = float(long)

    local_freq_tab = freq_tab.copy()

    speed_interval = {interval.right - interval.left for interval in local_freq_tab.index}
    if len(speed_interval) is not 1:
        import warnings
        warnings.warn("All speed bins not of equal lengths")
    speed_interval = speed_interval.pop()

    local_freq_tab.index = [interval.right for interval in local_freq_tab.index]

    mean_wind_speed = _calc_mean_speed_of_freq_tab(freq_tab)
    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    version = '0.1.0'  # __bw.__version__ # to be changed for 'pip' install
    sectors = len(local_freq_tab.columns)

    tab_string = "{0} created using brightwind version {1} at {2}. Mean wind speed for tab file is: {3} m/s." \
                 "\n{4} {5} {6}\n " \
                 "{7} {8} {9}\n ".format(str(file_name_print), version, str(current_timestamp),
                                         "{:.3f}".format(mean_wind_speed),
                                         "{:.2f}".format(lat), "{:.2f}".format(long), "{:.2f}".format(height),
                                         str(sectors), "{:.2f}".format(speed_interval), "{:.2f}".format(dir_offset))

    sum_of_sectors = local_freq_tab.sum(axis=0)
    tab_string += " ".join("{:.2f}".format(sector_percent) for sector_percent in sum_of_sectors.values) + "\n"

    for column in local_freq_tab.columns:
        local_freq_tab[column] = (local_freq_tab[column] / sum(local_freq_tab[column])) * 1000.0

    tab_string += local_freq_tab.to_string(header=False, float_format='%.2f', na_rep=0.00)
    tab_string_strip = re.sub(' +', ' ', tab_string)

    with open(str(file_path), "w") as file:
        file.write(tab_string_strip)
    print('Export of tab file successful.')


def export_csv(data, file_name=None, folder_path=None, **kwargs):
    """
    Export a DataFrame, Series or Array to a .csv file or a .tab. The pandas.to_csv documentation can be found at
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html

    :param data: Dataframe, Series or Array
    :type data: panda.Dataframe or pandas.Series, array or list-like objects
    :param file_name: The file name under which the CSV will be saved, or use the default,
           i.e '2019-06-07_brightwind_csv_export.csv'
    :type file_name: str
    :param folder_path: The directory where the CSV will be saved, default is the working directory
    :type folder_path: str
    :param kwargs: All the kwargs that can be passed to pandas.to_csv.
    :return exports a .csv or .tab file

    **Example usage**
    ::
        import brightwind as bw
        df = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)
        folder = r'C:\\some\\folder\\'

        # to export a .csv file with a specified name to a specific folder
        bw.export_csv(df, file_name='brightwind_calculations', folder_path=folder)

        # to export a .csv file using default settings (to the working directory using a default name)
        bw.export_csv(df)

        # to export a .tab file
        bw.export_csv(df, file_name='file_exported.tab', sep='\t')

    """
    if file_name is None:
        file_name = str(datetime.datetime.now().strftime("%Y-%m-%d")) + '_brightwind_csv_export.csv'

    if "." not in file_name:
        file_name = file_name + '.csv'

    if folder_path is None:
        folder_path = os.getcwd()

    if os.path.isdir(folder_path):
        file_path = os.path.normpath(os.path.join(folder_path, file_name))
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data.to_csv(file_path, **kwargs)
        else:
            pd.DataFrame(data).to_csv(file_path, header=None, index=None, **kwargs)
    else:
        raise NotADirectoryError("The destination folder doesn't seem to exist.")
    print('Export to csv successful.')
