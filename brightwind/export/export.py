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

__all__ = ['export_tab_file','export_csv']


def export_tab_file(freq_tab, height, lat, long, file_name=None, file_path=None, dir_offset=0.0):
    """
    Export a WaSP tab file from freq_table() function

    :param freq_tab: Tab file
    :param height: Height of the device in meters
    :type height: float
    :param lat: Latitude of the site location
    :type lat: float
    :param long: Longitude of the site location
    :type long: float
    :param file_name: Name of the file or location
    :param file_name: str
    :param file_path: Path of destination folder, default is the current working folder
    :type file_path: str
    :param dir_offset: Direction offset, default 0.0
    :type dir_offset: float
    :return: Creates a windographer file

    **Example Usage**
    ::
        import brightwind as bw
        df = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)
        graph, tab = bw.freq_table(df.Spd40mN, df.Dir38mS, return_data=True)
        bw.export_tab_file(tab, 100, l0, 10, file_name='campbell_tab_file', file_path=r'C:\...\brightwind')

    """

    if file_name is None:
        file_name = str(datetime.datetime.now().strftime("%Y-%m-%d")) + '_brightwind_tab_export.tab'

    if file_path is None:
        file_path = os.getcwd()

    if '.tab' not in file_name:
        file_name = file_name + '.tab'

    file_name_print = os.path.splitext(file_name)[0]
    path_file = os.path.join(file_path, file_name)

    local_freq_tab = freq_tab.copy()
    lat = float(lat)
    long = float(long)
    speed_interval = {interval.right - interval.left for interval in local_freq_tab.index}

    if len(speed_interval) is not 1:
        import warnings
        warnings.warn("All speed bins not of equal lengths")

    speed_interval = speed_interval.pop()
    sectors = len(local_freq_tab.columns)
    freq_sum = local_freq_tab.sum(axis=0)
    local_freq_tab.index = [interval.right for interval in local_freq_tab.index]
    tab_string = str(file_name_print) + "\n" + "{:.2f}".format(lat) + " " + "{:.2f}".format(
        long) + " " + "{:.2f}".format(height) + "\n" \
                 + str(sectors) + " " + "{:.2f}".format(speed_interval) + " " + "{:.2f}".format(dir_offset) + "\n"
    tab_string += " ".join("{:.2f}".format(percent) for percent in freq_sum.values) + "\n"

    for column in local_freq_tab.columns:
        local_freq_tab[column] = (local_freq_tab[column] / sum(local_freq_tab[column])) * 1000.0

    tab_string += local_freq_tab.to_string(header=False, float_format='%.2f', na_rep=0.00)
    tab_string_strip = re.sub(' +', ' ', tab_string)

    with open(str(path_file), "w") as file:
        file.write(tab_string_strip)



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
    print('Export successful.')
