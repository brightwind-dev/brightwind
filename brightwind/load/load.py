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
import requests
from typing import List
import errno
import os
import shutil
import json
from io import StringIO
import warnings
from dateutil.parser import parse
from brightwind.analyse import plot as plt


__all__ = ['load_csv',
           'load_campbell_scientific',
           'load_windographer_txt',
           'load_excel',
           'load_brightdata',
           'load_cleaning_file',
           'apply_cleaning',
           'apply_cleaning_windographer']


def _list_files(folder_path, file_type):
    """
    Return a list of file names retrieved from a folder filtering for a specific list of file types. This will walk
    through all sub-folders.

    :param folder_path: The path to the folder to search through.
    :type folder_path: str
    :param file_type: Is a list of file extensions to filter for e.g. ['.csv', '.txt']
    :type file_type: List[str]
    :return: List of file names with the full folder path.
    :rtype: List[str]

    """
    files_list: List[str] = []
    for root, dirs, files in os.walk(folder_path, topdown=True):
        for filename in files:
            extension = os.path.splitext(filename)[1]
            if extension in file_type:
                files_list.append(os.path.join(root, filename))
    if not files_list:
        if not os.path.isdir(folder_path):
            raise NotADirectoryError('Not valid folder.')
    return files_list


def _assemble_df_from_folder(source_folder, file_type, function_to_get_df, print_progress=False, **kwargs):
    """
    Assemble a DataFrame from from multiple data files scattered in subfolders filtering for a
    specific list of file types and reading those files with a specific function.

    :param source_folder: Is the main folder to search through.
    :type source_folder: str
    :param file_type: Is a list of file extensions to filter for e.g. ['.csv', '.txt']
    :type file_type: List[str]
    :param function_to_get_df: The function to call to read each data file into a DataFrame.
    :type function_to_get_df: python function
    :param print_progress: If you want print out statements of the files been processed set to true. Default is False.
    :type print_progress: bool, default False
    :param kwargs: All the kwargs that can be passed to this function.
    :return: A DataFrame with timestamps as it's index
    :rtype: pandas.DataFrame
    """
    files_list = _list_files(source_folder, file_type)
    ctr = 0
    assembled_df = pd.DataFrame()
    for file_name in files_list:
        df = function_to_get_df(file_name, **kwargs)
        assembled_df = assembled_df.append(df, verify_integrity=True)
        if print_progress:
            print("{0} file read and appended".format(file_name))
        ctr = ctr + 1
    if print_progress:
        print('Processed {0} files'.format(str(ctr)))
    return assembled_df.sort_index()


def _is_file(file_or_folder):
    """
    Returns True is file_or_folder is a file.
    :param file_or_folder: The file or folder path.
    :type file_or_folder: str
    :return: True if a file.
    """
    if os.path.isfile(file_or_folder):
        return True
    elif os.path.isdir(file_or_folder):
        return False
    else:
        raise FileNotFoundError("File or folder doesn't seem to exist.")


def _pandas_read_csv(filepath, **kwargs):
    """
    Wrapper function around the Pandas read_csv function.
    :param filepath: The file to read.
    :type filepath: str, StringIO
    :param kwargs: Extra key word arguments to be applied.
    :return: A pandas DataFrame.
    :rtype: pandas.DataFrame
    """
    try:
        return pd.read_csv(filepath, **kwargs)
    except FileNotFoundError:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)
    except Exception as error:
        raise error


def load_csv(filepath_or_folder, search_by_file_type=['.csv'], print_progress=True, **kwargs):
    """
    Load timeseries data from a csv file, or group of files in a folder, into a DataFrame.
    The format of the csv file should be column headings in the first row with the timestamp column as the first
    column, however these can be over written by sending your own arguments as this is a wrapper around the
    pandas.read_csv function. The pandas.read_csv documentation can be found at:
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

    :param filepath_or_folder: Location of the file folder containing the timeseries data.
    :type filepath_or_folder: str
    :param search_by_file_type: Is a list of file extensions to search for e.g. ['.csv', '.txt'] if a folder is sent.
    :type search_by_file_type: List[str], default ['.csv']
    :param print_progress: If you want to print out statements of the file been processed set to True. Default is True.
    :type print_progress: bool, default True
    :param kwargs: All the kwargs from pandas.read_csv can be passed to this function.
    :return: A DataFrame with timestamps as it's index.
    :rtype: pandas.DataFrame

    When assembling files from folders into a single DataFrame with timestamp as the index it automatically checks for
    duplicates and throws an error if any found.

    **Example usage**
    ::
        import brightwind as bw
        filepath = r'C:\\some\\folder\\some_data2.csv'
        df = bw.load_csv(filepath)
        print(df)

    To load a group of files from a folder other than a .csv file type::

        folder = r'C:\\some\\folder\\with\\txt\\files'
        df = bw.load_csv(folder, search_by_file_type=['.txt'], print_progress=True)

    If you want to load something that is different from a standard file where the column headings are not in the first
    row, the pandas.read_csv key word arguments (kwargs) can be used::

        filepath = r'C:\\some\\folder\\some_data_with_column_headings_on_second_line.csv'
        df = bw.load_csv(filepath, skiprows=0)
    """

    is_file = _is_file(filepath_or_folder)
    fn_arguments = {'header': 0, 'index_col': 0, 'parse_dates': True}
    merged_fn_args = {**fn_arguments, **kwargs}
    if is_file:
        return _pandas_read_csv(filepath_or_folder, **merged_fn_args)
    elif not is_file:
        return _assemble_df_from_folder(filepath_or_folder, search_by_file_type, _pandas_read_csv, print_progress,
                                        **merged_fn_args)


def load_windographer_txt(filepath, delimiter='tab', flag_text=9999, **kwargs):
    """
    Load a Windographer .txt data file exported fom the Windographer software into a DataFrame.

    - If flagged data was filtered out during the export from Windographer these can be replaced to work with Pandas.
    - If delimiter other than 'tab' is used during export you can specify 'comma', 'space' or user specific.
    - Once the file has been loaded into the DataFrame if the last column name contains 'Unnamed' it is removed. This is
      due to Windographer inserting an extra delimiter at the end of the column headings.
    - The function finds the line number of 'Date/Time' to know when the data starts. It ignores the header.

    :param filepath: Location of the file containing the Windographer timeseries data.
    :type filepath: str
    :param delimiter: Column delimiter or separator used to export the data from Windographer. These can be 'tab',
                      'comma', 'space' or user specified.
    :type delimiter: str, default 'tab'
    :param flag_text: This is the 'missing data point' text used during export if flagged data was filtered.
    :type flag_text: str, float
    :param kwargs: All the kwargs from pandas.read_csv can be passed to this function.
    :return: A DataFrame with timestamps as it's index.
    :rtype: pandas.DataFrame

    **Example usage**
    ::
        import brightwind as bw
        filepath = r'C:\\some\\folder\\brightwind\\datasets\\demo\\windographer_demo_site_data.txt'
        df = bw.load_windographer_txt(filepath)
        print(df)

    To load a file with delimiter and flagged text other than defaults::

        folder = r'C:\\some\\folder\\some_windographer.txt'
        df = bw.load_windographer_txt(filepath, delimiter=';', flag_text='***')

    """

    is_file = _is_file(filepath)
    if is_file:
        # Need to replace the flag text before loading into the DataFrame as this text could be a string or a number
        # and Pandas will throw and warning msg if data types in a column are mixed setting the column as string.
        with open(filepath, 'r') as file:
            file_contents = file.read().replace(str(flag_text), '')
        if 'Windographer' not in file_contents:
            warnings.warn("\nFile doesn't seem to be a Windographer file. This may load the data unexpectedly.",
                          Warning)
        number_of_header_rows_to_skip = 12
        for index, line in enumerate(file_contents.split('\n')):
            if 'Date/Time' in line:
                number_of_header_rows_to_skip = index
                break
        separators = [
            {'delimiter': 'tab', 'fn_argument': '\t'},
            {'delimiter': 'comma', 'fn_argument': ','},
            {'delimiter': 'space', 'fn_argument': ' '},
            {'delimiter': delimiter, 'fn_argument': delimiter}
        ]
        for separator in separators:
            if delimiter == separator['delimiter']:
                delimiter = separator['fn_argument']
        fn_arguments = {'skiprows': number_of_header_rows_to_skip, 'delimiter': delimiter,
                        'header': 0, 'index_col': 0, 'parse_dates': True}
        merged_fn_args = {**fn_arguments, **kwargs}
        df = _pandas_read_csv(StringIO(file_contents), **merged_fn_args)
        if len(df.columns) > 0 and 'Unnamed' in df.columns[-1]:
            df.drop(df.columns[-1], axis=1, inplace=True)
        return df
    elif not is_file:
        raise FileNotFoundError("File path seems to be a folder. Please load a single Windographer .txt data file.")


def load_campbell_scientific(filepath_or_folder, print_progress=True, **kwargs):
    """
    Load timeseries data from Campbell Scientific CR1000 formatted file, or group of files in a folder, into a
    DataFrame. If the file format is slightly different your own key word arguments can be sent as this is a wrapper
    around the pandas.read_csv function. The pandas.read_csv documentation can be found at:
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

    :param filepath_or_folder: Location of the file folder containing the timeseries data.
    :type filepath_or_folder: str
    :param print_progress: If you want to print out statements of the file been processed set to True. Default is True.
    :type print_progress: bool, default True
    :param kwargs: All the kwargs from pandas.read_csv can be passed to this function.
    :return: A DataFrame with timestamps as it's index
    :rtype: pandas.DataFrame

    When assembling files from folders into a single DataFrame with timestamp as the index it automatically checks for
    duplicates and throws an error if any found.

    **Example usage**
    ::
        import brightwind as bw
        filepath = r'C:\\some\\folder\\some_CR1000_data.csv'
        df = bw.load_campbell_scientific(filepath)
        print(df)

    To load a group of files from a folder::

        folder = r'C:\\some\\folder\\with\\CR1000\\files'
        df = bw.load_campbell_scientific(folder, print_progress=True)
    """

    is_file = _is_file(filepath_or_folder)
    fn_arguments = {'header': 0, 'index_col': 0, 'parse_dates': True, 'skiprows': [0, 2, 3]}
    merged_fn_args = {**fn_arguments, **kwargs}
    if is_file:
        return _pandas_read_csv(filepath_or_folder, **merged_fn_args)
    elif not is_file:
        return _assemble_df_from_folder(filepath_or_folder, ['.dat', '.csv'], _pandas_read_csv, print_progress,
                                        **merged_fn_args)


def _pandas_read_excel(filepath, **kwargs):
    """
    Wrapper function around the Pandas read_excel function.
    :param filepath: The file to read.
    :type filepath: str
    :param kwargs: Extra key word arguments to be applied.
    :return: A pandas DataFrame
    :rtype: pandas.DataFrame
    """
    try:
        return pd.read_excel(filepath, **kwargs)
    except FileNotFoundError:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)
    except Exception as error:
        raise error


def load_excel(filepath_or_folder, search_by_file_type=['.xlsx'], print_progress=True, sheet_name=0, **kwargs):
    """
    Load timeseries data from an Excel file, or group of files in a folder, into a DataFrame.
    The format of the Excel file should be column headings in the first row with the timestamp column as the first
    column, however these can be over written by sending your own arguments as this is a wrapper around the
    pandas.read_excel function. The pandas.read_excel documentation can be found at:
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_excel.html

    :param filepath_or_folder: Location of the file folder containing the timeseries data.
    :type filepath_or_folder: str
    :param search_by_file_type: Is a list of file extensions to search for e.g. ['.xlsx'] if a folder is sent.
    :type search_by_file_type: List[str], default .xlsx
    :param print_progress: If you want to print out statements of the file been processed set to True. Default is True.
    :type print_progress: bool, default True
    :param sheet_name: The Excel file sheet name you want to read from.
    :type sheet_name: string, int, mixed list of strings/ints, or None, default 0
    :param kwargs: All the kwargs from pandas.read_excel can be passed to this function.
    :return: A DataFrame with timestamps as it's index.
    :rtype: pandas.DataFrame

    When assembling files from folders into a single DataFrame with timestamp as the index it automatically checks for
    duplicates and throws an error if any found.

    **Example usage**
    ::
        import brightwind as bw
        filepath = r'C:\\some\\folder\\some_data.xlsx'
        df = bw.load_excel(filepath)
        print(df)

    To load a group of files from a folder other than a .csv file type::

        folder = r'C:\\some\\folder\\with\\excel\\files'
        df = bw.load_excel(folder, print_progress=True)

    If you want to load something that is different from a standard file where the column headings are not in the first
    row, the pandas.read_excel key word arguments (kwargs) can be used::

        filepath = r'C:\\some\\folder\\some_data_with_column_headings_on_second_line.xlsx'
        df = bw.load_excel(filepath, skiprows=0)
    """

    is_file = _is_file(filepath_or_folder)
    fn_arguments = {'index_col': 0, 'parse_dates': True, 'sheet_name': sheet_name}
    merged_fn_args = {**fn_arguments, **kwargs}
    if is_file:
        return _pandas_read_excel(filepath_or_folder, **merged_fn_args)
    elif not is_file:
        return _assemble_df_from_folder(filepath_or_folder, search_by_file_type, _pandas_read_excel, print_progress,
                                        **merged_fn_args)


def load_nrg_txt():
    return 'not yet implemented'


def _assemble_files_to_folder(source_folder, destination_folder, file_type, print_filename=False):
    """
    Assemble files scattered in subfolders of a certain directory and copy them to a single folder filtering for a
    specific list of file types. If there are duplicates, the largest file will be kept.

    :param source_folder: Is the main folder to search through.
    :type source_folder: str
    :param destination_folder: Is where you want all the files found to be copied to. If there are duplicates, the
           largest file will be kept.
    :type destination_folder: str
    :param file_type: Is a list of file extensions to filter for e.g. ['.csv', '.txt']
    :type file_type: List[str]
    :param print_filename: If you want all the file names found to be printed set to true. Default is False.
    :type print_filename: bool, default False
    :return: None

    """
    files_list = _list_files(source_folder, file_type)
    x = 0
    for file in files_list:
        filename = os.path.split(file)[1]
        filepath = os.path.split(file)[0]
        new_file = os.path.join(destination_folder, filename)
        if print_filename:
            print(new_file)
        if filepath == destination_folder:
            if print_filename:
                print('File to be moved is itself. Skipping.')
        elif os.path.exists(new_file):
            new_file_size = os.path.getsize(new_file)
            current_file_size = os.path.getsize(file)
            if new_file_size <= current_file_size:
                if print_filename:
                    print('File from source_folder is greater size than an existing one in destination_folder. File '
                          'in destination_folder will be overwritten.')
                os.remove(new_file)
                shutil.copyfile(file, new_file)
        else:
            try:
                shutil.copyfile(file, new_file)
            except FileNotFoundError:
                if not os.path.isdir(destination_folder):
                    raise NotADirectoryError('Destination folder is not valid folder.')
            except Exception as error:
                raise error
            x = x + 1
    if print_filename:
        print('Number of files processed: ' + str(len(files_list)) + '. Number of files moved: ' + str(x))


def _append_files_together(source_folder, assembled_file_name, file_type, append_first_line=True):
    """
    Assemble files scattered in subfolders of a certain directory and copy them to a single file filtering for a
    specific list of file types.

    :param source_folder: Is the main folder to search through.
    :type source_folder: str
    :param assembled_file_name: Name of the newly created file with all the appended data.
    :type assembled_file_name: str
    :param file_type: Is a list of file extensions to filter for e.g. ['.csv', '.txt']
    :type file_type: List[str]
    :param append_first_line: Append the first line (usually the column names) after the first file to the
                              assembled file.
    :type append_first_line: bool
    :return:
    """
    list_of_files = _list_files(source_folder, file_type)

    file_handler = open(os.path.join(source_folder, assembled_file_name), 'a+')
    for file_number, file in enumerate(list_of_files):
        file_handler2 = open(file, 'r')
        if file_number == 0:
            file_handler.write(file_handler2.read())
        if file_number > 0:
            if append_first_line:
                file_handler.write(file_handler2.read())
            else:
                file_handler.writelines(file_handler2.readlines()[1:])
        file_handler2.close()
    file_handler.close()
    return


class Reanalysis:
    """Object defining a Reanalysis dataset

    :param latitude: object.latitude gives the latitude of the location of dataset
    :type latitude: str
    :param longitude: object.longitude gives the longitude of the location of dataset
    :type longitude: str
    :param data: object.data returns the data from the location
    :type data: pandas.DataFrame
    :param source: Source of the dataset like MERA, MERRA2, etc.
    :type source: str

    """
    def __init__(self, latitude, longitude, data, source):
        self.latitude = latitude
        self.longitude = longitude
        self.data = data
        self.source = source


def _get_brightdata_credentials():
    if 'BRIGHTDATA_USERNAME' not in os.environ:
        raise Exception('BRIGHTDATA_USERNAME environmental variable is not set.')
    if 'BRIGHTDATA_PASSWORD' not in os.environ:
        raise Exception('BRIGHTDATA_PASSWORD environmental variable is not set.')
    return os.getenv('BRIGHTDATA_USERNAME'), os.getenv('BRIGHTDATA_PASSWORD')


def _get_environment_variable(name):
    if name not in os.environ:
        raise Exception('{} environmental variable is not set.'.format(name))
    return os.getenv(name)


def _get_brightdata(dataset, lat, long, nearest, from_date, to_date):
    """
    Get merra2 or era5 data from the brightdata platform and format it for use.
    :param lat:
    :param long:
    :param nearest:
    :param from_date:
    :param to_date:
    :return:
    """
    username, password = _get_brightdata_credentials()
    base_url = 'http://api.brightwindanalysis.com/brightdata'
    # base_url = 'http://localhost:5000'
    response = requests.get(base_url,  auth=(username, password), params={
        'dataset': dataset,
        'latitude': lat,
        'longitude': long,
        'nearest': nearest,
        'from_date': from_date,
        'to_date': to_date
    })
    try:
        json_response = response.json()
    except Exception:
        if response.status_code == 401:
            raise Exception('Please check your BRIGHTDATA_USERNAME and BRIGHTDATA_PASSWORD are correct.')
        raise Exception('Http code {}, something is wrong with the server.'.format(str(response.status_code)))
    reanalysis_list = []

    for node in json_response:
        temp_reanalysis = Reanalysis('', '', pd.DataFrame(), '')
        try:
            temp_reanalysis.latitude = node['latitude']
            temp_reanalysis.longitude = node['longitude']
            temp_reanalysis.source = dataset
            temp_reanalysis.data = pd.read_json(json.dumps(node['data']), orient='index')
        except Exception as error:
            if 'errors' in node:
                raise TypeError(json_response)
            else:
                raise error
        reanalysis_list.append(temp_reanalysis)

    return reanalysis_list


def load_brightdata(dataset, lat, long, nearest, from_date=None, to_date=None):
    """
    Retrieve timeseries datasets available from the brightdata platform. Returns a list of Reanalysis objects in order
    of closest distance to the requested lat, long.

    :param dataset: dataset type to be retrieved from brightdata e.g. merra2, era5.
    :type dataset: str
    :param lat: latitude of your point of interest.
    :type lat: float
    :param long: longitude of your point of interest.
    :type long: float
    :param nearest: the number of nearest nodes to your point of interest to retrieve. Currently only 1 to 4 is
                    accepted.
    :type nearest: int
    :param from_date: date from in 'yyyy-mm-dd' format.
    :type from_date: str
    :param to_date: date to in 'yyyy-mm-dd' format.
    :type to_date: str
    :return: a list of Reanalysis objects in order of closest distance to the requested lat, long.
    :rtype: List(Reanalysis)

    To use load_brightdata the BRIGHTDATA_USERNAME and BRIGHTDATA_PASSWORD environmental variables need to be set. In
    Windows this can be done by running the command prompt in Administrator mode and running:

    >>> setx BRIGHTDATA_USERNAME "username"
    >>> setx BRIGHTDATA_PASSWORD "password"

    **Example usage**
    ::
        import brightwind as bw
        nodes = bw.load_brightdata('era5', 53.4, -7.2, 4, '2018-01-01', '2018-10-01')
        for node in nodes:
            print(node.data)

    """

    handlers = [
        {'dataset': 'era5', 'process_fn': _get_brightdata, 'fn_arguments': {
            'dataset': dataset, 'lat': lat, 'long': long, 'from_date': from_date, 'to_date': to_date, 'nearest': nearest
            }
         },
        {'dataset': 'merra2', 'process_fn': _get_brightdata, 'fn_arguments': {
            'dataset': dataset, 'lat': lat, 'long': long, 'from_date': from_date, 'to_date': to_date, 'nearest': nearest
            }
         }
    ]
    for handler in handlers:
        if dataset == handler['dataset']:
            try:
                return handler['process_fn'](**handler['fn_arguments'])
            except Exception as error:
                raise error

    raise NotImplementedError('dataset not identified.')


class _LoadBWPlatform:

    _base_url = 'https://api.brightwindanalysis.com/platform'

    @staticmethod
    def _get_token():
        username = _get_environment_variable('BW_PLATFORM_USERNAME')
        password = _get_environment_variable('BW_PLATFORM_PASSWORD')

        params = {'username': username, 'password': password}
        json_response = requests.post('https://api.brightwindanalysis.com/auth/login', json=params).json()

        if json_response.get('error_description'):
            raise ValueError(json_response['error_description'])
        return json_response['access_token']

    @staticmethod
    def get_plants():
        """
        Get all the wind or solar plants you have access to. A list of dictionaries of all your plants are returned.
        Format of dictionary is:

        {
            'alias': None,
            'connection_details': None,
            'country': 'Ireland',
            'id': '78g2j9b2-70fb-425d-b0d9-33c26e94bd4e',
            'is_location_verified': True,
            'is_operational': False,
            'latitude': -8,
            'longitude': 54,
            'mec_mw': None,
            'name': 'wind farm name',
            'notes': None,
            'operator_uuid': None,
            'plant_type': 'wind',
            'region': None,
            'specifications': None,
            'trader_uuid': None
        }

        :return:
        """
        access_token = _LoadBWPlatform._get_token()
        headers = {'Authorization': 'Bearer ' + access_token}
        response = requests.get(_LoadBWPlatform._base_url + '/api/plants', headers=headers)
        if response.headers.get('content-type') != 'application/json.':
            response.raise_for_status()

        response_json = response.json()
        if 'Error' in response_json:    # catch if error comes back e.g. measurement_location_uuid isn't found
            raise ValueError(response_json['Error'])
        return response_json

    @staticmethod
    def get_meas_locs():
        """
        Get all the measurement locations you have access to. A list of dictionaries of all your sites are returned.
        Format of dictionary is:

        {
            'notes': None,
            'longitude': 54,
            'id': '55a8b5b2-70fb-415d-b0d9-33c26e94bd9e',
            'measurement_station_type': 'mast',
            'plant_uuid': '78g2j9b2-70fb-425d-b0d9-33c26e94bd4e',
            'name': 'Mast name',
            'latitude': -8
        }

        :return: A list of all the measurement locations you have access to.
        :rtype: List(Dict())
        """
        access_token = _LoadBWPlatform._get_token()
        headers = {'Authorization': 'Bearer ' + access_token}
        response = requests.get(_LoadBWPlatform._base_url + '/api/measurement-locations', headers=headers)

        response_json = response.json()
        if 'Error' in response_json:    # catch if error comes back e.g. measurement_location_uuid isn't found
            raise ValueError(response_json['Error'])
        return response_json

    @staticmethod
    def get_meas_points(meas_loc_uuid):
        """
        Get measurement points for a particular measurement location uuid. Return is a list of dictionaries.
        Format of dictionary is:

        {
            'id': '071d559a-8096-47bd-91cf-f7f9137a6689',
            'measurement_location_uuid': 'e927041f-8736-4fa3-9471-f806497633d5',
            'measurement_type': 'wind speed',
            'mounting_arrangement': {'boom_orientation_deg': 15,
                                     'height_metres': 100.125},
            'name': 'Spd1_100m15',
            'notes': None
        }

        :param meas_loc_uuid:
        :return:
        """
        access_token = _LoadBWPlatform._get_token()
        headers = {'Authorization': 'Bearer ' + access_token}
        response = requests.get(_LoadBWPlatform._base_url + '/api/measurement-points', headers=headers, params={
            'measurement_location_uuid': meas_loc_uuid
        })
        response_json = response.json()
        if 'Error' in response_json:    # catch if error comes back e.g. measurement_location_uuid isn't found
            raise ValueError(response_json['Error'])
        return response_json

    @staticmethod
    def get_sensor_configs(meas_loc_uuid):
        """
        Get all the sensor configurations for a certain measurement location uuid.

        {
            'calibration': None,
            'column_names': {'An1_100_315;wind_speed;Avg': {'is_ignored': False, 'metric': 'Avg'},
                             'An1_100_315;wind_speed;Count': {'is_ignored': False, 'metric': 'Count'},
                             'An1_100_315;wind_speed;Max': {'is_ignored': False, 'metric': 'Max'},
                             'An1_100_315;wind_speed;Min': {'is_ignored': False, 'metric': 'Min'},
                             'An1_100_315;wind_speed;StdDev': {'is_ignored': False, 'metric': 'StdDev'}},
            'date_from': '2017-07-26T00:00:00+00:00',
            'date_to': None,
            'desired_adj': None,
            'id': '2d6e5057-8319-4326-aa4e-1b5a753ae0a6',
            'logger_config': {'logger_offset': 0.2575,
                              'logger_slope': 0.04598,
                              'logger_stated_height': 100,
                              'measurement_units': 'm/s'},
            'logger_main_config_uuid': 'ed62d180-dac4-4615-820f-05313bb8ffff',
            'measurement_point_uuid': '071d559a-8096-47bd-91cf-f7f9137a6689',
            'notes': None,
            'sensor_info': {'sensor_model': 'Thies Anemometer First Class Advanced',
                            'sensor_type': 'anemometer'},
            'sensor_name': 'An1_100_315;wind_speed'
        }

        :param meas_loc_uuid:
        :return:
        """
        access_token = _LoadBWPlatform._get_token()
        headers = {'Authorization': 'Bearer ' + access_token}
        response = requests.get(_LoadBWPlatform._base_url + '/api/sensor-configs', headers=headers, params={
            'measurement_location_uuid': meas_loc_uuid
        })
        response_json = response.json()
        if 'Error' in response_json:    # catch if error comes back e.g. measurement_location_uuid isn't found
            raise ValueError(response_json['Error'])
        return response_json

    @staticmethod
    def get_data(measurement_location_uuid, from_date=None, to_date=None):
        """
        Retrieve measurement data from the brightwind platform and return it in a DataFrame with index as Timestamp.

        :param measurement_location_uuid: The measurement location uuid.
        :type measurement_location_uuid: str or uuid
        :param from_date: Datetime representing the start of the measurement period you want.
        :type from_date: datetime or str
        :param to_date: Datetime representing the end of the measurement period you want.
        :type to_date: datetime or str
        :return: DataFrame with index as a timestamp.
        :rtype: pd.DataFrame

        **Example usage**
        ::
            import brightwind as bw

            meas_loc_uuid = '55a8b5b2-70fb-415d-b0d9-33c26e94bd9e'

            # To load with a specific start and end date.
            df = bw.load.load._LoadBWPlatform.get_data(meas_loc_uuid, '2019-07-01', '2019-07-02')
            df

        Different date formats can be sent however it is recommended to use the format 'YYYY-MM-DD' to avoid
        your date interpreted incorrectly. E.g. '1-7-2019' will be interpreted as Jan 7th, 2019.

        If no dates are sent a false date of 1900-01-01 and todays date will be sent instead. It is recommended
        to always specify and end date to make your work repeatable, unless every time you run your code you
        want the most recent data. E.g.::

            df = bw.load.load._LoadBWPlatform.get_data(meas_loc_uuid, to_date='2019-07-02')
            df

        """
        access_token = _LoadBWPlatform._get_token()
        headers = {'Authorization': 'Bearer ' + access_token}

        # set max min dates, parse dates that are typed in and set to datetime obj
        if from_date is None or to_date is None:
            from_date, to_date = _if_null_max_the_date(from_date, to_date)
        if isinstance(from_date, str):
            from_date = parse(from_date)
        if isinstance(to_date, str):
            to_date = parse(to_date)
        print(from_date)
        print(to_date)
        response = requests.get(_LoadBWPlatform._base_url + '/api/resource-data-measurement-location', params={
            'measurement_location_uuid': measurement_location_uuid,
            'date_from': from_date.isoformat(),
            'date_to': to_date.isoformat(),
        }, headers=headers)

        response_json = response.json()
        if 'Error' in response_json:    # catch if error comes back e.g. measurement_location_uuid isn't found
            raise ValueError(response_json['Error'])

        df = pd.DataFrame(data=response_json)
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])   # this throws error if return doesn't have 'Timestamp'
            df.set_index('Timestamp', inplace=True)
        except Exception as error:
            if 'errors' in response_json:
                raise TypeError(response_json['errors'])
            else:
                raise error
        return df

    @staticmethod
    def _get_meas_points_in_df(meas_loc_uuid, Include_Tilt_Angle='N'):
        # Next we get the height of each instrument from the database and return it to a dataframe. In cases where a height does not exist
        # a dash is placed.

        pddict = dict()
        if Include_Tilt_Angle == 'Y':
            pddict = {'Sensor Name': [], 'Height [m]': [], 'Measurement Type': [], 'Tilt Angle [°]': [],
                      'Boom Orientation [°]': [], 'Sensor_UUID': []}
        else:
            pddict = {'Sensor Name': [], 'Height [m]': [], 'Measurement Type': [], 'Boom Orientation [°]': [],
                      'Sensor_UUID': []}

        meas_points = _LoadBWPlatform.get_meas_points(meas_loc_uuid)

        for mp in meas_points:
            # print(mp['mounting_arrangement']['height_metres'])

            pddict['Sensor Name'].append(mp['name'])
            pddict['Sensor_UUID'].append(mp['id'])
            pddict['Measurement Type'].append(mp['measurement_type'])

            if mp.get('mounting_arrangement') and 'boom_orientation_deg' in mp['mounting_arrangement'].keys():
                pddict['Boom Orientation [°]'].append(mp['mounting_arrangement']['boom_orientation_deg'])
            else:
                pddict['Boom Orientation [°]'].append('-')

            if Include_Tilt_Angle == 'Y':
                if mp.get('mounting_arrangement') and 'tilt_angle_deg' in mp['mounting_arrangement'].keys():
                    pddict['Tilt Angle [°]'].append(mp['mounting_arrangement']['tilt_angle_deg'])
                else:
                    pddict['Tilt Angle [°]'].append('-')

            if mp.get('mounting_arrangement') and mp['mounting_arrangement'].get('height_metres'):
                pddict['Height [m]'].append(mp['mounting_arrangement']['height_metres'])
            else:
                pddict['Height [m]'].append('-')

        Instrument_height = pd.DataFrame(pddict).set_index('Sensor_UUID')
        return Instrument_height

    @staticmethod
    def _get_sen_configs_in_df(meas_loc_uuid):
        # Next we get the relvant information we need from the database to populate the configuration table for the monthly report.

        pddict = {'Sensor Model': [], 'Units': [], 'Serial Number': [], 'Measurement_point_UUID': [], 'Date From': [],
                  'Date To': []}
        # pddict = {'Units':[],'Measurement_point_UUID':[]}

        sen_configs = _LoadBWPlatform.get_sensor_configs(meas_loc_uuid)

        for mp in sen_configs:
            # print(mp['mounting_arrangement']['height_metres'])
            if mp.get('measurement_point_uuid'):
                pddict['Measurement_point_UUID'].append(mp['measurement_point_uuid'])
                # pddict['Measurement Type'].append(mp['measurement_type'])

                if mp['logger_config'].get('measurement_units'):
                    # Note need to convert m2 symbol so that it can displaued properly in table. This will have to be done for any special units
                    mp['logger_config']['measurement_units'] = mp['logger_config']['measurement_units'].replace('m²', '$m^2$') if '²' in mp['logger_config']['measurement_units'] else mp['logger_config']['measurement_units']
                    pddict['Units'].append(mp['logger_config']['measurement_units'])
                else:
                    pddict['Units'].append('-')

                if mp['sensor_info'] and mp['sensor_info'].get('sensor_serial_number'):
                    pddict['Serial Number'].append(mp['sensor_info']['sensor_serial_number'])
                else:
                    pddict['Serial Number'].append('-')

                if mp['sensor_info'] and mp['sensor_info'].get('sensor_model'):
                    pddict['Sensor Model'].append(mp['sensor_info']['sensor_model'])
                else:
                    pddict['Sensor Model'].append('-')

                if mp.get('date_from'):
                    pddict['Date From'].append(mp['date_from'])
                else:
                    pddict['Date From'].append('-')

                if mp.get('date_to'):
                    pddict['Date To'].append(mp['date_to'])
                else:
                    pddict['Date To'].append(datetime.datetime.now())

        Sensor_config = pd.DataFrame(pddict).set_index('Measurement_point_UUID')
        return Sensor_config

    @staticmethod
    def get_sensor_table(meas_loc_uuid, measurement_type='wind speed', Include_Tilt_Angle='N', return_data=False):
        """
        Get the sensor setup in a formatted table for a measurement location uuid.

        :param meas_loc_uuid:
        :param measurement_type:
        :param return_data:
        :return:
        """
        meas_points_df = _LoadBWPlatform._get_meas_points_in_df(meas_loc_uuid, Include_Tilt_Angle=Include_Tilt_Angle)
        sen_configs_df = _LoadBWPlatform._get_sen_configs_in_df(meas_loc_uuid)
        sensor_table = meas_points_df.join(sen_configs_df)

        if Include_Tilt_Angle == 'Y':
            sensor_table = sensor_table[['Sensor Name', 'Units', 'Sensor Model', 'Measurement Type', 'Height [m]',
                                         'Boom Orientation [°]', 'Tilt Angle [°]', 'Serial Number', 'Date From']]
        else:
            sensor_table = sensor_table[['Sensor Name', 'Units', 'Sensor Model', 'Measurement Type', 'Height [m]',
                                         'Boom Orientation [°]', 'Serial Number', 'Date From']]

        sensor_table = sensor_table.set_index('Sensor Name')
        sensor_table['Date From'] = pd.to_datetime(sensor_table['Date From'])
        sensor_table['Date From'] = sensor_table['Date From'].dt.strftime("%d-%b-%Y")

        sensor_table.reset_index(inplace=True)
        sensor_table.drop(columns=['Measurement Type'], inplace=True)
        sensor_table.sort_values(by=['Sensor Name'], inplace=True)
        table = plt.render_table(sensor_table, header_columns=0, col_width=3.3)

        if return_data:
            return table, sensor_table.set_index('Sensor Name')
        else:
            return table


def _if_null_max_the_date(date_from, date_to):
    if pd.isnull(date_from):
        date_from = datetime.datetime(1900, 1, 1)
    if pd.isnull(date_to):
        date_to = datetime.datetime.today()
    return date_from, date_to


def load_cleaning_file(filepath, date_from_col_name='Start', date_to_col_name='Stop', **kwargs):
    """
    Load a cleaning file which contains a list of sensor names with corresponding periods of flagged data.
    This file is a simple comma separated file with the sensor name along with the start and end timestamps for the
    flagged period. There may be other columns in the file however these will be ignores.  E.g.:
    | Sensor |      Start          |       Stop
    ----------------------------------------------------
    | Spd80m | 2018-10-23 12:30:00 | 2018-10-25 14:20:00
    | Dir78m | 2018-12-23 02:40:00 |

    :param filepath:  File path of the file which contains the the list of sensor names along with the start and
           end timestamps of the periods that are flagged.
    :type filepath: str
    :param date_from_col_name: The column name of the date_from or the start date of the period to be cleaned.
    :type date_from_col_name: str, default 'Start'
    :param date_to_col_name: The column name of the date_to or the end date of the period to be cleaned.
    :type date_to_col_name: str, default 'Stop'
    :param kwargs: All the kwargs from pandas.read_csv can be passed to this function.
    :return: A DataFrame where each row contains the sensor name and the start and end timestamps of the flagged data.
    :rtype: pandas.DataFrame

    **Example usage**
    ::
        import brightwind as bw
        cleaning_file = r'C:\\some\\folder\\cleaning_file.csv'
        cleaning_df = bw.load_cleaning_file(cleaning_file)
        print(cleaning_df)

    """
    cleaning_df = _pandas_read_csv(filepath, **kwargs)
    # Issue when date format is DD-MM-YYYY and the MM is 12 or less.
    cleaning_df[date_from_col_name] = pd.to_datetime(cleaning_df[date_from_col_name])
    cleaning_df[date_to_col_name] = pd.to_datetime(cleaning_df[date_to_col_name])
    return cleaning_df


def apply_cleaning(data, cleaning_file_or_df, inplace = False, sensor_col_name='Sensor', date_from_col_name='Start',
                   date_to_col_name='Stop', all_sensors_descriptor='All', replacement_text='NaN'):
    """
    Apply cleaning to a DataFrame using predetermined flagged periods for each sensor listed in a cleaning file.
    The flagged data will be replaced with NaN values which then do not appear in any plots or effect calculations.

    This file is a simple comma separated file with the sensor name along with the start and end timestamps for the
    flagged period. There may be other columns in the file however these will be ignores.  E.g.:
    | Sensor |      Start          |       Stop
    ----------------------------------------------------
    | Spd80m | 2018-10-23 12:30:00 | 2018-10-25 14:20:00
    | Dir78m | 2018-12-23 02:40:00 |

    :param data: Data to be cleaned.
    :type data: pandas.DataFrame
    :param cleaning_file_or_df: File path of the csv file or a pandas DataFrame which contains the list of sensor
                                names along with the start and end timestamps of the periods that are flagged.
    :type cleaning_file_or_df: str, pd.DataFrame
    :param inplace: If 'inplace' is True, the original data, 'data', will be modified and and replaced with the cleaned
                    data. If 'inplace' is False, the original data will not be touched and instead a new object
                    containing the cleaned data is created. To store this cleaned data, please ensure it is assigned
                    to a new variable.
    :type inplace: Boolean
    :param sensor_col_name: The column name which contains the list of sensor names that have flagged periods.
    :type sensor_col_name: str, default 'Sensor'
    :param date_from_col_name: The column name of the date_from or the start date of the period to be cleaned.
    :type date_from_col_name: str, default 'Start'
    :param date_to_col_name: The column name of the date_to or the end date of the period to be cleaned.
    :type date_to_col_name: str, default 'Stop'
    :param all_sensors_descriptor: A text descriptor that represents ALL sensors in the DataFrame.
    :type all_sensors_descriptor: str, default 'All'
    :param replacement_text: Text used to replace the flagged data.
    :type replacement_text: str, default 'NaN'
    :return: DataFrame with the flagged data removed.
    :rtype: pandas.DataFrame

    **Example usage**
    ::
        import brightwind as bw

    Load data:
        data = bw.load_csv(r'C:\\Users\\Stephen\\Documents\\Analysis\\demo_data')
        cleaning_file = r'C:\\Users\\Stephen\\Documents\\Analysis\\demo_cleaning_file.csv'

    To apply cleaning to 'data' and store the cleaned data in 'data_cleaned':
        data_cleaned = bw.apply_cleaning(data, cleaning_file)
        print(data_cleaned)

    To modify 'data' and replace it with the cleaned data:
        bw.apply_cleaning(data, cleaning_file, inplace=True)
        print(data)

    To apply cleaning where the cleaning file has column names other than defaults::
        cleaning_file = r'C:\\some\\folder\\cleaning_file.csv'
        data = bw.apply_cleaning(data, cleaning_file, sensor_col_name='Data column',
                                 date_from_col_name='Start Time', date_to_col_name='Stop Time')

    """

    if inplace is False:
        data = data.copy(deep=True)

    if isinstance(cleaning_file_or_df, str):
        cleaning_df = load_cleaning_file(cleaning_file_or_df, date_from_col_name, date_to_col_name)
    elif isinstance(cleaning_file_or_df, pd.DataFrame):
        cleaning_df = cleaning_file_or_df
    else:
        return TypeError("Can't recognise the cleaning_file_or_df. Please make sure it is a file path or a DataFrame.")

    if replacement_text == 'NaN':
        replacement_text = np.nan

    for k in range(0, len(cleaning_df)):
        date_from, date_to = _if_null_max_the_date(cleaning_df[date_from_col_name][k], cleaning_df[date_to_col_name][k])

        pd.options.mode.chained_assignment = None
        if cleaning_df[sensor_col_name][k] == all_sensors_descriptor:
            data.loc[(data.index >= date_from) & (data.index < date_to), data.columns] = replacement_text
        else:
            for col in data.columns:
                if cleaning_df[sensor_col_name][k] in col:
                    data[col][(data.index >= date_from) & (data.index < date_to)] = replacement_text
        pd.options.mode.chained_assignment = 'warn'

    return data


def apply_cleaning_windographer(data, windog_cleaning_file, inplace=False, flags_to_exclude=['Synthesized'],
                                replacement_text='NaN'):
    """
    Apply cleaning to a DataFrame using the Windographer flagging log file after Windographer was used to clean and
    filter the data.
    The flagged data will be replaced with NaN values which then do not appear in any plots or effect calculations.

    :param data: Data to be cleaned.
    :type data: pandas.DataFrame
    :param windog_cleaning_file: File path of the Windographer flagging log file which contains the list of sensor
                                 names along with the start and end timestamps of the periods that are flagged.
    :type windog_cleaning_file: str
    :param inplace: If 'inplace' is True, the original data, 'data', will be modified and and replaced with the cleaned
                    data. If 'inplace' is False, the original data will not be touched and instead a new object
                    containing the cleaned data is created. To store this cleaned data, please ensure it is assigned
                    to a new variable.
    :type inplace: Boolean
    :param flags_to_exclude: List of flags you do not want to use to clean the data e.g. Synthesized.
    :type flags_to_exclude: List[str], default ['Synthesized']
    :param replacement_text: Text used to replace the flagged data.
    :type replacement_text: str, default 'NaN'
    :return: DataFrame with the flagged data removed.
    :rtype: pandas.DataFrame

    **Example usage**
    ::
        import brightwind as bw

    Load data:
        data = bw.load_csv(r'C:\\Users\\Stephen\\Documents\\Analysis\\demo_data')
        windog_cleaning_file = r'C:\\some\\folder\\windog_cleaning_file.txt'

    To apply cleaning to 'data' and store the cleaned data in 'data_cleaned':
        data_cleaned = bw.apply_cleaning_windographer(data, windog_cleaning_file)
        print(data_cleaned)

    To modify 'data' and replace it with the cleaned data:
        bw.apply_cleaning_windographer(data, windog_cleaning_file, inplace=True)
        print(data)

    Apply cleaning where you do not want the flag 'Tower shading' to be used::

        cleaning_file = r'C:\\some\\folder\\cleaning_file.csv'
        data = bw.apply_cleaning_windographer(data, windog_cleaning_file,
                                              flags_to_exclude=['Synthesized', 'Tower shading'],)
        print(data)

    """
    if inplace is False:
        data = data.copy(deep=True)

    sensor_col_name = 'Data Column'
    flag_col_name = 'Flag Name'
    date_from_col_name = 'Start Time'
    date_to_col_name = 'End Time'
    cleaning_df = load_cleaning_file(windog_cleaning_file, date_from_col_name, date_to_col_name, sep='\t')

    if replacement_text == 'NaN':
        replacement_text = np.nan

    for k in range(0, len(cleaning_df)):
        date_from, date_to = _if_null_max_the_date(cleaning_df[date_from_col_name][k], cleaning_df[date_to_col_name][k])

        pd.options.mode.chained_assignment = None
        for col in data.columns:
            if cleaning_df[sensor_col_name][k] in col:
                if cleaning_df[flag_col_name][k] not in flags_to_exclude:
                    data[col][(data.index >= date_from) & (data.index < date_to)] = replacement_text
        pd.options.mode.chained_assignment = 'warn'

    return data
