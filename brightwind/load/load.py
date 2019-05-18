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
import requests
from typing import List
import errno
import os
import shutil
import json
from io import StringIO
import warnings


__all__ = ['load_csv', 'load_campbell_scientific', 'load_windographer_txt', 'load_excel', 'load_brightdata']


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
    :type search_by_file_type: List[str], default .csv
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
        df = bw.load_csv(filepath, skiprows=[0])
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
        separators = [
            {'delimiter': 'tab', 'fn_argument': '\t'},
            {'delimiter': 'comma', 'fn_argument': ','},
            {'delimiter': 'space', 'fn_argument': ' '},
            {'delimiter': delimiter, 'fn_argument': delimiter}
        ]
        for separator in separators:
            if delimiter == separator['delimiter']:
                delimiter = separator['fn_argument']
        fn_arguments = {'skiprows': 12, 'delimiter': delimiter,
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
        df = bw.load_excel(filepath, skiprows=[0])
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
        new_file: str = os.path.join(destination_folder, filename)
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


def _append_files_together(source_folder, assembled_file_name, file_type):
    """
    Assemble files scattered in subfolders of a certain directory and copy them to a single file filtering for a
    specific list of file types.

    :param source_folder: Is the main folder to search through.
    :type source_folder: str
    :param assembled_file_name: Name of the newly created file with all the appended data.
    :type assembled_file_name: str
    :param file_type: Is a list of file extensions to filter for e.g. ['.csv', '.txt']
    :type file_type: List[str]
    :return:
    """
    list_of_files = _list_files(source_folder, file_type)

    file_handler = open(os.path.join(source_folder, assembled_file_name), 'a+')
    for file in list_of_files:
        file_handler2 = open(file, 'r')
        file_handler.write(file_handler2.read())
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
