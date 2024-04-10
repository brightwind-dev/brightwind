import pandas as pd
import numpy as np
from brightwind.utils import utils
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
from brightwind.analyse import plot as bw_plt
import time
import concurrent
import math


__all__ = ['load_csv',
           'load_campbell_scientific',
           'load_windographer_txt',
           'load_excel',
           'LoadBrightdata',
           'LoadBrightHub',
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


def load_csv(filepath_or_folder, search_by_file_type=['.csv'], print_progress=True, dayfirst=False, **kwargs):
    """
    Load timeseries data from a csv file, or group of files in a folder, into a DataFrame. The timezone is removed from
    the timestamps if it is present.
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
    :param dayfirst: If your timestamp starts with the day first e.g. DD/MM/YYYY then set this to true. Pandas defaults
            to reading 10/11/12 as 2012-10-11 (11-Oct-2012). If True, pandas parses dates with the day
            first, eg 10/11/12 is parsed as 2012-11-10. More info on pandas.read_csv parameters.
    :type dayfirst: bool, default False
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
    fn_arguments = {'header': 0, 'index_col': 0, 'parse_dates': True, 'dayfirst': dayfirst}
    merged_fn_args = {**fn_arguments, **kwargs}
    if is_file:
        return _pandas_read_csv(filepath_or_folder, **merged_fn_args).tz_localize(None)
    elif not is_file:
        return _assemble_df_from_folder(filepath_or_folder, search_by_file_type, _pandas_read_csv, print_progress,
                                        **merged_fn_args).tz_localize(None)


def load_windographer_txt(filepath, delimiter='tab', flag_text=9999, dayfirst=False, **kwargs):
    """
    Load a Windographer .txt data file exported from the Windographer software into a DataFrame. The timezone is removed
    from the timestamps if it is present.

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
    :param dayfirst: If your timestamp starts with the day first e.g. DD/MM/YYYY then set this to true. Pandas defaults
            to reading 10/11/12 as 2012-10-11 (11-Oct-2012). If True, pandas parses dates with the day
            first, eg 10/11/12 is parsed as 2012-11-10. More info on pandas.read_csv parameters.
    :type dayfirst: bool, default False
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
                        'header': 0, 'index_col': 0, 'parse_dates': True, 'dayfirst': dayfirst}
        merged_fn_args = {**fn_arguments, **kwargs}
        df = _pandas_read_csv(StringIO(file_contents), **merged_fn_args)
        if len(df.columns) > 0 and 'Unnamed' in df.columns[-1]:
            df.drop(df.columns[-1], axis=1, inplace=True)
        return df.tz_localize(None)
    elif not is_file:
        raise FileNotFoundError("File path seems to be a folder. Please load a single Windographer .txt data file.")


def load_campbell_scientific(filepath_or_folder, print_progress=True, dayfirst=False,  **kwargs):
    """
    Load timeseries data from Campbell Scientific CR1000 formatted file, or group of files in a folder, into a
    DataFrame. The timezone is removed from the timestamps if it is present.
    If the file format is slightly different your own key word arguments can be sent as this is a wrapper
    around the pandas.read_csv function. The pandas.read_csv documentation can be found at:
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

    :param filepath_or_folder: Location of the file folder containing the timeseries data.
    :type filepath_or_folder: str
    :param print_progress: If you want to print out statements of the file been processed set to True. Default is True.
    :type print_progress: bool, default True
    :param dayfirst: If your timestamp starts with the day first e.g. DD/MM/YYYY then set this to true. Pandas defaults
            to reading 10/11/12 as 2012-10-11 (11-Oct-2012). If True, pandas parses dates with the day
            first, eg 10/11/12 is parsed as 2012-11-10. More info on pandas.read_csv parameters.
    :type dayfirst: bool, default False
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
    fn_arguments = {'header': 0, 'index_col': 0, 'parse_dates': True, 'skiprows': [0, 2, 3],  'dayfirst': dayfirst}
    merged_fn_args = {**fn_arguments, **kwargs}
    if is_file:
        return _pandas_read_csv(filepath_or_folder, **merged_fn_args).tz_localize(None)
    elif not is_file:
        return _assemble_df_from_folder(filepath_or_folder, ['.dat', '.csv'], _pandas_read_csv, print_progress,
                                        **merged_fn_args).tz_localize(None)


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


class LoadBrightdata:
    """
    LoadBrightdata allows you to pull meta data and timeseries data of reanalysis datasets from brightdata. This
    is a fast way to get access to the available reanalysis datasets.

    To use LoadBrightdata, you need to request a username and password from stephen@brightwindanalysis.com.

    For security purposes LoadBrightdata uses stored environmental variables for your log in details. The
    BRIGHTDATA_USERNAME and BRIGHTDATA_PASSWORD environmental variables need to be set. In Windows this can be
    done by opening the command prompt in Administrator mode and running:

    > setx BRIGHTDATA_USERNAME "username"
    > setx BRIGHTDATA_PASSWORD "password"

    If Anaconda or your Python environment is running you will need to restart it for the environmental variables to
    take effect.

    """

    _BASE_URI = 'http://api.brightwindanalysis.com/brightdata/'
    # _BASE_URI = 'http://localhost:5000/'

    class Node:
        """
        Object defining a reanalysis node from Brightdata

        :param dataset:   Dataset type e.g. merra2, era5, etc.
        :type dataset:    str
        :param latitude:  Is the latitude of the node location for the dataset.
        :type latitude:   str
        :param longitude: Is the longitude of the node location for the dataset.
        :type longitude:  str
        :param data:      Contains the timeseries data from the node in a DataFrame.
        :type data:       pandas.DataFrame
        :param info:      Information relevant to the dataset.
        :type info:       dict
        """
        def __init__(self, dataset, latitude, longitude, data, info):
            self.dataset = dataset
            self.latitude = latitude
            self.longitude = longitude
            self.data = data
            self.info = info

    @staticmethod
    def _get_brightdata(sub_uri, **query_params):
        """
        Get merra2 or era5 data from brightdata and format it for use.
        :param sub_uri: sub part of uri string
        :param query_params: dictionary of the query parameters to be sent
        :return: List(Node)
        """
        username = utils.get_environment_variable('BRIGHTDATA_USERNAME')
        password = utils.get_environment_variable('BRIGHTDATA_PASSWORD')

        uri = LoadBrightdata._BASE_URI + sub_uri

        params = dict()
        for key in query_params:
            params[key] = query_params[key]

        response = requests.get(uri, auth=(username, password), params=params)

        try:
            json_response = response.json()
        except Exception:
            if response.status_code == 401:
                raise Exception('Please check your BRIGHTDATA_USERNAME and BRIGHTDATA_PASSWORD are correct.')
            raise Exception('Http code {}, something is wrong with the server.'.format(str(response.status_code)))

        nodes_list = []
        for node in json_response:
            temp_node_obj = LoadBrightdata.Node('', '', '', pd.DataFrame(), dict())
            try:
                for key in node:
                    if key in temp_node_obj.__dict__:   # if params returned are within the Node obj, add them
                        if key == 'data':
                            temp_node_obj.data = pd.read_json(json.dumps(node['data']), orient='index')
                        else:
                            setattr(temp_node_obj, key, node[key])
                    else:
                        temp_node_obj.info[key.replace('-', '_')] = node[key]
            except Exception as error:
                if 'Error' in node or 'message' in node:
                    raise TypeError(json_response)
                else:
                    raise error
            nodes_list.append(temp_node_obj)

        return nodes_list

    @staticmethod
    def _parse_variables(variables_list):
        var_parsed = None
        if variables_list is not None:
            var_parsed = variables_list[0]
            for variable in variables_list[1:]:
                var_parsed = var_parsed + ',' + variable
        return var_parsed

    @staticmethod
    def timeseries(dataset, lat, long, nearest=None, from_date=None, to_date=None, variables=None):
        """
            Retrieve timeseries datasets available from brightdata. Returns a list of Node objects in order
            of closest distance to the requested lat, long.

            :param dataset:   Dataset type to be retrieved from brightdata e.g. merra2, era5.
            :type  dataset:   str
            :param lat:       Latitude of your point of interest.
            :type  lat:       float
            :param long:      Longitude of your point of interest.
            :type  long:      float
            :param nearest:   The number of nearest nodes to your point of interest to retrieve. Currently only 1 to 4
                              is accepted. Default in the api is set to 1.
            :type  nearest:   int
            :param from_date: Start date of time period for which data will be retrieved.
                              Data will be retrieved that is ≥ this date.
                              If empty, the earliest date available will be retrieved.
            :type  from_date: str, in the datetime format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
            :param to_date:   End date of time period for which data will be retrieved.
                              Data will be retrieved that is < this date.
                              If empty, the latest date available will be retrieved.
            :type  to_date:   str, in the datetime format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
            :param variables: Specify variables to be retrieved. Be advised that all variables are not available for all
                              latitudes and longitudes or datasets.
                              Empty value will return Spd_50m_mps for merra2 and
                              Spd_100m_mps for era5.
                              Variables for each dataset are:
                                  merra2                    era5
                                - Spd_50m_mps               - Spd_100m_mps
                                - Dir_50m_deg               - Dir_100m_deg
                                - Tmp_2m_degC               - Tmp_2m_degC
                                - Prs_0m_hPa                - Prs_0m_hPa
                                - Spd_850pa_mps
                                - Dir_850pa_deg
                                - Spd_10m_mps
                                - Dir_10m_deg
            :type  variables: list

            :return: A list of Node objects in order of closest distance to the requested lat, long.
            :rtype: List(Node)

            To use LoadBrightdata the BRIGHTDATA_USERNAME and BRIGHTDATA_PASSWORD environmental variables need to be
            set. In Windows this can be done by running the command prompt in Administrator mode and running:

            >> setx BRIGHTDATA_USERNAME "username"
            >> setx BRIGHTDATA_PASSWORD "password"

            **Example usage**
            ::
                import brightwind as bw
                nodes = bw.LoadBrightdata.timeseries('era5', 53.4, -7.2, nearest=4,
                                                     from_date='2018-10-01', to_date='2018-10-02')
                for node in nodes:
                    print(node.dataset)
                    print(node.latitude)
                    print(node.longitude)
                    print(node.data)

                # get only temperature and pressure for the nearest node to my location
                merra2_nodes = bw.LoadBrightdata.timeseries('merra2', 60, 14.78, nearest=1,
                                                            from_date='2018-12-01', to_date='2018-12-02',
                                                            variables=['Tmp_2m_degC', 'Prs_0m_hPa'])
                print(merra2_nodes[0].dataset, merra2_nodes[0].latitude, merra2_nodes[0].longitude)
                merra2_nodes[0].data

                # get only temperature and pressure for the nearest node to my location up to most recent date
                merra2_nodes = bw.LoadBrightdata.timeseries('merra2', 49.15, 4.78, nearest=1,
                                                            from_date='2018-12-01',
                                                            variables=['Tmp_2m_degC', 'Prs_0m_hPa'])
                print(merra2_nodes[0].dataset, merra2_nodes[0].latitude, merra2_nodes[0].longitude)
                merra2_nodes[0].data

        """
        var_parsed = LoadBrightdata._parse_variables(variables)
        fn_arguments = {
            'dataset': dataset, 'latitude': lat, 'longitude': long, 'from-date': from_date,
            'to-date': to_date, 'nearest': nearest, 'variables': var_parsed
        }
        try:
            return LoadBrightdata._get_brightdata(sub_uri='timeseries', **fn_arguments)
        except Exception as error:
            raise error

    @staticmethod
    def monthly_means(dataset, lat, long, nearest=None, from_date=None, to_date=None, variables=None):
        """
            Retrieve monthly means from brightdata datasets such as merra2 and era5. Returns a list of Node objects
            in order of closest distance to the requested lat, long. Monthly coverage is also returned in the data.

            :param dataset:   Dataset type to be retrieved from brightdata e.g. merra2, era5.
            :type  dataset:   str
            :param lat:       Latitude of your point of interest.
            :type  lat:       float
            :param long:      Longitude of your point of interest.
            :type  long:      float
            :param nearest:   The number of nearest nodes to your point of interest to retrieve. Currently only 1 to 4
                              is accepted. Default in the api is set to 1.
            :type  nearest:   int
            :param from_date: Start date of time period for which data will be retrieved.
                              Data will be retrieved that is ≥ this date.
                              If empty, the earliest date available will be retrieved.
            :type  from_date: str, in the datetime format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
            :param to_date:   End date of time period for which data will be retrieved.
                              Data will be retrieved that is < this date.
                              If empty, the latest date available will be retrieved.
            :type  to_date:   str, in the datetime format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
            :param variables: Specify variables to be retrieved. Be advised that all variables are not available for all
                              latitudes and longitudes or datasets.
                              Empty value will return Spd_50m_mps for merra2 and
                              Spd_100m_mps for era5.
                              Variables for each dataset are:
                                  merra2                    era5
                                - Spd_50m_mps               - Spd_100m_mps
                                - Tmp_2m_degC               - Tmp_2m_degC
                                - Prs_0m_hPa                - Prs_0m_hPa
                                - Spd_850pa_mps
                                - Spd_10m_mps
            :type  variables: list

            :return: A list of Node objects in order of closest distance to the requested lat, long.
            :rtype: List(Node)

            To use LoadBrightdata the BRIGHTDATA_USERNAME and BRIGHTDATA_PASSWORD environmental variables need to be
            set. In Windows this can be done by running the command prompt in Administrator mode and running:

            >> setx BRIGHTDATA_USERNAME "username"
            >> setx BRIGHTDATA_PASSWORD "password"

            **Example usage**
            ::
                import brightwind as bw
                nodes = bw.LoadBrightdata.monthly_means('era5', 53.4, -7.2, nearest=4,
                                                        from_date='2018-01-01', to_date='2019-01-01')
                for node in nodes:
                    print(node.dataset)
                    print(node.latitude)
                    print(node.longitude)
                    print(node.data)

                # get only temperature and pressure for the nearest node to my location
                merra2_nodes = bw.LoadBrightdata.monthly_means('merra2', 60, 14.78, nearest=1,
                                                               from_date='2018-01-01', to_date='2019-01-01',
                                                               variables=['Tmp_2m_degC', 'Prs_0m_hPa'])
                print(merra2_nodes[0].dataset, merra2_nodes[0].latitude, merra2_nodes[0].longitude)
                merra2_nodes[0].data

                # get only temperature and pressure for the nearest node to my location up to most recent date
                merra2_nodes = bw.LoadBrightdata.monthly_means('merra2', 49.15, 4.78, nearest=1,
                                                               from_date='2018-12-01',
                                                               variables=['Tmp_2m_degC', 'Prs_0m_hPa'])
                print(merra2_nodes[0].dataset, merra2_nodes[0].latitude, merra2_nodes[0].longitude)
                merra2_nodes[0].data

        """
        var_parsed = LoadBrightdata._parse_variables(variables)
        fn_arguments = {
            'dataset': dataset, 'latitude': lat, 'longitude': long, 'from-date': from_date,
            'to-date': to_date, 'nearest': nearest, 'variables': var_parsed
        }
        try:
            return LoadBrightdata._get_brightdata(sub_uri='timeseries/monthly-means', **fn_arguments)
        except Exception as error:
            raise error

    @staticmethod
    def momm(dataset, lat, long, nearest=None, from_date=None, to_date=None, variables=None):
        """
            Retrieve the mean of monthly means from brightdata datasets such as merra2 and era5. Returns a list of
            Node objects in order of closest distance to the requested lat, long.

            :param dataset:   Dataset type to be retrieved from brightdata e.g. merra2, era5.
            :type  dataset:   str
            :param lat:       Latitude of your point of interest.
            :type  lat:       float
            :param long:      Longitude of your point of interest.
            :type  long:      float
            :param nearest:   The number of nearest nodes to your point of interest to retrieve. Currently only 1 to 4
                              is accepted. Default in the api is set to 1.
            :type  nearest:   int
            :param from_date: Start date of time period for which data will be retrieved.
                              Data will be retrieved that is ≥ this date.
                              If empty, the earliest date available will be retrieved.
            :type  from_date: str, in the datetime format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
            :param to_date:   End date of time period for which data will be retrieved.
                              Data will be retrieved that is < this date.
                              If empty, the latest date available will be retrieved.
            :type  to_date:   str, in the datetime format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
            :param variables: Specify variables to be retrieved. Be advised that all variables are not available for all
                              latitudes and longitudes or datasets.
                              Empty value will return Spd_50m_mps for merra2 and
                              Spd_100m_mps for era5.
                              Variables for each dataset are:
                                  merra2                    era5
                                - Spd_50m_mps               - Spd_100m_mps
                                - Tmp_2m_degC               - Tmp_2m_degC
                                - Prs_0m_hPa                - Prs_0m_hPa
                                - Spd_850pa_mps
                                - Spd_10m_mps
            :type  variables: list

            :return: A list of Node objects in order of closest distance to the requested lat, long.
            :rtype: List(Node)

            To use LoadBrightdata the BRIGHTDATA_USERNAME and BRIGHTDATA_PASSWORD environmental variables need to be
            set. In Windows this can be done by running the command prompt in Administrator mode and running:

            >> setx BRIGHTDATA_USERNAME "username"
            >> setx BRIGHTDATA_PASSWORD "password"

            **Example usage**
            ::
                import brightwind as bw
                nodes = bw.LoadBrightdata.momm('era5', 53.4, -7.2, nearest=4,
                                               from_date='2018-01-01', to_date='2019-01-01')
                for node in nodes:
                    print(node.dataset)
                    print(node.latitude)
                    print(node.longitude)
                    print(node.data)
                    print(node.info)

                # get only temperature and pressure for the nearest node to my location
                merra2_nodes = bw.LoadBrightdata.momm('merra2', 60, 14.78, nearest=1,
                                                      from_date='2018-01-01', to_date='2019-01-01',
                                                      variables=['Tmp_2m_degC', 'Prs_0m_hPa'])
                print(merra2_nodes[0].dataset, merra2_nodes[0].latitude, merra2_nodes[0].longitude)
                merra2_nodes[0].data

                # get only temperature and pressure for the nearest node to my location up to most recent date
                merra2_nodes = bw.LoadBrightdata.momm('merra2', 49.15, 4.78, nearest=1,
                                                      from_date='2018-12-01',
                                                      variables=['Tmp_2m_degC', 'Prs_0m_hPa'])
                print(merra2_nodes[0].dataset, merra2_nodes[0].latitude, merra2_nodes[0].longitude)
                merra2_nodes[0].data

        """
        var_parsed = LoadBrightdata._parse_variables(variables)
        fn_arguments = {
            'dataset': dataset, 'latitude': lat, 'longitude': long, 'from-date': from_date,
            'to-date': to_date, 'nearest': nearest, 'variables': var_parsed
        }
        try:
            return LoadBrightdata._get_brightdata(sub_uri='timeseries/momms', **fn_arguments)
        except Exception as error:
            raise error

    @staticmethod
    def monthly_norms(dataset, lat, long, nearest=None, from_date=None, to_date=None, ref_from_date=None,
                      ref_to_date=None, ref_no_years=None, variables=None):
        """
        Return the monthly mean wind speeds normalised to a specific reference period. The reference period can be
        between two specific dates or it could be a number of rolling years preceding each month of interest.

        :param dataset:       Dataset type to be retrieved from brightdata e.g. merra2, era5.
        :type  dataset:       str
        :param lat:           Latitude of your point of interest.
        :type  lat:           float
        :param long:          Longitude of your point of interest.
        :type  long:          float
        :param nearest:       The number of nearest nodes to your point of interest to retrieve. Currently only 1 to 4
                              is accepted. Default in the api is set to 1.
        :type  nearest:       int
        :param from_date:     Start date of time period for which data will be retrieved.
                              Data will be retrieved that is ≥ this date.
                              If empty, the earliest date available will be retrieved.
        :type  from_date:     str, in the datetime format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
        :param to_date:       End date of time period for which data will be retrieved.
                              Data will be retrieved that is < this date.
                              If empty, the latest date available will be retrieved.
        :type  to_date:       str, in the datetime format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
        :param ref_from_date: Start date of the reference period used to calculate the long term mean wind speed.
                              Data will be retrieved that is ≥ this date.
                              If empty, the earliest date available will be used.
        :type  ref_from_date: str, in the datetime format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
        :param ref_to_date:   End date of the reference period used to calculate the long term mean wind speed.
                              Data will be retrieved that is < this date.
                              If empty, the latest date available will be used.
        :type  ref_to_date:   str, in the datetime format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
        :param ref_no_years:  The number of years to define the reference period. This will be a rolling long term
                              up to the month been calculated at the time i.e. for ref_no_years=10 the reference
                              period for month Sep-2018 will be the 10 years preceding it, Sep-2008 to Aug-2018, and for
                              month Oct-2018 it will be Oct-2008 to Sep-2018.
                              Only used when ref_from_date and ref_to_date are both None. Default is 20.
                              Note, that if no data exists for the reference period this will cause an error, for
                              example if using a from-date of 2000-01-01 as no data exists in brightdata before
                              this date for most regions.
        :type  ref_no_years:  int
        :param variables:     Specify variables to be retrieved. Be advised that all variables are not available for all
                              latitudes and longitudes or datasets.
                              Empty value will return Spd_50m_mps for merra2 and
                              Spd_100m_mps for era5.
                              Variables for each dataset are:
                                  merra2                    era5
                                - Spd_50m_mps               - Spd_100m_mps
                                - Tmp_2m_degC               - Tmp_2m_degC
                                - Prs_0m_hPa                - Prs_0m_hPa
                                - Spd_850pa_mps
                                - Spd_10m_mps
        :type  variables:     list

        :return: A list of Node objects in order of closest distance to the requested lat, long.
        :rtype: List(Node)

        The long term mean wind speed is calculated using the mean of monthly mean function, bw.momm().

        To use LoadBrightdata the BRIGHTDATA_USERNAME and BRIGHTDATA_PASSWORD environmental variables need to be
        set. In Windows this can be done by running the command prompt in Administrator mode and running:

        >> setx BRIGHTDATA_USERNAME "username"
        >> setx BRIGHTDATA_PASSWORD "password"

        **Example usage**
        ::
            import brightwind as bw
            nodes = bw.LoadBrightdata.monthly_norms('merra2', 60, 14.78, nearest=3,
                                                    from_date='2019-01-01', to_date='2019-06-30',
                                                    ref_from_date='2002-01-01', ref_to_date='2019-01-01')
            for node in nodes:
                print(node.dataset)
                print(node.latitude)
                print(node.longitude)
                print(node.info)
                print(node.data)

            # use a reference period of a rolling 10 years
            merra2_nodes = bw.LoadBrightdata.monthly_norms('merra2', 60, 14.78, nearest=1,
                                                           from_date='2019-01-01', to_date='2019-07-01',
                                                           ref_no_years='10')
            print(merra2_nodes[0].dataset, merra2_nodes[0].latitude, merra2_nodes[0].longitude, merra2_nodes[0].info)
            merra2_nodes[0].data

            # use the 850pa wind speed instead
            merra2_nodes = bw.LoadBrightdata.monthly_norms('merra2', 60, 14.78, nearest=1,
                                                           from_date='2019-01-01', to_date='2019-07-01',
                                                           ref_no_years='10',
                                                           variables=['Spd_850pa_mps', 'Tmp_2m_degC'])
            print(merra2_nodes[0].dataset, merra2_nodes[0].latitude, merra2_nodes[0].longitude, merra2_nodes[0].info)
            merra2_nodes[0].data

        """
        var_parsed = LoadBrightdata._parse_variables(variables)
        fn_arguments = {
            'dataset': dataset, 'latitude': lat, 'longitude': long,
            'from-date': from_date, 'to-date': to_date, 'ref-from-date': ref_from_date, 'ref-to-date': ref_to_date,
            'ref-no-years': ref_no_years, 'nearest': nearest, 'variables': var_parsed
        }
        try:
            return LoadBrightdata._get_brightdata(sub_uri='timeseries/monthly-norms', **fn_arguments)
        except Exception as error:
            raise error


class _BrighthubAuth:
    """
    This class is used to define general functions that are then called by LoadBrightHub. Functions in this
    class are outside of LoadBrightHub and will be called only once during the analysis and this will avoid making
    multiple login to the Brighthub user pool.

    """

    # List possible errors encountered on Login
    __BRIGHTHUB_LOGIN_ERROR_MAP = {
        "not_authorized": "The Brighthub Email or Password is incorrect",
        "user_not_confirmed": "The User is not confirmed. Please confirm your email and try again.",
        "unexpected_error": "An unexpected error occurred.",
        "new_password_required": "Your password has expired or needs to be reset. "
                                 "Kindly reset your Brighthub password and try again.",
        "password_not_verified": "Could not verify your password. "
                                 "Please ensure you have confirmed your email and the password is correct."
    }
    ID_TOKEN = ''
    REFRESH_TOKEN = ''
    USERNAME = ''
    PASSWORD = ''

    @staticmethod
    def _get_cognito_request():
        """
        Function to form a request for Cognito Auth APIs
        """
        url = "https://cognito-idp.eu-west-1.amazonaws.com/"
        headers = {
            'X-Amz-Target': 'AWSCognitoIdentityProviderService.InitiateAuth',
            'Content-Type': 'application/x-amz-json-1.1'
        }
        client_id = "3qkkpikve578cbok46p136au3g"

        return url, headers, client_id

    @staticmethod
    def _get_id_token():
        """
        Function to login to the Brighthub user pool.
        Assign a id_token and a refresh_token to the global variables ID_TOKEN, REFRESH_TOKEN which can be used to
        make requests to the APIs.
        In case of an error, a error message will be returned

        """
        url, headers, client_id = _BrighthubAuth._get_cognito_request()

        if not _BrighthubAuth.USERNAME:
            _BrighthubAuth.USERNAME = utils.get_environment_variable('BRIGHTHUB_EMAIL')

        if not _BrighthubAuth.PASSWORD:
            _BrighthubAuth.PASSWORD = utils.get_environment_variable('BRIGHTHUB_PASSWORD')

        body = {
            "AuthParameters": {
                "USERNAME": _BrighthubAuth.USERNAME,
                "PASSWORD": _BrighthubAuth.PASSWORD
            },
            "AuthFlow": "USER_PASSWORD_AUTH",
            "ClientId": client_id
        }

        response = requests.post(url, headers=headers, json=body)
        login_response = response.json()

        # a login error occurred
        if login_response.get("__type"):
            if login_response["__type"] == "NotAuthorizedException":
                return ImportError(_BrighthubAuth.__BRIGHTHUB_LOGIN_ERROR_MAP["not_authorized"])
            elif login_response["__type"] == "UserNotConfirmedException":
                return ImportError(_BrighthubAuth.__BRIGHTHUB_LOGIN_ERROR_MAP["user_not_confirmed"])
            else:
                return ImportError(_BrighthubAuth.__BRIGHTHUB_LOGIN_ERROR_MAP["unexpected_error"])

        # challenge returned
        if login_response.get("ChallengeName"):
            if login_response["ChallengeName"] == "NEW_PASSWORD_REQUIRED":
                return ImportError(_BrighthubAuth.__BRIGHTHUB_LOGIN_ERROR_MAP["new_password_required"])
            elif login_response["ChallengeName"] == "PASSWORD_VERIFIER":
                return ImportError(_BrighthubAuth.__BRIGHTHUB_LOGIN_ERROR_MAP["password_verifier"])
            else:
                return ImportError(_BrighthubAuth.__BRIGHTHUB_LOGIN_ERROR_MAP["unexpected_error"])

        # login successful
        id_token = login_response['AuthenticationResult']['IdToken']
        refresh_token = login_response['AuthenticationResult']['RefreshToken']

        _BrighthubAuth.ID_TOKEN = id_token
        _BrighthubAuth.REFRESH_TOKEN = refresh_token

        return {}

    @staticmethod
    def _brighthub_refresh_token():
        """
        Function to generate a new token if the current id_token has expired. The new tokens are assigned to the global
        variables ID_TOKEN, REFRESH_TOKEN.
        In case of an error, a error message will be returned

        """
        url, headers, client_id = _BrighthubAuth._get_cognito_request()

        body = {
            "AuthParameters": {
                "REFRESH_TOKEN": _BrighthubAuth.REFRESH_TOKEN
            },
            "AuthFlow": "REFRESH_TOKEN_AUTH",
            "ClientId": client_id
        }
        response = requests.post(url, headers=headers, json=body)
        login_response = response.json()

        # a login error occurred
        if login_response.get("__type"):
            if login_response["__type"] == "NotAuthorizedException":
                return ImportError(_BrighthubAuth.__BRIGHTHUB_LOGIN_ERROR_MAP["not_authorized"])
            elif login_response["__type"] == "UserNotConfirmedException":
                return ImportError(_BrighthubAuth.__BRIGHTHUB_LOGIN_ERROR_MAP["user_not_confirmed"])
            else:
                return ImportError(_BrighthubAuth.__BRIGHTHUB_LOGIN_ERROR_MAP["unexpected_error"])

        id_token = login_response['AuthenticationResult']['IdToken']
        _BrighthubAuth.ID_TOKEN = id_token
        new_refresh_token = ""

        # refresh token only expires after 30 days, it is not returned in the response if it is still valid
        if login_response['AuthenticationResult'].get('RefreshToken'):
            new_refresh_token = login_response['AuthenticationResult']['RefreshToken']

        # add the refresh token only if it has been generated again
        # this functionality hasn't been tested as we could not replicate an expired refresh token
        if new_refresh_token:
            _BrighthubAuth.REFRESH_TOKEN = new_refresh_token

        return {}


class LoadBrightHub:
    """
    LoadBrightHub allows you to pull meta data and timeseries data of measurements from the BrightHub
    platform. This is a fast way to get access to the available open datasets on the platform.

    To use LoadBrightHub, first sign up on www.brighthub.io and note your email and password.

    For security purposes LoadBrightHub uses stored environmental variables for your log in details. The
    BRIGHTHUB_EMAIL and BRIGHTHUB_PASSWORD environmental variables need to be set. In Windows this can be
    done by opening the command prompt in Administrator mode and running:

    > setx BRIGHTHUB_EMAIL "your email")
    > setx BRIGHTHUB_PASSWORD "your password")

    If Anaconda or your Python environment is running you will need to restart it for the environmental variables to
    take effect.

    You can start by pulling all the available measurement stations available to you by running:

    bw.LoadBrightHub.get_measurement_stations()

    """

    __BASE_URI = 'https://api.brighthub.io'

    @staticmethod
    def _brighthub_request(url_end, params=None):
        """
        Function to make a GET request to the Brighthub endpoints

        :param url_end:     The end of the url to be concatenated with the __BASE_URI in order to make the request
        :type url_end:      str
        :param params:      Optional. A dictionary, list of tuples or bytes to send as a query string. Default None
        :type params:       dict, list(tuples), bytes
        :return response:   The requests reponse object returned by requests.get()
        :rtype:             requests.Response object
        """

        if not _BrighthubAuth.ID_TOKEN:
            login_response = _BrighthubAuth._get_id_token()
            if "error" in login_response:
                return ImportError(login_response)

        url = "{}{}".format(LoadBrightHub.__BASE_URI, url_end)
        headers = {"authorization": _BrighthubAuth.ID_TOKEN}

        response = requests.get(url=url, headers=headers, params=params)

        # If there is an auth error due to expired token
        if response.status_code == 401 and response.json().get("message") == "The incoming token has expired":
            # generate the token again
            refresh_token_response = _BrighthubAuth._brighthub_refresh_token()

            # if an error occurred
            if refresh_token_response.get("error"):
                return ImportError(refresh_token_response)
            else:
                # token refreshed successfully
                headers = {"authorization": _BrighthubAuth.ID_TOKEN}

                # make the request again
                response = requests.get(url=url, headers=headers, params=params)
                return response

        return response

    @staticmethod
    def get_plants(plant_type=None, plant_uuid=None):
        """
        Get plants available to you on BrightHub.

        :param plant_type: Filter for plant type: 'onshore_wind', 'offshore_wind' or 'solar'.
        :type plant_type:  str
        :param plant_uuid: Filter for a specific plant by sending it's uuid. This preferences over plant_type.
        :type plant_uuid:  str
        :return:           A table showing the available plants.
        :rtype:            pd.DataFrame

        To use LoadBrightHub, first sign up on www.brighthub.io and note your email and password.

        For security purposes LoadBrightHub uses stored environmental variables for your log in details. The
        BRIGHTHUB_EMAIL and BRIGHTHUB_PASSWORD environmental variables need to be set. In Windows this can be
        done by opening the command prompt in Administrator mode and running:

        > setx BRIGHTHUB_EMAIL "your email")
        > setx BRIGHTHUB_PASSWORD "your password")

        If Anaconda or your Python environment is running you will need to restart it for the environmental variables to
        take effect.

        **Example usage**
        ::
            import brightwind as bw

        To get all available plants
        ::
            bw.LoadBrightHub.get_plants()

        To get all available offshore plants
        ::
            bw.LoadBrightHub.get_plants(plant_type='offshore_wind')

        To get a specific plant
        ::
            bw.LoadBrightHub.get_plants(plant_uuid='7a58497e-bee1-42a2-8084-c47a5cf213b7')

        """
        # get all the plants and display them as a dataframe
        plants = None
        if plant_type is None and plant_uuid is None:
            # get all
            plants = LoadBrightHub._brighthub_request(url_end="/plants")
        elif plant_type in ['onshore_wind', 'offshore_wind', 'solar'] and plant_uuid is None:
            # get all for a specific plant_type
            plants = LoadBrightHub._brighthub_request(url_end="/plants", params={'plant_type_id': plant_type})
        elif plant_uuid is not None:
            # get all for plant_uuid
            plants = LoadBrightHub._brighthub_request(url_end="/plants/{}".format(plant_uuid))

        if plants.headers.get('content-type') != 'application/json.':
            plants.raise_for_status()

        plants_json = plants.json()
        if 'Error' in plants_json:    # catch if error comes back e.g. measurement_location_uuid isn't found
            raise ValueError(plants_json['Error'])
        plants_df = pd.read_json(json.dumps(plants_json))
        required_cols = ['name', 'country_id', 'region', 'plant_type_id',
                         'latitude_ddeg', 'longitude_ddeg', 'alias', 'uuid', 'notes']
        plants_df = plants_df[required_cols]
        plants_df.set_index(['name'], inplace=True)
        plants_df.sort_index(ascending=True, inplace=True)
        plants_df.rename(columns={'uuid': 'plant_uuid'}, inplace=True)
        return plants_df

    @staticmethod
    def get_measurement_stations(plant_uuid=None, measurement_station_uuid=None):
        """
        Get measurement stations available to you on BrightHub.

        :param plant_uuid:               Filter for measurement stations of a specific plant.
        :type plant_uuid:                str
        :param measurement_station_uuid: Filter for a specific measurement station by sending it's uuid. This
                                         preferences over plant_uuid.
        :type measurement_station_uuid:  str
        :return:                         A table showing the available measurement stations.
        :rtype:                          pd.DataFrame

        To use LoadBrightHub, first sign up on www.brighthub.io and note your email and password.

        For security purposes LoadBrightHub uses stored environmental variables for your log in details. The
        BRIGHTHUB_EMAIL and BRIGHTHUB_PASSWORD environmental variables need to be set. In Windows this can be
        done by opening the command prompt in Administrator mode and running:

        > setx BRIGHTHUB_EMAIL "your email")
        > setx BRIGHTHUB_PASSWORD "your password")

        If Anaconda or your Python environment is running you will need to restart it for the environmental variables to
        take effect.

        **Example usage**
        ::
            import brightwind as bw

        To get all available measurement stations
        ::
            bw.LoadBrightHub.get_measurement_stations()

        To get all available measurement stations for a specific plant
        ::
            bw.LoadBrightHub.get_measurement_stations(plant_uuid='7a58497e-bee1-42a2-8084-c47a5cf213b7')

        To get a specific measurement station
        ::
            bw.LoadBrightHub.get_measurement_stations(measurement_station_uuid='9344e576-6d5a-45f0-9750-2a7528ebfa14')

        """
        measurement_locations = None
        if plant_uuid is None and measurement_station_uuid is None:
            # get all
            measurement_locations = LoadBrightHub._brighthub_request(url_end="/measurement-locations")
        elif plant_uuid is not None and measurement_station_uuid is None:
            # get all for plant_uuid
            measurement_locations = LoadBrightHub._brighthub_request(url_end="/measurement-locations",
                                                                     params={'plant_uuid': plant_uuid})
        elif measurement_station_uuid is not None:
            # get for measurement_station_uuid
            measurement_locations = LoadBrightHub._brighthub_request(
                url_end="/measurement-locations/{}".format(measurement_station_uuid))

        if measurement_locations.headers.get('content-type') != 'application/json.':
            measurement_locations.raise_for_status()

        meas_loc_json = measurement_locations.json()
        if 'Error' in meas_loc_json:    # catch if error comes back e.g. measurement_location_uuid isn't found
            raise ValueError(meas_loc_json['Error'])
        meas_loc_df = pd.read_json(json.dumps(meas_loc_json))
        required_cols = ['name', 'measurement_station_type_id',
                         'latitude_ddeg', 'longitude_ddeg', 'plant_uuid', 'uuid', 'notes']
        meas_loc_df = meas_loc_df[required_cols]
        meas_loc_df.set_index(['name'], inplace=True)
        meas_loc_df.sort_index(ascending=True, inplace=True)
        meas_loc_df.rename(columns={'uuid': 'measurement_station_uuid',
                                    'measurement_station_type_id': 'measurement_station_type'}, inplace=True)
        return meas_loc_df

    @staticmethod
    def get_start_end_dates(measurement_station_uuid):
        """
        Get the start and end dates for the period of measurements from a particular measurement station.

        :param measurement_station_uuid: A specific measurement station's uuid.
        :type measurement_station_uuid:  str
        :return:                         The start and end dates. E.g. {'start_date': '2015-12-22T00:01:00',
                                                                        'end_date': '2016-12-19T23:01:00'}
        :rtype:                          dict
        """
        start_end_dates = LoadBrightHub._brighthub_request(
            url_end="/measurement-locations/{}/start-end-dates".format(measurement_station_uuid))

        if start_end_dates.headers.get('content-type') != 'application/json.':
            start_end_dates.raise_for_status()
        start_end_dates_json = start_end_dates.json()
        if 'Error' in start_end_dates_json:    # catch if error comes back e.g. measurement_location_uuid isn't found
            raise ValueError(start_end_dates_json['Error'])

        return start_end_dates.json()

    @staticmethod
    def get_data_model(measurement_station_uuid):
        """
        Get the IEA Wind: Task 43 WRA Data Model for the measurement station.

        Information about the data model can be found at: https://github.com/IEA-Task-43/digital_wra_data_standard

        Once the data model is retrieved you can use the brightwind MeasurementStation class to view and
        use the data from it.

        :param measurement_station_uuid: A specific measurement station's uuid.
        :type measurement_station_uuid:  str
        :return:                         The data model for the measurement station.
        :rtype:                          dict

        **Example usage**
        ::
            import brightwind as bw

        To get the data model for a specific measurement station
        ::
            data_model_json - bw.LoadBrightHub.get_data_model(
                                                    measurement_station_uuid='9344e576-6d5a-45f0-9750-2a7528ebfa14')

        Using the data model
        ::
            demo_mast = bw.MeasurementStation(data_model_json)
            demo_mast.get_table()

        """
        data_model = LoadBrightHub._brighthub_request(
            url_end="/measurement-locations/{}/data-model".format(measurement_station_uuid))

        if data_model.headers.get('content-type') != 'application/json.':
            data_model.raise_for_status()
        data_model_json = data_model.json()
        if 'Error' in data_model_json:  # catch if error comes back e.g. measurement_location_uuid isn't found
            raise ValueError(data_model_json['Error'])

        return data_model.json()

    # Pulling the timeseries data.
    __DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

    @staticmethod
    def get_timeseries_data(measurement_station_uuid, date_from=None, date_to=None):
        """
        Get the timeseries data from BrightHub for a particular measurement station.

        When using the date filters, the brightwind convention is greater than and equal to 'date_from'
        to less than 'date_to'.

        :param measurement_station_uuid: A specific measurement station's uuid.
        :type measurement_station_uuid:  str
        :param date_from:                Optional filter to retrieve data from and including this date onwards.
        :type date_from:                 str
        :param date_to:                  Optional filter to retrieve data up to this date.
        :return:                         The timeseries data.
        :rtype:                          pd.DataFrame
        """
        response = LoadBrightHub._brighthub_request(url_end="/measurement-locations/{}/timeseries-data"
                                                    .format(measurement_station_uuid),
                                                    params={"date_from": date_from, "date_to": date_to})

        response_json = response.json()
        if 'error' in response_json:  # catch if error comes back e.g. measurement_location_uuid isn't found
            raise ValueError(response_json['error'])

        pre_signed_url = response_json["url"]
        timeseries_response = requests.get(pre_signed_url)
        df = pd.read_csv(StringIO(timeseries_response.text, ))
        return df.set_index('Timestamp')

    @staticmethod
    def __get_date_range_as_chunks(date_from, date_to, interval):
        # Based on: https://stackoverflow.com/a/29721341/7373512
        diff = (date_to - date_from) / interval
        for i in range(interval):
            yield (date_from + diff * i).strftime(LoadBrightHub.__DATE_FORMAT)
        yield date_to.strftime(LoadBrightHub.__DATE_FORMAT)

    @staticmethod
    def get_data(measurement_station_uuid, date_from=None, date_to=None):
        """
        Get the timeseries data from BrightHub for a particular measurement station.

        :param measurement_station_uuid: A specific measurement station's uuid.
        :type measurement_station_uuid:  str
        :param date_from:                Optional filter to retrieve data from this date onwards.
        :type date_from:                 str
        :param date_to:                  Optional filter to retrieve data up to this date.
        :type date_to:                   str
        :return:                         The timeseries data.
        :rtype:                          pd.DataFrame

        **Example usage**
        ::
            import brightwind as bw

        To get all the data for the specific measurement station
        ::
            data = bw.LoadBrightHub.get_data(measurement_station_uuid='9344e576-6d5a-45f0-9750-2a7528ebfa14')
            data.head()

        To get data for a specific time period
        ::
            data = bw.LoadBrightHub.get_data(measurement_station_uuid='9344e576-6d5a-45f0-9750-2a7528ebfa14',
                                             date_from='2016-06-01',
                                             date_to='2016-07-01')

        To get data from a specific date
        ::
            data = bw.LoadBrightHub.get_data(measurement_station_uuid='9344e576-6d5a-45f0-9750-2a7528ebfa14',
                                             date_from='2016-06-01')

        To get data up to a specific date
        ::
            data = bw.LoadBrightHub.get_data(measurement_station_uuid='9344e576-6d5a-45f0-9750-2a7528ebfa14',
                                             date_to='2016-07-01')

        """
        if not date_from or not date_to:
            start_end_dates = LoadBrightHub.get_start_end_dates(measurement_station_uuid)

            if not date_from:
                date_from = start_end_dates['start_date']
            if not date_to:
                date_to = start_end_dates['end_date']

        date_from_dt = pd.to_datetime(date_from)
        date_to_dt = pd.to_datetime(date_to)

        time_diff = date_to_dt - date_from_dt
        time_diff_in_secs = time_diff.total_seconds()

        number_of_days_per_chunk = 30
        secs_in_single_day = 60 * 60 * 24
        chunk_size_in_secs = secs_in_single_day * number_of_days_per_chunk

        if time_diff_in_secs > chunk_size_in_secs:
            # Round up to ensure no data lost
            number_of_chunks = math.ceil(time_diff_in_secs / chunk_size_in_secs)
            chunk_dates = list(
                LoadBrightHub.__get_date_range_as_chunks(date_from_dt, date_to_dt, number_of_chunks)
            )
            chunks = [
                (chunk_date, chunk_dates[idx + 1])
                for (idx, chunk_date) in enumerate(chunk_dates)
                if idx < len(chunk_dates) - 1
            ]
        # Chunk into 3 if between 5 and 30 days data requested
        elif time_diff_in_secs > secs_in_single_day * 5:
            chunk_dates = list(LoadBrightHub.__get_date_range_as_chunks(date_from_dt, date_to_dt, 3))
            chunks = [
                (chunk_date, chunk_dates[idx + 1])
                for (idx, chunk_date) in enumerate(chunk_dates)
                if idx < len(chunk_dates) - 1
            ]
        # Don't bother splitting if less than 5 days data requested
        else:
            chunks = [(date_from_dt.strftime(LoadBrightHub.__DATE_FORMAT),
                       date_to_dt.strftime(LoadBrightHub.__DATE_FORMAT))]

        def get_chunk(_date_from, _date_to):
            response = LoadBrightHub._brighthub_request(
                url_end="/resource-data",
                params={"measurement_location_uuid": measurement_station_uuid,
                        "date_from": _date_from, "date_to": _date_to})
            pre_signed_url = response.json()["url"]
            response = requests.get(pre_signed_url)
            return response.text

        master_df = pd.DataFrame()
        with concurrent.futures.ThreadPoolExecutor() as executor:  # optimally defined number of threads
            res = [executor.submit(get_chunk, chunk[0], chunk[1]) for chunk in chunks]
            concurrent.futures.wait(res)
            for result in res:
                chunk_df = pd.read_csv(StringIO(result.result()),)
                chunk_df["Timestamp"] = pd.to_datetime(chunk_df["Timestamp"])
                chunk_df.index = pd.DatetimeIndex(chunk_df.pop("Timestamp"))
                if master_df.empty:
                    master_df = chunk_df
                else:
                    master_df = pd.concat([master_df, chunk_df])
        return master_df


class _LoadBWPlatform:
    """
    LoadBWPlatform allows you to pull meta data and timeseries data of measurements from the brightwind platform.

    To use LoadBWPlatform the BW_PLATFORM_USERNAME and BW_PLATFORM_PASSWORD environmental variables need to be set. In
    Windows this can be done by opening the command prompt in Administrator mode and running:

    > setx BW_PLATFORM_USERNAME "username"
    > setx BW_PLATFORM_PASSWORD "password"

    """

    _base_url = 'https://api.brightwindanalysis.com/platform'
    _ACCESS_TOKEN = {'token': '', 'expires_in': '', 'issued_at': ''}

    @staticmethod
    def _get_token():
        username = utils.get_environment_variable('BW_PLATFORM_USERNAME')
        password = utils.get_environment_variable('BW_PLATFORM_PASSWORD')

        params = {'username': username, 'password': password}
        if not _LoadBWPlatform._ACCESS_TOKEN.get('token') or (_LoadBWPlatform._ACCESS_TOKEN['issued_at'].timestamp()
                                                              + _LoadBWPlatform._ACCESS_TOKEN['expires_in']
                                                              < datetime.datetime.now().timestamp()):
            json_response = requests.post('https://api.brightwindanalysis.com/auth/login', json=params).json()
            if json_response.get('error_description'):
                raise ValueError(json_response['error_description'])
            _LoadBWPlatform._ACCESS_TOKEN['token'] = json_response['access_token']
            _LoadBWPlatform._ACCESS_TOKEN['expires_in'] = json_response['expires_in']
            _LoadBWPlatform._ACCESS_TOKEN['issued_at'] = datetime.datetime.now()

        return _LoadBWPlatform._ACCESS_TOKEN['token']

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
        plants_df = pd.read_json(json.dumps(response_json))
        plants_df['uuid'] = plants_df.id
        plants_df.drop(['id', 'alias', 'connection_details', 'is_location_verified', 'operator_uuid', 'specifications',
                        'trader_uuid'], axis=1, inplace=True)
        plants_df.set_index(['uuid'], inplace=True)
        return plants_df

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
        meas_locs_df = pd.read_json(json.dumps(response_json))
        meas_locs_df['uuid'] = meas_locs_df['id']
        meas_locs_df.drop(['id'], axis=1, inplace=True)
        meas_locs_df.set_index(['uuid'], inplace=True)
        return meas_locs_df

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
    def get_sensor_configs(meas_point_uuid):
        """
        Get all the sensor configurations for a certain measurement point uuid.

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

        :param meas_point_uuid:
        :return:
        """
        access_token = _LoadBWPlatform._get_token()
        headers = {'Authorization': 'Bearer ' + access_token}
        response = requests.get(_LoadBWPlatform._base_url + '/api/sensor-configs', headers=headers, params={
            'measurement_point_uuid': meas_point_uuid
        })
        response_json = response.json()
        if 'Error' in response_json:    # catch if error comes back e.g. measurement_location_uuid isn't found
            raise ValueError(response_json['Error'])
        return response_json

    @staticmethod
    def get_data(measurement_location_uuid, from_date=None, to_date=None):
        """
        Retrieve measurement data from the brightwind platform and return it in a DataFrame with index as Timestamp.

        :param measurement_location_uuid:   The measurement location uuid.
        :type measurement_location_uuid:    str or uuid
        :param from_date:                   Datetime representing the start of the measurement period you want
                                            (included).
        :type from_date:                    datetime or str
        :param to_date:                     Datetime representing the end of the measurement period you want
                                            (not included).
        :type to_date:                      datetime or str
        :return:                            DataFrame with index as a timestamp.
        :rtype:                             pd.DataFrame

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
            to_date = parse(to_date) - datetime.timedelta(seconds=1)

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
    def _get_sen_configs_in_df(meas_points_df):
        # Next we get the relvant information we need from the database to populate the configuration table for the monthly report.

        pddict = {'Sensor OEM': [], 'Units': [], 'Serial Number': [], 'Measurement_point_UUID': [], 'Date From': [],
                  'Date To': []}
        # pddict = {'Units':[],'Measurement_point_UUID':[]}

        limit_counter = 0
        for index, row in meas_points_df.iterrows():
            sen_configs = _LoadBWPlatform.get_sensor_configs(index)

            for sc in sen_configs:
                # print(mp['mounting_arrangement']['height_metres'])
                if sc.get('measurement_point_uuid'):
                    pddict['Measurement_point_UUID'].append(sc['measurement_point_uuid'])
                    # pddict['Measurement Type'].append(mp['measurement_type'])

                    if sc['logger_config'].get('measurement_units'):
                        # Note need to convert m2 symbol so that it can displaued properly in table. This will have to be done for any special units
                        sc['logger_config']['measurement_units'] = sc['logger_config']['measurement_units'].replace('m²', '$m^2$') if '²' in sc['logger_config']['measurement_units'] else sc['logger_config']['measurement_units']
                        pddict['Units'].append(sc['logger_config']['measurement_units'])
                    else:
                        pddict['Units'].append('-')

                    if sc['sensor_info'] and sc['sensor_info'].get('sensor_serial_number'):
                        pddict['Serial Number'].append(sc['sensor_info']['sensor_serial_number'])
                    else:
                        pddict['Serial Number'].append('-')

                    if sc['sensor_info'] and sc['sensor_info'].get('sensor_oem'):
                        pddict['Sensor OEM'].append(sc['sensor_info']['sensor_oem'])
                    else:
                        pddict['Sensor OEM'].append('-')

                    if sc.get('date_from'):
                        pddict['Date From'].append(sc['date_from'])
                    else:
                        pddict['Date From'].append('-')

                    if sc.get('date_to'):
                        pddict['Date To'].append(sc['date_to'])
                    else:
                        pddict['Date To'].append(datetime.datetime.now())

            # sleep(0.2)
            # limit_counter = limit_counter + 1
            # if limit_counter > 1:
            #     break

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
        sen_configs_df = _LoadBWPlatform._get_sen_configs_in_df(meas_points_df)
        sensor_table = meas_points_df.join(sen_configs_df)

        if Include_Tilt_Angle == 'Y':
            sensor_table = sensor_table[['Sensor Name', 'Units', 'Sensor OEM', 'Measurement Type', 'Height [m]',
                                         'Boom Orientation [°]', 'Tilt Angle [°]', 'Serial Number', 'Date From']]
        else:
            sensor_table = sensor_table[['Sensor Name', 'Units', 'Sensor OEM', 'Measurement Type', 'Height [m]',
                                         'Boom Orientation [°]', 'Serial Number', 'Date From']]

        sensor_table = sensor_table.set_index('Sensor Name')
        sensor_table['Date From'] = pd.to_datetime(sensor_table['Date From'])
        sensor_table['Date From'] = sensor_table['Date From'].dt.strftime("%d-%b-%Y")

        sensor_table.reset_index(inplace=True)
        sensor_table.drop(columns=['Measurement Type'], inplace=True)
        sensor_table['Date From'] = pd.to_datetime(sensor_table['Date From'])
        sensor_table.sort_values(by=['Sensor Name', 'Date From'], inplace=True)
        sensor_table['Date From'] = sensor_table['Date From'].dt.strftime("%d-%b-%Y")  # this code is shit!!!
        table = bw_plt.render_table(sensor_table, header_columns=0, col_width=3.3)

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


def load_cleaning_file(filepath, date_from_col_name='Start', date_to_col_name='Stop', dayfirst=False, **kwargs):
    """
    Load a cleaning file which contains a list of sensor names with corresponding periods of flagged data. The timezone
    is removed from the timestamps if it is present.
    This file is a simple comma separated file with the sensor name along with the start and end timestamps for the
    flagged period. There may be other columns in the file however these will be ignores.  E.g.:
    | Sensor |      Start          |       Stop
    ----------------------------------------------------
    | Spd80m | 2018-10-23 12:30:00 | 2018-10-25 14:20:00
    | Dir78m | 2018-12-23 02:40:00 |

    :param filepath:            File path of the file which contains the the list of sensor names along with the start
                                and end timestamps of the periods that are flagged.
    :type filepath: str
    :param date_from_col_name:  The column name of the date_from or the start date of the period to be cleaned.
    :type date_from_col_name:   str, default 'Start'
    :param date_to_col_name:    The column name of the date_to or the end date of the period to be cleaned.
    :type date_to_col_name:     str, default 'Stop'
    :param dayfirst:            If your timestamp starts with the day first e.g. DD/MM/YYYY then set this to true.
                                Pandas defaults to reading 10/11/12 as 2012-10-11 (11-Oct-2012). If True, pandas parses
                                dates with the day first, eg 10/11/12 is parsed as 2012-11-10. More info on
                                pandas.read_csv parameters.
    :type dayfirst:             bool, default False
    :param kwargs:              All the kwargs from pandas.read_csv can be passed to this function.
    :return:                    A DataFrame where each row contains the sensor name and the start and end timestamps of
                                the flagged data.
    :rtype:                     pandas.DataFrame

    **Example usage**
    ::
        import brightwind as bw
        cleaning_file = r'C:\\some\\folder\\cleaning_file.csv'
        cleaning_df = bw.load_cleaning_file(cleaning_file)
        print(cleaning_df)

    """
    if pd.__version__ < '2.0.0':
        date_format = None
    else:
        date_format = 'mixed'

    cleaning_df = _pandas_read_csv(filepath, **kwargs)
    # Issue when the date format is not the same in the full dataset.
    cleaning_df[date_from_col_name] = pd.to_datetime(cleaning_df[date_from_col_name],
                                                     dayfirst=dayfirst, format=date_format).dt.tz_localize(None)
    cleaning_df[date_to_col_name] = pd.to_datetime(cleaning_df[date_to_col_name],
                                                   dayfirst=dayfirst, format=date_format).dt.tz_localize(None)
    return cleaning_df


def apply_cleaning(data, cleaning_file_or_df, inplace=False, sensor_col_name='Sensor', date_from_col_name='Start',
                   date_to_col_name='Stop', all_sensors_descriptor='All', replacement_text='NaN', dayfirst=False):
    """
    Apply cleaning to a DataFrame using predetermined flagged periods for each sensor listed in a cleaning file.
    The flagged data will be replaced with NaN values which then do not appear in any plots or effect calculations.

    This file is a simple comma separated file with the sensor name along with the start and end timestamps for the
    flagged period. There may be other columns in the file however these will be ignores.  E.g.:
    | Sensor |      Start          |       Stop
    ----------------------------------------------------
    | Spd80m | 2018-10-23 12:30:00 | 2018-10-25 14:20:00
    | Dir78m | 2018-12-23 02:40:00 |

    :param data:                    Data to be cleaned.
    :type data:                     pandas.DataFrame
    :param cleaning_file_or_df:     File path of the csv file or a pandas DataFrame which contains the list of sensor
                                    names along with the start and end timestamps of the periods that are flagged.
    :type cleaning_file_or_df:      str, pd.DataFrame
    :param inplace:                 If 'inplace' is True, the original data, 'data', will be modified and and replaced
                                    with the cleaned data. If 'inplace' is False, the original data will not be touched
                                    and instead a new object containing the cleaned data is created. To store this
                                    cleaned data, please ensure it is assigned to a new variable.
    :type inplace: Boolean
    :param sensor_col_name:         The column name which contains the list of sensor names that have flagged periods.
    :type sensor_col_name:          str, default 'Sensor'
    :param date_from_col_name:      The column name of the date_from or the start date of the period to be cleaned.
    :type date_from_col_name:       str, default 'Start'
    :param date_to_col_name:        The column name of the date_to or the end date of the period to be cleaned.
    :type date_to_col_name:         str, default 'Stop'
    :param all_sensors_descriptor:  A text descriptor that represents ALL sensors in the DataFrame.
    :type all_sensors_descriptor:   str, default 'All'
    :param replacement_text:        Text used to replace the flagged data.
    :type replacement_text:         str, default 'NaN'
    :param dayfirst:                If your timestamp starts with the day first e.g. DD/MM/YYYY then set this to true.
                                    Pandas defaults to reading 10/11/12 as 2012-10-11 (11-Oct-2012). If True, pandas
                                    parses dates with the day first, eg 10/11/12 is parsed as 2012-11-10.
                                    More info on pandas.read_csv parameters.
    :type dayfirst:                 bool, default False
    :return:                        DataFrame with the flagged data removed.
    :rtype:                         pandas.DataFrame

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
        cleaning_df = load_cleaning_file(cleaning_file_or_df, date_from_col_name, date_to_col_name, dayfirst=dayfirst)
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
                                replacement_text='NaN', dayfirst=False):
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
    :param dayfirst: If your timestamp starts with the day first e.g. DD/MM/YYYY then set this to true. Pandas defaults
            to reading 10/11/12 as 2012-10-11 (11-Oct-2012). If True, pandas parses dates with the day
            first, eg 10/11/12 is parsed as 2012-11-10. More info on pandas.read_csv parameters.
    :type dayfirst: bool, default False
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
    cleaning_df = load_cleaning_file(windog_cleaning_file, date_from_col_name, date_to_col_name,
                                     dayfirst=dayfirst, sep='\t')

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
