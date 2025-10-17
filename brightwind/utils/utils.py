import numpy as np
import pandas as pd
import os
import json
from jsonschema import Draft7Validator
from typing import Union

__all__ = ['slice_data',
           'validate_coverage_threshold',
           'is_file',
           'is_file_extension',
           'validate_json',
           'linear_transform']


def _range_0_to_360(direction):
    if direction < 0:
        return direction+360
    elif direction > 360:
        return direction % 360
    else:
        return direction


def get_direction_bin_array(sectors):
    bin_start = 180.0/sectors
    direction_bins = np.arange(bin_start, 360, 360.0/sectors)
    direction_bins = np.insert(direction_bins, 0, 0)
    direction_bins = np.append(direction_bins, 360)
    return direction_bins


def _get_dir_sector_mid_pts(sector_idx):
    """Accepts a list of direction sector as strings and returns a list of
    mid points for that sector of type float
    """
    sectors = [idx.split('-') for idx in sector_idx]
    sector_mid_pts = []
    for sector in sectors:
        sector[0] = float(sector[0])
        sector[1] = float(sector[1])
        if sector[0] > sector[1]:
            mid_pt = ((360.0 + sector[0]+sector[1])/2.0) % 360
        else:
            mid_pt = 0.5*(sector[0]+sector[1])
        sector_mid_pts.append(mid_pt)
    return sector_mid_pts


def validate_coverage_threshold(coverage_threshold):
    """
    Validate that coverage_threshold is between 0 and 1 and if it is None set to zero.

    :param coverage_threshold: Should be number between or equal to 0 and 1.
    :type coverage_threshold:  float, int or None
    :return:                   coverage_threshold
    :rtype:                    float or int
    """
    coverage_threshold = 0 if coverage_threshold is None else coverage_threshold
    if coverage_threshold < 0 or coverage_threshold > 1:
        raise TypeError("Invalid coverage_threshold, this should be between or equal to 0 and 1.")
    return coverage_threshold


def slice_data(data, date_from=None, date_to=None):
    """
    Returns the slice of data between the two date or datetime ranges.

    :param data:        Pandas DataFrame or Series with timestamp as index.
    :type data:         pandas.DataFrame or pandas.Series
    :param date_from:   Start date as string in format YYYY-MM-DD or YYYY-MM-DD hh:mm. Start date is included in the
                        sliced data. If format of date_from is YYYY-MM-DD, then the first timestamp of the date is used
                        (e.g if date_from=2023-01-01 then 2023-01-01 00:00 is the first timestamp of the sliced data).
                        If date_from is not given then sliced data are taken from the first timestamp of the dataset.
    :type:              str
    :param date_to:     End date as string in format YYYY-MM-DD or YYYY-MM-DD hh:mm. End date is not included in the
                        sliced data. If format date_to is YYYY-MM-DD, then the last timestamp of the previous day is
                        used (e.g if date_to=2023-02-01 then 2023-01-31 23:50 is the last timestamp of the sliced data).
                        If date_to is not given then sliced data are taken up to the last timestamp of the dataset.
    :type:              str
    :returns:           Sliced data
    :rtype:             pandas.Dataframe or pandas.Series

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)

        # Return the slice of data between two input datetimes
        data_sliced = bw.utils.utils.slice_data(DATA, date_from='2016-11-23 00:30', date_to='2017-10-23 12:20')

        # Return the slice of data between two input dates
        data_sliced = bw.utils.utils.slice_data(DATA, date_from='2016-11-23', date_to='2017-10-23')

        # Return the slice of data from an input date up to the end of the dataset.
        data_sliced = bw.utils.utils.slice_data(DATA, date_from='2016-11-23')

    """
    if pd.__version__ < '2.0.0':
        date_format = "%Y-%m-%d %H:%M"
    else:
        date_format = 'ISO8601'

    if pd.isnull(date_from):
        date_from = data.index[0]
    else:
        date_from = pd.to_datetime(date_from, format=date_format)

    if pd.isnull(date_to):
        date_to = data.index[-1]
    else:
        date_to = pd.to_datetime(date_to, format=date_format)

    if date_to < date_from:
        raise ValueError('date_to must be greater than date_from')

    if date_to == data.index[-1]:
        return data[(data.index >= date_from)]
    else:
        return data[(data.index >= date_from) & (data.index < date_to)]


def is_float_or_int(value):
    """
    Returns True if the value is a float or an int, False otherwise.
    :param value:
    :return:
    """
    if type(value) is float:
        return True
    elif type(value) is int:
        return True
    else:
        return False


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


def get_environment_variable(name):
    if name not in os.environ:
        raise Exception('{} environmental variable is not set.'.format(name))
    return os.getenv(name)


def bold(text):
    """
    Function to return text as bold

    :param text: str to bold
    :type text: str
    :return: str in bold
    """
    return '\x1b[1;30m'+text+'\x1b[0m' if text else '\x1b[1;30m'+'\x1b[0m'


def is_file(file_or_folder):
    """
    Returns True is file_or_folder is a file.

    :param file_or_folder:  The file or folder path.
    :type file_or_folder:   str
    :return:                If is a file.
    :rtype:                 bool
    """
    if os.path.isfile(file_or_folder):
        return True
    elif os.path.isdir(file_or_folder):
        return False
    else:
        raise FileNotFoundError("File or folder doesn't seem to exist.")


def is_file_extension(file_or_folder, extension_required):
    """
    Returns True if file_or_folder is a file of the desired extension type.

    :param file_or_folder:      The file or folder path.
    :type file_or_folder:       str
    :param extension_required:  The file extension needed.
    :type extension_required:   str
    :return:                    If is a file with desired extension type.
    :rtype:                     bool
    """
    if is_file(file_or_folder):    
        _, extension = os.path.splitext(file_or_folder)
        if extension.lower() == extension_required.lower():
            return True
        else:
            raise ValueError(f"File extension must be {extension_required}, got: {extension}")
    else:
        raise ValueError(f"Input must be a {extension_required} file, got: {file_or_folder}")


def validate_json(json_to_check, schema):
    """
    Validates JSON data against a JSON schema.

    :param json_to_check:   The JSON data to validate.
    :type json_to_check:    dict
    :param schema:          The JSON schema to validate against.
    :type schema:           str | dict
    :return:                List of validation results, each containing:
                                - item_index (int): Index of the item in the list or 0 if single item
                                - is_valid (bool): True if validation passes, False otherwise
                                - error_message (str): Error message if validation fails, empty string otherwise
    :rtype:                 bool

    **Example usage**
    ::
        import brightwind as bw
        import json

        with open(bw.demo_datasets.demo_cleaning_rules_file) as file:
            cleaning_json = json.load(file)
        bw.utils.utils.validate_json(cleaning_json[0], bw.load.cleaning_rules_schema)
    """
    if isinstance(schema, str):
        if is_file(schema):
            with open(schema) as file:
                schema = json.load(file)
    elif not isinstance(schema, dict):
        raise ValueError("Incorrect schema type used, this must be a str or a dict.")
    
    errors = []
    validator = Draft7Validator(schema)
    for error in validator.iter_errors(json_to_check):
        error_path = " → ".join(str(path) for path in error.path) if error.path else "root"
        error_detail = {
            "path": error_path,
            "message": error.message,
            "schema_path": " → ".join(str(path) for path in error.schema_path)
        }
        errors.append(error_detail)
        
    data_is_valid = len(errors) == 0
    if not data_is_valid:
        print(f"Total of {len(errors)} errors.\n")
        for error in reversed(errors):
            print(f"Validation error at path: {error.get('path')}")
            print(f"Error message: {error.get('message')}")
            print(f"Failed schema part: {error.get('schema_path')}\n")
    
    return data_is_valid


def linear_transform(x_target : Union[float, int, np.ndarray, pd.Series],
                     x_ref : Union[float, int, np.ndarray, pd.Series],
                     y_ref : Union[float, int, np.ndarray, pd.Series],
                     slope : Union[float, int]) -> Union[float, int, np.ndarray, pd.Series]:
    """
    Calculate y_target value(s) based on the input x_target value(s) where a point or array of points (x_ref, y_ref) 
    and the slope of the line are known.

    Function applies a linear transformation based on the equation of a straight line in point-slope form below:

        y = m(x - x1) + y1 where m is the slope and (x1, y1) a known point on the line.
    
    :param x_target:    target x value(s) at which to calculate y_target
    :type x_target:     float or int or numpy.ndarray or pandas.Series
    :param x_ref:       reference x value(s) of known reference point(s) on the line
    :type x_ref:        float or int or numpy.ndarray or pandas.Series
    :param y_ref:       reference y value(s) of known reference point(s) on the line
    :type y_ref:        float or int or numpy.ndarray or pandas.Series
    :param slope:       slope of the line equal to (y_target - y_ref) / (x_target - x_ref) where 
                        ref and target x and y are any two points on the line
    :type slope:        float or int
    :return:            value(s) of y_target at specified x_target
    :rtype:             float or int or numpy.ndarray or pandas.Series

        **Example usage**
    ::
    import brightwind as bw
    
    data = bw.load_csv(bw.demo_datasets.demo_data)

    # calculate y_target for x_target = 10 where (x_ref, y_ref) = (5, 10) is a point on the line and the slope is -0.5.
    bw.utils.utils.linear_transform(x_target=10, x_ref=5, y_ref=10, slope=-0.5)
    # 7.5

    # calculate y_target for x_target as a pandas.Series and (x_ref, y_ref) = (2, 20) is a point on the line
    # and the slope is -0.0065.
    bw.utils.utils.linear_transform(
        x_target=data.T2m.loc['2016-01-09 17:10':'2016-01-09 18:00'],
        x_ref=2,
        y_ref=20,
        slope=-0.0065)

    # calculate y_target for x_target and x_ref as a int and y_ref as a np.ndarray where (x_ref, y_ref) = (2, y_ref)
    # are points on the line and the slope is -0.0065.
    bw.utils.utils.linear_transform(
        x_target=20,
        x_ref=2,
        y_ref=data.T2m.loc['2016-01-09 17:10':'2016-01-09 18:00'].values,
        slope=-0.0065)
    """
    # check input types
    for var, var_name in zip([x_ref, y_ref, x_target], ['x_ref', 'y_ref', 'x_target']):
        if not (isinstance(var, float) or isinstance(var, int) or isinstance(var, pd.Series) or 
                isinstance(var, np.ndarray)):
            raise TypeError(f"{var_name} must be a float or int or a numpy.ndarray or pandas.Series.")
        
    if not (isinstance(slope, float) or isinstance(slope, int)):
        raise TypeError("slope must be a float.")

    # check dimensions of x_ref and x_target if arrays or Series
    if (isinstance(x_ref, np.ndarray) or isinstance(x_ref, pd.Series)) and (
        isinstance(x_target, np.ndarray) or isinstance(x_target, pd.Series)):
        if len(x_ref) != len(x_target):
            raise ValueError("x_ref and x_target must have the same dimensions.")

    # check dimensions of y_ref if arrays or Series
    if isinstance(x_target - x_ref, np.ndarray) or isinstance(x_target - x_ref, pd.Series):
        if isinstance(y_ref, np.ndarray) or isinstance(y_ref, pd.Series):
            if len(y_ref) != len(x_target - x_ref):
                raise ValueError("y_ref must have the same dimensions as x_target or x_ref.")

    y_target = slope*(x_target - x_ref) + y_ref
    
    return y_target
