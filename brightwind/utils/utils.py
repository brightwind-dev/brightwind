import numpy as np
import pandas as pd
import os

__all__ = ['slice_data']


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


def slice_data(data, date_from: str='', date_to: str=''):
    """
    Returns the slice of data between the two date ranges,
    Date format: YYYY-MM-DD
    """
    import datetime
    date_from = pd.to_datetime(date_from, format="%Y-%m-%d")
    date_to = pd.to_datetime(date_to, format="%Y-%m-%d")

    if pd.isnull(date_from):
        date_from = data.index[0]

    if pd.isnull(date_to):
        date_to = data.index[-1]

    if date_to < date_from:
        raise ValueError('date_to must be greater than date_from')

    return data.loc[date_from:date_to, :]

    # if (isinstance(date_from, datetime.date) or isinstance(date_from, datetime.datetime)) \
    #         and (isinstance(date_to, datetime.date) or isinstance(date_to, datetime.datetime)):
    #     sliced_data = data.loc[date_from:date_to, :]
    # elif date_from and date_to:
    #     import datetime as dt
    #     date_from = dt.datetime.strptime(date_from[:10], "%Y-%m-%d")
    #     date_to = dt.datetime.strptime(date_to[:10], "%Y-%m-%d")
    #     sliced_data = data.loc[date_from:date_to, :]
    # else:
    #     return data
    # return sliced_data


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



