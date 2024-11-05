import os
import pandas as pd
import datetime
import re
import brightwind

__all__ = ['export_tab_file', 'export_csv', 'export_tws_file']


def _calc_mean_speed_of_freq_tab(freq_tab):
    """
    Calculate the mean wind speed derived from a frequency table.

    :param freq_tab: Frequency distribution by wind speed and direction sector. Data output from the bw.freq_table().
    :return:
    """
    local_freq_tab = freq_tab.copy()
    local_freq_tab.index = [(interval.right + interval.left)/2 for interval in local_freq_tab.index]
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
        df = bw.load_campbell_scientific(bw.demo_datasets.demo_campbell_scientific_site_data)
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
    if len(speed_interval) != 1:
        import warnings
        warnings.warn("All speed bins not of equal lengths")
    speed_interval = speed_interval.pop()

    local_freq_tab.index = [interval.right for interval in local_freq_tab.index]

    mean_wind_speed = _calc_mean_speed_of_freq_tab(freq_tab)
    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    version = brightwind.__version__
    sectors = len(local_freq_tab.columns)

    tab_string = "{0} created using brightwind version {1} at {2}. Mean wind speed derived from this tab file is: " \
                 "{3} m/s.\n{4} {5} {6}\n " \
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
    print('Export of tab file "{0}" successful.\nMean wind speed derived from this tab file is: ' 
          '{1} m/s.\nLatitude: {2}N, Longitude: {3}E, Height: {4} m\n'
          ''.format(str(file_name), "{:.3f}".format(mean_wind_speed),
                    str(lat), str(long), str(height)))


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
        df = bw.load_campbell_scientific(bw.demo_datasets.demo_campbell_scientific_site_data)
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


def export_tws_file(eastings, northings, height, wspd_series, direction_series, wspd_std_series=None, site_name=None, file_name=None, folder_path=None):
    """Export a WindSim timeseries tws file using a timeseries of wind speed and direction.

    :param eastings: Eastings of the measurement location in meters.
    :type eastings: float
    :param northings: Northings of the measurement location in meters.
    :type northings: float
    :param height: Height that the timeseries represents in meters.
    :type height: float
    :param wspd_series: Series of wind speed variable.
    :type wspd_series: pandas.Series
    :param direction_series: Series of wind directions between [0-360].
    :type direction_series: pandas.Series
    :param wspd_std_series: Series of wind speed standard deviations.
    :type wspd_std_series: pandas.Series
    :param site_name: The site name to include in the file, or use the default "brightwind_site",
                      i.e 'Demo Data'.
    :type site_name: str
    :param file_name: The file name under which the tab file will be saved, or use the default,
                      i.e '2019-06-07_brightwind_tws_export.tws'.
    :type file_name: str
    :param folder_path: The directory where the tab file will be saved, default is the working directory.
    :type folder_path: str

    :return: Creates a WindSim timeseries tws file which can be used in the WindSim software.

    **Example Usage**
    ::
        import brightwind as bw
        df = bw.load_campbell_scientific(bw.demo_datasets.demo_campbell_scientific_data)

        bw.export_tws_file(626100, 827971, 80, df.Spd80mN, df.Dir78mS, site_name='Demo Data', file_name='campbell_tws_file')"""
    
    if site_name is None:
        site_name = 'brightwind_site'
    
    if file_name is None:
        file_name = str(datetime.datetime.now().strftime("%Y-%m-%d")) + '_brightwind_tws_export.tws'

    if folder_path is None:
        folder_path = os.getcwd()

    if '.tws' not in file_name:
        file_name = file_name + '.tws'

    include_sd = True if wspd_std_series is not None else False
    
    file_name_print = os.path.splitext(file_name)[0]
    file_path = os.path.join(folder_path, file_name)

    eastings, northings = int(rouund(eastings,0)), int(rouund(northings,0))
    
    local_wspd_series = brightwind.analyse.analyse._convert_df_to_series(wspd_series.copy())
    local_direction_series = brightwind.analyse.analyse._convert_df_to_series(direction_series.copy())
    if include_sd: local_wspd_sd_series = brightwind.analyse.analyse._convert_df_to_series(wspd_std_series.copy())

    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    version = brightwind.__version__
    
    tws_string = f"{str(file_name_print)} created using brightwind version {version} at {current_timestamp}.\n"

    tws_output = pd.concat([round(local_direction_series,4),round(local_wspd_series,4)],axis=1, keys=['dir:','speed:']).dropna()
    if include_sd: tws_output = pd.concat([round(local_direction_series,4),round(local_wspd_series,4), round(local_wspd_sd_series,4)],axis=1, keys=['dir:','speed:','SDspeed:']).dropna()

    tws_output['rec nr:'] = range(1, len(tws_output)+1)
    start_id, end_id = tws_output.index[0].strftime('%B %y'), tws_output.index[-1].strftime('%B %y')
    tws_output['year:'], tws_output['mon:'], tws_output['date:'] = tws_output.index.strftime('%Y'), tws_output.index.strftime('%m'), tws_output.index.strftime('%d')
    tws_output['hour:'], tws_output['min:'] = tws_output.index.strftime('%H'), tws_output.index.strftime('%M')
    tws_output.set_index('rec nr:', inplace=True)

    if include_sd:
        tws_output = tws_output[['year:','mon:','date:','hour:','min:','dir:','speed:', 'SDspeed:']]
    else:
        tws_output = tws_output[['year:','mon:','date:','hour:','min:','dir:','speed:']]
    
    tws_string += 'version            : 48\n'
    tws_string += 'site name          : ' + site_name + '\n'
    tws_string += 'measurement period : ' + start_id + ' - ' + end_id + '\n'
    tws_string += 'site position      : ' + str(eastings) + '    ' + str(northings) + '\n'
    tws_string += 'coordinate system  : 3\n'
    tws_string += 'measurement height : ' + str(height) + '\n\n'
    
    if include_sd:
        tws_string += 'rec nr: year: mon: date: hour: min: dir: speed: SDspeed:\n'
    else:
        tws_string += 'rec nr: year: mon: date: hour: min: dir: speed:\n'
    
    with open(str(file_path), "w") as file:
        file.write(tws_string)
        tws_output.to_csv(file, header=False, sep=" ", mode='a', lineterminator='\n')
    
    print(f'Export of tws file "{str(file_name)}" successful.\nEastings: {eastings}E, Northings: {northings}N, Height: {str(height)} m\n')
