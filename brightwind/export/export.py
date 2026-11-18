import os
import pandas as pd
import numpy as np
import datetime
import re
import warnings
from typing import Optional, Union
import brightwind

__all__ = ['export_tab_file', 'export_csv', 'export_tws_file', 'export_wti_file']

# WindFarmer .wti turbulence file format, DNV WindFarmer user guide:
# https://mysoftware.dnv.com/download/public/renewables/windfarmer/manuals/latest/UserGuide/FileFormats/WTI.html
WINDFARMER_WTI_MIN_SPEED = 0.5  # the first wind speed row is 1 m/s, covering 0.5 to 1.5 m/s
# The user guide does not state a maximum. DNV's example file, M3_100.wti, has rows from 1 to 24 m/s, and WindFarmer
# fails to load a file with rows above 24 m/s ("Illegal file format found - load failed"):
# https://github.com/dnv-opensource/WindFarmer-automation/tree/main/DemoData/Hawaii
WINDFARMER_WTI_MAX_SPEED = 24  # the last wind speed row is 24 m/s, covering 23.5 to 24.5 m/s


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

       
def export_tws_file(
        wspd_series, direction_series, eastings, northings, height, wspd_std_series=None, site_name=None, 
        file_name=None, folder_path=None
        ):
    """
    Export a WindSim timeseries tws file using a timeseries of wind speed, wind direction and optionally 
    standard deviation of wind speed.

    :param wspd_series:         Series of wind speed variable with datetimeindex.
    :type wspd_series:          pandas.Series
    :param direction_series:    Series of wind direction variable between [0-360] with datetimeindex.
    :type direction_series:     pandas.Series
    :param eastings:            Eastings of the measurement location in meters.
    :type eastings:             float
    :param northings:           Northings of the measurement location in meters.
    :type northings:            float
    :param height:              Height that the timeseries represents in meters.
    :type height:               float
    :param wspd_std_series:     Series of wind speed standard deviations with datetimeindex.
    :type wspd_std_series:      pandas.Series
    :param site_name:           The site name to include in the file, or use the default "brightwind_site",
                                i.e 'Demo Data'.
    :type site_name:            str
    :param file_name:           The file name under which the tws file will be saved, or use the default,
                                i.e '2019-06-07_brightwind_tws_export.tws'.
    :type file_name:            str
    :param folder_path:         The directory where the tws file will be saved, default is the working directory.
    :type folder_path:          str
    :return:                    Creates a WindSim timeseries tws file which can be used in the WindSim software.

    **Example Usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)
        data = bw.apply_cleaning(data, bw.demo_datasets.demo_cleaning_file)

        bw.export_tws_file(
            data.Spd40mN, data.Dir38mS, 626100, 827971, 80, data.Spd40mNStd, site_name='Demo Data', file_name='demo.tws'
            )
        bw.export_tws_file(
            data.Spd40mN, data.Dir38mS, 626100, 827971, 80, site_name='Demo Data', file_name='demo_no_std.tws'
            )
    """
        
    
    # Type checks
    names    = ['eastings', 'northings', 'height', 'wspd_series', 'direction_series']
    objs     = [ eastings,   northings,   height,   wspd_series,   direction_series]
    expected = [(int,float), (int,float), (int,float), pd.Series,  pd.Series]

    for name, obj, exp in zip(names, objs, expected):
        if not isinstance(obj, exp):
            raise TypeError(f"'{name}' must be type: {exp}, received: {type(obj).__name__}")
            
    if wspd_std_series is not None and not isinstance(wspd_std_series, pd.Series):         
        raise TypeError(f"'wspd_std_series' must be type: {pd.Series}, received: {type(wspd_std_series).__name__}")
    if len(wspd_series) != len(direction_series):
        raise ValueError("The lengths of 'wspd_series' and 'direction_series' are not the same.")
    if wspd_std_series is not None and len(wspd_series) != len(wspd_std_series):
        raise ValueError("The length of 'wspd_std_series' is not equal to 'wspd_series' and 'direction_series'.")


    # Value checks
    if not direction_series.dropna().between(0, 360).all():
        raise ValueError("direction_series values must be between 0 and 360")
        
    if height not in range(0, 400):
        raise ValueError("height value must be between 0 and 400")
            
    # Optional inputs
    site_name = 'brightwind_site' if site_name is None else site_name

    file_name = str(datetime.datetime.now().strftime("%Y-%m-%d")) + '_brightwind_tws_export.tws' if file_name is None else file_name
    
    folder_path = os.getcwd() if folder_path is None else folder_path

    file_name = file_name + '.tws' if file_name[-4:] != '.tws' else file_name

    include_sd = True if wspd_std_series is not None else False
    
    # Setup
    file_name_print = os.path.splitext(file_name)[0]
    file_path = os.path.join(folder_path, file_name)
    
    local_wspd_series = brightwind.analyse.analyse._convert_df_to_series(wspd_series.copy())
    local_direction_series = brightwind.analyse.analyse._convert_df_to_series(direction_series.copy())
    if include_sd: local_wspd_sd_series = brightwind.analyse.analyse._convert_df_to_series(wspd_std_series.copy())
    
    # Get header info
    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    version = brightwind.__version__
    
    tws_string = f"{file_name_print} created using brightwind version {version} at {current_timestamp}.\n"
    
    # Prepare timeseries output
    tws_output = pd.concat([local_direction_series, local_wspd_series],axis=1, keys=['dir:','speed:'])
    if include_sd: 
        tws_output['SDspeed:'] = local_wspd_sd_series
    
    tws_output = tws_output.dropna()

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
    tws_string += 'site position      : ' + f"{eastings:.0f}" + '    ' + f"{northings:.0f}" + '\n'
    tws_string += 'coordinate system  : 3\n'
    tws_string += 'measurement height : ' + str(height) + '\n\n'
    
    if include_sd:
        tws_string += 'rec nr: year: mon: date: hour: min: dir: speed: SDspeed:\n'
    else:
        tws_string += 'rec nr: year: mon: date: hour: min: dir: speed:\n'
    
    tws_output['speed:'] = tws_output['speed:'].map('{:.3f}'.format)
    if include_sd:
        tws_output['SDspeed:'] = tws_output['SDspeed:'].map('{:.3f}'.format)
    tws_output['dir:'] = tws_output['dir:'].map('{:.1f}'.format)

    # Write to file
    with open(str(file_path), "w") as file:
        file.write(tws_string)
        tws_output.to_csv(file, header=False, sep=" ", mode='a', lineterminator='\n')
    
    # Print success message
    print(f'Export of tws file {file_name} successful.\nEastings: {eastings:.0f} E, Northings: {northings:.0f} N, Height: {height} m\n')


def _format_wti_table(table: pd.DataFrame, value_format: str, row_prefix: str = '') -> str:
    header = '#  \t' + '\t'.join(f'{sector_centre:7.1f}' for sector_centre in table.columns) + '\n'
    rows = ''.join(f'{row_prefix}{speed:2d}\t' + '\t'.join(format(value, value_format) for value in values) + '\n'
                   for speed, values in zip(table.index, table.values))
    return header + rows


def export_wti_file(wspd: Union[pd.Series, pd.DataFrame], wspd_std: Union[pd.Series, pd.DataFrame],
                    wdir: Union[pd.Series, pd.DataFrame], sectors: int = 12, count_threshold: int = 5,
                    file_name: Optional[str] = None, folder_path: Optional[str] = None) -> None:
    """
    Export a WindFarmer .wti turbulence file of mean turbulence intensity (TI) and sigma-sigma by wind speed and
    direction sector, from timeseries of wind speed, wind speed standard deviation and wind direction.

    The file follows the WindFarmer .wti file format described in the DNV WindFarmer user guide:
    https://mysoftware.dnv.com/download/public/renewables/windfarmer/manuals/latest/UserGuide/FileFormats/WTI.html

    The file contains three tab-separated tables, each with wind speed rows and direction sector columns:

        * Mean TI in percent,
        * Sigma-sigma, the standard deviation of the wind speed standard deviation, in the units of wspd_std,
        * Bin counts, the number of TI values in each wind speed bin and direction sector. This table is commented
          out and is for information only.

    The values are derived using bw.TI.by_speed_and_sector(). Wind speed rows are 1 m/s bins centred on whole
    numbers, always from 1 m/s (0.5 to 1.5 m/s) to 24 m/s (23.5 to 24.5 m/s), and TI is calculated for wind speeds
    from 0.5 m/s up to 24.5 m/s. Wind speeds above 24.5 m/s are not used. The user guide does not state a maximum
    wind speed, but WindFarmer fails to load a file with rows above 24 m/s, and the DNV example file, M3_100.wti, in
    the dnv-opensource/WindFarmer-automation repository also has rows from 1 to 24 m/s. The first direction sector is
    centred on 0 degrees.

    Where a wind speed bin and direction sector has fewer than count_threshold TI values, or no value, its mean TI
    and sigma-sigma are replaced by the values for that wind speed bin across all direction sectors. This matches the
    DNV example file. A wind speed bin with no data in any direction sector, below the highest wind speed bin with
    data, takes values interpolated linearly from the neighbouring wind speed bins.

    The highest wind speed bins are usually measured by very few TI values, or none. Wind speed bins above the
    highest bin with at least count_threshold TI values across all direction sectors therefore hold that bin's mean
    TI and sigma-sigma constant, for every direction sector, up to 24 m/s. If no wind speed bin has count_threshold
    TI values, the values of the highest wind speed bin with data are held. The bin counts table always reports the
    measured counts.

    :param wspd:            Wind speed timeseries.
    :type wspd:             pandas.Series or pandas.DataFrame
    :param wspd_std:        Wind speed standard deviation timeseries. Must have the same index as wspd.
    :type wspd_std:         pandas.Series or pandas.DataFrame
    :param wdir:            Wind direction timeseries between [0-360]. Must have the same index as wspd.
    :type wdir:             pandas.Series or pandas.DataFrame
    :param sectors:         Number of direction sectors, evenly spaced with the first centred on 0 degrees. Default
                            is 12.
    :type sectors:          int
    :param count_threshold: Minimum number of TI values in a wind speed bin and direction sector for its own mean TI
                            and sigma-sigma to be used. Below this, the values for that wind speed bin across all
                            direction sectors are used. It also sets the minimum number of TI values across all
                            direction sectors for a wind speed bin not to be replaced by the values held constant
                            from a lower wind speed bin. Default is 5. Set to 0 to only replace cells with no value
                            and only hold values above the highest wind speed bin with data.
    :type count_threshold:  int
    :param file_name:       The file name under which the wti file will be saved, or use the default,
                            i.e '2019-06-07_brightwind_wti_export.wti'. The '.wti' extension is added if missing.
    :type file_name:        str or None
    :param folder_path:     The directory where the wti file will be saved, default is the working directory.
    :type folder_path:      str or None
    :return:                Creates a WindFarmer .wti turbulence file.
    :rtype:                 None

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)
        data = bw.apply_cleaning(data, bw.demo_datasets.demo_cleaning_file)

        # Export a .wti file with 12 direction sectors to the working directory
        bw.export_wti_file(data.Spd80mN, data.Spd80mNStd, data.Dir78mS, file_name='demo_80m.wti')

        # Export a .wti file with 16 direction sectors, using each cell's own values wherever it has at least
        # 10 TI values
        bw.export_wti_file(data.Spd80mN, data.Spd80mNStd, data.Dir78mS, sectors=16, count_threshold=10,
                           file_name='demo_80m_16_sectors')

    """
    if not isinstance(sectors, (int, np.integer)) or isinstance(sectors, bool) or sectors < 1:
        raise ValueError(f"The input `sectors` must be a positive integer, got: {sectors}")
    if not isinstance(count_threshold, (int, np.integer)) or isinstance(count_threshold, bool) or count_threshold < 0:
        raise ValueError(f"The input `count_threshold` must be a non-negative integer, got: {count_threshold}")

    # contiguous 1 m/s rows from 1 to 24 m/s; TI.by_speed_and_sector raises an error if there is no data to bin
    speed_rows = list(range(1, WINDFARMER_WTI_MAX_SPEED + 1))
    ti_dist_kwargs = dict(speed_bin_array=np.arange(WINDFARMER_WTI_MIN_SPEED, WINDFARMER_WTI_MAX_SPEED + 1, 1),
                          speed_bin_labels=speed_rows, min_speed=WINDFARMER_WTI_MIN_SPEED, return_data=True)
    _, ti_by_sector = brightwind.TI.by_speed_and_sector(wspd, wspd_std, wdir, sectors=sectors, **ti_dist_kwargs)
    _, ti_all_sectors = brightwind.TI.by_speed_and_sector(wspd, wspd_std, wdir, direction_bin_array=[0, 360],
                                                          **ti_dist_kwargs)
    ti_by_sector = ti_by_sector.reindex(speed_rows)
    ti_all_sectors = ti_all_sectors.reindex(speed_rows)
    all_sector_counts = ti_all_sectors['TI_Count'].iloc[:, 0].fillna(0)
    ti_all_sectors = ti_all_sectors.interpolate(method='index', limit_direction='both')
    sector_centres = np.arange(sectors) * 360.0 / sectors

    counts = ti_by_sector['TI_Count'].fillna(0).astype('int64')
    counts.columns = sector_centres
    tables = {}
    for stat in ['Mean_TI', 'Sigma_Sigma']:
        table = ti_by_sector[stat].copy()
        table.columns = sector_centres
        all_sectors = ti_all_sectors[stat].iloc[:, 0]
        table = table.mask((counts < count_threshold) | table.isna(), all_sectors, axis=0)
        tables[stat] = table
    tables['Mean_TI'] = tables['Mean_TI'] * 100

    # hold the values of the highest wind speed bin with at least count_threshold TI values across all sectors, or
    # else the highest wind speed bin with data, constant for the higher wind speed bins
    enough_data = all_sector_counts >= max(count_threshold, 1)
    if not enough_data.any():
        enough_data = all_sector_counts > 0
    hold_from_row = enough_data[enough_data].index.max()
    for stat in tables:
        tables[stat].loc[tables[stat].index > hold_from_row] = tables[stat].loc[hold_from_row].values

    if ((tables['Mean_TI'] <= 0) | (tables['Mean_TI'] >= 100)).any().any():
        warnings.warn("Some mean TI values are not greater than 0 % and less than 100 %, as WindFarmer requires. "
                      "Check the wspd and wspd_std inputs.", UserWarning, stacklevel=2)

    if file_name is None:
        file_name = str(datetime.datetime.now().strftime("%Y-%m-%d")) + '_brightwind_wti_export.wti'
    if not file_name.endswith('.wti'):
        file_name = file_name + '.wti'
    if folder_path is None:
        folder_path = os.getcwd()
    file_path = os.path.join(folder_path, file_name)

    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wti_string = (f"# {os.path.splitext(file_name)[0]} created using brightwind version {brightwind.__version__} "
                  f"at {current_timestamp}.\n{sectors}\n\n")
    wti_string += _format_wti_table(tables['Mean_TI'], '7.4f')
    wti_string += '\n#Sigma Sigma\n' + _format_wti_table(tables['Sigma_Sigma'], '7.4f')
    wti_string += '\n#Bin Counts\n' + _format_wti_table(counts, '7d', row_prefix='#')

    with open(file_path, 'w', newline='\n') as file:
        file.write(wti_string)
    held = f'values held constant above {hold_from_row} m/s' if hold_from_row < speed_rows[-1] else 'no values held'
    print(f'Export of wti file "{file_name}" successful.\nWind speed rows: 1 to {speed_rows[-1]} m/s, {held}, '
          f'direction sectors: {sectors}\n')


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
