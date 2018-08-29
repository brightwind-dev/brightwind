""" Retrieve MERRA-2 data from the merra-two api on the Lightsail instance where data is stored in a Cassandra table.
    The api used is the one which takes in the latitude, longitude, from date and to date. These variables need to be
    set at the start of the python file. The retrieved file will be placed in the same folder as this python file.
"""
import requests
from datetime import datetime
import pandas as pd
from urllib.request import urlopen
import json
from typing import List, Dict
from transform.transform import _average_data_by_period
#import netCDF4
import numpy as np
import math
import os


class reanalysis:
    """Class defining the attributes of data from a reference site MERA or MERRA2
    :param: latitude for storing latitude
    :param: longitude for storing longitude
    :param: data for storing Pandas dataframe containing data
    :param: source can be MERA or MERRA2"""
    def __init__(self, latitude, longitude, data, source):
        self.latitude = latitude
        self.longitude = longitude
        self.data = data
        self.source = source


def extract_mera_grib_to_txt(lat: str, long: str, height: int, dest_folder: str, mera_data_src: str):
    """Extracts MERA data from a GRIB format into text files for 4 nearest grid points. You need to run this function
    on a linux machine with grib_api installed (as grib_api currently works only on linux based machines).
    :param: lat : latitude of the site
    :param: long: longitude of the site (range of longitudes in MERA is from 0 to 360)
    :param: height: the height for which to extract u and v components for can be one of the following[10, 50, 100, 200]
    :param: dest_folder: path to the folder where extracted txt files are saved
    :param: mera_data_src: path to the folder containing MERA data
    Creates a folder with name Site_lat<lat>_long<long> in the dest_folder and extracts data in subfolders with names <variable_subfolder>_processed
    """
    import subprocess
    import os
    import time
    if height <100:
        height = '0'+str(height)
    else: height = str(height)
    variable_subfolders = ['u'+height, 'v'+height]
    folders = os.listdir(mera_data_src)
    for variable_subfolder in variable_subfolders:
        if variable_subfolder in folders:
            folder_path = os.path.join(mera_data_src, variable_subfolder)
            files = os.listdir(folder_path)
            begin = time.time()
            if(not os.path.exists(os.path.join(dest_folder,"Site_lat"+lat+"_long"+long))):
                os.makedirs(os.path.join(dest_folder,"Site_lat"+lat+"_long"+long))
            destination_folder = os.path.join(dest_folder,"Site_lat"+lat+"_long"+long,variable_subfolder+"_processed")
            os.makedirs(destination_folder)
            for file in files:
                if (file.endswith('ANALYSIS')):
                    print((len(os.listdir(destination_folder)) / len(files) * 200.0), "% processed")
                    file_path = os.path.join(mera_data_src, folder_path, file)
                    with open(os.path.join(destination_folder, file + '.txt'), 'w+') as f:
                        subprocess.call(args=['grib_ls', '-p', 'mars.date,mars.time', '-l '+lat+','+long+',4', file_path],
                                        stdout=f)
            end = time.time()
            print("Time taken (minutes)", (time.time() - begin) / 60.0)
        else:
            print("INVALID height.")


def mera_date_parser(date: str, t: str):
    """Date parser for MERA data. Can be passed to pandas.read_csv
        mars.date   mars.time
        19811101    0
        19811101    300
    """
    import datetime as dt
    day  = dt.datetime.strptime(date[:8],"%Y%m%d")
    if t=='0':
        t = dt.datetime.strptime('00', "%H")
    elif t=='100':
        t = dt.datetime.strptime('01', "%H")
    elif t=='200':
        t = dt.datetime.strptime('02', "%H")
    elif t=='300':
        t = dt.datetime.strptime('03', "%H")
    elif t=='400':
        t = dt.datetime.strptime('04', "%H")
    elif t=='500':
        t = dt.datetime.strptime('05', "%H")
    elif t=='600':
        t = dt.datetime.strptime('06', "%H")
    elif t=='700':
        t = dt.datetime.strptime('07', "%H")
    elif t=='800':
        t = dt.datetime.strptime('08', "%H")
    elif t=='900':
        t = dt.datetime.strptime('09', "%H")
    elif t=='1000':
        t = dt.datetime.strptime('10', "%H")
    elif t=='1100':
        t = dt.datetime.strptime('11', "%H")
    elif t=='1200':
        t = dt.datetime.strptime('12', "%H")
    elif t=='1300':
        t = dt.datetime.strptime('13', "%H")
    elif t=='1400':
        t = dt.datetime.strptime('14', "%H")
    elif t=='1500':
        t = dt.datetime.strptime('15', "%H")
    elif t=='1600':
        t = dt.datetime.strptime('16', "%H")
    elif t=='1700':
        t = dt.datetime.strptime('17', "%H")
    elif t=='1800':
        t = dt.datetime.strptime('18', "%H")
    elif t=='1900':
        t = dt.datetime.strptime('19', "%H")
    elif t=='2000':
        t = dt.datetime.strptime('20', "%H")
    elif t=='2100':
        t = dt.datetime.strptime('21', "%H")
    elif t=='2200':
        t = dt.datetime.strptime('22', "%H")
    elif t=='2300':
        t = dt.datetime.strptime('23', "%H")
    elif t=='2400':
        t = dt.datetime.strptime('24', "%H")
    else:
        t = dt.datetime.strptime('00', "%H")
    return dt.datetime.combine(day.date(), t.time())


def get_mera_4nearest_dataframe(mera_txt_data_src: str, save_csv: bool=True)-> list:
    """Reads the extracted text files and returns a list of objects containing data and other attributes described below:
    :param: mera_txt_data_src: folder containing extracted MERA files for a site
    :returns: a list of 4 reanalysis objects with the following attributes:
                    latitude, longitude, data, source"""
    variables = os.listdir(mera_txt_data_src)
    #If csv files are already present ignore them
    variables = [variable for variable in variables if(not variable.endswith('.csv'))]
    reanalysis_objs = []
    for location in [1, 2, 3, 4]:
        for v, variable in enumerate(variables):
            months = os.listdir(os.path.join(mera_txt_data_src, variable))
            if (months[0] == 'desktop.ini'):
                months = months[1:]
            for m, month in enumerate(months):
                #print("Processing file:", os.path.join(mera_txt_data_src, variable, month))
                df_tmp = pd.read_csv(os.path.join(mera_txt_data_src, variable, month), sep=" ", skipinitialspace=True,
                                     header=1, skiprows=0, skipfooter=10,
                                     parse_dates=[[0, 1]], infer_datetime_format=True, date_parser=mera_date_parser,
                                     usecols=[0, 1, int(location + 1)], engine='python')
                df_tmp.columns = ['Timestamp', variable[:-10]]
                df_tmp = df_tmp.set_index('Timestamp')
                if m == 0:
                    file = os.path.join(mera_txt_data_src, variable, month)
                    with open(file, 'r') as f:
                        node_info = f.readlines()[-4:][location - 1]
                    node_info = node_info.split(' ')
                    node_lat = [line.split('=')[1] for line in node_info if line.startswith('latitude=')]
                    node_long = [line.split('=')[1] for line in node_info if line.startswith('longitude=')]
                    df_var = df_tmp.copy()
                else:
                    df_var = pd.concat([df_var, df_tmp])
            if v == 0:
                df_location = df_var.copy()
            else:
                df_location = pd.concat([df_location, df_var], axis=1)
        heights = set([column[1:] for column in df_location.columns])
        for height in heights:
            direction = 270.0 - (np.arctan2(df_location['u' + height], df_location['v' + height]) * (180.0 / math.pi))
            direction[direction > 360] = direction[direction > 360] - 360.0
            df_location[height + "m_wind_direction"] = direction
            df_location[height + "m_wind_speed"] = np.power(np.power(df_location['u' + height], 2) + np.power(df_location['v' + height], 2),
                                                     0.5)
            df_location = df_location.drop(['u' + height, 'v' + height], axis=1)
        if save_csv:
            df_location.to_csv(os.path.join(mera_txt_data_src,"Node_lat"+node_lat[0]+"_long"+node_long[0]+".csv"))
        reanalysis_objs.append(reanalysis(node_lat, node_long, df_location, "MERA"))
    #os.remove(mera_txt_data_src)
    return (reanalysis_objs)


def get_merra2_save_csv(lat: str, long: str, from_date: str, to_date: str) -> str:
    """ Return the filename when the MERRA-2 data is retrieved from the LightSail server through the api
        and saved to a csv file.
    """
    print(str(datetime.now()) + " - Run start.")
    apiurl = "http://52.16.60.214:3306/merra/" + lat + "/" + long + "/" + from_date + "/" + to_date
    print(str(datetime.now()) + " - " + apiurl)

    response = requests.get(apiurl)
    data = response.json()

    start_date: datetime = datetime.strptime(data["from"], '%Y-%m-%dT%H:%M:%S.%fZ')
    print("Start date: " + start_date.strftime('%Y-%m-%d'))
    end_date: datetime = datetime.strptime(data["to"], '%Y-%m-%dT%H:%M:%S.%fZ')
    print("End date: " + end_date.strftime('%Y-%m-%d'))
    print("LocationId: " + str(data["locationId"]))

    filename: str = 'MERRA-2_' + lat.replace('.', '-') + "N_" + long.replace('.', '-') + "E_" + \
                    start_date.strftime('%Y-%m-%d') + "_" + end_date.strftime('%Y-%m-%d') + ".csv"
    print("Outputed filename: " + filename)
    with open(filename, 'w') as the_file:
        the_file.write('DateTime,WS50m_m/s,WD50m_deg,T2M_degC,PS_hPa\n')
        for item in data["timeseriesData"]:
            timestamp = datetime.strptime(item["DateTime"], '%Y-%m-%dT%H:%M:%S.%fZ')
            # print(timestamp.strftime('%Y-%m-%d %H:%M:%S') + "," + str(item["WS50m_ms"]) + "," +
            #       str(item["WD50m_deg"]) + "," + str(item["T2M_degC"]) + "," + str(item["PS_hPa"]))
            the_file.write(timestamp.strftime('%Y-%m-%d %H:%M:%S') + "," + str(item["WS50m_ms"]) + "," +
                           str(item["WD50m_deg"]) + "," + str(item["T2M_degC"]) + "," + str(item["PS_hPa"]) + "\n")

    print(str(datetime.now()) + " - Finished")
    return filename


def get_merra2_dataframe(lat: str, long: str, from_date: str, to_date: str) -> reanalysis:
    """ Return a reanalysis object containing the MERRA-2 data retrieved from the LightSail server through the api with the following parameters:
                latitude, longitude, data, source=MERRA2.
    """
    print(str(datetime.now()) + " - Run start for get_merra_2_dataframe.")
    apiurl: str = "http://52.16.60.214:3306/merra/" + lat + "/" + long + "/" + from_date + "/" + to_date
    print(str(datetime.now()) + " - " + apiurl)

    response = urlopen(apiurl).read()
    data_json = json.loads(response)
    data_df: pd.DataFrame = pd.DataFrame(data_json["timeseriesData"])
    data_df.set_index('DateTime', inplace=True)
    data_df.index = pd.to_datetime(data_df.index)

    print(str(datetime.now()) + " - Finished")
    data = reanalysis(lat,long, data_df, "MERRA2")
    return data


def calc_distance(lat1, long1, lat2, long2):
    lat1, long1, lat2, long2 = float(lat1), float(long1), float(lat2), float(long2)
    return math.sqrt((lat2 - lat1)**2 + (long2 - long1)**2)


def get_merra2_nearest_nodes(lat: str, long: str, from_date: str, to_date: str, nearest_4: bool=True) ->\
        List[reanalysis]:
    """Returns a list of 4 reanalysis objects for 4 locations nearest to the one passed in parameters. A date range
    can also be used to get within a specified period.
    :param: lat: Latitude of the site
    :param: long: Longitude of the site
    :param: from_data: Start date for he dataset in format YYYY-MM-DD
    :param: to_date: End date for data in the format YYYY-MM-DD
    :param
    :returns: A list of reanalysis objects"""
    import pymysql.cursors

    reanalysis_objs = []
    serverBrightData = "52.45.243.214";
    serverPort = 3306;
    dbBrightData = "brightdata_db";
    userBrightData = "brightdata_app";
    passBrightData = "GaLe-GwEeHa_2";
    query = ("SELECT * FROM (" +
             "SELECT * " +
             "FROM locations " +
             "WHERE Latitude >= {latitude} AND Longitude < {longitude} AND ReanalysisId=2 " +
             "ORDER BY Latitude, Longitude DESC " +
             "LIMIT 1 " +
             ") AS x " +
             "UNION " +
             "SELECT * FROM ( " +
             "SELECT *  " +
             "FROM locations  " +
             "WHERE Latitude >= {latitude} AND Longitude >= {longitude} AND ReanalysisId=2 " +
             "ORDER BY Latitude, Longitude " +
             "LIMIT 1 " +
             ") AS x " +
             "UNION " +
             "SELECT * FROM ( " +
             "SELECT *  " +
             "FROM locations  " +
             "WHERE Latitude < {latitude} AND Longitude >= {longitude} AND ReanalysisId=2 " +
             "ORDER BY Latitude DESC, Longitude " +
             "LIMIT 1 " +
             ") AS x " +
             "UNION " +
             "SELECT * FROM ( " +
             "SELECT *  " +
             "FROM locations  " +
             "WHERE Latitude < {latitude} AND Longitude < {longitude} AND ReanalysisId=2 " +
             "ORDER BY Latitude DESC, Longitude DESC " +
             "LIMIT 1 " +
             ") AS x").format(latitude=lat, longitude=long)
    connection = pymysql.connect(host=serverBrightData, user=userBrightData, password=passBrightData,
                                 database=dbBrightData, port=serverPort)
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            query_output = cursor.fetchall()
    finally:
        connection.close()
    if nearest_4:
        for location in query_output:
            location_lat = str(location[-2])
            location_long = str(location[-1])
            reanalysis_objs.append(get_merra2_dataframe(location_lat,location_long, from_date, to_date))
    else:
        nearest_dist = 1e8
        for location in query_output:
            location_lat = str(location[-2])
            location_long = str(location[-1])
            distance = calc_distance(location_lat, location_long, lat, long)
            if distance < nearest_dist:
                nearest_dist = distance
                nearest_lat = location_lat
                nearest_long = location_long
        reanalysis_objs.append(get_merra2_dataframe(nearest_lat, nearest_long, from_date, to_date))
    return reanalysis_objs


def import_merra2_from_brightwind_csv(filename: str) -> pd.DataFrame:
    """ Return a dataframe with the MERRA-2 data from the csv read and appended to it.
    """
    temp_df: pd.DataFrame = pd.read_csv(filename, header=0, index_col=0, parse_dates=True)
    return temp_df


def add_index(df: pd.DataFrame) -> pd.DataFrame:
    """ Return a dataframe with an Index column added to the dataframe provided.
        Index column will be an average of all the columns in a row.
    """
    df['Index'] = df.mean(axis=1)
    return df


def merra2_monthly_means_index_momm_to_csv(nodes: List[Dict[str, str]], date_from: str,
                                           date_to: str, column: str='WS50m_ms'):
    """ gets MERRA-2 data for a number of nodes and exports the monthly means and mean of monthly means into csv
        files. The default column is the wind speed column but other columns can be specified.
    """
    merra2_dfs_list: List[pd.DataFrame] = []
    for node in nodes:
        merra2_dfs_list.append(get_merra2_dataframe(node['latitude'], node['longitude'], date_from, date_to).data)

    # merge wind speeds from all dataframes into one dataframe
    merra2_merged_df: pd.DataFrame = pd.DataFrame()
    for df_number, df in enumerate(merra2_dfs_list):
        df = df[[column]]
        df.columns = [column + '_' + str(df_number)]
        if df_number == 0:
            merra2_merged_df = df
        else:
            merra2_merged_df = pd.merge(merra2_merged_df, df, left_index=True, right_index=True)

    # calculate monthly means
    monthly_mean_df: pd.DataFrame = _average_data_by_period(merra2_merged_df, period='1M', drop_count=False)
    # create an index of the datasets
    monthly_mean_df = add_index(monthly_mean_df)
    print('Monthly means:')
    print(monthly_mean_df)
    monthly_mean_df.to_csv('Monthly_Means.csv')
    print()

    # calculate Mean Of Monthly Means
    momm_df: pd.DataFrame = mean_of_monthly_means(monthly_mean_df)
    print('Mean of monthly means:')
    print(momm_df)
    momm_df.to_csv('Mean_Of_Monthly_Means.csv')


def read_netcdf(nc_filename: str, latitude: float=0.0, longitude: float=0.0) -> pd.DataFrame():
    """ Read in a netCDF file and return a DataFrame.
    """
    nc = netCDF4.Dataset(nc_filename)
    time = nc.variables['time']
    longitude_varobj = nc.variables['longitude']
    latitude_varobj = nc.variables['latitude']
    u10 = nc.variables['u10']
    v10 = nc.variables['v10']

    # print('Variables of netCDF file:')
    # print(nc.variables.keys())
    print(u10)
    # print('time:')
    # print(time[:])
    # print(time.units)
    timestamp = netCDF4.num2date(time[:], time.units)
    # print(timestamp)
    # print(str(time[0]) + ', ' + str(time[1]) + ', ' + str(time[2]))
    # print('longitude:')
    # print(longitude)
    # print('Dimensions of netCDF file:')
    # for dimension in nc.dimensions.items():
    #     print(dimension)
    print('Latitude values:')
    print(latitude_varobj[:])
    lat_index: int = np.where(latitude_varobj[:] == latitude)[0][0]
    print('Longitude values:')
    print(longitude_varobj[:])
    long_index: int = np.where(longitude_varobj[:] == longitude)[0][0]

    df = pd.DataFrame({'timestamp': timestamp[:], 'u10': u10[:, lat_index, long_index],
                       'v10': v10[:, lat_index, long_index]})
    df.set_index('timestamp', inplace=True)
    df['WS10m_m/s'] = round(np.sqrt(np.square(df['u10']) + np.square(df['v10'])), 3)
    df['WD10m_deg'] = np.where((270 - np.arctan2(df['v10'], df['u10']) * 180 / np.pi) > 360,
                                             270 - np.arctan2(df['v10'], df['u10']) * 180 / np.pi - 360,
                                             270 - np.arctan2(df['v10'], df['u10']) * 180 / np.pi)
    df['WD10m_deg'] = round(df['WD10m_deg'], 0)
    # df = pd.DataFrame(index=time[:], data=[u10[:,1,2],v10[:,1,2]])
    return df


def read_grib(file_path=r'F:\MERA_data\u010'):
    """Reads a grib file and returns the data as strings. Need eccodes (or gribapi) to be installed on the system
    """
    import subprocess
    import os
    file_path = os.path.join(file_path,os.listdir(file_path)[0])
    #stdout=PIPE sens the contents of the file to the variable data instead of just printing them on the screen
    data = subprocess.run(args=['grib_get_data', file_path],stdout=subprocess.PIPE)
    return data


def read_era5(date="2017-10-01", area="55/-10/50/-5", param="246.228", target="test_function"):
    """Download grib files from ERA-5 dataset. For more information about the request see ERA-Interim and ERA-5 notes.
    """
    import ecmwfapi
    server = ecmwfapi.ECMWFDataServer()
    request = {"class":"ea","dataset":"era5","expver":"1","grid":"N320","levtype":"sfc","stream":"oper",
               "time":"00:00:00/01:00:00/02:00:00/03:00:00/04:00:00/05:00:00/06:00:00/07:00:00/08:00:00/09:00:00/" \
                      "10:00:00/11:00:00/12:00:00/13:00:00/14:00:00/15:00:00/16:00:00/17:00:00/18:00:00/19:00:00/" \
                      "20:00:00/21:00:00/22:00:00/23:00:00",
               "type":"an", "date":date, "area":area, "param":param, "target":target}
    status=server.retrieve(request)
    print("Status",status)

    #For saving netcdf as csv
    """
    import xarray as xa
    data = xa.open_dataset(os.path.join(os.getcwd(),'netcdf_file'))
    df = data.to_dataframe()
    df.to_csv("grib_N320.csv")
    """

    #era-interim
    example_era_interim = {
        "class": "ei",
        "dataset": "interim",
        "date": "2017-09-01/to/2017-09-30",
        "expver": "1",
        "grid": "0.75/0.75",
        "levelist": "10",
        "levtype": "ml",
        "param": "131.128/132.128",
        "step": "0",
        "stream": "oper",
        "time": "18:00:00",
        "type": "an",
        "target": "output_netcdf",
        "format": "netcdf"

    }

    #era-5
    example_grib ={
        "class": "ea",
        "dataset": "era5",
        "date": "2017-10-01",
        "expver": "1",
        "grid" : "N320",
        "levtype": "sfc",
        "param": "246.228",
        "stream": "oper",
        "time": "00:00:00",
        "type": "an",
        "target": "grib_vertical"
    }
    example_netcdf = {
        "class": "ea",
        "dataset": "era5",
        "date": "2017-10-01",
        "expver": "1",
        "grid": "0.3/0.1",
        "levtype": "sfc",
        "param": "246.228",
        "stream": "oper",
        "time": "00:00:00",
        "type": "an",
        "target": "netcdf_.3.1",
        "area": "55/-10/50/-5",
        "format": "netcdf"
    }
