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


from brightwind.load.load import _is_file
import pandas as pd
import numpy as np
import requests
import os
import json


__all__ = ['DataModel']


def _flatten_sensor_dict(sensor):
    """
        Flatten the sensor dictionary retrieved from jason
        assigning all the sub-dictionaries to the main dictionary.

        :param sensor: The sensor dictionary retrieved for a single configuration
                           option and meas_point id.
        :type sensor: dict
        :return: output
        :rtype: dict

    """
    output = {key: value for key, value in sensor.items() if (type(value) != list) or (value == {})}
    for key, value in zip(sensor.keys(), sensor.values()):
        if type(value) == list:
            if key == 'calibration':
                value = {key + "_" + k: v for k, v in value[0].items()}
            output.update(value)
    return output


def _rename_variables(input_dict, root_name):
    for var_to_rename in ['height_m', 'serial_number', 'update_at', 'notes']:
        if var_to_rename in list(input_dict.keys()):
            input_dict[ root_name + '_' + var_to_rename] = input_dict.pop(var_to_rename)
    return input_dict


def _replace_none_date(input_dict):
    for date_str in ['date_from', 'date_to']:
        if input_dict[date_str] is None:
            input_dict[date_str] = '2100-12-31'
    return input_dict


def _get_meas_points(meas_points):
    meas_points_flatten = []
    for meas_point in meas_points:
        #         meas_point = _flatten_meas_point_dict(meas_point)
        sen_configs = sorted(meas_point['sensor_config'], key=lambda i: i['date_from'])
        sen_configs = [_replace_none_date(_rename_variables(sen_config, 'sen_config')) for sen_config in sen_configs]
        sensors = [_replace_none_date(_rename_variables(_flatten_sensor_dict(sensor), 'sensor')) for sensor in
                   meas_point['sensor']]
        if meas_point['mounting_arrangement'] is not None:
            mounting_arrangements = [_replace_none_date(_rename_variables(mntg_arrang, 'mounting_arrangement'))
                                     for mntg_arrang in meas_point['mounting_arrangement']]
        else:
            mounting_arrangements = {}

        date_from = [sen_config['date_from'] for sen_config in sen_configs]
        date_to = [sen_config['date_to'] for sen_config in sen_configs]
        for sensor in sensors:
            date_from.append(sensor['date_from'])
            date_to.append(sensor['date_to'])
        for mntg_arrang in mounting_arrangements:
            date_from.append(mntg_arrang['date_from'])
            date_to.append(mntg_arrang['date_to'])

        date_from.extend(date_to)
        dates = np.unique(date_from)
        for i in range(len(dates) - 1):
            good_sen_config = {}
            for sen_config in sen_configs:
                if (sen_config['date_from'] <= dates[i]) & (sen_config['date_to'] > dates[i]):
                    good_sen_config = sen_config.copy()
            if good_sen_config != {}:
                for sensor in sensors:
                    if (sensor['date_from'] <= dates[i]) & (sensor['date_to'] > dates[i]):
                        good_sen_config.update(sensor)
                for mntg_arrang in mounting_arrangements:
                    if (mntg_arrang['date_from'] <= dates[i]) & (mntg_arrang['date_to'] > dates[i]):
                        good_sen_config.update(mntg_arrang)
                good_sen_config['date_to'] = dates[i + 1]
                good_sen_config['date_from'] = dates[i]
                good_sen_config.update(meas_point)
                del good_sen_config['sensor_config']
                del good_sen_config['sensor']
                meas_points_flatten.append(good_sen_config)
    return meas_points_flatten


def _format_sensor_table(meas_points, table_type='full'):
    if table_type == 'full':
        header = ['name', 'measurement_units', 'oem',
                  'height_m', 'boom_orientation_deg', 'vane_dead_band_orientation_deg',
                  'date_from', 'date_to', 'connection_channel', 'sen_config_height_m', 'slope', 'offset',
                  'calibration_slope',
                  'calibration_offset']
        header_for_report = ['Instrument Name', 'Units', 'Sensor OEM',
                             'Height [m]', 'Boom Orient. [deg, mag N]', 'Dead Band Orient. [deg, mag N]',
                             'Date From', 'Date To', 'Logger Channel', 'Logger Stated Height [m]', 'Logger Slope',
                             'Logger Offset', 'Calibration Slope',
                             'Calibration Offset']
    elif table_type == 'meas_points':
        header = ['name', 'measurement_type_id', 'height_m', 'boom_orientation_deg']
        header_for_report = ['Instrument Name', 'Measurement Type', 'Height [m]', 'Boom Orient. [deg, mag N]']
    elif table_type == 'speed_info':
        header = ['name', 'measurement_units', 'oem', 'model', 'sensor_serial_number',
                  'height_m', 'boom_orientation_deg',
                  'date_from', 'date_to', 'slope', 'offset', 'calibration_slope',
                  'calibration_offset', 'measurement_type_id']
        header_for_report = ['Instrument Name', 'Units', 'Sensor Make', 'Sensor Model', 'Serial No',
                             'Height [m]', 'Boom Orient. [deg, mag N]',
                             'Date From', 'Date To', 'Logger Slope', 'Logger Offset', 'Calibration Slope',
                             'Calibration Offset', 'measurement_type_id']
    elif table_type == 'direction_info':
        header = ['name', 'measurement_units', 'oem', 'model', 'sensor_serial_number',
                  'height_m', 'boom_orientation_deg', 'vane_dead_band_orientation_deg',
                  'date_from', 'date_to', 'offset', 'measurement_type_id']
        header_for_report = ['Instrument Name', 'Units', 'Sensor Make', 'Sensor Model', 'Serial No',
                             'Height [m]', 'Boom Orient. [deg, mag N]', 'Dead Band Orient. [deg, mag N]',
                             'Date From', 'Date To', 'Logger Offset', 'measurement_type_id']

    sensors_table_report = pd.DataFrame(meas_points)

    if any(elem not in sensors_table_report.columns for elem in header):
        ind_to_remove = [ind for ind, elem in enumerate(header) if elem not in sensors_table_report.columns]
        del header[ind_to_remove[0]]
        del header_for_report[ind_to_remove[0]]

    sensors_table_report = pd.DataFrame(sensors_table_report[header])
    if table_type == 'speed_info':
        sensors_table_report = sensors_table_report[sensors_table_report['measurement_type_id'] == 'wind_speed']
        del sensors_table_report['measurement_type_id']
    if table_type == 'direction_info':
        sensors_table_report = sensors_table_report[sensors_table_report['measurement_type_id'] == 'wind_direction']
        del sensors_table_report['measurement_type_id']

    if sensors_table_report.columns.contains('date_from'):
        sensors_table_report['date_from'] = pd.to_datetime(sensors_table_report['date_from']).dt.strftime("%d-%b-%Y")
        sensors_table_report['date_to'] = pd.to_datetime(sensors_table_report['date_to']).dt.strftime("%d-%b-%Y")

    sensors_table_report = sensors_table_report.replace({np.nan: '-', 'NaT': '-', '31-Dec-2100': '-'})
    sensors_table_report.rename(columns={k: h for k, h in zip(header, header_for_report)}, inplace=True)
    index_name = 'Instrument Name'
    sensors_table_report = sensors_table_report.set_index(index_name)

    return sensors_table_report


class DataModel:
    class __DotDict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    def __init__(self, wra_data_model):
        """
        Load a IEA Wind Resource Assessment Data Model.

        The IEA Wind: Task 43 Work Package 4 WRA Data Model was first released in January 2021. Versions of the Data Model
        Schema can be found at https://github.com/IEA-Task-43/digital_wra_data_standard

        :param wra_data_model: The filepath to an implementation of the WRA Data Model as a .json file or a json string.
        :type wra_data_model:  str
        :return:               Python Dict
        :rtype:                Dict???????
        """
        self.__data_model = self._load_wra_data_model(wra_data_model)
        # self.__header = self._get_header()
        self.__header = self._Header(self.__data_model)
        self.__measurement_locations = self._MeasurementLocations(self.__data_model)
        self.__measurements = self._Measurements(self.__data_model)
        self.__wspds = self._Wspds(self.__data_model)

    @staticmethod
    def _load_wra_data_model(wra_data_model):
        """
        Load a IEA Wind Resource Assessment Data Model.

        The IEA Wind: Task 43 Work Package 4 WRA Data Model was first released in January 2021. Versions of the Data Model
        Schema can be found at https://github.com/IEA-Task-43/digital_wra_data_standard

        *** SHOULD INCLUDE CHECKING AGAINST THE JSON SCHEMA (WHICH WOULD MEAN GETTING THE CORRECT VERSION FROM GITHUB)
            AND MAKE SURE PROPER JSON
        :param wra_data_model: The filepath to an implementation of the WRA Data Model as a .json file or a json string.
        :type wra_data_model:  str
        :return:               Python Dict
        :rtype:                Dict
        """
        # Assess whether filepath or json str sent.
        if '.json' == wra_data_model[-5:]:
            if _is_file(wra_data_model):
                with open(wra_data_model) as json_file:
                    dm = json.load(json_file)
        else:
            dm = json.loads(wra_data_model)
        return dm

    # def __getattr__(self):
    #     return self.data_model

    @property
    def data_model(self):
        return self.__data_model

    # @data_model.setter
    # def data_model(self, a):
    #     self.__data_model = a

    # def _get_header(self):
    #     # extract the header info from the _Header class
    #     return self._Header(self.__data_model)

    @property
    def header(self):
        # return the header info
        return self.__header

    @property
    def measurement_locations(self):
        return self.__measurement_locations

    @property
    def measurements(self):
        return self.__measurements

    @property
    def wspds(self):
        return self.__wspds

    class _Header:
        def __init__(self, dm):
            """
            Extract the header info from the data model and return either a dict or DataFrame

            """
            index = []
            col_value = []
            header_dict = {}
            for key, value in dm.items():
                if key != 'measurement_location':
                    index.append(key)
                    col_value.append(value)
                    header_dict[key] = value
            self._info = header_dict
            self._index = index
            self._col_value = col_value

        @property
        def info(self):
            return self._info

        @property
        def table(self):
            df = pd.DataFrame({'': self._col_value}, index=self._index)
            df_styled = df.style.set_properties(**{'text-align': 'left'})
            df_styled = df_styled.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
            return df_styled

    class _MeasurementLocations:
        def __init__(self, dm):
            meas_locs = []
            for meas_loc in dm.get('measurement_location'):
                # **** CHANGE METHOD TO USE THE SAME AS HEADER ***
                meas_locs.append({
                    'Name': meas_loc['name'],
                    'Latitude [Decimal Degrees]': meas_loc['latitude_ddeg'],
                    'Longitude [Decimal Degrees]': meas_loc['longitude_ddeg'],
                    'Station Type': meas_loc['measurement_station_type_id'],
                    'Notes': meas_loc['notes'],
                    'Last Updated': meas_loc['update_at'],
                    'Mast Type': meas_loc['mast_properties']['mast_geometry_id'],
                    'Mast Height': meas_loc['mast_properties']['mast_height_m'],
                    'Mast OEM': meas_loc['mast_properties']['mast_oem']
                })
            self._info = meas_locs

        @property
        def info(self):
            return self._info

        @property
        def table(self):
            meas_locs_df = pd.DataFrame(self._info)
            meas_locs_df.set_index('Name', inplace=True)
            return meas_locs_df

    class _Measurements:
        def __init__(self, dm):
            # for meas_loc in dm['measurement_location']:
            self._info = _get_meas_points(dm.get('measurement_location')[0].get('measurement_point'))

        @property
        def info(self):
            return self._info

        @property
        def table(self):
            sensors_table = _format_sensor_table(self._info, table_type='meas_points')
            return sensors_table.drop_duplicates()

        @property
        def table_detailed(self):
            sensors_table = _format_sensor_table(self._info)
            return sensors_table

    class _Wspds:
        def __init__(self, dm):
            """
            Extract the wind speed measurement points

            """
            meas_points = _get_meas_points(dm.get('measurement_location')[0].get('measurement_point'))
            wspds = []
            for meas_point in meas_points:
                if meas_point.get('measurement_type_id') == 'wind_speed':
                    wspds.append(meas_point)
            self._info = wspds

        @property
        def info(self):
            return self._info

        @property
        def names(self):
            wspd_names = []
            for wspd in self._info:
                if wspd.get('name') not in wspd_names:
                    wspd_names.append(wspd.get('name'))
            return wspd_names

        @property
        def table(self):
            sensors_table = _format_sensor_table(self._info, table_type='speed_info')
            return sensors_table.drop_duplicates()


