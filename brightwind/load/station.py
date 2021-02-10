#     brightwind is a library that provides wind analysts with easy to use tools for working with meteorological data.
#     Copyright (C) 2021 Stephen Holleran
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
from brightwind.analyse.plot import COLOR_PALETTE
import pandas as pd
import numpy as np
import requests
import os
import json
import gmaps


__all__ = ['MeasurementStation',
           'plot_meas_loc_on_gmap']


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
        header = ['name', 'measurement_units', 'oem', 'model',
                  'height_m', 'boom_orientation_deg', 'vane_dead_band_orientation_deg',
                  'date_from', 'date_to', 'connection_channel', 'sen_config_height_m', 'slope', 'offset',
                  'calibration_slope',
                  'calibration_offset']
        header_for_report = ['Instrument Name', 'Units', 'Sensor OEM', 'Sensor Model',
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
                  'date_from', 'date_to', 'connection_channel', 'slope', 'offset',
                  'calibration_slope', 'calibration_offset', 'measurement_type_id']
        header_for_report = ['Instrument Name', 'Units', 'Sensor Make', 'Sensor Model', 'Serial No',
                             'Height [m]', 'Boom Orient. [deg, mag N]',
                             'Date From', 'Date To', 'Logger Channel', 'Logger Slope', 'Logger Offset',
                             'Calibration Slope', 'Calibration Offset', 'measurement_type_id']
    elif table_type == 'direction_info':
        header = ['name', 'measurement_units', 'oem', 'model', 'sensor_serial_number',
                  'height_m', 'boom_orientation_deg', 'vane_dead_band_orientation_deg',
                  'date_from', 'date_to', 'connection_channel', 'offset', 'measurement_type_id']
        header_for_report = ['Instrument Name', 'Units', 'Sensor Make', 'Sensor Model', 'Serial No',
                             'Height [m]', 'Boom Orient. [deg, mag N]', 'Dead Band Orient. [deg, mag N]',
                             'Date From', 'Date To', 'Logger Channel', 'Logger Offset', 'measurement_type_id']

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

    if 'date_from' in sensors_table_report.columns:
        sensors_table_report['date_from'] = pd.to_datetime(sensors_table_report['date_from']).dt.strftime("%d-%b-%Y")
        sensors_table_report['date_to'] = pd.to_datetime(sensors_table_report['date_to']).dt.strftime("%d-%b-%Y")

    sensors_table_report = sensors_table_report.replace({np.nan: '-', 'NaT': '-', '31-Dec-2100': '-'})
    sensors_table_report.rename(columns={k: h for k, h in zip(header, header_for_report)}, inplace=True)
    index_name = 'Instrument Name'
    sensors_table_report = sensors_table_report.set_index(index_name)

    return sensors_table_report


def _get_title(property_name, schema):
    """
    Get the title for the property name from the WRA Data Model Schema.

    If the property name is not found it will return itself.

    *** Bug: 'sensitivity' finds 'Logger Sensitivity' when it could be 'Calibration Sensitivity' ***

    :param property_name: The property name to find.
    :type property_name:  str
    :param schema:        The WRA Data Model Schema.
    :type schema:         dict
    :return:              The title as stated in the schema.
    :rtype:               str
    """
    # search through definitions first
    if schema.get('definitions') is not None:
        if property_name in schema.get('definitions').keys():
            return schema.get('definitions').get(property_name).get('title')
    # search through properties
    if schema.get('properties') is not None:
        if property_name in schema.get('properties').keys():
            return schema.get('properties').get(property_name).get('title')
        # if not found in properties, loop through properties to find an array or object to move down to
        for k, v in schema.get('properties').items():
            if v.get('type') is not None and 'array' in v['type']:
                # move down into an array
                result = _get_title(property_name, v['items'])
                if result != property_name:
                    return result
            elif v.get('type') is not None and 'object' in v['type']:
                # move down into an object
                result = _get_title(property_name, v)
                if result != property_name:
                    return result
    # can't find the property_name in the schema, return itself
    return property_name


def _create_coord_dict(name, latitude, longitude):
    return {'name': name, 'coords': (latitude, longitude)}


def plot_meas_loc_on_gmap(meas_loc_data_model, reference_nodes=None, map_type='TERRAIN',
                          zoom_level=9, rename_label=True):
    """
    Visualise on Google maps image the location of one or more measurement locations and reference data nodes.

    # Requirement for the function is to have gmaps library installed, instructions to install it are at the link below:
    # https://jupyter-gmaps.readthedocs.io/en/latest/install.html

    :param meas_loc_data_model:       one or more measurement locations as extracted from the BrightWind platform
    :type meas_loc_data_model:        dict or list(dict) or Dataframe
    :param reference_nodes: merra2 or/and era5 reference nodes. If both reference datasets are provided then they
                            must be given as input as list of lists. ie. reference_nodes=[merra2_nodes, era5_nodes]
    :type reference_nodes:  list or list(list)
    :param map_type:        Google maps base map types to use for the image. Google maps offers three different base
                            map types: 'SATELLITE', 'HYBRID', 'TERRAIN'
                            (see https://jupyter-gmaps.readthedocs.io/en/latest/tutorial.html)
    :type map_type:         str
    :param zoom_level:      Google maps zoom_level to use for the image.
                            (see https://jupyter-gmaps.readthedocs.io/en/latest/tutorial.html)
    :type zoom_level:       int
    :param rename_label:    Rename the reference dataset labels using the label_nodes_with_bearing function before
                            reporting the labels on the google maps image.
    :type rename_label:     Boolean, default True

    :return:                Google maps image with input measurement and reference nodes locations.
    :rtype:                 fig

    **Example usage**
    ::
        import merlin

        meas_locs = [{'plant_uuid': 'xxxxx',
                    'latitude': 53.0,
                    'longitude': -7.1,
                    'measurement_station_type': 'mast',
                    'name': 'Test-MM1',
                    'notes': None},
                    {'plant_uuid': 'xxxxx',
                    'latitude': 53.5,
                    'longitude': -7.1,
                    'measurement_station_type': 'mast',
                    'name': 'Test-MM2',
                    'notes': None}]

        merlin.gis.plot_nodes_on_gmap(meas_locs[0])

        merlin.gis.plot_nodes_on_gmap(meas_locs)

        merlin.gis.plot_points_on_gmap(meas_locs, reference_nodes=merra2_nodes)

        merlin.gis.plot_nodes_on_gmap(meas_locs, reference_nodes=[merra2_nodes, era5_nodes])
    """
    gmaps_api_key = 'AIzaSyDvGrLQhqTDK24DmP-75RzmNEd6NMK5Svk'
    gmaps.configure(api_key=gmaps_api_key)
    ref_colors = {'merra2': COLOR_PALETTE.fourth, 'era5': COLOR_PALETTE.primary}  # colors used for merra2 and era5 nodes
    figure_layout = {
        'width': '900px',
        'height': '600px',
        'margin': '0 auto 0 auto',
        'padding': '1px'
    }

    # Plot meas locations
    meas_locs_list_points = []
    if type(meas_loc_data_model) == pd.DataFrame:
        for meas_loc in meas_loc_data_model.itertuples():
            point = _create_coord_dict(meas_loc.name, meas_loc.latitude, meas_loc.longitude)
            meas_locs_list_points.append(point)
    elif type(meas_loc_data_model) == dict:
        point = _create_coord_dict(meas_loc_data_model['name'], meas_loc_data_model['latitude_ddeg'],
                                   meas_loc_data_model['longitude_ddeg'])
        meas_locs_list_points.append(point)
    elif type(meas_loc_data_model) == list:
        for meas_loc in meas_loc_data_model:
            point = _create_coord_dict(meas_loc['name'], meas_loc['latitude_ddeg'], meas_loc['longitude_ddeg'])
            meas_locs_list_points.append(point)
    else:
        raise TypeError('Error with format of meas_loc, please input dataframe or dictionary or list of dictionaries')

    # print('NOTE: Click on the measurement location markers to visualise the labels.')

    # Assign center of figure as first meas_loc in list
    fig = gmaps.figure(center=meas_locs_list_points[0]['coords'], map_type=map_type,
                       zoom_level=zoom_level, layout=figure_layout)

    for i_color, meas_loc in enumerate(meas_locs_list_points):
        marker = gmaps.marker_layer([meas_loc['coords']],
                                    info_box_content=meas_loc['name'],
                                    display_info_box=True)
        # marker = gmaps.symbol_layer([meas_loc['coords']], fill_color=COLOR_PALETTE.color_list[i_color + 2],
        #                             stroke_color=COLOR_PALETTE.color_list[i_color + 2], scale=3,
        #                             info_box_content=meas_loc['name'],
        #                             display_info_box=True)
        fig.add_layer(marker)

    # Plot reference nodes
    if reference_nodes:
        raise NotImplementedError
        # if reference_nodes is a list, because only one reference dataset is given as input,
        # then convert it to a list of list
        # if not any(isinstance(element, list) for element in reference_nodes):
        #     reference_nodes = [reference_nodes]
        # for reanalysis_nodes in reference_nodes:
        #     reanalysis_nodes = label_nodes_with_bearing(meas_locs, reanalysis_nodes) \
        #         if rename_label else reanalysis_nodes
        #     # loop through all nodes of each reanalysis dataset (ie merra2_nodes)
        #     for node in reanalysis_nodes:
        #         if node.dataset in ref_colors.keys():
        #             node_color = ref_colors[node.dataset]
        #         else:
        #             node_color = 'white'
        #
        #         ref_marker = gmaps.symbol_layer([(node.latitude, node.longitude)], fill_color='black',
        #                                         stroke_color=node_color, stroke_opacity=0.7, scale=5,
        #                                         info_box_content=node.label, display_info_box=True)
        #         fig.add_layer(ref_marker)

    return fig


# class __DotDict(dict):
#     """
#     dot.notation access to dictionary attributes
#     dotmap is n alternative library https://github.com/drgrib/dotmap
#     """
#     __getattr__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__


class MeasurementStation:
    """
    Create a Station object by loading in an IEA Wind Resource Assessment Data Model.

    The IEA Wind: Task 43 Work Package 4 WRA Data Model was first released in January 2021. Versions of the
    Data Model Schema can be found at https://github.com/IEA-Task-43/digital_wra_data_standard

    The Schema associated with this data model file will be downloaded from GitHub and used to parse the data model.

    :param wra_data_model: The filepath to an implementation of the WRA Data Model as a .json file or a json string.
    :type wra_data_model:  str
    :return:               A simplified object to represent the data model
    :rtype:                DataModel
    """
    def __init__(self, wra_data_model):
        self.__data_model = self._load_wra_data_model(wra_data_model)
        version = self.__data_model.get('version')
        self.__schema = self._get_schema(version=version)
        # self.__header = self._get_header()
        self.__header = _Header(dm=self.__data_model, schema=self.__schema)
        self.__measurement_location = _MeasurementLocation(dm=self.__data_model)
        self.__logger_configs = _LoggerConfigs(dm_measurement_loc=self.__measurement_location.data_model,
                                               schema=self.__schema)
        self.__measurements = _Measurements(dm_measurement_loc=self.__measurement_location.data_model)
        self.__wspds = _Wspds(dm_measurement_loc=self.__measurement_location.data_model)
        self.__wdirs = _Wdirs(dm_measurement_loc=self.__measurement_location.data_model)

    @staticmethod
    def _load_wra_data_model(wra_data_model):
        """
        Load a IEA Wind Resource Assessment Data Model.

        The IEA Wind: Task 43 Work Package 4 WRA Data Model was first released in January 2021. Versions of the
        Data Model Schema can be found at https://github.com/IEA-Task-43/digital_wra_data_standard

        *** SHOULD INCLUDE CHECKING AGAINST THE JSON SCHEMA (WHICH WOULD MEAN GETTING THE CORRECT VERSION FROM GITHUB)
            AND MAKE SURE PROPER JSON
        :param wra_data_model: The filepath to an implementation of the WRA Data Model as a .json file or a json string.
        :type wra_data_model:  str
        :return:               Python dictionary of the data model.
        :rtype:                dict
        """
        # Assess whether filepath or json str sent.
        dm = dict()
        if '.json' == wra_data_model[-5:]:
            if _is_file(wra_data_model):
                with open(wra_data_model) as json_file:
                    dm = json.load(json_file)
        else:
            dm = json.loads(wra_data_model)
        return dm

    # def __getattr__(self):
    #     return self.data_model

    @staticmethod
    def _get_schema(version):
        """
        Get the JSON Schema from GitHub based on the version number in the data model.

        :param version: The version from the header information from the data model json file.
        :type version:  str
        :return:        The IEA Wind Task 43 WRA Data Model Schema.
        :rtype:         dict
        """
        schema_link = 'https://github.com/IEA-Task-43/digital_wra_data_standard/releases/download/v{}' \
                      '/iea43_wra_data_model.schema.json'
        # THE VERSION NUMBER IN THE DEMO MODEL IS INCORRECT
        if version != '0.1.0-2021.01':
            version = '0.1.0-2021.01'
        response = requests.get(schema_link.format(version))
        if response.status_code == 404:
            raise ValueError('Schema could not be downloaded from GitHub. Please check the version number in the '
                             'data model json file.')
        schema = json.loads(response.content)
        return schema

    @property
    def data_model(self):
        return self.__data_model

    @property
    def schema(self):
        return self.__schema

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
    def measurement_location(self):
        return self.__measurement_location

    @property
    def logger_configs(self):
        return self.__logger_configs

    @property
    def measurements(self):
        return self.__measurements

    @property
    def wspds(self):
        return self.__wspds

    @property
    def wdirs(self):
        return self.__wdirs


class _Header:
    def __init__(self, dm, schema):
        """
        Extract the header info from the data model and return either a dict or table

        """
        self._schema = schema
        keys = []
        values = []
        header_dict = {}
        for key, value in dm.items():
            if key != 'measurement_location':
                keys.append(key)
                values.append(value)
                header_dict[key] = value
        self._info = header_dict
        self._keys = keys
        self._values = values

    @property
    def info(self):
        return self._info

    @property
    def table(self):
        # get titles for each property
        titles = []
        for key in self._keys:
            titles.append(_get_title(key, self._schema))
        df = pd.DataFrame({'': self._values}, index=titles)
        df_styled = df.style.set_properties(**{'text-align': 'left'})
        df_styled = df_styled.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
        return df_styled


class _MeasurementLocation:
    def __init__(self, dm):
        if len(dm.get('measurement_location')) > 1:
            raise Exception('More than one measurement location found in the data model. Only processing'
                            'the first one found. Please remove extra measurement locations.')
        self._data_model = dm.get('measurement_location')[0]

    @property
    def data_model(self):
        return self._data_model

    @property
    def table(self):
        # Rename for column headings?
        if self._data_model.get('mast_properties') is None:
            mast_geometry = 'Unknown'
            mast_height = 'Unknown'
            mast_oem = 'Unknown'
        else:
            mast_geometry = self._data_model.get('mast_properties').get('mast_geometry_id')
            mast_height = self._data_model.get('mast_properties').get('mast_height_m')
            mast_oem = self._data_model.get('mast_properties').get('mast_oem')
        meas_locs = [{
                'Name': self._data_model['name'],
                'Latitude [ddeg]': self._data_model['latitude_ddeg'],
                'Longitude [ddeg]': self._data_model['longitude_ddeg'],
                'Measurement Station Type': self._data_model['measurement_station_type_id'],
                'Notes': self._data_model['notes'],
                # 'Date of Update': self._data_model['update_at'],
                'Mast Geometry': mast_geometry,
                'Mast Height [m]': mast_height,
                'Mast OEM': mast_oem
            }]
        meas_locs_df = pd.DataFrame(meas_locs)
        meas_locs_df.set_index('Name', inplace=True)
        return meas_locs_df


class _LoggerConfigs:
    def __init__(self, dm_measurement_loc, schema):
        self._data_model = dm_measurement_loc.get('logger_main_config')
        self._schema = schema

    @property
    def data_model(self):
        """
        This is the original data model unchanged from this level down.

        :return: The data model from this level down.
        :rtype:  Dict or List
        """
        return self._data_model

    @property
    def table(self):
        # build a replica list of loggers which have the schema titles
        temp_loggers = []
        temp_logger = {}
        for logger in self._data_model:
            for k, v in logger.items():
                temp_logger[_get_title(k, self._schema)] = v
            temp_loggers.append(temp_logger)
        log_configs_df = pd.DataFrame(temp_loggers)
        log_configs_df.set_index('Logger Name', inplace=True)
        return log_configs_df


class _Measurements:
    def __init__(self, dm_measurement_loc):
        # for meas_loc in dm['measurement_location']:
        self._data_model = _get_meas_points(dm_measurement_loc.get('measurement_point'))

    @property
    def data_model(self):
        return self._data_model

    @property
    def table(self):
        sensors_table = _format_sensor_table(self._data_model, table_type='meas_points')
        return sensors_table.drop_duplicates()

    @property
    def table_detailed(self):
        sensors_table = _format_sensor_table(self._data_model)
        return sensors_table


class _Wspds:
    def __init__(self, dm_measurement_loc):
        """
        Extract the wind speed measurement points

        :param dm_measurement_loc: The measurement location from the WRA Data Model
        :type dm_measurement_loc:  Dict

        """
        meas_points = _get_meas_points(dm_measurement_loc.get('measurement_point'))
        wspds = []
        for meas_point in meas_points:
            if meas_point.get('measurement_type_id') == 'wind_speed':
                wspds.append(meas_point)
        self._data_model = wspds
        self._names = self._get_names()

    @property
    def data_model(self):
        return self._data_model

    def _get_names(self):
        wspd_names = []
        for wspd in self._data_model:
            if wspd.get('name') not in wspd_names:
                wspd_names.append(wspd.get('name'))
        return wspd_names

    @property
    def names(self):
        return self._names

    def get_heights(self):
        wspd_heights = []
        for wspd_name in self.names:
            for wspd in self._data_model:
                if wspd.get('name') == wspd_name:
                    wspd_heights.append(wspd.get('height_m'))
                    break
        return wspd_heights

    @property
    def table(self):
        sensors_table = _format_sensor_table(self._data_model, table_type='speed_info')
        return sensors_table.drop_duplicates()


class _Wdirs:
    def __init__(self, dm_measurement_loc):
        """
        Extract the wind speed measurement points

        :param dm_measurement_loc: The measurement location from the WRA Data Model
        :type dm_measurement_loc:  Dict

        """
        meas_points = _get_meas_points(dm_measurement_loc.get('measurement_point'))
        wdirs = []
        for meas_point in meas_points:
            if meas_point.get('measurement_type_id') == 'wind_direction':
                wdirs.append(meas_point)
        self._data_model = wdirs

    @property
    def data_model(self):
        return self._data_model

    @property
    def names(self):
        wdir_names = []
        for wdir in self._data_model:
            if wdir.get('name') not in wdir_names:
                wdir_names.append(wdir.get('name'))
        return wdir_names

    @property
    def table(self):
        sensors_table = _format_sensor_table(self._data_model, table_type='direction_info')
        return sensors_table.drop_duplicates()
