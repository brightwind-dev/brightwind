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
from brightwind.utils import utils
import pandas as pd
import numpy as np
import requests
import json
import gmaps
from dotmap import DotMap


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
            input_dict[root_name + '_' + var_to_rename] = input_dict.pop(var_to_rename)
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


def plot_meas_loc_on_gmap(meas_station, map_type='TERRAIN',
                          zoom_level=9):
    """
    Visualise on Google Maps the location of one or more measurement locations.

    Unfortunately, to use this function you must have a Google Maps API key which should be free. To get one, follow
    the 'Get Started' instructions on the Google Maps Platform or go to the 'Credentials' section:

    https://cloud.google.com/maps-platform

    Once you have it, the GMAPS_API_KEY environmental variable will need to be set. In Windows this can be done
    by running the command prompt in Administrator mode and running:

    >> setx GMAPS_API_KEY "yourlonggooglemapsapikey"

    If Anaconda or your Python environment is running you will need to restart it for the environmental variables to
    take effect.

    Additionally, the function requires the 'gmaps' library installed, instructions to install it are at the link below:
    https://jupyter-gmaps.readthedocs.io/en/latest/install.html

    :param meas_station:    A measurement station object which contains the latitude and longitude of a
                            measurement location.
    :type meas_station:     MeasurementStation
    :param map_type:        Google maps base map types to use for the image. Google maps offers three different base
                            map types: 'SATELLITE', 'HYBRID', 'TERRAIN'
                            (see https://jupyter-gmaps.readthedocs.io/en/latest/tutorial.html)
    :type map_type:         str
    :param zoom_level:      Google maps zoom_level to use for the image.
                            (see https://jupyter-gmaps.readthedocs.io/en/latest/tutorial.html)
    :type zoom_level:       int
    :return:                Google maps image with input measurement and reference nodes locations.
    :rtype:                 fig

    **Example usage**
    ::
        mm1 = bw.MeasurementStation(bw.demo_datasets.demo_wra_data_model)
        bw.plot_meas_loc_on_gmap(mm1)

        bw.plot_meas_loc_on_gmap(mm1, map_type='SATELLITE')

    """
    gmaps.configure(api_key=utils.get_environment_variable('GMAPS_API_KEY'))
    figure_layout = {
        'width': '900px',
        'height': '600px',
        'margin': '0 auto 0 auto',
        'padding': '1px'
    }

    # Plot meas locations
    meas_loc_points = []
    if isinstance(meas_station, MeasurementStation):
        point = _create_coord_dict(meas_station.name, meas_station.lat, meas_station.long)
        meas_loc_points.append(point)
    else:
        raise TypeError('Error with format of meas_loc, please input dataframe or dictionary or list of dictionaries')

    # Assign center of figure as first meas_loc in list
    fig = gmaps.figure(center=meas_loc_points[0]['coords'], map_type=map_type,
                       zoom_level=zoom_level, layout=figure_layout)

    for i_color, meas_loc_point in enumerate(meas_loc_points):
        marker = gmaps.marker_layer([meas_loc_point['coords']],
                                    info_box_content=meas_loc_point['name'],
                                    display_info_box=True)
        fig.add_layer(marker)
    return fig


def _get_title(property_name, schema, property_section):
    """
    Get the title for the property name from the WRA Data Model Schema.

    If the property name is not found it will return itself.

    *** Bug: 'sensitivity' finds 'Logger Sensitivity' when it could be 'Calibration Sensitivity'
        Send a parent property name as an option ***

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


def _rename_keys(dictionary, schema, property_section_with_prefix=None):
    """
    Rename the keys in a dictionary with the title from the schema. If there are prefixes from raising a child
    property up to a parent level, state the property section that was raised up.

    :param dictionary:                   Dictionary with keys to rename.
    :type dictionary:                    dict
    :param schema:                       The WRA Data Model Schema.
    :type schema:                        dict
    :param property_section_with_prefix: The property section that was raised up to a parent level. These may have had
                                         prefixes added.
    :type property_section_with_prefix:  str or None
    :return:                             A renamed dictionary.
    :rtype:                              dict
    """
    renamed_dict = {}
    if property_section_with_prefix is not None:
        prefix = PREFIX_DICT[property_section_with_prefix]['prefix']
        keys_that_were_prefixed = PREFIX_DICT[property_section_with_prefix]['keys_to_prefix']
        prefixed_keys = []
        for key in keys_that_were_prefixed:
            prefixed_keys.append(prefix + key)
        title_prefix = PREFIX_DICT[property_section_with_prefix]['title_prefix']
        for k, v in dictionary.items():
            if k in prefixed_keys:
                # remove the prefix, get the tile and then add title_prefix
                renamed_dict[title_prefix + ' ' + _get_title(k.replace(prefix, ''), schema)] = v
            else:
                renamed_dict[_get_title(k, schema)] = v
    else:
        for k, v in dictionary.items():
            renamed_dict[_get_title(k, schema)] = v

    return renamed_dict


def _extract_keys_to_unique_list(lists_of_dictionaries):
    """
    Extract the keys for a list of dictionaries and merge them into a unique list.

    :param lists_of_dictionaries: List of dictionaries to pull unique keys from.
    :type lists_of_dictionaries:  list(dict)
    :return: Merged list of keys into a unique list.
    :rtype:  list
    """
    merged_list = list(lists_of_dictionaries[0].keys())
    for idx, d in enumerate(lists_of_dictionaries):
        if idx != 0:
            merged_list = merged_list + list(set(list(d.keys())) - set(merged_list))
    return merged_list


def _add_prefix(dictionary, keys_to_prefix, prefix_keys_with):
    """
    Add a prefix to certain keys in the dictionary.

    :param dictionary:       The dictionary containing the keys to rename.
    :type dictionary:        dict
    :param keys_to_prefix:   List of specific keys to rename
    :type keys_to_prefix:    list(str)
    :param prefix_keys_with: Text string to place before the key's name pulled from the schema.
    :type prefix_keys_with:  str
    :return:                 The dictionary with the keys prefixed.
    :rtype:                  dict
    """
    prefixed_dict = {}
    for k, v in dictionary.items():
        if k in keys_to_prefix:
            prefixed_dict[prefix_keys_with + k] = v
        else:
            prefixed_dict[k] = v
    return prefixed_dict


def _merge_two_dicts(x, y):
    """
    Given two dictionaries, merge them into a new dict as a shallow copy.
    """
    z = x.copy()
    z.update(y)
    return z


def _filter_parent_level(dictionary):
    """
    Pull only the parent level keys and values i.e. do not return any child lists or dictionaries or nulls/Nones.

    :param dictionary:
    :return:
    """
    parent = {}
    for key, value in dictionary.items():
        if (type(value) != list) and (type(value) != dict) and (value is not None):
            parent.update({key: value})
    return parent


def _flatten_dict(dictionary, property_to_bring_up, keys_to_prefix=None, prefix_keys_with=None):
    """
    Bring a child level in a dictionary up to the parent level.

    This is usually when there is an array of child levels and so the parent level is repeated.

    keys_to_prefix=['notes', 'update_at'],
    prefix_keys_with='Mast'

    :param dictionary:           Dictionary with keys to prefix.
    :type dictionary:            dict
    :param property_to_bring_up: The child property name to raise up to the parent level.
    :type property_to_bring_up:  str
    :param keys_to_prefix:       The list of keys to add a prefix to.
    :type keys_to_prefix:        list(str) or None
    :param prefix_keys_with:     The prefix to add.
    :type prefix_keys_with:      str or None
    :return:                     A list of merged dictionaries
    :rtype:                      list(dict)
    """
    result = []
    parent = _filter_parent_level(dictionary)
    for key, value in dictionary.items():
        if (type(value) == list) and (key == property_to_bring_up):
            for item in value:
                child = _filter_parent_level(item)
                if keys_to_prefix is not None and prefix_keys_with is not None:
                    child = _add_prefix(child, keys_to_prefix=keys_to_prefix,
                                        prefix_keys_with=prefix_keys_with)
                result.append(_merge_two_dicts(parent, child))
        if (type(value) == dict) and (key == property_to_bring_up):
            child = _filter_parent_level(value)
            if keys_to_prefix is not None and prefix_keys_with is not None:
                child = _add_prefix(child, keys_to_prefix=keys_to_prefix,
                                    prefix_keys_with=prefix_keys_with)
            result.append(_merge_two_dicts(parent, child))
    if not result:
        result.append(parent)
    return result


def _create_coord_dict(name, latitude, longitude):
    return {'name': name, 'coords': (latitude, longitude)}


PREFIX_DICT = {
    'mast_properties': {
        'prefix': 'mast_',
        'title_prefix': 'Mast ',
        'keys_to_prefix': ['notes', 'update_at']
    },
    'vertical_profiler_properties': {
        'prefix': 'vertical_profiler_properties_',
        'title_prefix': 'Vert. Prof. Prop. ',
        'keys_to_prefix': ['notes', 'update_at']
    },
    'lidar_config': {
        'prefix': 'lidar_config_',
        'title_prefix': 'Lidar Specific Configs ',
        'keys_to_prefix': ['date_from', 'date_to', 'notes', 'update_at']
    }
}


class MeasurementStation:
    """
    Create a Measurement Station object by loading in an IEA Wind Resource Assessment Data Model.

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
        self.__meas_loc_data_model = self._get_meas_loc_data_model(dm=self.__data_model)
        self.__meas_loc_properties = self.__get_properties()
        self.__logger_configs = _LoggerConfigs(meas_loc_dm=self.__meas_loc_data_model,
                                               schema=self.__schema, station_type=self.type)
        self.__measurements = _Measurements(meas_loc_dm=self.__meas_loc_data_model, schema=self.__schema)
        # if self.type in ['mast']:
            # self.__wspds = _Wspds(meas_loc_dm=self.__meas_loc_data_model)
            # self.__wdirs = _Wdirs(meas_loc_dm=self.__meas_loc_data_model)
            # self.__mast_section_geometry = _MastSectionGeometry()

    # def __getattr__(self):
    #     return self.data_model

    # @data_model.setter
    # def data_model(self, a):
    #     self.__data_model = a

    # def _get_header(self):
    #     # extract the header info from the _Header class
    #     return self._Header(self.__data_model)

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
        response = requests.get(schema_link.format(version))
        if response.status_code == 404:
            raise ValueError('Schema could not be downloaded from GitHub. Please check the version number in the '
                             'data model json file.')
        schema = json.loads(response.content)
        return schema

    @staticmethod
    def _get_meas_loc_data_model(dm):
        if len(dm.get('measurement_location')) > 1:
            raise Exception('More than one measurement location found in the data model. Only processing'
                            'the first one found. Please remove extra measurement locations.')
        return dm.get('measurement_location')[0]

    @property
    def data_model(self):
        """
        The data model from the measurement_location onwards i.e. excluding the header.
        :return:
        """
        return self.__meas_loc_data_model

    @property
    def schema(self):
        return self.__schema

    @property
    def name(self):
        return self.__meas_loc_data_model.get('name')

    @property
    def lat(self):
        return self.__meas_loc_data_model.get('latitude_ddeg')

    @property
    def long(self):
        return self.__meas_loc_data_model.get('longitude_ddeg')

    @property
    def type(self):
        return self.__meas_loc_data_model.get('measurement_station_type_id')

    def __get_properties(self):
        list_for_df = []
        if self.type == 'mast':
            list_for_df = _flatten_dict(self.__meas_loc_data_model, property_to_bring_up='mast_properties',
                                        keys_to_prefix=PREFIX_DICT['mast_properties']['keys_to_prefix'],
                                        prefix_keys_with=PREFIX_DICT['mast_properties']['prefix'])
        elif self.type in ['lidar', 'sodar', 'flidar']:
            list_for_df = _flatten_dict(self.__meas_loc_data_model, property_to_bring_up='vertical_profiler_properties',
                                        keys_to_prefix=PREFIX_DICT['mast_properties']['keys_to_prefix'],
                                        prefix_keys_with=PREFIX_DICT['vertical_profiler_properties']['prefix'])
        return list_for_df

    def get_table(self, horizontal_table_orientation=False):
        """
        Get a table representation of the attributes for the measurement station and it's mast or vertical profiler
        properties.

        :param horizontal_table_orientation: horizontal or vertical table orientation.
        :type horizontal_table_orientation:  bool
        :return:                             A table showing all the information for the measurement station. If a
                                             horizontal table then a pd.DataFrame is returned. If a vertical table
                                             then a styled pd.DataFrame is returned which does not have the same
                                             properties as a standard DataFrame.
        :rtype:                              pd.DataFrame or pd.io.formats.style.Styler
        """
        list_for_df = self.__meas_loc_properties

        prop_sec_with_prefix = ''
        if self.type == 'mast':
            prop_sec_with_prefix = 'mast_properties'
        elif self.type in ['lidar', 'sodar', 'flidar']:
            prop_sec_with_prefix = 'vertical_profiler_properties'

        df = pd.DataFrame()
        if horizontal_table_orientation:
            list_for_df_with_titles = []
            for row in list_for_df:
                list_for_df_with_titles.append(_rename_keys(dictionary=row, schema=self.__schema,
                                                            property_section_with_prefix=prop_sec_with_prefix))
            df = pd.DataFrame(list_for_df_with_titles, columns=_extract_keys_to_unique_list(list_for_df_with_titles))
        elif horizontal_table_orientation is False:
            for idx, row in enumerate(list_for_df):
                titles = list(_rename_keys(dictionary=row, schema=self.__schema,
                                           property_section_with_prefix=prop_sec_with_prefix).keys())
                df_temp = pd.DataFrame({idx + 1: list(row.values())}, index=titles)
                df = pd.concat([df, df_temp], axis=1, sort=False)
            df = df.style.set_properties(**{'text-align': 'left'})
            df = df.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
        return df

    @property
    def properties(self):
        return self.__meas_loc_properties

    @property
    def header(self):
        # return the header info
        return self.__header

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

    @property
    def mast_section_geometry(self):
        return self.__mast_section_geometry


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
        self._header_properties = header_dict
        self._keys = keys
        self._values = values

    @property
    def properties(self):
        return self._header_properties

    def get_table(self):
        # get titles for each property
        titles = []
        for key in self._keys:
            titles.append(_get_title(key, self._schema))
        df = pd.DataFrame({'': self._values}, index=titles)
        df_styled = df.style.set_properties(**{'text-align': 'left'})
        df_styled = df_styled.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
        return df_styled


class _LoggerConfigs:
    def __init__(self, meas_loc_dm, schema, station_type):
        self._log_cfg_data_model = meas_loc_dm.get('logger_main_config')
        self._schema = schema
        self._type = station_type
        self.__log_cfg_properties = self.__get_properties()

    @property
    def data_model(self):
        """
        This is the original data model unchanged from this level down.

        :return: The data model from this level down.
        :rtype:  Dict or List
        """
        return self._data_model

    def __get_properties(self):
        list_for_df = []
        if self._type == 'mast':
            # if mast, there are no child dictionaries
            list_for_df = self._log_cfg_data_model  # logger config data model is already a list
        elif self._type in ['lidar', 'flidar']:
            for log_config in self._log_cfg_data_model:
                log_configs_flat = _flatten_dict(log_config, property_to_bring_up='lidar_config',
                                                 keys_to_prefix=PREFIX_DICT['lidar_config']['keys_to_prefix'],
                                                 prefix_keys_with=PREFIX_DICT['lidar_config']['prefix'])
                for log_config_flat in log_configs_flat:
                    list_for_df.append(log_config_flat)
        return list_for_df

    def get_table(self, horizontal_table_orientation=False):
        """
        Get a table representation of the attributes for the logger configurations.

        If a LiDAR then the lidar specific configurations are also presented.

        :param horizontal_table_orientation: horizontal or vertical table orientation.
        :type horizontal_table_orientation:  bool
        :return:                             A table showing all the information for the measurement station. If a
                                             horizontal table then a pd.DataFrame is returned. If a vertical table
                                             then a styled pd.DataFrame is returned which does not have the same
                                             properties as a standard DataFrame.
        :rtype:                              pd.DataFrame or pd.io.formats.style.Styler
        """
        list_for_df = self.__log_cfg_properties

        prop_sec_with_prefix = ''
        if self._type == 'mast':
            # if mast, there are no child dictionaries
            prop_sec_with_prefix = None
        elif self._type in ['lidar', 'flidar']:
            prop_sec_with_prefix = 'lidar_config'

        df = pd.DataFrame()
        if horizontal_table_orientation:
            list_for_df_with_titles = []
            for row in list_for_df:
                list_for_df_with_titles.append(_rename_keys(dictionary=row, schema=self._schema,
                                                            property_section_with_prefix=prop_sec_with_prefix))
            df = pd.DataFrame(list_for_df_with_titles, columns=_extract_keys_to_unique_list(list_for_df_with_titles))
            df.set_index('Logger Name', inplace=True)
        elif horizontal_table_orientation is False:
            for idx, row in enumerate(list_for_df):
                titles = list(_rename_keys(dictionary=row, schema=self._schema,
                                           property_section_with_prefix=prop_sec_with_prefix).keys())
                df_temp = pd.DataFrame({idx + 1: list(row.values())}, index=titles)
                df = pd.concat([df, df_temp], axis=1, sort=False)
            df = df.style.set_properties(**{'text-align': 'left'})
            df = df.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
        return df

    @property
    def properties(self):
        return self.__log_cfg_properties


class _Measurements:
    def __init__(self, meas_loc_dm, schema):
        # for meas_loc in dm['measurement_location']:
        self._data_model = meas_loc_dm.get('measurement_point')
        self._schema = schema

    @property
    def data_model(self):
        return self._data_model

    def get_table(self, detailed=False):
        df = pd.DataFrame()
        # if detailed is False:
        #     list_for_df = []
        #     for meas_point in self._data_model:
        #         list_for_df.append(_filter_parent_level(meas_point))
        #     list_for_df_with_titles = []
        #     for row in list_for_df:
        #         list_for_df_with_titles.append(_rename_keys(row, schema=self._schema, keys_to_rename=list(row.keys()),
        #                                                     prefix_keys_with=None))
        #     df = pd.DataFrame(list_for_df_with_titles, columns=_extract_keys_to_unique_list(list_for_df_with_titles))
        #     df.set_index('Name', inplace=True)
        #     df.fillna('-', inplace=True)
        #     df.sort_values(['Measurement Type', 'Height [m]'], ascending=[False, False], inplace=True)
        # elif detailed:
        #     df = _format_sensor_table(self._data_model)
        return df


class _Wspds:
    def __init__(self, meas_loc_dm):
        """
        Extract the wind speed measurement points

        :param meas_loc_dm: The measurement location from the WRA Data Model
        :type meas_loc_dm:  Dict

        """
        meas_points = _get_meas_points(meas_loc_dm.get('measurement_point'))
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
    def __init__(self, meas_loc_dm):
        """
        Extract the wind speed measurement points

        :param meas_loc_dm: The measurement location from the WRA Data Model
        :type meas_loc_dm:  Dict

        """
        meas_points = _get_meas_points(meas_loc_dm.get('measurement_point'))
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


class _MastSectionGeometry:
    def __init__(self):
        raise NotImplementedError
