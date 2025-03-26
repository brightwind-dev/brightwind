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


from brightwind.utils.utils import is_file
import numpy as np
import pandas as pd
import requests
import json
import copy

__all__ = ['MeasurementStation']


def _replace_none_date(list_or_dict):
    if isinstance(list_or_dict, list):
        renamed = []
        for item in list_or_dict:
            renamed.append(_replace_none_date(item))
        return renamed
    elif isinstance(list_or_dict, dict):
        for date_str in ['date_from', 'date_to']:
            if (date_str in list_or_dict.keys()) and (list_or_dict.get(date_str) is None):
                list_or_dict[date_str] = DATE_INSTEAD_OF_NONE
    return list_or_dict


def _get_title(property_name, schema, property_section=None):
    """
    Get the title for the property name from the WRA Data Model Schema. Optionally, you can send the section of the
    schema where the property should be found. This avoids finding the wrong property name when the name
    is not unique.

    If the property name is not found it will return itself.

    :param property_name:    The property name to find.
    :type property_name:     str
    :param schema:           The WRA Data Model Schema.
    :type schema:            dict
    :param property_section: The section in the schema where the property can be found. This avoids the case where the
                             property_name is not unique in the schema.
    :type property_section:  str or None
    :return:                 The title as stated in the schema.
    :rtype:                  str
    """
    # search through definitions first
    if schema.get('definitions') is not None:
        if property_name.split('_id')[0] in schema.get('definitions').keys():
            return schema.get('definitions').get(property_name.split('_id')[0]).get('title')
    # search through properties
    if schema.get('properties') is not None:
        # is property_name in the main properties
        if property_name in schema.get('properties').keys() and property_section is None:
            return schema.get('properties').get(property_name).get('title')
        # is property_section part of the main properties
        if property_section in schema.get('properties').keys():
            property_type = schema.get('properties').get(property_section).get('type')
            if property_type is not None and 'array' in property_type:
                # move down into an array
                result = _get_title(property_name, schema.get('properties').get(property_section)['items'])
                if result != property_name:
                    return result
            elif property_type is not None and 'object' in property_type:
                # move down into an object
                result = _get_title(property_name, schema.get('properties').get(property_section))
                if result != property_name:
                    return result
        # don't recognise either property_name or property_section.
        # loop through each property to find an array or object to move down to
        for k, v in schema.get('properties').items():
            if v.get('type') is not None and 'array' in v['type']:
                # move down into an array
                result = _get_title(property_name, v['items'], property_section)
                if result != property_name:
                    return result
            elif v.get('type') is not None and 'object' in v['type']:
                # move down into an object
                result = _get_title(property_name, v, property_section)
                if result != property_name:
                    return result
    # can't find the property_name in the schema, return itself
    return property_name


def _rename_to_title(list_or_dict, schema):
    """
    Rename the names in a list to it's equivalent title in the schema or the keys in a dictionary. If there are
    prefixes from raising a child property up to a parent level, this will find the normal schema title and add
    the prefixed title to it.

    :param list_or_dict: List of names or dictionary with keys to rename.
    :type list_or_dict:  list or dict
    :param schema:       The WRA Data Model Schema.
    :type schema:        dict
    :return:             A renamed list or keys in dictionary.
    :rtype:              list or dict
    """
    prefixed_names = {}
    # find all possible prefixed names and build a dict to contain it and the separator and title.
    for key in PREFIX_DICT.keys():
        for col in PREFIX_DICT[key]['keys_to_prefix']:
            prefixed_name = key + PREFIX_DICT[key]['prefix_separator'] + col
            prefixed_names[prefixed_name] = {'prefix_separator': PREFIX_DICT[key]['prefix_separator'],
                                             'title_prefix': PREFIX_DICT[key]['title_prefix']}
    list_special_cases_no_prefix = ['logger_measurement_config.slope', 'logger_measurement_config.offset',
                                    'logger_measurement.sensitivity',
                                    'calibration.slope', 'calibration.offset', 'calibration.sensitivity']
    if isinstance(list_or_dict, dict):
        renamed_dict = {}
        for k, v in list_or_dict.items():
            if k in list(prefixed_names.keys()):
                # break out the property name and the name, get the title and then add title_prefix to it.
                property_section = k[0:k.find(prefixed_names[k]['prefix_separator'])]
                property_name = k[k.find(prefixed_names[k]['prefix_separator']) + 1:]
                if k in list_special_cases_no_prefix:
                    # Special cases don't add a title prefix as there is already one in the schema title
                    renamed_dict[_get_title(property_name, schema, property_section)] = v
                else:
                    renamed_dict[prefixed_names[k]['title_prefix'] + _get_title(property_name, schema,
                                                                                property_section)] = v
            else:
                # if not in the list of prefixed_names then just find the title as normal.
                renamed_dict[_get_title(k, schema)] = v
        return renamed_dict
    elif isinstance(list_or_dict, list):
        renamed_list = []
        for name in list_or_dict:
            if name in list(prefixed_names.keys()):
                # break out the property name and the name, get the title and then add title_prefix to it.
                property_section = name[0:name.find(prefixed_names[name]['prefix_separator'])]
                property_name = name[name.find(prefixed_names[name]['prefix_separator']) + 1:]
                if name in list_special_cases_no_prefix:
                    # Special cases don't add a title prefix as there is already one in the schema title
                    renamed_list.append(_get_title(property_name, schema, property_section))
                else:
                    renamed_list.append(prefixed_names[name]['title_prefix'] + _get_title(property_name, schema,
                                                                                          property_section))
            else:
                # if not in the list of prefixed_names then just find the title as normal.
                renamed_list.append(_get_title(name, schema))
        return renamed_list


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


def _add_prefix(dictionary, property_section):
    """
    Add a prefix to certain keys in the dictionary.

    :param dictionary:       The dictionary containing the keys to rename.
    :type dictionary:        dict
    :return:                 The dictionary with the keys prefixed.
    :rtype:                  dict
    """
    prefixed_dict = {}
    for k, v in dictionary.items():
        if k in PREFIX_DICT[property_section]['keys_to_prefix']:
            prefixed_dict[property_section + PREFIX_DICT[property_section]['prefix_separator'] + k] = v
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
    Pull only the parent level keys and values i.e. do not return any child lists or dictionaries.

    :param dictionary:
    :return:
    """
    parent = {}
    for key, value in dictionary.items():
        if (type(value) != list) and (type(value) != dict):
            parent.update({key: value})
    return parent


def _flatten_dict(dictionary, property_to_bring_up):
    """
    Bring a child level in a dictionary up to the parent level.

    This is usually when there is an array of child levels and so the parent level is repeated.

    :param dictionary:           Dictionary with keys to prefix.
    :type dictionary:            dict
    :param property_to_bring_up: The child property name to raise up to the parent level.
    :type property_to_bring_up:  str
    :return:                     A list of merged dictionaries
    :rtype:                      list(dict)
    """
    result = []
    parent = _filter_parent_level(dictionary)
    for key, value in dictionary.items():
        if (type(value) == list) and (key == property_to_bring_up):
            for item in value:
                child = _filter_parent_level(item)
                child = _add_prefix(child, property_section=property_to_bring_up)
                result.append(_merge_two_dicts(parent, child))
        if (type(value) == dict) and (key == property_to_bring_up):
            child = _filter_parent_level(value)
            child = _add_prefix(child, property_section=property_to_bring_up)
            # return a dictionary and not a list
            result = _merge_two_dicts(parent, child)
            # result.append(_merge_two_dicts(parent, child))
    if not result:
        result.append(parent)
    return result


def _raise_child(dictionary, child_to_raise):
    """

    :param dictionary:
    :param child_to_raise:
    :return:
    """
    # FUTURE DEV: ACCOUNT FOR 'DATE_OF_CALIBRATION' WHEN RAISING UP MULTIPLE CALIBRATIONS
    if dictionary is None:
        return None
    new_dict = dictionary.copy()
    for key, value in dictionary.items():
        if (key == child_to_raise) and (value is not None):
            # Found the key to raise. Flattening dictionary.
            return _flatten_dict(dictionary, child_to_raise)
    # didn't find the child to raise. search down through each nested dict or list
    for key, value in dictionary.items():
        if (type(value) == dict) and (value is not None):
            # 'key' is a dict, looping through it's own keys.
            flattened_dicts = _raise_child(value, child_to_raise)
            if flattened_dicts:
                new_dict[key] = flattened_dicts
                return new_dict
        elif (type(value) == list) and (value is not None):
            # 'key' is a list, looping through it's items.
            temp_list = []
            for idx, item in enumerate(value):
                flattened_dicts = _raise_child(item, child_to_raise)
                if flattened_dicts:
                    if isinstance(flattened_dicts, list):
                        for flat_dict in flattened_dicts:
                            temp_list.append(flat_dict)
                    else:
                        # it is a dictionary so just append it
                        temp_list.append(flattened_dicts)
            if temp_list:
                # Temp_list is not empty. Replacing 'key' with this.
                new_dict[key] = temp_list
                return new_dict
    return None


PREFIX_DICT = {
    'mast_properties': {
        'prefix_separator': '.',
        'title_prefix': 'Mast ',
        'keys_to_prefix': ['notes', 'update_at']
    },
    'vertical_profiler_properties': {
        'prefix_separator': '.',
        'title_prefix': 'Vert. Prof. Prop. ',
        'keys_to_prefix': ['notes', 'update_at']
    },
    'lidar_config': {
        'prefix_separator': '.',
        'title_prefix': 'Lidar Specific Configs ',
        'keys_to_prefix': ['date_from', 'date_to', 'notes', 'update_at']
    },
    'logger_measurement_config': {
        'prefix_separator': '.',
        'title_prefix': 'Logger ',
        'keys_to_prefix': ['height_m', 'serial_number', 'slope', 'offset', 'sensitivity', 'notes', 'update_at']
    },
    'column_name': {
        'prefix_separator': '.',
        'title_prefix': 'Column Name ',
        'keys_to_prefix': ['notes', 'update_at']
    },
    'sensor': {
        'prefix_separator': '.',
        'title_prefix': 'Sensor ',
        'keys_to_prefix': ['serial_number', 'notes', 'update_at']
    },
    'calibration': {
        'prefix_separator': '.',
        'title_prefix': 'Calibration ',
        'keys_to_prefix': ['slope', 'offset', 'sensitivity', 'report_file_name', 'report_link',
                           'uncertainty_k_factor', 'date_from', 'date_to', 'notes', 'update_at']
    },
    'calibration_uncertainty': {
        'prefix_separator': '.',
        'title_prefix': 'Calibration Uncertainty ',
        'keys_to_prefix': []
    },
    'mounting_arrangement': {
        'prefix_separator': '.',
        'title_prefix': 'Mounting Arrangement ',
        'keys_to_prefix': ['notes', 'update_at']
    },
    'interference_structures': {
        'prefix_separator': '.',
        'title_prefix': 'Interference Structure ',
        'keys_to_prefix': ['structure_type_id', 'orientation_from_mast_centre_deg', 'orientation_reference_id',
                           'distance_from_mast_centre_mm',
                           'date_from', 'date_to', 'notes', 'update_at']
    }
}
DATE_INSTEAD_OF_NONE = '2100-12-31'
SENSOR_TYPE_ORDER = ['anemometer', '2d_ultrasonic', '3d_ultrasonic', 'propeller_anemometer', 'gill_propeller',
                     'wind_vane', 'pyranometer', 'pyrheliometer', 'thermometer', 'hygrometer', 'barometer',
                     'rain_gauge', 'voltmeter', 'ammeter',
                     'ice_detection_sensor', 'fog_sensor', 'illuminance_sensor', 'gps', 'compass', 'other']
MEAS_TYPE_ORDER = ['wind_speed', 'wind_direction', 'vertical_wind_speed',
                   'global_horizontal_irradiance', 'direct_normal_irradiance', 'diffuse_horizontal_irradiance',
                   'global_tilted_irradiance', 'global_normal_irradiance', 'soiling_loss_index', 'illuminance',
                   'wind_speed_turbulence',
                   'air_temperature', 'temperature', 'relative_humidity', 'air_pressure', 'precipitation',
                   'ice_detection', 'voltage', 'current',
                   'fog', 'carrier_to_noise_ratio', 'doppler_spectral_broadening',
                   'gps_coordinates', 'orientation', 'compass_direction', 'true_north_offset',
                   'elevation', 'altitude', 'azimuth', 'status', 'counter', 'availability', 'quality',
                   'tilt_x', 'tilt_y', 'tilt_z', 'timestamp', 'other']


class MeasurementStation:
    """
    Create a Measurement Station object by loading in an IEA Wind Resource Assessment Data Model.

    The IEA Wind: Task 43 Work Package 4 WRA Data Model was first released in January 2021. Versions of the
    Data Model Schema can be found at https://github.com/IEA-Task-43/digital_wra_data_standard

    The Schema associated with this data model file will be downloaded from GitHub and used to parse the data model.

    :param wra_data_model: The filepath to an implementation of the WRA Data Model as a .json file or
                           a json formatted string or
                           a dictionary format of the data model.
    :type wra_data_model:  str or dict
    :return:               A simplified object to represent the data model
    :rtype:                MeasurementStation
    """
    def __init__(self, wra_data_model):
        self.__data_model = self._load_wra_data_model(wra_data_model)
        version = self.__data_model.get('version')
        self.__schema = self._get_schema(version=version)
        self.__header = _Header(dm=self.__data_model, schema=self.__schema)
        self.__meas_loc_data_model = self._get_meas_loc_data_model(dm=self.__data_model)
        self.__meas_loc_properties = self.__get_properties()
        self.__logger_main_configs = _LoggerMainConfigs(meas_loc_dm=self.__meas_loc_data_model,
                                                        schema=self.__schema, station_type=self.type)
        self.__measurements = _Measurements(meas_loc_dm=self.__meas_loc_data_model, schema=self.__schema)
        # self.__mast_section_geometry = _MastSectionGeometry()

    def __getitem__(self, item):
        return self.__meas_loc_properties[item]

    def __iter__(self):
        return iter(self.__meas_loc_properties)

    def __repr__(self):
        return repr(self.__meas_loc_properties)

    @staticmethod
    def _load_wra_data_model(wra_data_model):
        """
        Load a IEA Wind Resource Assessment Data Model.

        The IEA Wind: Task 43 Work Package 4 WRA Data Model was first released in January 2021. Versions of the
        Data Model Schema can be found at https://github.com/IEA-Task-43/digital_wra_data_standard

        *** SHOULD INCLUDE CHECKING AGAINST THE JSON SCHEMA (WHICH WOULD MEAN GETTING THE CORRECT VERSION FROM GITHUB)
            AND MAKE SURE PROPER JSON

        :param wra_data_model: The filepath to an implementation of the WRA Data Model as a .json file or
                               a json formatted string or
                               a dictionary format of the data model.
        :return:               Python dictionary of the data model.
        :rtype:                dict
        """
        # Assess whether filepath or json str sent.
        dm = dict()
        if isinstance(wra_data_model, str) and '.json' == wra_data_model[-5:]:
            if is_file(wra_data_model):
                with open(wra_data_model) as json_file:
                    dm = json.load(json_file)
        elif isinstance(wra_data_model, str):
            dm = json.loads(wra_data_model)
        else:
            # it is most likely already a dict so return itself
            dm = wra_data_model
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
            raise Exception('More than one measurement location found in the data model. Only processing '
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
        meas_loc_prop = []
        if self.type in ['mast', 'solar']:
            meas_loc_prop = _flatten_dict(self.__meas_loc_data_model, property_to_bring_up='mast_properties')
        elif self.type in ['lidar', 'sodar', 'floating_lidar']:
            meas_loc_prop = _flatten_dict(self.__meas_loc_data_model,
                                          property_to_bring_up='vertical_profiler_properties')
        return meas_loc_prop

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

        df = pd.DataFrame()
        if horizontal_table_orientation:
            list_for_df_with_titles = []
            if isinstance(list_for_df, dict):
                list_for_df_with_titles = [_rename_to_title(list_or_dict=list_for_df, schema=self.__schema)]
            elif isinstance(list_for_df, list):
                for row in list_for_df:
                    list_for_df_with_titles.append(_rename_to_title(list_or_dict=row, schema=self.__schema))
            df = pd.DataFrame(list_for_df_with_titles, columns=_extract_keys_to_unique_list(list_for_df_with_titles))
            df.set_index('Name', inplace=True)
        elif horizontal_table_orientation is False:
            if isinstance(list_for_df, dict):
                # if a dictionary, it only has 1 row of data
                titles = list(_rename_to_title(list_or_dict=list_for_df, schema=self.__schema).keys())
                df = pd.DataFrame({'': list(list_for_df.values())}, index=titles)
            elif isinstance(list_for_df, list):
                for idx, row in enumerate(list_for_df):
                    titles = list(_rename_to_title(list_or_dict=row, schema=self.__schema).keys())
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
    def logger_main_configs(self):
        return self.__logger_main_configs

    @property
    def measurements(self):
        return self.__measurements

    @property
    def mast_section_geometry(self):
        return 'Not yet implemented.'
        # return self.__mast_section_geometry


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

    def __getitem__(self, item):
        return self._header_properties[item]

    def __iter__(self):
        return iter(self._header_properties)

    def __repr__(self):
        return repr(self._header_properties)

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


class _LoggerMainConfigs:
    def __init__(self, meas_loc_dm, schema, station_type):
        self._log_cfg_data_model = meas_loc_dm.get('logger_main_config')
        self._schema = schema
        self._type = station_type
        self.__log_cfg_properties = self.__get_properties()

    def __getitem__(self, item):
        return self.__log_cfg_properties[item]

    def __iter__(self):
        return iter(self.__log_cfg_properties)

    def __repr__(self):
        return repr(self.__log_cfg_properties)

    @property
    def data_model(self):
        """
        This is the original data model unchanged from this level down.

        :return: The data model from this level down.
        :rtype:  Dict or List
        """
        return self._log_cfg_data_model

    def __get_properties(self):
        log_cfg_props = []
        if self._type == 'mast':
            # if mast, there are no child dictionaries
            log_cfg_props = self._log_cfg_data_model  # logger config data model is already a list
        elif self._type in ['lidar', 'floating_lidar']:
            for log_config in self._log_cfg_data_model:
                log_configs_flat = _flatten_dict(log_config, property_to_bring_up='lidar_config')
                for log_config_flat in log_configs_flat:
                    log_cfg_props.append(log_config_flat)
        return log_cfg_props

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

        df = pd.DataFrame()
        if horizontal_table_orientation:
            list_for_df_with_titles = []
            for row in list_for_df:
                list_for_df_with_titles.append(_rename_to_title(list_or_dict=row, schema=self._schema))
            df = pd.DataFrame(list_for_df_with_titles, columns=_extract_keys_to_unique_list(list_for_df_with_titles))
            df.set_index('Logger Name', inplace=True)
        elif horizontal_table_orientation is False:
            for idx, row in enumerate(list_for_df):
                titles = list(_rename_to_title(list_or_dict=row, schema=self._schema).keys())
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
        self._meas_data_model = meas_loc_dm.get('measurement_point')
        self._schema = schema
        self.__meas_properties = self.__get_properties()
        self.__meas_dict = self.__get_properties_as_dict()

    # Making _Measurements emulate a dictionary.
    # Not using super(_Measurements, self).__init__(*arg, **kw) as I do not want the user to __setitem__,
    # __delitem__, clear, update or pop. Therefore, writing out the specific behaviour I want for the dictionary.

    def __getitem__(self, key):
        return self.__meas_dict[key]

    def __iter__(self):
        return iter(self.__meas_dict)

    def __repr__(self):
        return repr(self.__meas_dict)

    def __len__(self):
        return len(self.__meas_dict)

    def __contains__(self, key):
        return key in self.__meas_dict

    # Don't allow copy as user needs to use copy.deepcopy to copy the dictionary, might also confuse with the object.
    # def copy(self):
    #     return self.__meas_dict.copy()

    def keys(self):
        return self.__meas_dict.keys()

    def values(self):
        return self.__meas_dict.values()

    def items(self):
        return self.__meas_dict.items()

    @property
    def data_model(self):
        return self._meas_data_model

    def __get_parent_properties(self):
        meas_props = []
        for meas_point in self._meas_data_model:
            meas_props.append(_filter_parent_level(meas_point))
        return meas_props

    @property
    def properties(self):
        return self.__meas_properties

    @property
    def names(self):
        """
        The names of all the measurements.

        :return: The list of names.
        :rtype:  list(str)
        """
        return self.get_names()

    @property
    def wspds(self):
        return self.__get_properties_as_dict(measurement_type_id='wind_speed')

    @property
    def wspd_names(self):
        return self.get_names(measurement_type_id='wind_speed')

    @property
    def wspd_heights(self):
        return self.get_heights(measurement_type_id='wind_speed')

    @property
    def wdirs(self):
        return self.__get_properties_as_dict(measurement_type_id='wind_direction')

    @property
    def wdir_names(self):
        return self.get_names(measurement_type_id='wind_direction')

    @property
    def wdir_heights(self):
        return self.get_heights(measurement_type_id='wind_direction')

    @staticmethod
    def __meas_point_merge(logger_measurement_configs, sensors=None, mounting_arrangements=None):
        """
        Merge the properties from logger_measurement_configs, sensors and mounting_arrangements. This will account for
        when each property was changed over time.

        :param logger_measurement_configs:  Logger measurement config properties
        :type logger_measurement_configs:   list
        :param sensors:                     Sensor properties
        :type sensors:                      list
        :param mounting_arrangements:       Mounting arrangement properties
        :type mounting_arrangements:        list
        :return:                            The properties merged together.
        :rtype:                             list(dict)
        """
        logger_measurement_configs = _replace_none_date(logger_measurement_configs)
        sensors = _replace_none_date(sensors)
        mounting_arrangements = _replace_none_date(mounting_arrangements)
        date_from = [log_meas_cfg.get('date_from') for log_meas_cfg in logger_measurement_configs]
        date_to = [log_meas_cfg.get('date_to') for log_meas_cfg in logger_measurement_configs]
        if sensors is not None:
            for sensor in sensors:
                date_from.append(sensor.get('date_from'))
                date_to.append(sensor.get('date_to'))
        if mounting_arrangements is not None:
            for mounting_arrangement in mounting_arrangements:
                date_from.append(mounting_arrangement['date_from'])
                date_to.append(mounting_arrangement['date_to'])
        date_from.extend(date_to)
        dates = list(set(date_from))
        dates.sort()
        meas_points_merged = []
        for i in range(len(dates) - 1):
            good_log_meas_cfg = {}
            for logger_measurement_config in logger_measurement_configs:
                if (logger_measurement_config['date_from'] <= dates[i]) & (
                        logger_measurement_config.get('date_to') > dates[i]):
                    good_log_meas_cfg = logger_measurement_config.copy()
            if good_log_meas_cfg != {}:
                if sensors is not None:
                    for sensor in sensors:
                        if (sensor['date_from'] <= dates[i]) & (sensor['date_to'] > dates[i]):
                            good_log_meas_cfg.update(sensor)
                if mounting_arrangements is not None:
                    for mounting_arrangement in mounting_arrangements:
                        if (mounting_arrangement['date_from'] <= dates[i]) & (
                                mounting_arrangement['date_to'] > dates[i]):
                            good_log_meas_cfg.update(mounting_arrangement)
                good_log_meas_cfg['date_to'] = dates[i + 1]
                good_log_meas_cfg['date_from'] = dates[i]
                meas_points_merged.append(good_log_meas_cfg)
        # replace 'date_to' if equals to 'DATE_INSTEAD_OF_NONE'
        for meas_point in meas_points_merged:
            if meas_point.get('date_to') is not None and meas_point.get('date_to') == DATE_INSTEAD_OF_NONE:
                meas_point['date_to'] = None
        return meas_points_merged

    def __get_properties(self):
        meas_props = []
        for meas_point in self._meas_data_model:
            logger_meas_configs = _raise_child(meas_point, child_to_raise='logger_measurement_config')
            calib_raised = _raise_child(meas_point, child_to_raise='calibration')
            if calib_raised is None:
                sensors = _raise_child(meas_point, child_to_raise='sensor')
            else:
                sensors = _raise_child(calib_raised, child_to_raise='sensor')
                sensors = [sensors_needed for sensors_needed in sensors if sensors_needed['measurement_type_id'] == meas_point['measurement_type_id']]
            mounting_arrangements = _raise_child(meas_point, child_to_raise='mounting_arrangement')

            if mounting_arrangements is None:
                meas_point_merged = self.__meas_point_merge(logger_measurement_configs=logger_meas_configs,
                                                            sensors=sensors)
            else:
                meas_point_merged = self.__meas_point_merge(logger_measurement_configs=logger_meas_configs,
                                                            sensors=sensors,
                                                            mounting_arrangements=mounting_arrangements)
            for merged_meas_point in meas_point_merged:
                meas_props.append(merged_meas_point)
        return meas_props

    def __get_properties_by_type(self, measurement_type_id):
        merged_properties = copy.deepcopy(self.__meas_properties)
        meas_list = []
        for meas_point in merged_properties:
            meas_type = meas_point.get('measurement_type_id')
            if meas_type is not None and meas_type == measurement_type_id:
                meas_list.append(meas_point)
        return meas_list

    def __get_properties_as_dict(self, measurement_type_id=None):
        """
        Get the flattened properties as a dictionary with name as the key. This is for easy use for accessing a
        measurement point.

        e.g. mm1.measurements['Spd1']

        :return: Flattened properties as a dictionary
        :rtype:  dict
        """
        meas_dict = {}
        merged_properties = copy.deepcopy(self.__meas_properties)
        for meas_point in merged_properties:
            meas_point_name = meas_point['name']
            if meas_point['measurement_type_id'] == measurement_type_id or measurement_type_id is None:
                if meas_point_name in meas_dict.keys():
                    meas_dict[meas_point_name].append(meas_point)
                else:
                    meas_dict[meas_point_name] = [meas_point]
        return meas_dict

    def __get_table_for_cols(self, columns_to_show):
        """
        Get table of measurements for specific columns.
        :param columns_to_show: Columns required to show in table.
        :type columns_to_show:  list(str)
        :return:                Table as a pandas DataFrame
        :rtype:                 pd.DataFrame
        """
        temp_df = pd.DataFrame(self.__meas_properties)
        # select the common columns that are available
        avail_cols = [col for col in columns_to_show if col in temp_df.columns]
        if not avail_cols:
            raise KeyError('No data to show from the list of columns provided')
        # Drop all rows that have no data for the avail_cols
        temp_df.dropna(axis=0, subset=avail_cols, how='all', inplace=True)
        if temp_df.empty:
            raise KeyError('No data to show from the list of columns provided')
        # Name needs to be included in the grouping but 'date_from' and 'date_to' should not be
        # as we filter for them later
        required_in_avail_cols = {'include': ['name'], 'remove': ['date_from', 'date_to']}
        for include_col in required_in_avail_cols['include']:
            if include_col not in avail_cols:
                avail_cols.insert(0, include_col)
        for remove_col in required_in_avail_cols['remove']:
            if remove_col in avail_cols:
                avail_cols.remove(remove_col)
        # Remove duplicates resulting from other info been dropped.
        temp_df.sort_values(['name', 'date_from'], ascending=[True, True], inplace=True)
        temp_df.fillna('-', inplace=True)  # groupby drops nan so need to fill them in
        # group duplicate data for the columns available
        grouped_by_avail_cols = temp_df.groupby(avail_cols)
        # get date_to from the last row in each group to assign to the first row.
        new_date_to = grouped_by_avail_cols.last()['date_to']
        df = grouped_by_avail_cols.first()[['date_from', 'date_to']]
        df['date_to'] = new_date_to
        df.reset_index(level=avail_cols, inplace=True)
        df.sort_values(['name', 'date_from'], ascending=[True, True], inplace=True)
        # get titles
        title_cols = _rename_to_title(list_or_dict=list(df.columns), schema=self._schema)
        df.columns = title_cols
        df.set_index('Name', inplace=True)
        df.replace(DATE_INSTEAD_OF_NONE, '-', inplace=True)
        return df

    def get_table(self, detailed=False, wind_speeds=False, wind_directions=False, calibrations=False,
                  mounting_arrangements=False, columns_to_show=None):
        """
        Get tables to show information about the measurements made.

        :param detailed:              For a more detailed table that includes how the sensor is programmed into the
                                      logger, information about the sensor itself and how it is mounted on the mast
                                      if it was. 
                                      If detailed=False then table is showing details only for measurement points.
        :type detailed:               bool
        :param wind_speeds:           Wind speed specific details.
        :type wind_speeds:            bool
        :param wind_directions:       Wind direction specific details.
        :type wind_directions:        bool
        :param calibrations:          Calibration specific details.
        :type calibrations:           bool
        :param mounting_arrangements: Mounting arrangement specific details.
        :type mounting_arrangements:  bool
        :param columns_to_show:       Optionally provide a list of column names you want to see in a table. This list
                                      should be pulled from the list of keys available in the measurements.properties.
                                      'name', 'date_from' and 'date_to' are always inserted so no need to include them
                                      in your list.
        :type columns_to_show:        list(str) or None
        :return:                      A table showing information about the measurements made by this measurement station.
        :rtype:                       pd.DataFrame

        **Example usage**
        ::
            import brightwind as bw
            mm1 = bw.MeasurementStation(bw.demo_datasets.demo_wra_data_model)
            mm1.measurements.get_table()

        To get a more detailed table::
            mm1.measurements.get_table(detailed=True)

        To get wind speed specific details::
            mm1.measurements.get_table(wind_speeds=True)

        To get wind speed specific details::
            mm1.measurements.get_table(wind_directions=True)

        To get calibration specific details::
            mm1.measurements.get_table(calibrations=True)

        To get mounting specific details::
            mm1.measurements.get_table(mounting_arrangements=True)

        To make your own table::
            columns = ['calibration.slope', 'calibration.offset', 'calibration.report_file_name', 'date_of_calibration']
            mm1.measurements.get_table(columns_to_show=columns)

        """
        df = pd.DataFrame()
        if detailed is False and wind_speeds is False and wind_directions is False \
                and calibrations is False and mounting_arrangements is False and columns_to_show is None:
            # default summary table
            list_for_df = self.__get_parent_properties()
            list_for_df_with_titles = []
            for row in list_for_df:
                list_for_df_with_titles.append(_rename_to_title(list_or_dict=row, schema=self._schema))
            df = pd.DataFrame(list_for_df_with_titles, columns=_extract_keys_to_unique_list(list_for_df_with_titles))
            # order rows
            order_index = dict(zip(MEAS_TYPE_ORDER, range(len(MEAS_TYPE_ORDER))))
            df['meas_type_rank'] = df['Measurement Type'].map(order_index)
            df.sort_values(['meas_type_rank', 'Height [m]'], ascending=[True, False], inplace=True)
            df.drop('meas_type_rank', axis=1, inplace=True)
            df.set_index('Name', inplace=True)
            df.fillna('-', inplace=True)
        elif detailed is True:
            cols_required = ['name', 'oem', 'model', 'sensor_type_id', 'sensor.serial_number',
                             'height_m', 'boom_orientation_deg',
                             'date_from', 'date_to', 'connection_channel', 'measurement_units_id',
                             'logger_measurement_config.slope', 'logger_measurement_config.offset',
                             'calibration.slope', 'calibration.offset',
                             'logger_measurement_config.notes', 'sensor.notes']
            df = pd.DataFrame(self.__meas_properties).set_index(
                ['date_from', 'date_to']).reset_index()
            # get what is common from both lists and use this to filter df
            cols_required = [col for col in cols_required if col in df.columns]
            df = df[cols_required]
            # order rows
            if 'sensor_type_id' in df.columns:
                order_index = dict(zip(SENSOR_TYPE_ORDER, range(len(SENSOR_TYPE_ORDER))))
                df['sensor_rank'] = df['sensor_type_id'].map(order_index)
                df.sort_values(['sensor_rank', 'height_m'], ascending=[True, False], inplace=True)
                df.drop('sensor_rank', axis=1, inplace=True)
            else:
                df.sort_values(['name', 'height_m'], ascending=[True, False], inplace=True)
            # get titles
            title_cols = _rename_to_title(list_or_dict=list(df.columns), schema=self._schema)
            df.columns = title_cols
            # tidy up
            df.set_index('Name', inplace=True)
            df.fillna('-', inplace=True)
            df.replace(DATE_INSTEAD_OF_NONE, '-', inplace=True)
        elif wind_speeds is True:
            cols_required = ['name', 'measurement_type_id', 'oem', 'model', 'sensor.serial_number', 'is_heated',
                             'height_m', 'boom_orientation_deg', 'mounting_type_id',
                             'date_from', 'date_to', 'connection_channel',
                             'logger_measurement_config.slope', 'logger_measurement_config.offset',
                             'calibration.slope', 'calibration.offset',
                             'logger_measurement_config.notes', 'sensor.notes']
            df = pd.DataFrame(self.__meas_properties)
            df = df[df['measurement_type_id'] == 'wind_speed'].set_index(
                ['date_from', 'date_to']).reset_index()
            # get what is common from both lists and use this to filter df
            cols_required = [col for col in cols_required if col in df.columns]
            df = df[cols_required]
            df.drop('measurement_type_id', axis=1, inplace=True)
            # order rows
            df.sort_values(['height_m', 'name'], ascending=[False, True], inplace=True)
            # get titles
            title_cols = _rename_to_title(list_or_dict=list(df.columns), schema=self._schema)
            df.columns = title_cols
            # tidy up
            df.set_index('Name', inplace=True)
            df.fillna('-', inplace=True)
            df.replace(DATE_INSTEAD_OF_NONE, '-', inplace=True)
        elif wind_directions is True:
            cols_required = ['name', 'measurement_type_id', 'oem', 'model', 'sensor.serial_number', 'is_heated',
                             'height_m', 'boom_orientation_deg', 'vane_dead_band_orientation_deg',
                             'orientation_reference_id',
                             'date_from', 'date_to', 'connection_channel',
                             'logger_measurement_config.slope', 'logger_measurement_config.offset',
                             'logger_measurement_config.notes', 'sensor.notes']
            df = pd.DataFrame(self.__meas_properties)
            df = df[df['measurement_type_id'] == 'wind_direction'].set_index(
                ['date_from', 'date_to']).reset_index()
            # get what is common from both lists and use this to filter df
            cols_required = [col for col in cols_required if col in df.columns]
            df = df[cols_required]
            df.drop('measurement_type_id', axis=1, inplace=True)
            # order rows
            df.sort_values(['height_m', 'name'], ascending=[False, True], inplace=True)
            # get titles
            title_cols = _rename_to_title(list_or_dict=list(df.columns), schema=self._schema)
            df.columns = title_cols
            # tidy up
            df.set_index('Name', inplace=True)
            df.fillna('-', inplace=True)
            df.replace(DATE_INSTEAD_OF_NONE, '-', inplace=True)
        elif calibrations is True:
            cols_required = ['calibration.slope', 'calibration.offset', 'calibration.report_file_name',
                             'date_of_calibration', 'calibration_organisation', 'place_of_calibration',
                             'calibration.uncertainty_k_factor', 'calibration.update_at', 'calibration.notes']
            df = self.__get_table_for_cols(cols_required)
        elif mounting_arrangements is True:
            cols_required = ['mounting_type_id', 'boom_orientation_deg', 'orientation_reference_id', 'boom_oem',
                             'boom_model', 'boom_diameter_mm', 'boom_length_mm', 'distance_from_mast_to_sensor_mm',
                             'upstand_height_mm', 'upstand_diameter_mm', 'vane_dead_band_orientation_deg',
                             'mounting_arrangement.notes']
            df = self.__get_table_for_cols(cols_required)
        elif columns_to_show is not None:
            df = self.__get_table_for_cols(columns_to_show)
        return df

    def get_names(self, measurement_type_id=None):
        """
        Get the names of measurements for a particular measurement_type, or all of them if measurement_type_id is None.

        :param measurement_type_id: The measurement_type_id (as defined by the IEA Wind Task 43
                                    WRA Data Model) to filter for the names.
        :type measurement_type_id:  str or None
        :return:                    The list of names.
        :rtype:                     list(str)

        **Example usage**
        ::
            import brightwind as bw
            mm1 = bw.MeasurementStation(bw.demo_datasets.iea43_wra_data_model_v1_0)

            # To get all measurement point names:
            mm1.measurements.get_names(measurement_type_id=None)

            # To get measurement point names only for measurement_type_id='air_temperature':
            mm1.measurements.get_names(measurement_type_id='air_temperature')

        """

        if type(measurement_type_id) is not str and measurement_type_id is not None:
            raise TypeError('measurement_type_id must be a string or None')
        
        names = []
        for meas_point in self.__meas_properties:  # use __meas_properties as it is a list and holds it's order
            meas_type = meas_point.get('measurement_type_id')
            meas_name = meas_point.get('name')
            if measurement_type_id is not None:
                if meas_type is not None and meas_type == measurement_type_id:
                    if meas_name not in names:
                        names.append(meas_point.get('name'))
            else:
                if meas_name not in names:
                    names.append(meas_point.get('name'))
        return names

    def get_heights(self, names=None, measurement_type_id=None):
        """
        Get the heights of the measurements.

        A list of measurement names can be provided to filter the results. If names are provided
        the measurement_type_id is ignored.

        A measurement_type_id can be provided to filter the results. Measurement types available can be
        found by running the following after initialisation:
        MeasurementStation.measurements.get_table()

        :param names:               Optional list of measurement names to filter by.
        :type names:                list(str) or str or None
        :param measurement_type_id: Optional measurement type to filter by.
        :type measurement_type_id:  str
        :return:                    The heights of the measurements.
        :rtype:                     list(float)

        **Example usage**
        ::
            import brightwind as bw
            mm1 = bw.MeasurementStation(bw.demo_datasets.iea43_wra_data_model_v1_0)

            # To get heights for all measurements:
            mm1.measurements.get_heights(names=None, measurement_type_id=None)

            # To get heights only for defined names=['Spd_80mSE', 'Dir_76mNW']:
            mm1.measurements.get_heights(names=['Spd_80mSE', 'Dir_76mNW'])

            # To get heights only for defined names='Spd_40mSE':
            mm1.measurements.get_heights(names='Spd_40mSE')

            # To get heights only for measurement_type_id='air_temperature':
            mm1.measurements.get_heights(measurement_type_id='air_temperature')

        """

        if type(measurement_type_id) is not str and measurement_type_id is not None:
            raise TypeError('measurement_type_id must be a string or None')
        
        heights = []
        if names is None:
            names = self.get_names(measurement_type_id=measurement_type_id)
        if isinstance(names, str):
            names = [names]
        for name in names:
            name_found = False  # used to fill in a NaN if the name isn't found to keep consistent order
            for meas_point in self.__meas_properties:  # use __meas_properties as it is a list and holds it's order
                if meas_point['name'] == name and meas_point.get('height_m') is not None:
                    heights.append(meas_point['height_m'])
                    name_found = True
                    break
                elif meas_point['name'] == name and meas_point.get('height_m') is None:
                    heights.append(np.NaN)
                    name_found = True
                    break
            if name_found is False:
                heights.append(np.NaN)
        return heights


class _MastSectionGeometry:
    def __init__(self):
        raise NotImplementedError
