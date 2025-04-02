import pytest
import brightwind as bw
import pandas as pd
import numpy as np
import warnings
import json
import copy

MM1 = bw.MeasurementStation(bw.demo_datasets.iea43_wra_data_model_v1_0)
MM2 = bw.MeasurementStation(bw.demo_datasets.demo_wra_data_model)
FL1 = bw.MeasurementStation(bw.demo_datasets.floating_lidar_iea43_wra_data_model_v1_2)
SO1 = bw.MeasurementStation(bw.demo_datasets.sodar_iea43_wra_data_model_v1_3)
SS1 = bw.MeasurementStation(bw.demo_datasets.solar_iea43_wra_data_model_v1_3)

def _get_schema(schema):
    with open(schema) as json_file:
        schema = json.load(json_file)
    return schema


def test_get_title():
    property_name_title = {
        "mast_geometry_id": "Mast Geometry",
        "latitude_ddeg": "Latitude [ddeg]",
        "pole_diameter_mm": "Pole Diameter [mm]",
        "device_datum_plane_height_m": "Device Datum Plane Height [m]",
        "logger_measurement_config": "Logger Measurement Configuration",
        "statistic_type_id": "Statistic Type",
        "sensitivity": "Logger Sensitivity",
        "date_of_calibration": "Date of Calibration",
        "reference_unit": "Reference Unit",
        "vane_dead_band_orientation_deg": "Vane Dead Band Orientation [deg]",
        "author": "Author",
        "date_from": "Date From",
        "author2": "author2"
    }

    property_name_title_v1_2 = {
        "device_vertical_orientation": "Device Vertical Orientation",
        "logger_firmware_version": "Logger Firmware Version",
        "logger_stated_boom_orientation_deg": "Logger Stated Boom Orientation [deg]",
        "sensor_body_size_mm": "Sensor Body Size [mm]",
        "diameter_of_interference_structure_mm": "Diameter of Interference Structure [mm]"
    }

    for name, title in property_name_title.items():
        assert bw.load.station._get_title(name, _get_schema(bw.demo_datasets.iea43_wra_data_model_schema_v1_0)) == title

    for name, title in property_name_title.items():
        assert bw.load.station._get_title(name, _get_schema(bw.demo_datasets.iea43_wra_data_model_schema_v1_2)) == title

    for name, title in property_name_title_v1_2.items():
        assert bw.load.station._get_title(name, _get_schema(bw.demo_datasets.iea43_wra_data_model_schema_v1_2)) == title


def test_get_table():
  
    ## tests for MeasurementStation.get_table()
    MM1.get_table(horizontal_table_orientation=False).data.loc['Name'].values[0] == 'Test_MM1'
    for value in MM1.get_table(horizontal_table_orientation=True).T['Test_MM1'].to_dict().values():
        assert value in MM1.properties.values()

    # Test data for solar measurement station
    table_test = SS1.get_table().data
    assert table_test.loc["Measurement Station Type"].values[0] == "solar"
    assert table_test.loc["Name"].values[0] == "test-solar-site"

    ## tests for logger_main_configs.get_table()

    assert MM1.logger_main_configs.get_table(horizontal_table_orientation=False).data.loc['Logger Name'].values[0] == 'AName_MM1'

    for value in MM1.logger_main_configs.get_table(horizontal_table_orientation=True).T['AName_MM1'].to_dict().values():
        assert value in MM1.logger_main_configs.properties[0].values()
        
    # Test data representing a solar measurement station      
    table_test = SS1.logger_main_configs.get_table()
    assert isinstance(table_test, (pd.DataFrame, pd.io.formats.style.Styler))
    assert table_test.data.loc["Logger Model Name", 1] == "CR800"
    assert table_test.data.loc["Logger Serial Number", 1] == "13588"
    
    # Test data representing a sodar measurement station 
    table_test = SO1.logger_main_configs.get_table()
    assert isinstance(table_test, (pd.DataFrame, pd.io.formats.style.Styler))
    assert table_test.data.loc["Logger OEM", 1] == "Other"
    assert table_test.data.loc["Logger Serial Number", 1] == "Fulcrum3D"
        
    ## tests for measurements.get_table()

    for value in MM1.measurements.get_table(detailed=True).T['Dir_76mNW'].to_dict().values():
        if value != '-':
            assert value in MM1.measurements.properties[10].values()

    MM2.measurements.get_table()

    for value in FL1.measurements.get_table(detailed=True).T['Dir_80m_MC'].to_dict().values():
        if value != '-':
            assert value in FL1.measurements.properties[4].values()

    test_null_values_json = _get_schema(bw.demo_datasets.floating_lidar_iea43_wra_data_model_v1_2)
    for measurement_point in test_null_values_json['measurement_location'][0]['measurement_point']:
        measurement_point['height_m'] = None
    flr2 = bw.MeasurementStation(test_null_values_json)
    measurement_table = flr2.measurements.get_table(detailed=False, wind_speeds=False, wind_directions=False, 
                                                   calibrations=False, mounting_arrangements=False, columns_to_show=None)
    assert "meas_type_rank" not in measurement_table.columns
    # test that the height reference column is present in table even if all values are None
    assert "Height Reference" in measurement_table.columns
    assert "Name" in measurement_table.index.name

    for measurement_point in test_null_values_json['measurement_location'][0]['measurement_point']:
        measurement_point['height_ref_m'] = None
    with pytest.raises(ValueError) as except_info:
        bw.MeasurementStation(test_null_values_json)
    assert str(except_info.value) == "There is a problem with the validity of the supplied WRA data model please check the errors above."

    MM2.measurements.get_table(detailed=True)
    measurement_table_detailed = MM1.measurements.get_table(detailed=True)
    assert "Name" in measurement_table_detailed.index.name
    assert "Height [m]" in measurement_table_detailed.columns
    assert "sensor_rank" not in measurement_table_detailed.columns

    MM2.measurements.get_table(wind_speeds=True)
    measurement_table_wspd = MM1.measurements.get_table(wind_speeds=True)
    assert "measurement_type_id" not in measurement_table_wspd.columns

    measurement_table_wspd = SO1.measurements.get_table(wind_speeds=True)
    assert "Logger Notes" in measurement_table_wspd.columns

    MM2.measurements.get_table(wind_directions=True)
    measurement_table_wdir = MM1.measurements.get_table(wind_directions=True)
    assert "measurement_type_id" not in measurement_table_wdir.columns

    MM2.measurements.get_table(calibrations=True)
    measurement_table_calib = MM1.measurements.get_table(calibrations=True)
    assert "Calibration Slope" in measurement_table_calib.columns

    MM2.measurements.get_table(mounting_arrangements=True)
    measurement_table_calib = MM1.measurements.get_table(mounting_arrangements=True)
    assert "Boom Orientation [deg]" in measurement_table_calib.columns

    columns = ['calibration.slope', 'calibration.offset', 'calibration.report_file_name', 'date_of_calibration']
    MM2.measurements.get_table(columns_to_show=columns)
    measurement_table_input_cols = MM1.measurements.get_table(columns_to_show=columns)
    assert "Calibration Slope" in measurement_table_input_cols.columns

    columns = ['calibration.slope', 'calibration.offset', 'logger_measurement_config.sensitivity',
               'logger_measurement_config.height_m','calibration.sensitivity']
    measurement_table_input_cols = SS1.measurements.get_table(columns_to_show=columns)
    assert "Calibration Sensitivity" in measurement_table_input_cols.columns
    assert "Logger Sensitivity" in measurement_table_input_cols.columns

    
def test_properties():

    # Check thermometer has correct properties assigned
    properties = MM1.measurements.properties
    for measurement in properties:
        if "Tmp" in measurement['name'] and "logger" not in measurement['name']:
            assert measurement['measurement_type_id'] == "air_temperature"
            assert measurement['sensor_type_id'] == 'thermometer'

    # Check thermohygrometer has correct properties assigned
    obj_test = copy.deepcopy(MM1.measurements)
    for dm in obj_test._meas_data_model:
        if dm['name'] == 'Tmp_5m':
            dm['sensor'][0]['sensor_type_id'] = 'thermohygrometer'
            dm['sensor'][0]['calibration'] = [{'slope': 100.0, 'offset': -30.0, 'measurement_type_id': 'air_temperature'},
                                              {'slope': 100.0, 'offset': 0.0, 'measurement_type_id': 'relative_humidity'}]

    properties = obj_test._Measurements__get_properties()
    for measurement in properties:
        if "Tmp_5m" in measurement['name']:
            assert measurement['measurement_type_id'] == "air_temperature"
            assert measurement['sensor_type_id'] == 'thermohygrometer'

            
def test_get_names():
    names1 = FL1.measurements.names
    expected_names = ['Spd_80', 'Dir_80m', 'VSpd_80', 'Spd_80_MC', 'Dir_80m_MC', 'VSpd_80_MC', 'Water_Tmp',
                               'Wave_Height_Sig', 'Wave_Height_Max', 'Wave_Dir', 'Wave_Spread', 'Wave_PP', 'WtrSpd5m',
                               'VWtrSpd5m', 'WtrDir5m', 'Echo_5m', 'SNR_5m', 'Salinity_0m', 'Conductivity_0m', 
                               'Pressure_0m','Water_Level'] 
    assert names1 == expected_names

    assert 'Prs_76m' in MM1.measurements.get_names(measurement_type_id=None)
    assert MM1.measurements.get_names(measurement_type_id='air_temperature') == ['Tmp_78m', 'Tmp_5m']
    
    measurement_type_id2= 'vertical_wind_speed'
    names2 = FL1.measurements.get_names(measurement_type_id2)
    assert names2 == ['VSpd_80']
    
    measurement_type_id3= 'wind_speed'
    names3 = FL1.measurements.get_names(measurement_type_id3)
    assert names3 == ['Spd_80']

    measurement_type_id3= None
    names3 = FL1.measurements.get_names(measurement_type_id3)
    assert names3 == expected_names


def test_get_heights():

    # To get heights for all measurements:
    expected_heights =  [80.1, 80.2, 60.1, 60.2, 40.1, 30.1, 40.2, 76.1, 56.1, 78, 5, 5, 76, np.nan]
    assert MM1.measurements.get_heights(names=None, measurement_type_id=None) == expected_heights

    # To get heights only for defined names=['Spd_80mSE', 'Dir_76mNW']:
    assert MM1.measurements.get_heights(names=['Spd_80mSE', 'Dir_76mNW']) == [80.2, 76.1]

    # To get heights only for defined names='Spd_40mSE':
    assert MM1.measurements.get_heights(names='Spd_40mSE') == [40.2]

    # To get heights only for measurement_type_id='air_temperature':
    assert MM1.measurements.get_heights(measurement_type_id='air_temperature') == [78, 5]

    
