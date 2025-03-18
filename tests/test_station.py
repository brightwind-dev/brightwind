import pytest
import brightwind as bw
import pandas as pd
import numpy as np
import warnings
import json


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
    mm1 = bw.MeasurementStation(bw.demo_datasets.iea43_wra_data_model_v1_0)

    for value in mm1.get_table(horizontal_table_orientation=True).T['Test_MM1'].to_dict().values():
        assert value in mm1.properties.values()

    for value in mm1.logger_main_configs.get_table(horizontal_table_orientation=True).T['AName_MM1'].to_dict().values():
        assert value in mm1.logger_main_configs.properties[0].values()

    for value in mm1.measurements.get_table(detailed=True).T['Dir_76mNW'].to_dict().values():
        if value is not '-':
            assert value in mm1.measurements.properties[10].values()

    mm2 = bw.MeasurementStation(bw.demo_datasets.floating_lidar_iea43_wra_data_model_v1_2)

    for value in mm2.measurements.get_table(detailed=True).T['Dir_80m_MC'].to_dict().values():
        if value is not '-':
            assert value in mm2.measurements.properties[4].values()

def test_get_table_drop_methodology(monkeypatch):
    # Test data with known empty columns and rows
    test_data = {
        "version": "1.2.0-2023.01",
        "measurement_location": [{
            "measurement_point": [
                {
                    "name": "WS1",
                    "height_m": 80,
                    "measurement_type_id": "wind_speed",
                    "empty_col1": None,
                    "empty_col2": None,
                    "logger_measurement_config": [
                        {
                            "date_from": "2020-01-01",
                            "date_to": "2020-12-31",
                            "slope": 1.0
                        }
                    ]
                },
                {
                    "name": "WS2", 
                    "height_m": 60,
                    "measurement_type_id": "wind_speed",
                    "empty_col1": None,
                    "empty_col2": None,
                    "logger_measurement_config": [
                        {
                            "date_from": "2020-01-01",
                            "date_to": None,
                            "slope": None
                        }
                    ]
                }
            ]
        }]
    }
    
    mm = bw.MeasurementStation(test_data)
    
    # Test 1: Default table drops empty columns
    df = mm.measurements.get_table()
    assert "empty_col1" not in df.columns
    assert "empty_col2" not in df.columns
    
    # Test 2: Detailed table drops empty columns but keeps required ones
    df_detailed = mm.measurements.get_table(detailed=True)
    assert "Name" in df_detailed.index.name
    assert "Height [m]" in df_detailed.columns
    
    # Test 3: Wind speeds table drops measurement_type_id after filtering
    df_ws = mm.measurements.get_table(wind_speeds=True)
    assert "measurement_type_id" not in df_ws.columns
    assert "Measurement Type" not in df_ws.columns
    
    # Test 4: Ranking column is dropped after sorting
    df = mm.measurements.get_table()
    assert "meas_type_rank" not in df.columns
    
    # Test 5: Custom columns handling empty data
    cols = ["name", "height_m", "empty_col1", "empty_col2"]
    df_custom = mm.measurements.get_table(columns_to_show=cols)
    assert df_custom.shape[0] > 0  # Should still return rows even with empty columns
    assert "Name" in df_custom.index.name

def test_get_table_pandas_version_consistency():
    # Create test data that could be affected by pandas version differences
    test_data = {
        "version": "1.2.0-2023.01",
        "measurement_location": [{
            "measurement_point": [
                {
                    "name": "Test1",
                    "height_m": 100,
                    "measurement_type_id": "wind_speed",
                    "null_column": None,
                    "sensor_type_id": 100,
                    "logger_measurement_config": [
                        {
                            "date_from": "2020-01-01",
                            "date_to": "2020-12-31",
                            "some_value": np.nan
                        }
                    ]
                }
            ]
        }]
    }
    test_wind_direction_data = {
        "version": "1.2.0-2023.01",
        "measurement_location": [{
            "measurement_point": [
                {
                    "name": "Test1",
                    "height_m": 100,
                    "measurement_type_id": "wind_direction",
                    "null_column": None,
                    "sensor_type_id": 100,
                    "logger_measurement_config": [
                        {
                            "date_from": "2020-01-01",
                            "date_to": "2020-12-31",
                            "some_value": np.nan
                        }
                    ]
                }
            ]
        }]
    }
    
    mm = bw.MeasurementStation(test_data)
    mm2 = bw.MeasurementStation(test_wind_direction_data)
    
    df1 = mm.measurements.get_table()
    df2 = mm.measurements.get_table(detailed=True)
    df3 = mm.measurements.get_table(wind_speeds=True)
    df4 = mm2.measurements.get_table(wind_directions=True)
    
    assert df1.isnull().sum().sum() == 0
    
    assert df1.index.name == 'Name'
    assert df2.index.name == 'Name'
    
    assert "null_column" not in df1.columns
    assert "null_column" not in df3.columns
    assert "null_column" not in df4.columns

    assert "measurement_type_id" not in df3.columns
    assert "measurement_type_id" not in df4.columns
