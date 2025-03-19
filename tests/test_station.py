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

    with open(bw.demo_datasets.floating_lidar_iea43_wra_data_model_v1_2) as json_file:
        test_data_null_values = json.load(json_file)
    for measurement_point in test_data_null_values["measurement_location"][0]["measurement_point"]:
        measurement_point['notes'] = None
    mm2 = bw.MeasurementStation(test_data_null_values)
    measurement_table = mm2.measurements.get_table()
    assert "notes" not in measurement_table.columns

    measurement_table_detailed = mm2.measurements.get_table(detailed=True)
    assert "Name" in measurement_table_detailed.index.name
    assert "Height [m]" in measurement_table_detailed.columns
    
    df_ws = mm2.measurements.get_table(wind_speeds=True)
    assert "measurement_type_id" not in df_ws.columns
    
    # Ranking column is dropped after sorting
    df = mm2.measurements.get_table()
    assert "meas_type_rank" not in df.columns
    
    # Custom columns handling empty data
    cols = ["name", "height_m", "empty_col1", "empty_col2"]
    df_custom = mm2.measurements.get_table(columns_to_show=cols)
    assert df_custom.shape[0] > 0
    assert "Name" in df_custom.index.name

    df_wd = mm2.measurements.get_table(wind_directions=True)
    assert "measurement_type_id" not in df_wd.columns