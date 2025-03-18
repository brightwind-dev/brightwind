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

def test_solar_measurement_station_get_properties():
    # Test data for solar measurement station
    solar_data = bw.demo_datasets.solar_iea43_wra_data_model_v1_0
    
    # Create station and get properties
    station = bw.MeasurementStation(solar_data)
    table = station.get_table().data
    
    assert table.loc["Measurement Station Type"][0] == "solar"
    assert table.loc["Measurement Type"][0] == "global_horizontal_irradiance"
    assert table.loc["Measurement Units"][0] == "W/m2"