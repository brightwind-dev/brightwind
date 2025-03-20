import pytest
import brightwind as bw
import pandas as pd
import numpy as np
import warnings
import json

MM1 = bw.MeasurementStation(bw.demo_datasets.iea43_wra_data_model_v1_0)
MM2 = bw.MeasurementStation(bw.demo_datasets.demo_wra_data_model)
FL1 = bw.MeasurementStation(bw.demo_datasets.floating_lidar_iea43_wra_data_model_v1_2)
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
# df.loc['Logger Name']#['AName_MM1'].to_dict().values()

    for value in MM1.logger_main_configs.get_table(horizontal_table_orientation=True).T['AName_MM1'].to_dict().values():
        assert value in MM1.logger_main_configs.properties[0].values()
        
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
        measurement_point['height_reference'] = None
    flr2 = bw.MeasurementStation(test_null_values_json)
    measurement_table = flr2.measurements.get_table(detailed=False, wind_speeds=False, wind_directions=False, 
                                                   calibrations=False, mounting_arrangements=False, columns_to_show=None)
    assert "meas_type_rank" not in measurement_table.columns
    # test that the height reference column is present in table even if all values are None
    assert "Height Reference" in measurement_table.columns
    assert "Name" in measurement_table.index.name

    MM2.measurements.get_table(detailed=True)
    measurement_table_detailed = MM1.measurements.get_table(detailed=True)
    assert "Name" in measurement_table_detailed.index.name
    assert "Height [m]" in measurement_table_detailed.columns
    assert "sensor_rank" not in measurement_table_detailed.columns

    MM2.measurements.get_table(wind_speeds=True)
    measurement_table_wspd = MM1.measurements.get_table(wind_speeds=True)
    assert "measurement_type_id" not in measurement_table_wspd.columns

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
