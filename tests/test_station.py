import pytest
import brightwind as bw
import pandas as pd
import numpy as np
import warnings
import json


def _get_schema():
    with open(bw.demo_datasets.iea43_wra_data_model_schema_v1_0) as json_file:
        schema = json.load(json_file)
    return schema


SCHEMA = _get_schema()


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
    for name, title in property_name_title.items():
        assert bw.load.station._get_title(name, SCHEMA) == title
