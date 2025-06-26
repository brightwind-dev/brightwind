import brightwind as bw
import geopandas as gpd
import numpy as np
import rioxarray
import pandas as pd
import pytest
import xarray as xr

from shapely.geometry import Point, Polygon, LineString
from pyproj import CRS

import brightwind.utils.constants as Constants

DATA = bw.load_csv(bw.demo_datasets.demo_data)
DATA = bw.apply_cleaning(DATA, bw.demo_datasets.demo_cleaning_file)
WSPD_COLS = ['Spd80mN', 'Spd80mS', 'Spd60mN', 'Spd60mS', 'Spd40mN', 'Spd40mS']
WDIR_COLS = ['Dir78mS', 'Dir58mS', 'Dir38mS']

DATA_GEOSPATIAL = {
    'name': ['A', 'B', 'C', 'D'],
    'geometry': [
        Point(5, 23),
        Point(-3.9, 57.3),
        Polygon([(-3, 56), (-2, 56), (-2, 56.1), (-3, 56.1), (-3, 56)]),
        LineString([(-3, 56), (-2, 56), (-2, 56.1), (-3, 56.1), (-3, 56)])
    ]
}
TEST_WIND_MAP_EXTRACTION_GDF = gpd.GeoDataFrame(DATA_GEOSPATIAL, crs="EPSG:4326")

lons = np.linspace(-5, 0, 10)
lats = np.linspace(55, 58, 10)
rng = np.random.default_rng(seed=42)
data = rng.random((10, 10))

TEST_DATASET = xr.Dataset(
    {
        "wind_speed": (["y", "x"], data)
    },
    coords={
        "x": lons,
        "y": lats
    }
)

TEST_DATASET = TEST_DATASET.rio.write_crs("EPSG:4326")
wkt_4326 = CRS.from_epsg(4326).to_wkt()
TEST_DATASET = TEST_DATASET.assign_coords(
    crs=xr.DataArray(
        data=0,  # dummy scalar value
        attrs={'crs_wkt': wkt_4326}
    )
)
TEST_DATASET = TEST_DATASET.rio.write_crs("EPSG:4326", inplace=True)


def test_slice_data():
    data_sliced = bw.utils.utils.slice_data(DATA, date_from='2016-11-23 00:30', date_to='2017-10-23 12:20')

    assert data_sliced.index[0] == pd.Timestamp('2016-11-23 00:30')
    assert data_sliced.index[-1] == pd.Timestamp('2017-10-23 12:10')

    data_sliced = bw.utils.utils.slice_data(DATA, date_from='2016-11-23', date_to='2017-10-23')

    assert data_sliced.index[0] == pd.Timestamp('2016-11-23 00:00')
    assert data_sliced.index[-1] == pd.Timestamp('2017-10-22 23:50')

    data_sliced = bw.utils.utils.slice_data(DATA, date_from='2016-11-23')
    assert data_sliced.index[-1] == DATA.index[-1]

    data_sliced = bw.utils.utils.slice_data(DATA, date_to='2017-10-23')
    assert data_sliced.index[0] == DATA.index[0]


def test_get_country_code_for_geometry():

    actual_country = "GBR"
    geom = TEST_WIND_MAP_EXTRACTION_GDF.iloc[1].geometry
    country = bw.utils.wind_map.get_country_code_for_geometry(geom)
    assert actual_country == country

    actual_country = "DZA"
    geom = TEST_WIND_MAP_EXTRACTION_GDF.iloc[0].geometry
    country = bw.utils.wind_map.get_country_code_for_geometry(geom)
    assert actual_country == country

    actual_country = "GBR"
    geom = TEST_WIND_MAP_EXTRACTION_GDF.iloc[2].geometry
    country = bw.utils.wind_map.get_country_code_for_geometry(geom)
    assert actual_country == country


def get_country_code_from_coordinates():

    latitude = TEST_WIND_MAP_EXTRACTION_GDF.iloc[0].geometry.y
    longitude = TEST_WIND_MAP_EXTRACTION_GDF.iloc[0].geometry.x
    actual_country = "DZA"
    country = bw.utils.wind_map.get_country_code_from_coordinates(latitude, longitude)
    assert actual_country == country

    latitude = TEST_WIND_MAP_EXTRACTION_GDF.iloc[1].geometry.y
    longitude = TEST_WIND_MAP_EXTRACTION_GDF.iloc[1].geometry.x
    actual_country = "GBR"
    country = bw.utils.wind_map.get_country_code_from_coordinates(latitude, longitude)
    assert actual_country == country

def test_check_newa_location_valid():

    row = TEST_WIND_MAP_EXTRACTION_GDF.iloc[0]
    valid = bw.utils.wind_map.check_newa_location_valid(row, Constants.NEWA_EXTENT_BOUNDS)
    assert "Invalid location" in valid

    row = TEST_WIND_MAP_EXTRACTION_GDF.iloc[1]
    valid = bw.utils.wind_map.check_newa_location_valid(row, Constants.NEWA_EXTENT_BOUNDS)
    assert "Valid location" in valid


def test_download_newa_data():

    row = TEST_WIND_MAP_EXTRACTION_GDF.iloc[1]
    newa_returned = bw.utils.wind_map.download_newa_data(row, "wind_speed_avg", 100, "mesoscale")
    assert "Variable name not found please consult documentation" in newa_returned

    row = TEST_WIND_MAP_EXTRACTION_GDF.iloc[0]
    newa_returned = bw.utils.wind_map.download_newa_data(row, "wind_speed_mean", 100, "mesoscale")
    assert "Invalid location: NEWA covers" in newa_returned

    row = TEST_WIND_MAP_EXTRACTION_GDF.iloc[3]
    newa_returned = bw.utils.wind_map.download_newa_data(row, "wind_speed_mean", 100, "mesoscale")
    assert "Invalid geometry type must be Point or Polygon" in newa_returned
    
    # row = TEST_WIND_MAP_EXTRACTION_GDF.iloc[1]
    # newa_returned = bw.utils.wind_map.download_newa_data(row, "wind_speed_mean", 100, "mesoscale")
    # assert 
    
    # row = TEST_WIND_MAP_EXTRACTION_GDF.iloc[2]
    # newa_returned = bw.utils.wind_map.download_newa_data(row, "wind_speed_mean", 100, "mesoscale")
    # assert 


def test_reproject_xarray_dataset():

    reprojected_ds = bw.utils.wind_map._reproject_xarray_dataset(TEST_DATASET, "EPSG:27700", "x", "y")
    new_crs = reprojected_ds.rio.crs

    assert '["EPSG","27700"]' in str(new_crs), str(new_crs)


def test_call_wind_map_api():

    new_gdf_gwa = bw.utils.wind_map.call_wind_map_api("gwa", TEST_WIND_MAP_EXTRACTION_GDF, "wind-speed", 
                                                  "EPSG:4326", height_requested=100)
    expected_column_name = "gwa_wind-speed_100m"
    assert expected_column_name in new_gdf_gwa.columns
    assert np.allclose(new_gdf_gwa.iloc[0][expected_column_name], 6.683101654051394)
    assert expected_column_name in new_gdf_gwa.columns
    assert np.allclose(new_gdf_gwa.iloc[1][expected_column_name], 8.413563728348029)
    assert isinstance(new_gdf_gwa.iloc[2][expected_column_name], xr.DataArray)
    assert "Invalid geometry type" in new_gdf_gwa.iloc[3][expected_column_name]

    new_gdf_newa = bw.utils.wind_map.call_wind_map_api("newa-mesoscale", TEST_WIND_MAP_EXTRACTION_GDF, "wind_speed_avg", 
                                                  "EPSG:4326", height_requested=100)
    assert ("Variable name not found please consult documentation" in 
            new_gdf_newa.iloc[1]["newa-mesoscale_wind_speed_avg_100m"])

    # new_gdf_newa = bw.utils.wind_map.call_wind_map_api("newa-mesoscale", TEST_WIND_MAP_EXTRACTION_GDF, "wind_speed_mean", 
    #                                               "EPSG:4326", height_requested=100)
    # expected_column_name = "newa-mesoscale_wind_speed_mean_100m"
    # assert "Invalid location: NEWA covers" in new_gdf_newa.iloc[0][expected_column_name]
    # assert "Invalid geometry type" in new_gdf_newa.iloc[3][expected_column_name]
    # Other tests to be written when
    


def test_buffer_only_polygons():

    geom = TEST_WIND_MAP_EXTRACTION_GDF.iloc[1].geometry
    new_geom = bw.utils.wind_map._buffer_only_polygons(geom, Constants.WIND_MAP_BUFFER_EPSILON)
    assert geom == new_geom

    geom = TEST_WIND_MAP_EXTRACTION_GDF.iloc[0].geometry
    new_geom = bw.utils.wind_map._buffer_only_polygons(geom, Constants.WIND_MAP_BUFFER_EPSILON)
    assert geom == new_geom

    geom = TEST_WIND_MAP_EXTRACTION_GDF.iloc[2].geometry
    new_geom = bw.utils.wind_map._buffer_only_polygons(geom, Constants.WIND_MAP_BUFFER_EPSILON)
    assert geom != new_geom
    assert isinstance(new_geom, Polygon)


def test_extract_from_dataset_at_geometry():

    row = TEST_WIND_MAP_EXTRACTION_GDF.iloc[1]
    data_at_point = bw.utils.wind_map.extract_from_dataset_at_geometry(row, [TEST_DATASET["wind_speed"]])
    print(data_at_point)
    assert isinstance(data_at_point, float)

    row = TEST_WIND_MAP_EXTRACTION_GDF.iloc[0]
    data_at_point = bw.utils.wind_map.extract_from_dataset_at_geometry(row, [TEST_DATASET["wind_speed"]])
    assert "outside dataset bounds" in data_at_point

    row = TEST_WIND_MAP_EXTRACTION_GDF.iloc[2]
    data_at_point = bw.utils.wind_map.extract_from_dataset_at_geometry(row, [TEST_DATASET["wind_speed"]])
    assert isinstance(data_at_point, xr.DataArray)

    row = TEST_WIND_MAP_EXTRACTION_GDF.iloc[3]
    data_at_point = bw.utils.wind_map.extract_from_dataset_at_geometry(row, [TEST_DATASET["wind_speed"]])
    assert "Invalid geometry type" in data_at_point