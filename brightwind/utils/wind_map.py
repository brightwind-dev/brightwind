
from shapely.geometry import Point, Polygon, box
import pycountry
from geopy.geocoders import Nominatim
import geopy.exc
import requests
import time
import warnings

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import random
import xarray as xr
import rioxarray
from io import BytesIO
from pyproj import Transformer, CRS
import numpy as np

import brightwind.utils.constants as Constants


__all__ = ['call_wind_map_api']


def get_country_code_for_geometry(geom):
    """
    Function to return the country code for a shapely.geometry object. For Points, using the point itself; for Polygons, 
    using the centroid.

    :param geom:      Geometry object that the country code will be returned for
    :type geom:       Point | Polygon
    :return:          The three letter country code
    :rtype:           str
    """

    if geom.is_empty:
        return None
    point = geom if geom.geom_type == 'Point' else geom.centroid
    return get_country_code_from_coordinates(point.y, point.x)


def get_country_code_from_coordinates(latitude, longitude):
    """
    Gets the 3-letter country code for a given latitude, longitude pair

    :param latitude:    Latitude of point in question
    :type latitude:     float
    :param longitude:   Longitude of point in question
    :type longitude:    float
    :returns:           3-letter country code
    :rtype:             str
    """

    geolocator = Nominatim(user_agent="your_app_name")

    try:
        location = geolocator.reverse((latitude, longitude), language="en")
    except geopy.exc.GeocoderTimedOut:
        return None

    if location and "address" in location.raw:
        country_code = location.raw["address"].get("country_code")
        if country_code:
            # Convert the two-letter country code to three-letter code
            country = pycountry.countries.get(alpha_2=country_code.upper())
            if country:
                return country.alpha_3
    return None


def check_newa_location_valid(row, bounds):
    """
    Function which checks whether a shapely geometry is within the extent of NEWA wind map.

    :param row:        A row from a GeoDataFrame, expected to have a 'geometry' attribute containing a shapely geometry 
                       (Point or Polygon).
    :type row:         pandas.Series or geopandas.GeoSeries
    :param bounds:     The bounds of the NEWA wind map
    :type bounds:      tuple
    :return:           A descriptor if the location is or is not valid to extract NEWA wind map data at
    :rtype:            str
    """
    bbox = box(*bounds)
    def apply_func(geom):
        if geom.within(bbox):
            return "Valid location"
        else:
            return (f"Invalid location: NEWA covers ({Constants.NEWA_EXTENTS.north}°N to {Constants.NEWA_EXTENTS.south}°" 
                    f"N and {Constants.NEWA_EXTENTS.west}°W to {Constants.NEWA_EXTENTS.east}°W)")
    location_check = apply_func(row.geometry)
    return location_check



def download_newa_data(row, variable_requested, height_requested, model_type):
    """
    Function to download NEWA wind map data from either the meso scale or the micro scale model.

    :param row:                   A row from a GeoDataFrame, expected to have a 'geometry' attribute containing a 
                                  shapely geometry (Point or Polygon).
    :type row:                    pandas.Series or geopandas.GeoSeries
    :param variable_requested:    Variable name required
    :type variable_requested:     str
    :param height_requested:      Height required
    :type height_requested:       float
    :param model_type:            Model type to download from either "mesoscale" or "microscale"
    :type model_type:             str 
    :return:                      The extracted value(s) from the NEWA wind map for the given geometry and parameters.
                                  This may be a scalar value (for points), or an xarray.DataArray
    :rtype:                       float | xarray.DataArray | str
    """

    newa_data = check_newa_location_valid(row, Constants.NEWA_EXTENT_BOUNDS)
    
    if isinstance(row.geometry, Point):
        base_url = f"https://wps.neweuropeanwindatlas.eu/api/{model_type}-atlas/v1/get-data-point"
        url_params = {
            "latitude": row.geometry.y,
            "longitude": row.geometry.x,
            "variable": variable_requested
        }
    elif isinstance(row.geometry, Polygon):
        base_url = f"https://wps.neweuropeanwindatlas.eu/api/{model_type}-atlas/v1/get-data-bbox"
        polygon_bounds = row.geometry.bounds
        url_params = {
            "southBoundLatitude": polygon_bounds[3] - Constants.WIND_MAP_BUFFER_EPSILON,
            "northBoundLatitude": polygon_bounds[1] + Constants.WIND_MAP_BUFFER_EPSILON,
            "westBoundLongitude": polygon_bounds[0] - Constants.WIND_MAP_BUFFER_EPSILON,
            "eastBoundLongitude": polygon_bounds[2] + Constants.WIND_MAP_BUFFER_EPSILON,
            "variable": variable_requested
        }
    else:
        return "Invalid geometry type must be Point or Polygon"
    
    if "Invalid location" not in newa_data and "Invalid geometry type" not in newa_data:
        if variable_requested in Constants.NEWA_VARIABLES_BY_HEIGHT[model_type]:
            if height_requested is None or height_requested not in Constants.NEWA_VALID_HEIGHTS[model_type]:
                error_msg = f"Height must be one of {Constants.NEWA_VALID_HEIGHTS[model_type]}"
                warnings.warn(error_msg)
                return error_msg
            else:
                url_params["height"] = height_requested
                newa_data = call_api_with_retry_logic(base_url, url_params)
                if not isinstance(newa_data, str):
                    newa_data = xr.open_dataset(BytesIO(newa_data))
        elif variable_requested in Constants.NEWA_VARIABLES_WITHOUT_HEIGHT[model_type]:
            newa_data = call_api_with_retry_logic(base_url, url_params)
            if not isinstance(newa_data, str):
                newa_data = xr.open_dataset(BytesIO(newa_data))
        else:
            newa_data = "Variable name not found please consult documentation"
            warnings.warn(newa_data)

    if isinstance(row.geometry, Point) and not isinstance(newa_data, str):
        if "height" in newa_data.coords:
            newa_data = newa_data.isel(height=0)[variable_requested].values.flatten()[0]
        else:
            newa_data = newa_data[variable_requested].values.flatten()[0]
    if isinstance(row.geometry, Polygon) and not isinstance(newa_data, str):
        if "height" in newa_data.coords:
            newa_data = newa_data.isel(height=0)[variable_requested]
        else:
            newa_data = newa_data[variable_requested]
        newa_data = _reproject_xarray_dataset(newa_data, "EPSG:4326", "west_east", "south_north")

    return newa_data


def _reproject_xarray_dataset(ds, epsg_required, x_variable, y_variable):
    """
    Reprojects an xarray dataset from the projection stored in the crs attribute into a user supplied EPSG code.

    :param ds:                         The xarray Dataset to reproject
    :type ds:                          xr.Dataset
    :param epsg_required:              The epsg code the output dataset is required in
    :type epsg_required:               str
    :param x_variable:                 Name of the x-coordinate variable
    :type x_variable:                  str
    :param y_variable:                 Name of the y-coordinate variable
    :type y_variable:                  str
    :return:                           The re-projected Dataset
    :rtype:                            xr.Dataset
    """
    src_crs = CRS.from_wkt(ds.crs.attrs.get('crs_wkt', '')) if 'crs_wkt' in ds.crs.attrs else None
    transformer = Transformer.from_crs(src_crs, epsg_required, always_xy=True)
    x_coords = ds[x_variable].values
    y_coords = ds[y_variable].values
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
    lon_mesh, lat_mesh = transformer.transform(x_mesh, y_mesh)
    ds_reproj = ds.copy()
    ds_reproj = ds_reproj.assign_coords({
            x_variable: ([x_variable], lon_mesh[0, :]),
            y_variable: ([y_variable], lat_mesh[:, 0])
        })
    ds = ds_reproj.rio.write_crs(epsg_required)
    return ds
        

def call_api_with_retry_logic(base_url, url_params, max_retries = 5, backoff_factor = 2, timeout = 500):

    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        status_forcelist=[500, 502, 503, 504, 429],
        method_whitelist=["GET"],
        backoff_factor=backoff_factor,
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                delay = backoff_factor * (2 ** (attempt - 1)) + random.uniform(0, 1)
                print(f"Attempt {attempt + 1}/{max_retries + 1} - waiting {delay:.1f}s before retry...")
                time.sleep(delay)
            
            response = session.get(base_url, params=url_params, timeout=timeout)
            if response.status_code == 200:
                break
            else:
                if attempt == max_retries:
                    response.raise_for_status()  # Raise on final attempt
                continue
                
        except requests.exceptions.Timeout as e:
            last_exception = e
            if attempt == max_retries:
                print(f"Final attempt failed: {e}")
                return None
                
        except requests.exceptions.ConnectionError as e:
            last_exception = e
            if attempt == max_retries:
                return f"Final API call attempt failed: {e}"
                
        except requests.exceptions.RequestException as e:
            last_exception = e
            if attempt == max_retries:
                return f"Final API call attempt failed: {e}"
    else:
        return f"All {max_retries + 1} API call attempts failed. Last error: {last_exception}"
    return response.content


def call_wind_map_api(wind_map_name, location_to_query, variable_requested, input_crs, height_requested=None):
    """
    Function to facilitate the call of open access wind map APIs for the return of wind map data for a point location or
    area of interest.

    :param wind_map_name:        The name of the windmap that data will be extracted from. Options are currently:
                                 "gwa" or ""
    :type wind_map_name:         _type_
    :param location_to_query:    The point location or area that the user would like to return wind map data from
    :type location_to_query:     geodataframe
    :param variable_requested:
    :type variable_requested:    str
    :param input_crs:
    :type input_crs:    str
    :param height_requested:
    :type height_requested:    int
    """

    location_to_query = location_to_query.set_crs(input_crs) 
    location_to_query = location_to_query.to_crs("EPSG:4326")

    variable_ouput_name = f"{wind_map_name}_{variable_requested}_{height_requested}m" if height_requested else f"{wind_map_name}_{variable_requested}"

    if wind_map_name == "gwa":
        if variable_requested not in Constants.GWA_VARIABLES_WITH_HEIGHT and variable_requested not in Constants.GWA_VARIABLES_WITHOUT_HEIGHT:
            location_to_query[variable_ouput_name] = "Invalid variable requested"
            return location_to_query
        if variable_requested in Constants.GWA_VARIABLES_WITH_HEIGHT:
            if height_requested not in Constants.GWA_VARIABLE_HEIGHTS:
                location_to_query[variable_ouput_name] = ("Invalid height "
                f"requested for this variable, heights available are: {Constants.GWA_VARIABLE_HEIGHTS}")
                return location_to_query
        location_to_query['country_code'] = location_to_query.geometry.apply(get_country_code_for_geometry)
        countries_required = set(location_to_query['country_code'])
        gwa_datasets = []
        for country_code in countries_required:
            if variable_requested in Constants.GWA_VARIABLES_WITH_HEIGHT:
                gwa_api_url = rf"https://globalwindatlas.info/api/gis/country/{country_code}/{variable_requested}/{height_requested}"
            else:
                gwa_api_url = rf"https://globalwindatlas.info/api/gis/country/{country_code}/{variable_requested}"
            ds = rioxarray.open_rasterio(gwa_api_url)
            gwa_datasets.append(ds)

        location_to_query['old_geom'] = location_to_query.geometry
        location_to_query.geometry = location_to_query.geometry.apply(
            lambda g: _buffer_only_polygons(g, Constants.WIND_MAP_BUFFER_EPSILON)
            )
        location_to_query[variable_ouput_name] = location_to_query.apply(
            lambda row: extract_from_dataset_at_geometry(row, gwa_datasets), axis=1
            )
        location_to_query.geometry = location_to_query['old_geom']
        del location_to_query['old_geom']

    if "newa" in wind_map_name:
        model_type = wind_map_name.split("-")[-1]
        location_to_query[variable_ouput_name] = location_to_query.apply(
            lambda row: download_newa_data(row, variable_requested, height_requested, model_type), axis=1
            )
    
    return location_to_query
    

def _buffer_only_polygons(geom, buffer_dist):
    """
    Function which buffers a geometry only if it is a Polygon object.

    :param geom:          The geometry to buffer
    :type geom:           shapely.geometry.base.BaseGeometry
    :param buffer_dist:   The amount to buffer the polygon by the EPSG which the geometry is in 
                          (either degrees or meters)
    :type buffer_dist:    float
    :return:              The geometry with a buffer applied if input is a Polygon; otherwise, the original geometry.
    :rtype:               shapely.geometry.base.BaseGeometry
    """
    if geom.geom_type == 'Polygon':
        return geom.buffer(buffer_dist)
    return geom


def extract_from_dataset_at_geometry(row, dss):
    """
    Function to extract data from a Dataset or DataArray at a point location and return a float or at a Polygon and 
    return a Dataset or DataArray containing values clipped to this area. Other geometry types are not supported and
    result in return of "Invalid geometry type".

    :param row:                      A row from a GeoDataFrame, expected to have a 'geometry' attribute containing a 
                                     shapely geometry (Point or Polygon).
    :type row:                       pandas.Series or geopandas.GeoSeries
    :param dss:                      List of Dataset or DataArray to extract data from at the geometry specified
    :type dss:                       List[xr.Dataset] | List[xr.DataArray]
    :return:                         Extracted GWA value at Point, GWA surface for Polygon or "Invalid geometry type" for
                                     any other geometry type
    :rtype:                          str | float | xr.DataArray
    """
    
    for ds in dss:
        minx, maxx = float(ds.x.min()), float(ds.x.max())
        miny, maxy = float(ds.y.min()), float(ds.y.max())
        if isinstance(row.geometry, Point):
            if not (minx <= row.geometry.x <= maxx and miny <= row.geometry.y <= maxy):
                continue
            ds_at_geometry = ds.interp(
                x=row.geometry.x, y=row.geometry.y, method="linear"
                ).item()
            if not np.isnan(ds_at_geometry):
                return ds_at_geometry
        elif isinstance(row.geometry, Polygon):
            if not row.geometry.intersects(box(minx, miny, maxx, maxy)):
                continue
            
            data_at_centroid = ds.interp(
                x=row.geometry.centroid.x, y=row.geometry.centroid.y, method="linear"
                ).item()
            if not np.isnan(data_at_centroid):
                return ds.rio.clip([row.geometry], drop=True)
        else:
            return "Invalid geometry type"
    return "Geometry outside dataset bounds"