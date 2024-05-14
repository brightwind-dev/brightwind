from brightwind.load.station import MeasurementStation
from brightwind.utils import utils
import gmaps
import math


__all__ = ['plot_meas_station_on_gmap',
           'distance_between_points_haversine'
           ]


def _create_coord_dict(name, latitude, longitude):
    return {'name': name, 'coords': (latitude, longitude)}


def plot_meas_station_on_gmap(meas_station, map_type='TERRAIN', zoom_level=9):
    """
    Visualise on Google Maps the location of the measurement station.

    Unfortunately, to use this function you must have a Google Maps API key which should be free. To get one, follow
    the 'Get Started' instructions on the Google Maps Platform or go to the 'Credentials' section:

    https://cloud.google.com/maps-platform

    Once you have it, the GMAPS_API_KEY environmental variable will need to be set. In Windows this can be done
    by running the command prompt in Administrator mode and running:

    >> setx GMAPS_API_KEY "YourLongGoogleMapsAPIkey"

    If Anaconda or your Python environment is running you will need to restart it for the environmental variables to
    take effect.

    Additionally, the function requires the 'gmaps' library installed, instructions to install it are at the link below:
    https://jupyter-gmaps.readthedocs.io/en/latest/install.html

    :param meas_station:    A measurement station object which contains the latitude and longitude of it's location.
    :type meas_station:     bw.MeasurementStation
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
        import brightwind as bw
        mm1 = bw.MeasurementStation(bw.demo_datasets.demo_wra_data_model)
        bw.plot_meas_station_on_gmap(mm1)

        bw.plot_meas_station_on_gmap(mm1, map_type='SATELLITE')

    """
    gmaps.configure(api_key=utils.get_environment_variable('GMAPS_API_KEY'))
    figure_layout = {
        'width': '900px',
        'height': '600px',
        'margin': '0 auto 0 auto',
        'padding': '1px'
    }

    # Plot measurement locations
    meas_loc_points = []
    if isinstance(meas_station, MeasurementStation):
        point = _create_coord_dict(meas_station.name, meas_station.lat, meas_station.long)
        meas_loc_points.append(point)
    else:
        raise TypeError('Error with format of meas_station, please input a MeasurementStation object.')

    # Assign center of figure as first meas_loc in list
    fig = gmaps.figure(center=meas_loc_points[0]['coords'], map_type=map_type,
                       zoom_level=zoom_level, layout=figure_layout)

    for i_color, meas_loc_point in enumerate(meas_loc_points):
        marker = gmaps.marker_layer([meas_loc_point['coords']],
                                    info_box_content=meas_loc_point['name'],
                                    display_info_box=True)
        fig.add_layer(marker)
    return fig


def distance_between_points_haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two geographical points using the Haversine formula. The distance is returned in km.

    :param lat1:    Latitude of the first point in decimal degrees.
    :type lat1:     float
    :param lon1:    Longitude of the first point in decimal degrees.
    :type lon1:     float
    :param lat2:    Latitude of the second point in decimal degrees.
    :type lat2:     float
    :param lon2:    Longitude of the second point in decimal degrees.
    :type lon2:     float
    :return:        Distance in km.
    :rtype:         float
    """
    radius = 6371  # Radius of the Earth in kilometers

    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = radius * c

    return distance
