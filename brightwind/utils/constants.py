from typing import NamedTuple

class Extents(NamedTuple):
    north: float
    south: float
    west: float
    east: float

NEWA_EXTENTS = Extents(north=72.25, south=31.89, west=-19.5, east=47.80)
NEWA_EXTENT_BOUNDS = (NEWA_EXTENTS.west, NEWA_EXTENTS.south, NEWA_EXTENTS.east, NEWA_EXTENTS.north)

NEWA_VALID_HEIGHTS = {
    "mesoscale": [50, 75, 100, 150, 200, 250, 500],
    "microscale": [50, 100, 200],
    }

WIND_MAP_BUFFER_EPSILON = 0.05

NEWA_VARIABLES_BY_HEIGHT = {
    "mesoscale": [
                    "wind_speed_min",
                    "wind_speed_max",
                    "wind_speed_mean",
                    "wind_speed_std",
                    "power_density_min",
                    "power_density_max",
                    "power_density_mean",
                    "power_density_std",
                ],
    "microscale": ["wind_speed"],
}

NEWA_VARIABLES_WITHOUT_HEIGHT = {
    "mesoscale": [
        'WS10_min', 'WS10_max', 'WS10_mean', 'WS10_std', 'T2_min', 'T2_max', 'T2_mean', 'T2_std', 'T100_min', 'T100_max', 
        'T100_mean', 'T100_std', 'tke50_min', 'tke50_max', 'tke50_mean', 'tke50_std', 'rho_min', 'rho_max', 'rho_mean', 
        'rho_std', 'Q2_min', 'Q2_max', 'Q2_mean', 'Q2_std', 'Q100_min', 'Q100_max', 'Q100_mean', 'Q100_std', 
        'pbl_height_min', 'pbl_height_max', 'pbl_height_mean', 'pbl_height_std', 'elevation', 'landmask', 'lu_index'
        ],
    "microscale": [
        "elevation", "rix"
        ],
}

GWA_VARIABLE_HEIGHTS = [10, 50, 100, 150, 200]

GWA_VARIABLES_WITH_HEIGHT = [
    "wind-speed",
    "air-density",
    "power-density",
    "combined-Weibull-A",
    "combined-Weibull-k",
]
GWA_VARIABLES_WITHOUT_HEIGHT = [
    "elevation_w_bathymetry",
    "capacity-factor_IEC1",
    "capacity-factor_IEC2",
    "capacity-factor_IEC3",
    "IEC-class-fatigue-loads",
    "IEC-class-fatigue-loads-incl-wake",
    "IEC-class-extreme-loads",
]