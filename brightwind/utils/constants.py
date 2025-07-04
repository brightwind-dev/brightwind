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