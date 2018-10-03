import numpy as np


def _range_0_to_360(dir):
    if dir < 0:
        return dir+360
    elif dir > 360:
        return dir % 360
    else:
        return dir


def get_direction_bin_array(sectors):
    bin_start = 180.0/sectors
    direction_bins = np.arange(bin_start, 360, 360.0/sectors)
    direction_bins = np.insert(direction_bins, 0, 0)
    direction_bins = np.append(direction_bins, 360)
    return direction_bins