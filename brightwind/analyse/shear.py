from typing import List, Dict
import numpy as np


def calc_shear(wind_speeds: List[float], heights: List[float], plot=False) -> float:
    """
    Derive the best fit power law exponent (as 1/alpha) from a given time-step of speed data at 2 or more elevations
    :param wind_speeds: List of wind speeds [m/s]
    :param heights: List of heights [m above ground]. The position of the height in the list must be the same position in the list as its
    corresponding wind speed value.
    :return: The shear value (alpha), as the inverse exponent of the best fit power law, based on the form: (v1/v2) = (z1/z2)^(1/alpha)

    METHODOLOGY:
        Derive natural log of elevation and speed data sets
        Derive coefficients of linear best fit along log-log distribution
        Characterise new distribution of speed values based on linear best fit
        Derive 'alpha' based on gradient of first and last best fit points (function works for 2 or more points)
        Return alpha value
    """

    logheights = np.log(heights)  # take log of elevations
    logwind_speeds = np.log(wind_speeds)  # take log of speeds
    coeffs = np.polyfit(logheights, logwind_speeds, deg=1)  # get coefficients of linear best fit to log distribution
    poly = np.poly1d(coeffs)
    wind_speedsfit = lambda heights: np.exp(poly(np.log(heights)))  # characterise values of v along log best fit
    alpha = 1 / ((np.log(wind_speedsfit(heights[-1]) / wind_speedsfit(heights[0]))) / (np.log(heights[-1] / heights[0])))  # derive alpha based on (reciprocal of)
    # gradient defined by the 2 outer points on the log best fit

    if plot:
        return alpha, wind_speedsfit
    else:
        return alpha

