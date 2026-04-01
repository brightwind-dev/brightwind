from typing import Union
import numpy as np
import pandas as pd
from brightwind.utils.constants import (ACCEL_DUE_TO_GRAVITY, TEMP_LAPSE_RATE_STANDARD_ATMOSPHERE,
                                        GAS_CONST_DRY_AIR, AIR_DENSITY_LAPSE_RATE)

__all__ = ['apply_scale_factor',
           'linear_transform',
           'scale_air_density_to_height',
           'scale_air_temperature_to_height',
           'scale_air_pressure_to_height']

def apply_scale_factor(data: Union[float, int, pd.DataFrame, pd.Series, np.array],
                       scale_factor: Union[float, int]
                       ) -> Union[float, int, pd.DataFrame, pd.Series, np.array]:
    """
    Scales data by the scale_factor.

    If data input is pd.DataFrame, only numeric columns are scaled.

    :param data:            Data value(s) to scale by the scale_factor.
    :type data:             float or int or pandas.Series or pandas.DataFrame or numpy.array
    :param scale_factor:    Scaling factor to use for scaling data values.
    :type scale_factor:     float or int
    :returns:               Scaled data value(s). Output type depends on type(data).
    :rtype:                 float or int or pandas.Series or pandas.DataFrame or numpy.array

        **Example usage**
    ::
    import brightwind as bw
    import pandas as pd
    import numpy as np

    # scale float by scale_factor of 0.5
    bw.apply_scale_factor(3, 0.5)
    # 1.5

    # # scale np.array by scale_factor of 0.5
    bw.apply_scale_factor(np.array([0, 1, 2]), 0.5)
    # array([0. , 0.5, 1. ])

    # scale pd.Series by scale_factor of -10
    bw.apply_scale_factor(pd.Series([10, 20, 30, 40]), -10)
    # 0   -100
    # 1   -200
    # 2   -300
    # 3   -400
    # dtype: int64

    # scale pd.DataFrame by scale factor of 2
    df = pd.DataFrame({'a':[0.5, 1.2], 'b':[3, 4], 'c':['a', 'b']})
    bw.apply_scale_factor(df, 2)
    # 	a	b	c
    # 0	1.0	6	a
    # 1	2.4	8	b
    """

    if not isinstance(data, (float, int, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError('data should be a float or int or pd.DataFrame or pd.Series or np.ndarray')
    if not isinstance(scale_factor, (float, int)):
        raise ValueError('scale_factor should be a float or int')

    if isinstance(data, pd.DataFrame):
        # only apply scaling to numeric columns
        numeric_df = scale_factor * (data.select_dtypes(include='number'))
        result = pd.concat([numeric_df, data.select_dtypes(exclude='number')], axis=1)
        return result
    if isinstance(data, pd.Series):
        if not pd.api.types.is_numeric_dtype(data):
            raise ValueError('data inputted as a pandas.Series must be numeric')
    if isinstance(data, np.ndarray):
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError('data inputted as a np.ndarray must be numeric')
    return scale_factor * data


def linear_transform(x_target: Union[float, int, np.ndarray, pd.Series],
                     x_ref: Union[float, int, np.ndarray, pd.Series],
                     y_ref: Union[float, int, np.ndarray, pd.Series],
                     slope: Union[float, int]) -> Union[float, int, np.ndarray, pd.Series]:
    """
    Perform a linear transform of known (x_ref, y_ref) to calculate y_target,
    using a constant slope and known value(s) of x_target.

    Function applies a linear transformation based on the equation of a straight line with constant slope:
        y_target = slope * (x_target - x_ref) + y_ref,
    where (x_ref, y_ref) is effectively considered a known point on a line with the inputted slope.

    Note that if pd.Series or np.ndarray inputs are provided for x_target, x_ref and y_ref,
    then the transform is performed on a point by point basis, as in, the n^{th} value of
    the resulting y_target would be:
        y_target[n] = slope * (x_target[n] - x_ref[n]) + y_ref[n],
    where (x_ref[n], y_ref[n]) is effectively considered a known point on a line with the inputted slope.
    Therefore all three of x_target, x_ref and y_ref must have the same dimensions.
    The inputted slope is used even in the case of pd.Series or np.ndarray inputs as the purpose of this function
    is to perform a linear transformation using a predefined slope, not to fit a line to the input data.

    :param x_target:    Target x value(s) at which to calculate y_target.
    :type x_target:     float or int or numpy.ndarray or pandas.Series
    :param x_ref:       Reference x value(s) of known reference point(s) on the line.
    :type x_ref:        float or int or numpy.ndarray or pandas.Series
    :param y_ref:       Reference y value(s) of known reference point(s) on the line.
    :type y_ref:        float or int or numpy.ndarray or pandas.Series
    :param slope:       Slope of the line equal to (y_target - y_ref) / (x_target - x_ref) where
                        ref and target x and y are any two points on the line.
    :type slope:        float or int
    :return:            Value(s) of y_target at specified x_target.
    :rtype:             float or int or numpy.ndarray or pandas.Series

        **Example usage**
    ::
    import brightwind as bw

    data = bw.load_csv(bw.demo_datasets.demo_data)

    # calculate y_target for x_target = 11 where (x_ref, y_ref) = (5, 10) is a point on the line and the slope is -0.5.
    bw.linear_transform(x_target=11, x_ref=5, y_ref=10, slope=-0.5)
    # 7.0

    # calculate y_target for x_target as a pandas.Series and (x_ref, y_ref) = (2, 20) is a point on the line
    # and the slope is -0.0065.
    bw.linear_transform(
        x_target=data.T2m.loc['2016-01-09 17:10':'2016-01-09 18:00'],
        x_ref=2,
        y_ref=20,
        slope=-0.0065)
    # Timestamp
    # 2016-01-09 17:10:00    20.006799
    # 2016-01-09 17:20:00    20.007390
    # 2016-01-09 17:30:00    20.008249
    # 2016-01-09 17:40:00    20.007462
    # 2016-01-09 17:50:00    20.007988
    # 2016-01-09 18:00:00    20.007065
    # Name: T2m, dtype: float64

    # calculate y_target for x_target and x_ref as a int and y_ref as a np.ndarray where (x_ref, y_ref) = (2, y_ref)
    # are points on the line and the slope is -0.0065.
    bw.linear_transform(
        x_target=20,
        x_ref=2,
        y_ref=data.T2m.loc['2016-01-09 17:10':'2016-01-09 18:00'].values,
        slope=-0.0065)
    # array([0.837, 0.746, 0.614, 0.735, 0.654, 0.796])
    """
    # check input types
    for var, var_name in zip([x_ref, y_ref, x_target], ['x_ref', 'y_ref', 'x_target']):
        if not isinstance(var, (float, int, pd.Series, np.ndarray)):
            raise TypeError(f"{var_name} must be a float or int or a numpy.ndarray or pandas.Series.")

    if not isinstance(slope, (float, int)):
        raise TypeError("slope must be a float or int.")

    # check dimensions of x_ref and x_target if arrays or Series
    if (isinstance(x_ref, (np.ndarray, pd.Series))) and (
            isinstance(x_target, (np.ndarray, pd.Series))):
        if len(x_ref) != len(x_target):
            raise ValueError("x_ref and x_target must have the same dimensions.")

    # check dimensions of y_ref if arrays or Series
    if isinstance(x_target - x_ref, (np.ndarray, pd.Series)):
        if isinstance(y_ref, (np.ndarray, pd.Series)):
            if len(y_ref) != len(x_target - x_ref):
                raise ValueError("y_ref must have the same dimensions as x_target or x_ref.")

    y_target = slope * (x_target - x_ref) + y_ref

    return y_target


def scale_air_density_to_height(ref_air_density_kg_m3: Union[float, pd.Series],
                                ref_height_m: float,
                                target_height_m: float,
                                lapse_rate_kg_m3_m: float = (0.001 * AIR_DENSITY_LAPSE_RATE)
                                ) -> Union[float, pd.Series]:
    """
    Linearly scales reference air density measurement (ref_air_density_kg_m3) from its measurement height
    (ref_height_m) to the height specified as the target_height_m, by applying a constant lapse_rate_kg_m3_m.

    :param ref_air_density_kg_m3:     Reference air density value(s) in kg/m3.
    :type ref_air_density_kg_m3:      float or pandas.Series
    :param ref_height_m:              Measurement height (in metres) of ref_air_density_kg_m3.
    :type ref_height_m                float
    :param target_height_m:           Height (in metres) that ref_air_density_kg_m3 is scaled to.
    :type target_height_m:            float
    :param lapse_rate_kg_m3_m:        Lapse rate describes how air density changes with increasing height above the
                                        earth's surface in kg/m3/m.
                                        Default value of -0.113 kg/m3 per km above earth's surface (-0.000113 kg/m3/m)
                                        taken from WindFarmer Theory Manual Version 5.3, DNV GL (April 2014).
    :type lapse_rate_kg_m3_m:         float
    :return:                          Air density at specified height of target_height_m in kg/m3. Type depends on
                                        type(ref_air_density_kg_m3) input.
    :rtype:                           float or pandas.Series

        **Example usage**
    ::
    import brightwind as bw

    # scale float value of air density using default lapse_rate_kg_m3_m
    bw.scale_air_density_to_height(ref_air_density_kg_m3=1.224, ref_height_m=80, target_height_m=100)
    # 1.22174

    # scale float value of air density using non-default value for lapse_rate_kg_m3_m
    bw.scale_air_density_to_height(ref_air_density_kg_m3=1.224, ref_height_m=80, target_height_m=100,
                                    lapse_rate_kg_m3_m=-0.0002)
    # 1.22

    # derive air density and scale based on series input values for reference air density
    data = bw.load_csv(bw.demo_datasets.demo_data)
    test_density = bw.calc_air_density(data.T2m, data.P2m)
    bw.scale_air_density_to_height(ref_air_density_kg_m3=test_density.loc['2016-01-09 17:10':'2016-01-09 18:00'],
                                    ref_height_m=2, target_height_m=10)

    # Timestamp
    # 2016-01-09 17:10:00    1.186780
    # 2016-01-09 17:20:00    1.187175
    # 2016-01-09 17:30:00    1.187747
    # 2016-01-09 17:40:00    1.185950
    # 2016-01-09 17:50:00    1.186301
    # 2016-01-09 18:00:00    1.185686
    # Name: air_density_scaled, dtype: float64
    """

    scaled_air_density = linear_transform(x_target=target_height_m, x_ref=ref_height_m,
                                          y_ref=ref_air_density_kg_m3, slope=lapse_rate_kg_m3_m)
    if isinstance(scaled_air_density, pd.Series):
        air_density_series_name = ref_air_density_kg_m3.name if ref_air_density_kg_m3.name else "air_density"
        scaled_air_density.name = f"{air_density_series_name}_scaled"

    return scaled_air_density


def scale_air_temperature_to_height(ref_air_temperature: Union[float, pd.Series],
                                    ref_height_m: float,
                                    target_height_m: float,
                                    lapse_rate_deg_m: float = TEMP_LAPSE_RATE_STANDARD_ATMOSPHERE
                                    ) -> Union[float, pd.Series]:
    """
    Linearly scales reference air temperature measurement (ref_air_temperature) from its measurement height
    (ref_height_m) to the height specified as target_height_m, by applying the constant lapse_rate_deg_m.

    :param ref_air_temperature:   Air temperature value(s) in degrees [for example in Celsius or Kelvin].
    :type ref_air_temperature:    float or pandas.Series
    :param ref_height_m:          Measurement height (in metres) of ref_air_temperature.
    :type ref_height_m:           float
    :param target_height_m:       Height (in metres) that ref_air_temperature is scaled to.
    :type target_height_m:        float
    :param lapse_rate_deg_m:      Lapse rate describes how air temperature changes with increasing height
                                  above the earth's surface.
                                  Units should be degrees of temperature per unit of height, e.g. °C/m or K/m.
                                  Default value of -6.5 degrees Celsius per km above the earth's surface
                                  (or -0.0065 °C/m) is commonly used as an approximation of the
                                  atmospheric lapse rate.
                                  In particular, the IEC standards rely on the ISO2533:1975 Standard Atmosphere
                                  which states that a lapse rate of 6.5 K/km is valid for geopotential altitudes
                                  of up to 11 km above earth's surface.
                                  This value was also adopted in WASP 11:
                                  Mortensen, N. G., Heathfield, D. N., Rathmann, O., & Nielsen, M. (2014).
                                  Wind Atlas Analysis and Application Program: WAsP 11 Help Facility. Computer
                                  programme, Department of Wind Energy, Technical University of Denmark
                                  https://orbit.dtu.dk/en/publications/wind-atlas-analysis-and-application-program-wasp-11-help-facility
    :type lapse_rate_deg_m:       float
    :return:                      Air temperature at specified height of target_height_m in same unit as input
                                  ref_air_temperature [for example in Celsius or Kelvin]. Output type depends on
                                  type(ref_air_temperature) input.
    :rtype:                       float or pandas.Series

        **Example usage**
    ::

    import brightwind as bw

    # scale air temperature based on float input value for reference air temperature
    bw.scale_air_temperature_to_height(ref_air_temperature=10.0065, ref_height_m=10, target_height_m=11)
    # 10.0

    # scale air temperature based on float input value for reference air temperature with non-default lapse_rate_deg_m
    bw.scale_air_temperature_to_height(ref_air_temperature=10, ref_height_m=12, target_height_m=10,
                                        lapse_rate_deg_m=-0.001)
    # 10.002

    # scale air temperature based on series input values for reference air temperature
    data = bw.load_csv(bw.demo_datasets.demo_data)

    bw.scale_air_temperature_to_height(ref_air_temperature=data.T2m.loc['2016-01-09 17:10':'2016-01-09 18:00'],
                                         ref_height_m=2, target_height_m=20)
    # Timestamp
    # 2016-01-09 17:10:00    0.837
    # 2016-01-09 17:20:00    0.746
    # 2016-01-09 17:30:00    0.614
    # 2016-01-09 17:40:00    0.735
    # 2016-01-09 17:50:00    0.654
    # 2016-01-09 18:00:00    0.796
    # Name: T2m_scaled, dtype: float64
    """

    scaled_air_temp = linear_transform(x_target=target_height_m, x_ref=ref_height_m,
                                       y_ref=ref_air_temperature, slope=lapse_rate_deg_m)
    if isinstance(scaled_air_temp, pd.Series):
        air_temp_series_name = ref_air_temperature.name if ref_air_temperature.name else "air_temperature"
        scaled_air_temp.name = f"{air_temp_series_name}_scaled"
    return scaled_air_temp


def scale_air_pressure_to_height(ref_air_pressure_hPa: Union[float, pd.Series],
                                 ref_air_temp_degC: Union[float, pd.Series],
                                 ref_height_m: Union[float, int],
                                 target_height_m: Union[float, int],
                                 ref_air_temp_height_m: Union[float, int, None] = None
                                 ) -> Union[float, int, pd.Series]:
    """
    Calculates air pressure at target height (target_height_m) using reference air pressure (ref_air_pressure_hPa)
    and air temperature (ref_air_temp_degC) values at a reference height (ref_height_m).

    Calculation based on ISO:2533-1975 Standard Atmosphere (https://www.iso.org/obp/ui/#iso:std:iso:2533:en)
    as suggested by IEC 61400-12-1:

    scaled_air_pressure = ref_air_pressure_hPa*((1 + (L/ref_air_temp_K)*(target_height_m - ref_height_m))**(-g/(L*R)))
    where:
    g = 9.80665 is acceleration due to gravity (m/s^2)
    L = -0.0065 is the temperature lapse rate (K/m) (denoted beta in ISO:2533 notation)
    R = 287.05 is the specific gas constant for dry air (J/K/kg or m2/K/s2)

    If air temperature is measured at a different height than air pressure, the `ref_air_temp_height_m` parameter
    can be provided. In this case, the air temperature is first scaled from `ref_air_temp_height_m` to
    `ref_height_m` internally using `scale_air_temperature_to_height` before the pressure scaling is applied.

    :param ref_air_pressure_hPa:    Reference air pressure value(s) in hPa (1mbar = 1hPa = 100Pa).
    :type ref_air_pressure_hPa:     float or int or pandas.Series
    :param ref_air_temp_degC:       Reference air temperature value(s) in degrees celsius. If
                                    `ref_air_temp_height_m` is provided, this temperature is assumed to be
                                    measured at `ref_air_temp_height_m` and will be scaled to `ref_height_m`
                                    internally. Otherwise, temperature is assumed to be at `ref_height_m`.
    :type ref_air_temp_degC:        float or int or pandas.Series
    :param ref_height_m:            Height (in metres) of reference air pressure (ref_air_pressure_hPa).
                                    If `ref_air_temp_height_m` is None, this is also assumed to be the
                                    height of ref_air_temp_degC.
    :type ref_height_m:             float or int
    :param target_height_m:         Height (in metres) which ref_air_pressure_hPa is scaled to.
    :type target_height_m:          float or int
    :param ref_air_temp_height_m:   Height (in metres) at which ref_air_temp_degC is measured. If None
                                    (default), air temperature is assumed to be at ref_height_m and no internal
                                    temperature scaling is applied. If a float or int value is provided,
                                    the air temperature is first scaled from ref_air_temp_height_m to ref_height_m
                                    before being used in the pressure scaling calculation.
    :type ref_air_temp_height_m:    float or int or None
    :return:                        Air pressure at specified height of target_height_m in hPa (1mbar = 1hPa = 100Pa).
                                    Type depends on type(ref_air_pressure_hPa) and type(ref_air_temp_degC) inputs.
    :rtype:                         float or int or pandas.Series

        **Example usage**
    ::

    import brightwind as bw

    # scale float value of air pressure
    round(bw.scale_air_pressure_to_height(ref_air_pressure_hPa=1000, ref_air_temp_degC=12, ref_height_m=10,
                                    target_height_m=200), 2)
    # 977.45

    # scale air pressure based on input series of reference air pressure and air temperature
    data = bw.load_csv(bw.demo_datasets.demo_data)

    bw.scale_air_pressure_to_height(ref_air_pressure_hPa=data['P2m'].loc['2016-01-09 17:10':'2016-01-09 18:00'],
                                    ref_air_temp_degC=data['T2m'].loc['2016-01-09 17:10':'2016-01-09 18:00'],
                                    ref_height_m=2, target_height_m=10).round(2)
    # Timestamp
    # 2016-01-09 17:10:00    933.07
    # 2016-01-09 17:20:00    933.07
    # 2016-01-09 17:30:00    933.07
    # 2016-01-09 17:40:00    932.07
    # 2016-01-09 17:50:00    932.07
    # 2016-01-09 18:00:00    932.07
    # Name: P2m_scaled, dtype: float64

    """

    # check input types
    for var, var_name in zip([ref_air_pressure_hPa, ref_air_temp_degC], ['ref_air_pressure_hPa', 'ref_air_temp_degC']):
        if not isinstance(var, (float, int, pd.Series)):
            raise TypeError(f"{var_name} must be a float or int or pandas.Series.")
    for var, var_name in zip([ref_height_m, target_height_m], ['ref_height_m', 'target_height_m']):
        if not isinstance(var, (float, int)):
            raise TypeError(f"{var_name} must be a float or int.")
    if ref_air_temp_height_m is not None and not isinstance(ref_air_temp_height_m, (float, int)):
        raise TypeError("ref_air_temp_height_m must be a float, int, or None.")

    # check dimensions of ref_air_pressure_hPa and ref_air_temp_degC if Series
    if isinstance(ref_air_pressure_hPa, pd.Series) and (isinstance(ref_air_temp_degC, pd.Series)):
        if len(ref_air_pressure_hPa) != len(ref_air_temp_degC):
            raise ValueError("ref_air_pressure_hPa and ref_air_temp_degC must have the same dimensions.")

    # if air temperature is at a different height than air pressure, scale it to the air pressure height first
    if ref_air_temp_height_m is not None:
        ref_air_temp_degC = scale_air_temperature_to_height(ref_air_temp_degC, ref_air_temp_height_m, ref_height_m)
        
    # Constants as outlined in ISO:2533
    g = ACCEL_DUE_TO_GRAVITY  # Acceleration due to gravity (m/s^2)
    L = TEMP_LAPSE_RATE_STANDARD_ATMOSPHERE  # Temperature lapse rate (K/m) (denoted beta in ISO:2533 notation)
    R = GAS_CONST_DRY_AIR  # Specific gas constant dry air

    ref_air_temp_K = ref_air_temp_degC + 273.15  # Convert temp units to K

    scaled_air_pressure_hPa = ref_air_pressure_hPa * ((1 + (L / ref_air_temp_K) * (target_height_m - ref_height_m)
                                                       ) ** (-g / (L * R)))
    if isinstance(scaled_air_pressure_hPa, pd.Series):
        air_pressure_series_name = ref_air_pressure_hPa.name if ref_air_pressure_hPa.name else "air_pressure"
        scaled_air_pressure_hPa.name = f"{air_pressure_series_name}_scaled"
    return scaled_air_pressure_hPa
