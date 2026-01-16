"""
This file contains constant numbers, strings, etc used elsewhere in the brightwind library
"""

#  Specific gas constant for dry air (J/K/kg or m2/K/s2) from ISO:2533-1975 Standard Atmosphere
GAS_CONST_DRY_AIR = 287.05

#  Specific gas constant for water vapour (J/K/kg or m2/K/s2)
GAS_CONST_WATER = 461.495

# Air density lapse rate (kg/m3/km) from WindFarmer Theory Manual Version 5.3, DNV GL (April 2014)
AIR_DENSITY_LAPSE_RATE = -0.113

# Acceleration due to gravity (m/s^2) from ISO:2533-1975 Standard Atmosphere
ACCEL_DUE_TO_GRAVITY = 9.80665

# Temperature lapse rate (K/m or degC/m) from ISO:2533-1975 Standard Atmosphere
TEMP_LAPSE_RATE_STANDARD_ATMOSPHERE = -0.0065