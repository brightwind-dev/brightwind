import pytest
import brightwind as bw
import pandas as pd


def test_Average():
    # Load data

    # load data as dataframe
    data = bw.load_csv(bw.datasets.demo_data)
    # Specify columns in data which contain the anemometer measurements from which to calculate shear
    anemometers = data[['Spd80mN', 'Spd60mN', 'Spd40mN']]
    # Specify the heights of these anemometers
    heights = [80, 60, 40]

    # Test initialisation
    shear_avg_power_law = bw.Shear.Average(anemometers, heights)
    shear_avg_log_law = bw.Shear.Average(anemometers, heights, calc_method='log_law')

    # Test attributes
    shear_avg_power_law.alpha
    shear_avg_log_law.slope

    # Test apply
    shear_avg_power_law.apply(data['Spd80mN'], 40, 60)
    shear_avg_log_law.apply(data['Spd80mN'], 40, 60)
    assert True


def test_BySector():

    # load data as dataframe
    data = bw.load_csv(bw.datasets.demo_data)
    # Specify columns in data which contain the anemometer measurements from which to calculate shear
    anemometers = data[['Spd80mN', 'Spd60mN', 'Spd40mN']]
    # Specify the heights of these anemometers
    heights = [80, 60, 40]
    # Specify directions
    directions = data['Dir78mS']
    # custom bins
    custom_bins = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]

    # Test initialisation
    shear_by_sector_power_law = bw.Shear.BySector(anemometers, heights, directions)
    shear_by_sector_log_law = bw.Shear.BySector(anemometers, heights, directions, calc_method='log_law')
    shear_by_sector_custom_bins = bw.Shear.BySector(anemometers, heights, directions,
                                                    direction_bin_array=custom_bins)
    # test attributes
    shear_by_sector_power_law.plot
    shear_by_sector_power_law.alpha
    shear_by_sector_custom_bins.plot
    shear_by_sector_custom_bins.alpha

    # Test apply
    shear_by_sector_power_law.apply(data['Spd80mN'],directions, 40, 60)
    shear_by_sector_log_law.apply(data['Spd80mN'], directions, 40, 60)
    shear_by_sector_custom_bins.apply(data['Spd80mN'], directions, 40, 60)

    assert True


def test_TimeOfDay():
    # Load data

    # load data as dataframe
    data = bw.load_csv(bw.datasets.demo_data)
    # Specify columns in data which contain the anemometer measurements from which to calculate shear
    anemometers = data[['Spd80mN', 'Spd60mN', 'Spd40mN']]
    # Specify the heights of these anemometers
    heights = [80, 60, 40]

    # Test initialisation
    shear_by_tod_power_law = bw.Shear.TimeOfDay(anemometers, heights)
    shear_by_tod_power_law = bw.Shear.TimeOfDay(anemometers, heights, by_month=False)
    shear_by_tod_log_law = bw.Shear.TimeOfDay(anemometers, heights, calc_method='log_law')
    shear_by_tod_log_law = bw.Shear.TimeOfDay(anemometers, heights, by_month=False, calc_method='log_law')

    # Test attributes
    shear_by_tod_power_law.alpha
    shear_by_tod_log_law.slope

    # Test apply
    shear_by_tod_power_law.apply(data['Spd80mN'], 40, 60)
    shear_by_tod_log_law.apply(data['Spd80mN'], 40, 60)
    assert True


def test_TimeSeries():
    # Load data

    # load data as dataframe
    data = bw.load_csv(bw.datasets.demo_data)
    # Specify columns in data which contain the anemometer measurements from which to calculate shear
    anemometers = data[['Spd80mN', 'Spd60mN', 'Spd40mN']]
    # Specify the heights of these anemometers
    heights = [80, 60, 40]
    anemometers = anemometers[:100]
    # Test initialisation
    shear_by_ts_power_law = bw.Shear.TimeSeries(anemometers, heights)
    shear_by_ts_power_law = bw.Shear.TimeSeries(anemometers, heights,  maximise_data=True)
    shear_by_ts_log_law = bw.Shear.TimeSeries(anemometers, heights, calc_method='log_law')
    shear_by_ts_log_law = bw.Shear.TimeSeries(anemometers, heights, calc_method='log_law',
                                              maximise_data=True)

    # Test attributes
    shear_by_ts_power_law.alpha
    shear_by_ts_log_law.slope

    # Test apply
    shear_by_ts_power_law.apply(data['Spd80mN'], 40, 60)
    shear_by_ts_log_law.apply(data['Spd80mN'], 40, 60)
    assert True


def test_scale():
    # load data as dataframe
    data = bw.load_csv(bw.datasets.demo_data)
    # Specify columns in data which contain the anemometer measurements from which to calculate shear
    bw.Shear.scale(.2, data['Spd40mN'], 40, 60)
    assert True
