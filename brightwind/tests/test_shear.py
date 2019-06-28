import pytest
import brightwind as bw
import pandas as pd


def test_PowerLaw_calc_alpha():
    # Load data

    # load data as dataframe
    data = bw.load_csv(bw.datasets.demo_data)
    # Specify columns in data which contain the anemometer measurements from which to calculate shear
    anemometers = data[['Spd80mN', 'Spd60mN', 'Spd40mN']]
    # Specify the heights of these anemometers
    heights = [80, 60, 40]

    # Test function
    shear_by_power_law = bw.Shear.PowerLaw.calc_alpha(anemometers, heights, return_object=True)

    # Test attributes
    shear_by_power_law.plot
    shear_by_power_law.alpha

    assert True


def test_PowerLaw_apply_alpha():
    # Load data

    # load data as dataframe
    data = bw.load_csv(bw.datasets.demo_data)
    # Specify columns in data which contain the anemometer measurements from which to calculate shear
    anemometers = data[['Spd80mN', 'Spd60mN', 'Spd40mN']]
    # Specify the heights of these anemometers
    heights = [80, 60, 40]

    # Test function
    shear_by_power_law = bw.Shear.PowerLaw.calc_alpha(anemometers, heights, return_object=True)

    shear_by_power_law.apply_alpha(data['Spd80mN'], 40, 60)
    assert True


def test_BySector_calc_alpha():
    # Load data

    # load data as dataframe
    data = bw.load_csv(bw.datasets.demo_data)
    # Specify columns in data which contain the anemometer measurements from which to calculate shear
    anemometers = data[['Spd80mN', 'Spd60mN', 'Spd40mN']]
    # Specify the heights of these anemometers
    heights = [80, 60, 40]
    # Specify directions
    directions = data['Dir78mS']

    # custom bins
    custom_bins = [0,30,60,90,120,150,180,210,240,270,300,330,360]
    # Test function
    shear_by_sector= bw.Shear.BySector.calc_alpha(anemometers, heights, directions, return_object=True)
    shear_by_sector_custom_bins = bw.Shear.BySector.calc_alpha(anemometers, heights, directions,
                                                               direction_bin_array=custom_bins, return_object=True)

    # Test attributes
    shear_by_sector.plot
    shear_by_sector.alpha
    shear_by_sector_custom_bins.plot
    shear_by_sector_custom_bins.alpha

    assert True


def test_BySector_apply_alpha():
    # Load data

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

    # Test function
    shear_by_sector = bw.Shear.BySector.calc_alpha(anemometers, heights, directions, return_object=True)

    # Test attributes
    shear_by_sector.apply_alpha(data['Spd80mN'], 40, 60,directions)

    shear_by_sector_custom_bins = bw.Shear.BySector.calc_alpha(anemometers, heights, directions,
                                                               direction_bin_array=custom_bins, return_object=True)

    shear_by_sector_custom_bins.apply_alpha(data['Spd80mN'], 40, 60, directions)

    assert True


def test_scale():
    # load data as dataframe
    data = bw.load_csv(bw.datasets.demo_data)
    # Specify columns in data which contain the anemometer measurements from which to calculate shear
    bw.Shear.scale(.2, data['Spd40mN'], 40, 60)
    assert True
