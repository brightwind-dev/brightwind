import pytest
import brightwind as bw

DATA = bw.load_csv(bw.demo_datasets.demo_data)
DATA = bw.apply_cleaning(DATA, bw.demo_datasets.demo_cleaning_file)
WSPD_COLS = ['Spd80mN', 'Spd80mS', 'Spd60mN', 'Spd60mS', 'Spd40mN', 'Spd40mS']
WDIR_COLS = ['Dir78mS', 'Dir58mS', 'Dir38mS']


def test_average():
    # Specify columns in data which contain the anemometer measurements from which to calculate shear
    anemometers = DATA[['Spd80mN', 'Spd60mN', 'Spd40mN']]
    # Specify the heights of these anemometers
    heights = [80, 60, 40]

    # Test initialisation
    shear_avg_power_law = bw.Shear.Average(anemometers, heights)
    shear_avg_log_law = bw.Shear.Average(anemometers, heights, calc_method='log_law')

    # Test attributes
    assert round(shear_avg_power_law.alpha, 4) == 0.1434
    assert round(shear_avg_log_law.roughness, 4) == 0.0549

    # Test apply
    shear_avg_power_law.apply(DATA['Spd80mN'], 40, 60)
    shear_avg_log_law.apply(DATA['Spd80mN'], 40, 60)

    assert True
    # Test specific values
    wspds = [7.74, 8.2, 8.57]
    heights = [60, 80, 100]
    specific_test = bw.Shear.Average(wspds, heights)
    assert round(specific_test.alpha, 9) == 0.199474297

    wspds = [8, 8.365116]
    heights = [80, 100]
    specific_test = bw.Shear.Average(wspds, heights)
    assert round(specific_test.alpha, 1) == 0.2
    specific_test_log = bw.Shear.Average(wspds, heights, calc_method='log_law')
    assert round(specific_test_log.roughness, 9) == 0.602156994


def test_by_sector():
    # Specify columns in data which contain the anemometer measurements from which to calculate shear
    anemometers = DATA[['Spd80mN', 'Spd60mN', 'Spd40mN']]
    # Specify the heights of these anemometers
    heights = [80, 60, 40]
    # Specify directions
    directions = DATA['Dir78mS']
    # custom bins
    custom_bins = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]

    # Test initialisation
    shear_by_sector_power_law = bw.Shear.BySector(anemometers, heights, directions)
    shear_by_sector_log_law = bw.Shear.BySector(anemometers, heights, directions, calc_method='log_law')
    shear_by_sector_custom_bins = bw.Shear.BySector(anemometers, heights, directions,
                                                    direction_bin_array=custom_bins)
    # test attributes
    shear_by_sector_power_law.plot
    assert round(shear_by_sector_power_law.alpha.mean(), 4) == 0.1235
    shear_by_sector_custom_bins.plot
    assert round(shear_by_sector_custom_bins.alpha.mean(), 4) == 0.1265

    # Test apply
    shear_by_sector_power_law.apply(DATA['Spd80mN'], directions, 40, 60)
    shear_by_sector_log_law.apply(DATA['Spd80mN'], directions, 40, 60)
    shear_by_sector_custom_bins.apply(DATA['Spd80mN'], directions, 40, 60)

    assert True


def test_time_of_day():
    # Specify columns in data which contain the anemometer measurements from which to calculate shear
    anemometers = DATA[['Spd80mN', 'Spd60mN', 'Spd40mN']]
    # Specify the heights of these anemometers
    heights = [80, 60, 40]

    # Test initialisation
    shear_by_tod_power_law = bw.Shear.TimeOfDay(anemometers, heights)
    shear_by_tod_power_law = bw.Shear.TimeOfDay(anemometers, heights, by_month=False)
    shear_by_tod_log_law = bw.Shear.TimeOfDay(anemometers, heights, calc_method='log_law')
    shear_by_tod_log_law = bw.Shear.TimeOfDay(anemometers, heights, by_month=False, calc_method='log_law')

    # Test attributes
    assert round(shear_by_tod_power_law.alpha.mean()[0], 4) == 0.1473
    shear_by_tod_log_law.roughness

    # Test apply
    shear_by_tod_power_law.apply(DATA['Spd80mN'], 40, 60)
    shear_by_tod_log_law.apply(DATA['Spd80mN'], 40, 60)
    assert True


def test_time_series():
    # Specify columns in data which contain the anemometer measurements from which to calculate shear
    anemometers = DATA[['Spd80mN', 'Spd60mN', 'Spd40mN']]
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
    assert round(shear_by_ts_power_law.alpha.mean(), 4) == 0.1786
    # Changed to support equality for very large numbers
    assert (shear_by_ts_log_law.roughness.mean() / 4.306534305567819e+68 - 1) < 1e-6

    # Test apply
    shear_by_ts_power_law.apply(DATA['Spd80mN'], 40, 60)
    shear_by_ts_log_law.apply(DATA['Spd80mN'], 40, 60)
    assert True


def test_scale():
    # Specify columns in data which contain the anemometer measurements from which to calculate shear
    bw.Shear.scale(DATA['Spd40mN'], 40, 60, alpha=.2)
    bw.Shear.scale(DATA['Spd40mN'], 40, 60, calc_method='log_law', roughness=.03)
    assert True
