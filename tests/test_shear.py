import pytest
import brightwind as bw
import numpy as np
import pandas as pd

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

    wspds = [8, np.nan]
    heights = [80, 100]
    with pytest.raises(ValueError) as except_info:
        bw.Shear.Average(wspds, heights)
    assert str(except_info.value) == "There is not valid data within the dataset provided to calculate shear"

    wspds = [8, 2]
    heights = [80, 100]
    with pytest.raises(ValueError) as except_info:
        bw.Shear.Average(wspds, heights)
    assert str(except_info.value) == "There is not valid data above 3 m/s within the dataset provided to calculate shear"


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

    data_test = DATA[['Spd80mN', 'Spd60mN', 'Spd40mN', 'Dir78mS']].copy()
    data_test.loc[(data_test.Dir78mS >= 15) & (data_test.Dir78mS <= 45), "Spd80mN"] = 2
    anemometers = data_test[['Spd80mN', 'Spd60mN', 'Spd40mN']]
    shear_by_sector_power_law = bw.Shear.BySector(anemometers, heights, directions)
    assert pd.isna(shear_by_sector_power_law.alpha.at["15.0-45.0"])#

    data_test = DATA[['Spd80mN', 'Spd60mN', 'Spd40mN', 'Dir78mS']].copy()
    data_test.loc[(data_test.Dir78mS >= 45) & (data_test.Dir78mS <= 75), "Spd80mN"] = np.nan
    anemometers = data_test[['Spd80mN', 'Spd60mN', 'Spd40mN']]
    shear_by_sector_power_law = bw.Shear.BySector(anemometers, heights, directions)
    assert pd.isna(shear_by_sector_power_law.alpha.at["45.0-75.0"])

    assert True


def test_time_of_day():
    # Specify columns in data which contain the anemometer measurements from which to calculate shear
    anemometers = DATA[['Spd80mN', 'Spd60mN', 'Spd40mN']]
    # Specify the heights of these anemometers
    heights = [80, 60, 40]

    # Test initialisation
    shear_by_tod_power_law1 = bw.Shear.TimeOfDay(anemometers, heights)
    assert shear_by_tod_power_law1.alpha['Jan'].round(6).to_list()[0:5] == [
        0.203745, 0.184187, 0.165501, 0.187611, 0.191284]
    shear_by_tod_power_law2 = bw.Shear.TimeOfDay(anemometers, heights, by_month=False)
    assert shear_by_tod_power_law2.alpha['12 Month Average'].round(6).to_list()[0:5] == [
        0.177502, 0.177530, 0.179454, 0.176763, 0.175528]
    shear_by_tod_power_law3 = bw.Shear.TimeOfDay(anemometers, heights, segment_start_time=8)
    assert shear_by_tod_power_law3.alpha['Jan'].round(6).to_list()[5:10] == [
        0.181648, 0.185978, 0.198043, 0.18695, 0.171201]
    shear_by_tod_log_law1 = bw.Shear.TimeOfDay(anemometers, heights, calc_method='log_law')
    assert shear_by_tod_log_law1.roughness['Jan'].round(6).to_list()[0:5] == [
        0.434260, 0.256827, 0.139850, 0.283929, 0.314679]
    shear_by_tod_log_law2 = bw.Shear.TimeOfDay(anemometers, heights, by_month=False, calc_method='log_law')
    assert shear_by_tod_log_law2.roughness['12_month_average'].round(6).to_list()[0:5] == [
        0.236910, 0.244433, 0.267135, 0.248233, 0.242695]
    shear_by_tod_log_law3 = bw.Shear.TimeOfDay(anemometers, heights, by_month=False,
                                               calc_method='log_law', segments_per_day=12)
    assert shear_by_tod_log_law3.roughness['12_month_average'].round(6).to_list()[5:10] == [
        0.036186, 0.023027, 0.023544, 0.065522, 0.140514]

    # Test attributes
    assert round(shear_by_tod_power_law2.alpha.mean()[0], 4) == 0.1473
    assert round(shear_by_tod_log_law2.roughness.mean()[0], 4) == 0.1450

    # Test apply
    assert (round(DATA['Spd80mN']['2017-11-23 10:10:00':'2017-11-23 10:40:00'] * (
            60 / 40) ** 0.141777, 5) == round(shear_by_tod_power_law1.apply(DATA['Spd80mN'][
                                    '2017-11-23 10:10:00':'2017-11-23 10:40:00'], 40, 60), 5)).all()
    assert (round(DATA['Spd80mN']['2017-11-23 10:10:00':'2017-11-23 10:40:00'] * (
            60/40) ** 0.126957, 5) == round(shear_by_tod_power_law2.apply(DATA['Spd80mN'][
                                    '2017-11-23 10:10:00':'2017-11-23 10:40:00'], 40, 60), 5)).all()
    assert (round(DATA['Spd80mN']['2017-11-23 10:10:00':'2017-11-23 10:40:00'] * (
            60 / 40) ** 0.141777, 5) == round(shear_by_tod_power_law1.apply(DATA['Spd80mN'][
                                    '2017-11-23 10:10:00':'2017-11-23 10:40:00'], 40, 60), 5)).all()
    assert  list(round(shear_by_tod_log_law1.apply(DATA['Spd80mN']['2017-11-23 10:10:00':'2017-11-23 10:40:00'],
                                                   40, 60), 5)) == [11.11452, 9.95853, 9.69339, 8.40695]
    assert list(round(shear_by_tod_log_law2.apply(DATA['Spd80mN']['2017-11-23 10:10:00':'2017-11-23 10:40:00'],
                                                  40, 60), 5)) == [11.16479, 10.00356, 9.73723, 8.44497]

    # Test errors
    with pytest.raises(ValueError) as except_info:
        bw.Shear.TimeOfDay(anemometers, heights, segments_per_day=23)
    assert str(except_info.value) == "'segments_per_day' must be a divisor of 24."
    with pytest.raises(ValueError) as except_info:
        bw.Shear.TimeOfDay(anemometers, heights, segment_start_time=24)
    assert str(except_info.value) == "'segment_start_time' must be an integer between 0 and 23 (inclusive)."
    with pytest.raises(ValueError) as except_info:
        bw.Shear.TimeOfDay(anemometers, heights, by_month=False, plot_type='12x24')
    assert str(except_info.value) == "12x24 plot is only possible when 'by_month=True'."

    data_test = DATA[['Spd80mN', 'Spd60mN', 'Spd40mN', 'Dir78mS']].copy()
    data_test.loc[data_test.index.hour == 2, "Spd80mN"] = 2
    anemometers = data_test[['Spd80mN', 'Spd60mN', 'Spd40mN']]
    shear_by_time_power_law = bw.Shear.TimeOfDay(anemometers, heights)
    assert shear_by_time_power_law.alpha.iloc[2].isna().all()

    data_test = DATA[['Spd80mN', 'Spd60mN', 'Spd40mN', 'Dir78mS']].copy()
    data_test.loc[data_test.index.hour == 5, "Spd80mN"] = np.nan
    anemometers = data_test[['Spd80mN', 'Spd60mN', 'Spd40mN']]
    shear_by_time_power_law = bw.Shear.TimeOfDay(anemometers, heights)
    assert shear_by_time_power_law.alpha.iloc[5].isna().all()


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

    DATA['Spd80mN'].iloc[0] = 2
    shear_ts = bw.Shear.TimeSeries(anemometers, heights)
    alpha = shear_ts.alpha
    assert pd.isna(alpha.iloc[0])

    DATA['Spd80mN'].iloc[0] = np.nan
    shear_ts = bw.Shear.TimeSeries(anemometers, heights, calc_method='log_law')
    roughness = shear_ts.roughness
    assert pd.isna(roughness.iloc[0])
    assert True


def test_scale():
    # Specify columns in data which contain the anemometer measurements from which to calculate shear
    bw.Shear.scale(DATA['Spd40mN'], 40, 60, alpha=.2)
    bw.Shear.scale(DATA['Spd40mN'], 40, 60, calc_method='log_law', roughness=.03)
    assert True
