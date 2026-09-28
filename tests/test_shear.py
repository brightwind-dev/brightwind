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

    # Test plot axis labels
    assert shear_avg_power_law.plot.axes[0].get_xlabel() == 'Wind Speed [m/s]'
    assert shear_avg_power_law.plot.axes[0].get_ylabel() == 'Height AGL [m]'
    assert shear_avg_log_law.plot.axes[0].get_xlabel() == 'Wind Speed [m/s]'
    assert shear_avg_log_law.plot.axes[0].get_ylabel() == 'Height AGL [m]'

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
    assert str(except_info.value) == "There is no valid data within the dataset provided to calculate the shear."

    wspds = [8, 2]
    heights = [80, 100]
    with pytest.raises(ValueError) as except_info:
        bw.Shear.Average(wspds, heights)
    assert str(except_info.value) == "There is no valid data above 3 m/s within the dataset provided to calculate the shear."


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
    assert shear_by_sector_power_law.alpha.to_dict() == pytest.approx({
        '345.0-15.0': 0.11937, '15.0-45.0': 0.145463, '45.0-75.0': 0.096945, '75.0-105.0': 0.044056,
        '105.0-135.0': 0.054538, '135.0-165.0': 0.116558, '165.0-195.0': 0.354113, '195.0-225.0': 0.213977,
        '225.0-255.0': 0.096221, '255.0-285.0': 0.054575, '285.0-315.0': 0.077513, '315.0-345.0': 0.10868}, abs=1e-6)
    assert shear_by_sector_power_law.alpha_count.to_list() == [
        1874, 3456, 2494, 3415, 3501, 1978, 8667, 13311, 8554, 10077, 7511, 1697]
    assert shear_by_sector_log_law.roughness.to_list() == pytest.approx(
        [0.013213, 0.058274, 0.001885, 0.0, 1e-06, 0.010458, 3.696403, 0.53858, 0.001695, 1e-06, 0.000143, 0.005848],
        abs=1e-6)
    assert shear_by_sector_custom_bins.alpha.to_list() == pytest.approx([
        0.147848, 0.122666, 0.071071, 0.032401, 0.073788, 0.291367, 0.286366, 0.164111, 0.061095, 0.059813, 0.101548,
        0.105536], abs=1e-6)
    period = slice('2017-06-15 12:00', '2017-06-15 12:40')
    assert shear_by_sector_power_law.apply(DATA['Spd80mN'][period], directions[period], 40,
                                           60).to_list() == pytest.approx([8.09578, 9.81557, 9.7012, 8.38067, 8.81738],
                                                                          abs=1e-5)

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
    shear_by_tod_power_law4 = bw.Shear.TimeOfDay(anemometers['2016-05-01':], heights)
    assert shear_by_tod_power_law4.alpha['Jan'].round(6).to_list()[0:5] == [
        0.219158, 0.183266, 0.159368, 0.159639, 0.170001]
    assert shear_by_tod_power_law4.alpha['May'].round(6).to_list()[0:5] == \
        shear_by_tod_power_law1.alpha['May'].round(6).to_list()[0:5]
    shear_by_tod_log_law4 = bw.Shear.TimeOfDay(anemometers['2016-05-01':], heights, calc_method='log_law')
    assert shear_by_tod_log_law4.roughness['Jan'].round(6).to_list()[0:5] == [
        0.613791, 0.252135, 0.111832, 0.111367, 0.162696]
    assert shear_by_tod_log_law4.roughness['Jun'].round(6).to_list()[0:5] == \
        shear_by_tod_log_law1.roughness['Jun'].round(6).to_list()[0:5]
    shear_by_tod_log_law5 = bw.Shear.TimeOfDay(anemometers[anemometers.index.month != 5], heights, calc_method='log_law')
    assert 'May' not in shear_by_tod_log_law5.roughness.columns
    

    # Test attributes
    assert round(shear_by_tod_power_law2.alpha.mean().iloc[0], 4) == 0.1473
    assert round(shear_by_tod_log_law2.roughness.mean().iloc[0], 4) == 0.1450

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
    assert (round(DATA['Spd80mN']['2016-05-10 10:10:00':'2016-05-10 10:40:00'] * (
            60 / 40) ** 0.096166, 5) == round(shear_by_tod_power_law4.apply(DATA['Spd80mN'][
                                    '2016-05-10 10:10:00':'2016-05-10 10:40:00'], 40, 60), 5)).all()
    assert (round(DATA['Spd80mN']['2017-01-23 01:10:00':'2017-01-23 01:40:00'] * (
            60 / 40) ** 0.183266, 5) == round(shear_by_tod_power_law4.apply(DATA['Spd80mN'][
                                    '2017-01-23 01:10:00':'2017-01-23 01:40:00'], 40, 60), 5)).all()
    
    # Test plot
    shear_plot4 = shear_by_tod_log_law4.plot
    legend = shear_plot4.get_axes()[0].get_legend()
    assert legend.get_lines()[0].get_color() == bw.analyse.plot._colormap_to_colorscale(
        bw.analyse.plot.COLOR_PALETTE.color_map_cyclical, 13)[0]
    assert legend.get_lines()[11].get_color() == bw.analyse.plot._colormap_to_colorscale(
        bw.analyse.plot.COLOR_PALETTE.color_map_cyclical, 13)[11]
    shear_plot5 = shear_by_tod_log_law5.plot
    legend = shear_plot5.get_axes()[0].get_legend()
    assert legend.get_lines()[0].get_color() == bw.analyse.plot._colormap_to_colorscale(
        bw.analyse.plot.COLOR_PALETTE.color_map_cyclical, 13)[0]
    assert legend.get_lines()[10].get_color() == bw.analyse.plot._colormap_to_colorscale(
        bw.analyse.plot.COLOR_PALETTE.color_map_cyclical, 13)[11]
    

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


def test_time_of_day_apply():
    anemometers = DATA[['Spd80mN', 'Spd60mN', 'Spd40mN']]
    heights = [80, 60, 40]
    # one timestamp for each month, at a different hour each time
    timestamps = pd.to_datetime(['2016-01-10 01:00', '2016-02-01 03:00', '2016-03-01 05:00', '2016-04-01 07:00',
                                 '2016-05-01 09:00', '2016-06-01 11:00', '2016-07-01 13:00', '2016-08-01 15:00',
                                 '2016-09-01 17:00', '2016-10-01 19:00', '2016-11-01 21:00', '2016-12-01 23:00'])
    cases = [
        ({}, 40, 80, 7.470713,
         [5.255952, 11.854113, 11.554538, 13.060339, 11.888655, 6.860667, 6.905871, 4.522068, 15.488296, 7.620377,
          7.488793, 5.23901]),
        ({'calc_method': 'log_law'}, 40, 80, 7.475318,
         [5.261172, 11.863411, 11.565096, 13.064503, 11.893347, 6.863552, 6.906965, 4.522997, 15.50949, 7.624915,
          7.493248, 5.242754]),
        ({'by_month': False}, 40, 100, 7.725035,
         [5.443171, 12.522484, 11.929505, 13.886765, 12.533407, 7.158539, 7.215577, 4.690742, 15.374625, 7.838723,
          7.694895, 5.354785]),
        ({'segments_per_day': 2, 'segment_start_time': 7}, 40, 80, 7.469702,
         [5.258712, 11.801394, 11.53134, 12.526182, 11.726311, 6.879178, 6.964924, 4.590044, 15.477017, 7.608653,
          7.48853, 5.183642]),
        ({'calc_method': 'log_law', 'segments_per_day': 6, 'segment_start_time': 3}, 40, 80, 7.474933,
         [5.262591, 11.906175, 11.579636, 12.746705, 12.006999, 6.836182, 6.926862, 4.590298, 15.389964, 7.606758,
          7.503646, 5.206709]),
    ]
    for kwargs, height, shear_to, expected_mean, expected_values in cases:
        shear_by_tod = bw.Shear.TimeOfDay(anemometers, heights, **kwargs)
        scaled = shear_by_tod.apply(DATA['Spd40mN'], height, shear_to)
        assert scaled.name == 'Spd40mN_scaled_to_' + str(shear_to) + 'm'
        assert scaled.index.equals(DATA.index)
        assert scaled.isna().sum() == DATA['Spd40mN'].isna().sum() == 449
        assert scaled.mean() == pytest.approx(expected_mean, abs=1e-6)
        np.testing.assert_allclose(scaled[timestamps], expected_values, rtol=0, atol=1e-6)

    # input out of time order gives the same result, sorted by time
    shear_by_tod = bw.Shear.TimeOfDay(anemometers, heights)
    scaled = shear_by_tod.apply(DATA['Spd40mN'], 40, 80)
    scaled_shuffled = shear_by_tod.apply(DATA['Spd40mN'].sample(frac=1, random_state=0), 40, 80)
    pd.testing.assert_series_equal(scaled_shuffled, scaled)

    # error if the object has no shear for a month in the input time series
    shear_by_tod = bw.Shear.TimeOfDay(anemometers[anemometers.index.month != 5], heights)
    with pytest.raises(ValueError) as except_info:
        shear_by_tod.apply(DATA['Spd40mN'], 40, 80)
    assert str(except_info.value) == ("The shear by TimeOfDay object doesn't have shear values for May. The shear "
                                      "cannot be applied to the input time series for this month.")


def test_calc_linear_fit():
    rng = np.random.default_rng(0)
    for heights in [[80, 40], [80, 60, 40], [40, 60, 80, 100, 120], [80, 80, 60, 40]]:
        log_heights = np.log(heights)
        log_wspds = np.log(rng.uniform(3, 20, size=(50, len(heights))))
        slope, intercept = bw.Shear._calc_linear_fit(log_heights, log_wspds)
        expected_slope, expected_intercept = np.polyfit(log_heights, log_wspds.T, deg=1)
        np.testing.assert_allclose(slope, expected_slope, rtol=0, atol=1e-12)
        np.testing.assert_allclose(intercept, expected_intercept, rtol=0, atol=1e-12)


def test_time_series_full_data():
    anemometers = DATA[['Spd80mN', 'Spd60mN', 'Spd40mN']]
    heights = [80, 60, 40]
    shear_by_ts_power_law = bw.Shear.TimeSeries(anemometers, heights)
    shear_by_ts_log_law = bw.Shear.TimeSeries(anemometers, heights, calc_method='log_law')

    # test against a np.polyfit fit of every valid timestamp
    valid = (anemometers > 3).all(axis=1)
    slope, intercept = np.polyfit(np.log(heights), anemometers[valid].values.T, deg=1)
    expected_roughness = bw.Shear._calc_roughness(slope=slope, intercept=intercept)
    expected_alpha = np.polyfit(np.log(heights), np.log(anemometers[valid].values.T), deg=1)[0]
    assert shear_by_ts_power_law.alpha.isna().to_list() == (~valid).to_list()
    np.testing.assert_allclose(shear_by_ts_power_law.alpha[valid], expected_alpha, rtol=0, atol=1e-12)
    realistic = expected_roughness < 10
    np.testing.assert_allclose(shear_by_ts_log_law.roughness[valid][realistic], expected_roughness[realistic],
                               rtol=1e-9, atol=1e-12)

    # test specific values
    assert shear_by_ts_power_law.alpha.count() == 79514
    assert shear_by_ts_power_law.alpha.mean() == pytest.approx(0.150953, abs=1e-6)
    assert shear_by_ts_log_law.roughness.median() == pytest.approx(0.118889, abs=1e-6)
    assert shear_by_ts_power_law.alpha['2017-06-15 12:00':'2017-06-15 12:40'].to_list() == pytest.approx([
        0.068212, 0.060139, 0.057766, 0.006736, 0.039497], abs=1e-6)
    assert shear_by_ts_log_law.roughness['2017-06-15 12:00':'2017-06-15 12:40'].to_list() == pytest.approx(
        [2.4e-05, 3e-06, 2e-06, 0.0, 0.0], abs=1e-6)


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
    assert abs(shear_by_ts_log_law.roughness.mean() / 4.306534305567819e+68 - 1) < 1e-6

    # Test plot is only created when first requested and is then reused
    assert shear_by_ts_power_law._plot is None
    plot = shear_by_ts_power_law.plot
    assert shear_by_ts_power_law.plot is plot
    assert plot.axes[0].get_xlabel() == 'Wind Speed [m/s]'
    assert plot.axes[0].get_ylabel() == 'Height AGL [m]'

    # Test apply
    shear_by_ts_power_law.apply(DATA['Spd80mN'], 40, 60)
    shear_by_ts_log_law.apply(DATA['Spd80mN'], 40, 60)

    DATA.loc[DATA.index[0], 'Spd80mN'] = 2
    shear_ts = bw.Shear.TimeSeries(anemometers, heights)
    alpha = shear_ts.alpha
    assert pd.isna(alpha.iloc[0])

    DATA.loc[DATA.index[0], 'Spd80mN'] = np.nan
    shear_ts = bw.Shear.TimeSeries(anemometers, heights, calc_method='log_law')
    roughness = shear_ts.roughness
    assert pd.isna(roughness.iloc[0])
    assert True


def test_scale():
    # Specify columns in data which contain the anemometer measurements from which to calculate shear
    bw.Shear.scale(DATA['Spd40mN'], 40, 60, alpha=.2)
    bw.Shear.scale(DATA['Spd40mN'], 40, 60, calc_method='log_law', roughness=.03)
    assert True
