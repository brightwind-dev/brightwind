import pytest
import brightwind as bw
import pandas as pd
import numpy as np

DATA = bw.load_csv(bw.demo_datasets.demo_data)
DATA = bw.apply_cleaning(DATA, bw.demo_datasets.demo_cleaning_file)
WSPD_COLS = ['Spd80mN', 'Spd80mS', 'Spd60mN', 'Spd60mS', 'Spd40mN', 'Spd40mS']
WDIR_COLS = ['Dir78mS', 'Dir58mS', 'Dir38mS']


def test_monthly_means():
    # Load data
    bw.monthly_means(DATA)
    bw.monthly_means(DATA[['Spd80mN']])

    bw.monthly_means(DATA[WSPD_COLS], return_data=True)
    bw.monthly_means(DATA.Spd80mN, return_data=True)
    assert True


def test_sector_ratio():
    bw.sector_ratio(DATA['Spd80mN'], DATA['Spd80mS'], DATA['Dir78mS'], sectors=72, boom_dir_1=0,
                    boom_dir_2=180, return_data=True)
    bw.sector_ratio(DATA[['Spd40mN']], DATA.Spd60mN, wdir=DATA[['Dir38mS']])
    bw.sector_ratio(DATA.Spd40mN, DATA.Spd60mN, wdir=DATA.Dir38mS,
                    direction_bin_array=[0, 45, 135, 180, 220, 360], boom_dir_1=160, boom_dir_2=340)
    assert True


def test_basic_stats():
    bw.basic_stats(DATA)
    bs2 = bw.basic_stats(DATA['Spd80mN'])
    assert (bs2['count'][0] == 95180.0) and (round(bs2['mean'][0], 6) == 7.518636) and \
           (round(bs2['std'][0], 6) == 3.994552) and (round(bs2['min'][0], 3) == 0.215) and \
           (round(bs2['max'][0], 1) == 29.0)


def test_time_continuity_gaps():
    gaps = bw.time_continuity_gaps(DATA['Spd80mN'])
    assert gaps.iloc[0, 0] == pd.Timestamp('2016-03-09 06:20:00')
    assert gaps.iloc[0, 1] == pd.Timestamp('2016-03-09 10:20:00')
    assert gaps.iloc[1, 0] == pd.Timestamp('2016-03-29 23:50:00')
    assert gaps.iloc[1, 1] == pd.Timestamp('2016-03-30 07:00:00')
    assert abs(gaps.iloc[0, 2] - 0.166667) < 1e-5
    assert abs(gaps.iloc[1, 2] - 0.298611) < 1e-5


def test_dist_12x24():
    bw.dist_12x24(DATA[['Spd40mN']], return_data=True)
    bw.dist_12x24(DATA.Spd40mN, var_name_label='wind speed', return_data=True)
    bw.dist_12x24(DATA.PrcpTot, aggregation_method='sum')

    def custom_agg(x):
        return x.mean() + (2 * x.std())

    bw.dist_12x24(DATA.PrcpTot, aggregation_method=custom_agg, return_data=True)
    assert True


def test_ti_twelve_by_24():
    bw.TI.twelve_by_24(DATA[['Spd60mN']], DATA[['Spd60mNStd']])
    bw.TI.twelve_by_24(DATA.Spd60mN, DATA.Spd60mNStd)
    bw.TI.twelve_by_24(DATA.Spd60mN, DATA.Spd60mNStd, return_data=True)
    bw.TI.twelve_by_24(DATA.Spd60mN, DATA.Spd60mNStd, return_data=True, var_name_label='Speed 60 m N m/s')
    bw.TI.twelve_by_24(DATA.Spd60mN, DATA.Spd60mNStd, var_name_label='Speed 60 m N m/s')
    bw.TI.twelve_by_24(DATA.Spd40mN, DATA.Spd40mNStd)
    assert 1 == 1


def test_coverage():
    # hourly coverage
    bw.coverage(DATA[['Spd80mN']], period='1H')
    bw.coverage(DATA.Spd80mN, period='1H')
    # monthly_coverage
    bw.coverage(DATA.Spd80mN, period='1M')
    # monthly_coverage of variance
    bw.coverage(DATA.Spd80mN, period='1M', aggregation_method='var')
    assert True


def test_dist_by_dir_sector():
    bw.dist_by_dir_sector(DATA[['Spd40mN']], DATA[['Dir38mS']])
    bw.dist_by_dir_sector(DATA.Spd40mN, DATA.Dir38mS)

    bw.dist_by_dir_sector(DATA.Spd40mN, DATA.Dir38mS,
                          direction_bin_array=[0, 90, 130, 200, 360],
                          direction_bin_labels=['northerly', 'easterly', 'southerly', 'westerly'],
                          return_data=True)

    bw.dist_by_dir_sector(DATA.Spd40mN, DATA.Dir38mS, aggregation_method='std', return_data=True)


def test_freq_table():
    bw.freq_table(DATA[['Spd40mN']], DATA[['Dir38mS']])
    bw.freq_table(DATA.Spd40mN, DATA.Dir38mS, return_data=True)

    # Calling with user defined dir_bin labels BUGGY
    graph, tab = bw.freq_table(DATA.Spd40mN, DATA.Dir38mS, direction_bin_array=[0, 90, 160, 210, 360],
                               direction_bin_labels=['lowest', 'lower', 'mid', 'high'], return_data=True)
    assert (tab.columns == ['lowest', 'lower', 'mid', 'high']).all()

    bw.freq_table(DATA.Spd40mN, DATA.Dir38mS, plot_bins=[0, 3, 6, 9, 12, 15, 41],
                  plot_labels=['0-3 m/s', '4-6 m/s', '7-9 m/s', '10-12 m/s', '13-15 m/s', '15+ m/s'],
                  return_data=True)
    # Calling with user defined var_bin labels
    bw.freq_table(DATA.Spd40mN, DATA.Dir38mS, var_bin_array=[0, 10, 15, 50],
                  var_bin_labels=['low', 'mid', 'high'], plot_bins=[0, 10, 15, 50], plot_labels=None,
                  return_data=True)

    bw.freq_table(DATA.Spd40mN, DATA.Dir38mS, var_bin_array=[0, 8, 14, 41], var_bin_labels=['low', 'mid', 'high'],
                  direction_bin_array=[0, 90, 130, 200, 360],
                  direction_bin_labels=['northerly', 'easterly', 'southerly', 'westerly'],
                  plot_bins=[0, 8, 14, 41], plot_labels=None, return_data=True)
    bw.freq_table(DATA.T2m, DATA.Dir78mS, var_bin_array=[-10, 0, 10, 20],
                  var_bin_labels=['low', 'mid', 'high'],
                  plot_bins=[-10, 0, 10, 20], plot_labels=None,
                  return_data=True)


def test_dist():
    bw.dist(DATA[['Spd40mN']], bins=[0, 8, 12, 21], bin_labels=['normal', 'gale', 'storm'])

    # For distribution of %frequency of wind speeds
    bw.dist(DATA.Spd40mN, bins=[0, 8, 12, 21], bin_labels=['normal', 'gale', 'storm'])

    # For distribution of mean temperature
    bw.dist(DATA.T2m, bins=[-10, 4, 12, 18, 30], aggregation_method='mean')

    # For custom aggregation function
    def custom_agg(x):
        return x.mean() + (2 * x.std())

    bw.dist(DATA.T2m, bins=[-10, 4, 12, 18, 30], aggregation_method=custom_agg)

    # For distribution of mean wind speeds with respect to temperature
    bw.dist(DATA.Spd40mN, var_to_bin_against=DATA.T2m,
            bins=[-10, 4, 12, 18, 30],
            bin_labels=['freezing', 'cold', 'mild', 'hot'], aggregation_method='mean')


def test_dist_of_wind_speed():
    bw.dist_of_wind_speed(DATA[['Spd80mN']], max_speed=30, max_y_value=10, return_data=False)
    bw.dist_of_wind_speed(DATA.Spd80mN, max_speed=30, max_y_value=10, return_data=False)
    assert True


def test_freq_distribution():
    bw.freq_distribution(DATA[['Spd80mN']], max_speed=30, max_y_value=10, return_data=False)
    bw.freq_distribution(DATA.Spd80mN, max_speed=30, max_y_value=10, return_data=False)
    assert True


def test_ti_by_speed():
    bw.TI.by_speed(DATA[['Spd80mN']], DATA[['Spd80mNStd']])
    bw.TI.by_speed(DATA.Spd80mN, DATA.Spd80mNStd)

    # 60 percentile
    bw.TI.by_speed(DATA.Spd80mN, DATA.Spd80mNStd, percentile=60, return_data=True)

    # bin_array
    bw.TI.by_speed(DATA.Spd80mN, DATA.Spd80mNStd, speed_bin_array=[0, 10, 14, 51],
                   speed_bin_labels=['low', 'mid', 'high'], return_data=True)
    # assert TI_by_speed.index == ['low', 'mid', 'high']
    assert True


def test_calc_air_density():
    bw.calc_air_density(DATA[['T2m']], DATA[['P2m']])
    bw.calc_air_density(DATA.T2m, DATA.P2m)
    bw.calc_air_density(DATA.T2m, DATA.P2m, elevation_ref=0, elevation_site=200)

    with pytest.raises(TypeError) as except_info:
        bw.calc_air_density(15, 1013, elevation_site=200)
    assert str(except_info.value) == 'elevation_ref should be a number'
    with pytest.raises(TypeError) as except_info:
        bw.calc_air_density(15, 1013, elevation_ref=200)
    assert str(except_info.value) == 'elevation_site should be a number'
    assert abs(bw.calc_air_density(15, 1013) - 1.225) < 1e-3
    assert abs(bw.calc_air_density(15, 1013, elevation_ref=0, elevation_site=200) - 1.203) < 1e-3
    assert (abs(bw.calc_air_density(pd.Series([15, 12.5, -5, 23]), pd.Series([1013, 990, 1020, 900])) -
                pd.Series([1.225, 1.208, 1.326, 1.059])) < 1e-3).all()


def test_dist_matrix_by_direction_sector():
    bw.dist_matrix_by_dir_sector(var_series=DATA.Spd80mN, var_to_bin_by_series=DATA.Spd80mN,
                                 direction_series=DATA.Dir78mS, aggregation_method='count')
    bw.dist_matrix_by_dir_sector(DATA.Spd40mN, DATA.T2m, DATA.Dir38mS,
                                 var_to_bin_by_array=[-8, -5, 5, 10, 15, 20, 26],
                                 direction_bin_array=[0, 90, 180, 270, 360],
                                 direction_bin_labels=['north', 'east', 'south', 'west'])
    bw.dist_matrix_by_dir_sector(DATA.Spd40mN, DATA.T2m, DATA.Dir38mS,
                                 var_to_bin_by_array=[-8, -5, 5, 10, 15, 20, 26], sectors=8)
    assert True


def test_average_wdirs():
    wdirs = np.array([350, 10])
    assert bw.average_wdirs(wdirs) == 0.0

    wdirs = np.array([0, 180])
    assert bw.average_wdirs(wdirs) is np.NaN

    wdirs = np.array([90, 270])
    assert bw.average_wdirs(wdirs) is np.NaN

    wdirs = np.array([45, 135])
    assert bw.average_wdirs(wdirs) == 90

    wdirs = np.array([135, 225])
    assert bw.average_wdirs(wdirs) == 180

    wdirs = np.array([45, 315, 225, 135])
    assert bw.average_wdirs(wdirs) is np.NaN

    wdirs = np.array([225, 315])
    assert bw.average_wdirs(wdirs) == 270

    wdirs = np.array([0, 10, 20, 340, 350, 360])
    assert bw.average_wdirs(wdirs) == 0.0

    wdirs_with_nan = [15, np.nan, 25]
    assert round(bw.average_wdirs(wdirs_with_nan), 3) == 20.000

    wspds = [3, 4, 5]
    assert round(bw.average_wdirs(wdirs_with_nan, wspds), 3) == 21.253

    wspds_with_nan = [3, 4, np.nan]
    assert round(bw.average_wdirs(wdirs_with_nan, wspds_with_nan), 3) == 15.0

    wspds_with_nan = [np.nan, np.nan, np.nan]
    assert bw.average_wdirs(wdirs_with_nan, wspds_with_nan) is np.NaN

    wspds_with_nan = [3, 4, np.nan]
    assert round(bw.average_wdirs(pd.Series(wdirs_with_nan), pd.Series(wspds_with_nan)), 3) == 15.0

    wdirs_with_nan = np.array(wdirs_with_nan)
    assert round(bw.average_wdirs(wdirs_with_nan), 3) == 20.000

    wdirs_with_nan = pd.Series(wdirs_with_nan)
    assert round(bw.average_wdirs(wdirs_with_nan), 3) == 20.000

    wdirs_series = pd.Series(wdirs)
    assert bw.average_wdirs(wdirs_series) == 0.0

    wspds = np.array([5, 5, 5, 5, 5, 5])
    assert bw.average_wdirs(wdirs, wspds) == 0.0

    wspds_series = pd.Series(wspds)
    assert bw.average_wdirs(wdirs_series, wspds_series) == 0.0

    wspds = np.array([5, 8.5, 10, 10, 6, 5])
    assert round(bw.average_wdirs(wdirs, wspds), 4) == 0.5774

    wdirs = np.array([[350, 10],
                      [0, 180],
                      [90, 270],
                      [45, 135],
                      [135, 225],
                      [15, np.nan]])
    wdirs_df = pd.DataFrame(wdirs)
    avg_wdirs = np.round(bw.average_wdirs(wdirs_df).values, 3)
    avg_wdirs = np.array([x for x in avg_wdirs if x == x])  # remove nans
    expected_result = [0., np.nan, np.nan, 90., 180., 15.]
    expected_result = np.array([x for x in expected_result if x == x])  # remove nans
    for i, j in zip(avg_wdirs, expected_result):
        assert i == j

    wspds = np.array([[1, 2],
                      [1, 2],
                      [1, 2],
                      [1, 2],
                      [1, 2],
                      [np.nan, 2]])
    wspds_df = pd.DataFrame(wspds)
    avg_wdirs = np.round(bw.average_wdirs(wdirs_df, wspds_df).values, 2)
    avg_wdirs = np.array([x for x in avg_wdirs if x == x])  # remove nans
    expected_result = np.array([3.36, 180., 270., 108.43, 198.43])
    for i, j in zip(avg_wdirs, expected_result):
        assert i == j
