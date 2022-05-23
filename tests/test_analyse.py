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

    assert list(round(bw.monthly_means(DATA[WSPD_COLS], return_data=True)[1][
                      :'2016-01-01'], 5).values[0]) == [9.25346, 9.22915, 8.51025, 8.68109, 8.10126, 8.20393]
    assert round(bw.monthly_means(DATA.Spd80mN, return_data=True)[1], 5)[0] == 9.25346
    # input data_resolution
    data_monthly = bw.average_data_by_period(DATA.Spd80mS, period='1M')
    data_monthly = data_monthly[data_monthly.index.month.isin([2, 4, 6, 8])]
    _, monthly_mean_data = bw.monthly_means(data_monthly, return_data=True,
                                            data_resolution=pd.Timedelta('1M'))
    assert (monthly_mean_data.dropna() == data_monthly).all()
    with pytest.raises(ValueError) as except_info:
        bw.monthly_means(data_monthly, return_data=True)
    assert str(except_info.value) == "The time period specified is less than the temporal resolution of the data. " \
                                     "For example, hourly data should not be averaged to 10 minute data."


def test_sector_ratio():
    bw.sector_ratio(DATA['Spd80mN'], DATA['Spd80mS'], DATA['Dir78mS'], sectors=72, boom_dir_1=0,
                    boom_dir_2=180, return_data=True)
    bw.sector_ratio(DATA[['Spd40mN']], DATA.Spd60mN, wdir=DATA[['Dir38mS']])
    bw.sector_ratio(DATA.Spd40mN, DATA.Spd60mN, wdir=DATA.Dir38mS,
                    direction_bin_array=[0, 45, 135, 180, 220, 360], boom_dir_1=160, boom_dir_2=340)
    bw.sector_ratio(DATA.Spd80mN, DATA.Spd80mS, wdir=DATA.Dir78mS, radial_limits=(0.8, 1.2),
                    figure_size=(10,10))
    bw.sector_ratio(DATA[['Spd80mN', 'Spd60mN']], DATA[['Spd80mS', 'Spd60mS']],
                    DATA['Dir78mS'], boom_dir_1=0, boom_dir_2=180, figure_size=(25, 25))
    bw.sector_ratio(DATA[['Spd80mN', 'Spd60mN', 'Spd40mN']], DATA[['Spd80mS', 'Spd60mS', 'Spd40mS']],
                    DATA[['Dir78mS', 'Dir58mS', 'Dir38mS']], boom_dir_1=0, boom_dir_2=180,
                    figure_size=(25,25))
    bw.sector_ratio(DATA[['Spd80mN', 'Spd60mN', 'Spd40mN']], DATA[['Spd80mS', 'Spd60mS', 'Spd40mS']],
                 DATA[['Dir78mS', 'Dir58mS', 'Dir38mS']], boom_dir_1=0, boom_dir_2=180, figure_size=(25, 25),
                 return_data=True)
    bw.sector_ratio(DATA['Spd80mN'], DATA['Spd80mS'], DATA['Dir78mS'], boom_dir_1=0, boom_dir_2=180, return_data=True)
    bw.sector_ratio(DATA[['Spd80mN', 'Spd60mN']], DATA[['Spd80mS', 'Spd60mS']],
                    DATA[['Dir78mS', 'Dir58mS']], boom_dir_1=[0, 350], boom_dir_2=[180, 170], figure_size=(25, 25))
    bw.sector_ratio(DATA['Spd80mN'], DATA['Spd80mS'], DATA['Dir78mS'], sectors=72, boom_dir_1=0,
                    boom_dir_2=180, annotate=False)
    bw.sector_ratio(DATA[['Spd80mN', 'Spd60mN', 'Spd60mN']], DATA[['Spd80mS', 'Spd60mS', 'Spd60mS']], DATA['Dir78mS'],
                    annotate=False, figure_size=(25, 25), boom_dir_1=0, boom_dir_2=180)
    assert True


def test_basic_stats():
    bw.basic_stats(DATA)
    bs2 = bw.basic_stats(DATA['Spd80mN'])
    assert (bs2['count'][0] == 95180.0) and (round(bs2['mean'][0], 6) == 7.518636) and \
           (round(bs2['std'][0], 6) == 3.994552) and (round(bs2['min'][0], 3) == 0.215) and \
           (round(bs2['max'][0], 1) == 29.0)


def test_time_continuity_gaps():
    gaps = bw.time_continuity_gaps(DATA['Spd80mN'])
    assert gaps.iloc[0, 0] == pd.Timestamp('2016-03-09 06:10:00')
    assert gaps.iloc[0, 1] == pd.Timestamp('2016-03-09 10:30:00')
    assert gaps.iloc[1, 0] == pd.Timestamp('2016-03-29 23:40:00')
    assert gaps.iloc[1, 1] == pd.Timestamp('2016-03-30 07:10:00')
    assert abs(gaps.iloc[0, 2] - 0.173611) < 1e-5
    assert abs(gaps.iloc[1, 2] - 0.305556) < 1e-5

    # test for when timesteps are irregular
    # THIS WILL RAISE 3 WARNINGS.
    data_test = DATA.copy()
    data_test.reset_index(inplace=True)
    data_test['Timestamp'][10] = data_test['Timestamp'][10] + pd.Timedelta('1 min')
    data_test['Timestamp'][20] = data_test['Timestamp'][20] + pd.Timedelta('9 min')
    data_test.set_index('Timestamp', inplace=True)
    gaps_irregular = bw.time_continuity_gaps(data_test)
    assert gaps_irregular.iloc[0, 0] == pd.Timestamp('2016-01-09 18:10:00')
    assert gaps_irregular.iloc[0, 1] == pd.Timestamp('2016-01-09 18:21:00')
    assert gaps_irregular.iloc[1, 0] == pd.Timestamp('2016-01-09 18:21:00')
    assert gaps_irregular.iloc[1, 1] == pd.Timestamp('2016-01-09 18:30:00')
    assert abs(gaps_irregular.iloc[0, 2] - 0.000694) < 1e-5
    assert np.isnan(gaps_irregular.iloc[1, 2])
    assert abs(gaps_irregular.iloc[2, 2] - 0.006250) < 1e-5

    # test for monthly timeseries
    data_monthly = bw.average_data_by_period(DATA[DATA.index.month.isin([1, 3, 4, 5, 6, 7, 8, 10, 12])],
                                             period='1M').dropna()
    gaps_irregular = bw.time_continuity_gaps(data_monthly)
    assert gaps_irregular.iloc[0, 0] == pd.Timestamp('2016-01-01')
    assert gaps_irregular.iloc[1, 1] == pd.Timestamp('2016-10-01')
    assert gaps_irregular.iloc[1, 2] == 30
    assert gaps_irregular.iloc[0, 2] == 29


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
    assert round(bw.coverage(DATA[['Spd80mN']], period='1H')[
           '2016-01-09 17:00':'2016-01-09 17:30'].values[0][0], 5) == 0.83333
    assert round(bw.coverage(DATA.Spd80mN, period='1H')[
           '2016-01-09 17:00':'2016-01-09 17:30'].values[0], 5) == 0.83333
    # monthly_coverage
    assert round(bw.coverage(DATA.Spd80mN, period='1M')['2016-05-01'], 5) == 0.36537
    # monthly_coverage of variance
    assert round(bw.coverage(DATA.Spd80mN, period='1M', aggregation_method='var')['2016-05-01'], 5) == 0.36537
    # input data_resolution
    data1 = DATA[:'2016-01-10'].copy()
    data1.reset_index(inplace=True)
    drop_indices = np.array([0, 66, 96, 43, 10, 21, 84, 11, 58, 120, 78, 166, 148,
                             176, 31, 93, 114, 107, 17, 49, 110, 16, 130, 69, 106, 18,
                             12, 77, 67, 29, 81, 28, 118, 95, 54, 179, 169, 72, 144,
                             90, 38, 92, 142, 30, 45, 151, 126, 42, 73, 171, 99, 83,
                             157, 75, 60, 82, 162, 22, 128, 52, 123, 153, 36, 20, 170,
                             65, 152, 61, 140, 85, 111, 8, 37, 121, 63, 112, 141, 183,
                             168, 74, 88, 119, 156, 51, 180, 143, 79, 134, 124, 117, 109,
                             108, 86, 161, 155, 9, 182, 50, 3, 150, 138, 19, 15, 40,
                             97, 158, 24, 113, 39, 1, 137, 4, 13, 57, 5, 35, 187,
                             172, 47, 132, 122, 116, 33, 6, 181, 62, 133, 32, 25, 89,
                             34, 94, 46, 14, 185, 76, 101, 100, 98, 167, 125, 164, 26,
                             136, 139, 174, 127, 104, 80, 2, 178, 160, 173, 41, 59, 163,
                             175, 64, 145, 27, 55, 149, 70, 146, 147, 103, 184, 165, 44,
                             23, 115, 48, 68, 102, 53, 7, 129, 56, 135, 91])
    data1 = data1.drop(drop_indices)
    data1 = data1.set_index('Timestamp')
    assert round(bw.coverage(data1.Spd80mS, period='1M',
                             data_resolution=pd.Timedelta('10min')).values[0], 8) == 0.00179211


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

