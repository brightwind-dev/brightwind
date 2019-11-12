import pytest
import brightwind as bw
import pandas as pd


def test_monthly_means():
    # Load data
    bw.monthly_means(bw.load_csv(bw.datasets.demo_data))
    bw.monthly_means(bw.load_csv(bw.datasets.demo_data)[['Spd80mS']])

    bw.monthly_means(bw.load_csv(bw.datasets.demo_data)[['Spd80mS', 'Spd80mN', 'Spd60mS', 'Spd60mN',
                                                         'Spd40mS', 'Spd40mN']], return_data=True)
    bw.monthly_means(bw.load_csv(bw.datasets.demo_data).Spd80mS, return_data=True)
    assert True


def test_sector_ratio():
    data = bw.load_csv(bw.datasets.demo_data)
    sec_rat_data = bw.sector_ratio(data['Spd80mN'], data['Spd80mS'], data['Spd60mN'],
                                   sectors=72, boom_dir_1=315, boom_dir_2=135, return_data=True)[1]
    data = bw.load_csv(bw.datasets.demo_data)
    bw.sector_ratio(data[['Spd40mN']], data.Spd60mN, wdir=data[['Dir38mS']])
    bw.sector_ratio(data.Spd40mN, data.Spd60mN, wdir=data.Dir38mS,
                    direction_bin_array=[0, 45, 135, 180, 220, 360], boom_dir_1=160, boom_dir_2=340)
    assert True


def test_basic_stats():
    data = bw.load_csv(bw.datasets.demo_data)
    bw.basic_stats(data)
    # bs1 = bw.basic_stats(data[['Spd_80mN']])
    bs2 = bw.basic_stats(data['Spd80mN'])
    assert (bs2['count'] == 95629.0).bool() and((bs2['mean']-7.498665) < 1e-6).bool() and \
           ((bs2['std']-3.998231) < 1e-6).bool() and (bs2['max'] == 29.00).bool() and (bs2['min'] == 0.215).bool()


def test_time_continuity_gaps():
    import pandas as pd
    data = bw.load_csv(bw.datasets.demo_data)
    # gaps_0 = bw.time_continuity_gaps(data[['WS70mA100NW_Avg']][:400])
    gaps = bw.time_continuity_gaps(data['Spd80mN'][:20000])
    assert gaps.iloc[0, 0] == pd.Timestamp('2016-01-09 15:50:00')
    assert gaps.iloc[0, 1] == pd.Timestamp('2016-01-09 16:50:00')
    assert gaps.iloc[1, 0] == pd.Timestamp('2016-05-11 23:10:00')
    assert gaps.iloc[1, 1] == pd.Timestamp('2016-05-31 15:10:00')
    assert abs(gaps.iloc[0, 2] - .041666666666666664) < 1e-5
    assert abs(gaps.iloc[1, 2] - 19.666666666666668) < 1e-5


def test_dist_12x24():
    df = bw.load_csv(bw.datasets.demo_data)
    graph, table12x24 = bw.dist_12x24(df[['Spd40mN']], return_data=True)
    graph, table12x24 = bw.dist_12x24(df.Spd40mN, var_name_label='wind speed', return_data=True)
    graph = bw.dist_12x24(df.PrcpTot, aggregation_method='sum')
    
    def custom_agg(x):
        return x.mean() + (2 * x.std())

    graph, table12x24 = bw.dist_12x24(df.PrcpTot, aggregation_method=custom_agg, return_data=True)

    assert True


def test_TI_twelve_by_24():
    df = bw.load_csv(bw.datasets.demo_data)
    bw.TI.twelve_by_24(df[['Spd60mN']], df[['Spd60mNStd']])
    bw.TI.twelve_by_24(df.Spd60mN, df.Spd60mNStd)
    bw.TI.twelve_by_24(df.Spd60mN, df.Spd60mNStd, return_data=True)
    bw.TI.twelve_by_24(df.Spd60mN, df.Spd60mNStd, return_data=True, var_name_label='Speed 60 m N m/s')
    bw.TI.twelve_by_24(df.Spd60mN, df.Spd60mNStd, var_name_label='Speed 60 m N m/s')
    bw.TI.twelve_by_24(df.Spd40mN, df.Spd40mNStd)
    assert 1 == 1


def test_coverage():
    data = bw.load_csv(bw.datasets.demo_data)

    # hourly coverage
    data_hourly = bw.coverage(data[['Spd80mN']], period='1H')
    data_hourly = bw.coverage(data.Spd80mN, period='1H')
    # monthly_coverage
    data_monthly = bw.coverage(data.Spd80mN, period='1M')
    # monthly_coverage of variance
    data_monthly= bw.coverage(data.Spd80mN, period='1M', aggregation_method='var')
    assert True


def test_dist_by_dir_sector():
    df = bw.load_csv(bw.datasets.demo_data)
    rose = bw.dist_by_dir_sector(df[['Spd40mN']], df[['Dir38mS']])
    rose = bw.dist_by_dir_sector(df.Spd40mN, df.Dir38mS)

    rose, distribution = bw.dist_by_dir_sector(df.Spd40mN, df.Dir38mS,
                                               direction_bin_array=[0, 90, 130, 200, 360],
                                               direction_bin_labels=['northerly', 'easterly', 'southerly',
                                                                     'westerly'],
                                               return_data=True)

    rose, distribution = bw.dist_by_dir_sector(df.Spd40mN, df.Dir38mS, aggregation_method='std', return_data=True)


def test_freq_table():
    df = bw.load_csv(bw.datasets.demo_data)

    graph = bw.freq_table(df[['Spd40mN']], df[['Dir38mS']])
    graph, tab = bw.freq_table(df.Spd40mN, df.Dir38mS, return_data=True)

    # Calling with user defined dir_bin labels BUGGY
    graph, tab = bw.freq_table(df.Spd40mN, df.Dir38mS, direction_bin_array=[0, 90, 160, 210, 360],
                           direction_bin_labels=['lowest', 'lower', 'mid', 'high'], return_data=True)
    assert (tab.columns == ['lowest','lower','mid','high']).all()

    tab = bw.freq_table(df.Spd40mN, df.Dir38mS, plot_bins=[0, 3, 6, 9, 12, 15, 41],
                        plot_labels=['0-3 m/s', '4-6 m/s', '7-9 m/s', '10-12 m/s', '13-15 m/s', '15+ m/s'],
                        return_data=True)
    #Calling with user defined var_bin labels
    graph, tab = bw.freq_table(df.Spd40mN, df.Dir38mS, var_bin_array=[0, 10, 15, 50],
                               var_bin_labels=['low', 'mid', 'high'], plot_bins=[0, 10, 15, 50], plot_labels=None,
                               return_data=True)

    tab = bw.freq_table(df.Spd40mN, df.Dir38mS, var_bin_array=[0, 8, 14, 41], var_bin_labels=['low', 'mid', 'high'],
                        direction_bin_array=[0, 90, 130, 200, 360],
                        direction_bin_labels=['northerly', 'easterly', 'southerly', 'westerly'],
                        plot_bins=[0, 8, 14, 41], plot_labels=None, return_data=True)
                        # var_bin_labels=['operating','shutdow','dangerous'],
    temp_rose, temp_freq_tab = bw.freq_table(df.T2m, df.Dir78mS, var_bin_array=[-10, 0, 10, 20],
                                             var_bin_labels=['low', 'mid', 'high'],
                                             plot_bins=[-10, 0, 10, 20], plot_labels=None,
                                             return_data=True)


def test_dist():
    df = bw.load_csv(bw.datasets.demo_data)
    dist = bw.dist(df[['Spd40mN']], bins=[0, 8, 12, 21], bin_labels=['normal', 'gale', 'storm'])

    # For distribution of %frequency of wind speeds
    dist = bw.dist(df.Spd40mN, bins=[0, 8, 12, 21], bin_labels=['normal', 'gale', 'storm'])

    # For distribution of mean temperature
    temp_dist = bw.dist(df.T2m, bins=[-10, 4, 12, 18, 30], aggregation_method='mean')

    # For custom aggregation function
    def custom_agg(x):
        return x.mean() + (2 * x.std())

    temp_dist = bw.dist(df.T2m, bins=[-10, 4, 12, 18, 30], aggregation_method=custom_agg)

    # For distribution of mean wind speeds with respect to temperature
    spd_dist = bw.dist(df.Spd40mN, var_to_bin_against=df.T2m,
                       bins=[-10, 4, 12, 18, 30],
                       bin_labels=['freezing', 'cold', 'mild', 'hot'], aggregation_method='mean')


def test_dist_of_wind_speed():
    df = bw.load_csv(bw.datasets.demo_data)
    spd_dist = bw.dist_of_wind_speed(df[['Spd80mN']], max_speed=30, max_y_value=10, return_data=False)
    spd_dist = bw.dist_of_wind_speed(df.Spd80mN, max_speed=30, max_y_value=10, return_data=False)
    assert True


def test_freq_distribution():
    df = bw.load_csv(bw.datasets.demo_data)
    spd_dist = bw.freq_distribution(df[['Spd80mN']], max_speed=30, max_y_value=10, return_data=False)
    spd_dist = bw.freq_distribution(df.Spd80mN, max_speed=30, max_y_value=10, return_data=False)
    assert True


def test_TI_by_speed():
    df = bw.load_csv(bw.datasets.demo_data)
    TI_by_speed = bw.TI.by_speed(df[['Spd80mN']], df[['Spd80mNStd']])
    TI_by_speed = bw.TI.by_speed(df.Spd80mN, df.Spd80mNStd)

    #60 percentile
    TI_by_speed_60 = bw.TI.by_speed(df.Spd80mN, df.Spd80mNStd, percentile=60, return_data=True)

    #bin_array
    TI_by_speed = bw.TI.by_speed(df.Spd80mN, df.Spd80mNStd, speed_bin_array=[0, 10, 14, 51],
                                      speed_bin_labels=['low', 'mid', 'high'], return_data=True)
    # assert TI_by_speed.index == ['low', 'mid', 'high']
    assert True


def test_calc_air_density():
    data = bw.load_csv(bw.datasets.demo_data)
    bw.calc_air_density(data[['T2m']], data[['P2m']])
    bw.calc_air_density(data.T2m, data.P2m)
    bw.calc_air_density(data.T2m, data.P2m, elevation_ref=0, elevation_site=200)

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
    df = bw.load_csv(bw.datasets.demo_data)
    bw.dist_matrix_by_dir_sector(var_series=df.Spd80mN, var_to_bin_by_series=df.Spd80mN, direction_series=df.Dir78mS,
                                 aggregation_method='count')
    matrix = bw.dist_matrix_by_dir_sector(df.Spd40mN, df.T2m, df.Dir38mS,
                                          var_to_bin_by_array=[-8, -5, 5, 10, 15, 20, 26],
                                          direction_bin_array=[0, 90, 180, 270, 360],
                                          direction_bin_labels=['north', 'east', 'south', 'west'])
    matrix = bw.dist_matrix_by_dir_sector(df.Spd40mN, df.T2m, df.Dir38mS,
                                          var_to_bin_by_array=[-8, -5, 5, 10, 15, 20, 26], sectors=8)
    assert True