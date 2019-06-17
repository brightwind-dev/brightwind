import pytest
import brightwind as bw
import pandas as pd



def test_monthly_means():
    #Load data
    bw.monthly_means(bw.load_csv(bw.datasets.shell_flats_80m_csv))
    bw.monthly_means(bw.load_csv(bw.datasets.shell_flats_80m_csv)[['WS70mA100NW_Avg','WS70mA100SE_Avg',
                                                                          'WS50mA100NW_Avg','WS50mA100SE_Avg',
                                                                          'WS20mA100CB1_Avg','WS20mA100CB2_Avg']],
                        return_data=True)
    bw.monthly_means(bw.load_csv(bw.datasets.shell_flats_80m_csv).WS80mWS425NW_Avg, return_data=True)
    assert True


def test_wspd_ratio_by_dir_sector():
    data = bw.load_csv(bw.datasets.shell_flats_80m_csv)
    bw.wspd_ratio_by_dir_sector(data['WS70mA100NW_Avg'], data['WS70mA100SE_Avg'], data['WD50mW200PNW_VAvg'],
                          sectors = 72, boom_dir_1 = 315, boom_dir_2 = 135,return_data=True)[1]
    data = bw.load_csv(bw.datasets.demo_data)
    bw.wspd_ratio_by_dir_sector(data.Spd40mN, data.Spd60mN, wdir=data.Dir38mS,
                             direction_bin_array=[0, 45, 135, 180, 220, 360], boom_dir_1=160, boom_dir_2=340)
    assert True


def test_basic_stats():
    data = bw.load_csv(bw.datasets.shell_flats_80m_csv)
    bw.basic_stats(data)
    bs2 = bw.basic_stats(data['WS70mA100NW_Avg'])
    assert (bs2['count']==58874.0).bool() and((bs2['mean']-9.169382)<1e-6).bool() and ((bs2['std']-4.932851)<1e-6).bool()\
           and (bs2['max']==27.66).bool() and (bs2['min'] == 0.0).bool()


def test_time_continuity_gaps():
    import pandas as pd
    data = bw.load_csv(bw.datasets.shell_flats_80m_csv)
    gaps = bw.time_continuity_gaps(data['WS70mA100NW_Avg'][:400])
    assert gaps.iloc[0, 0] == pd.Timestamp('2011-07-16 17:50:00')
    assert gaps.iloc[0, 1] == pd.Timestamp('2011-07-16 18:10:00')
    assert gaps.iloc[1, 0] == pd.Timestamp('2011-07-16 23:00:00')
    assert gaps.iloc[1, 1] == pd.Timestamp('2011-07-16 23:20:00')
    assert abs(gaps.iloc[0, 2] - 0.01388) < 1e-5
    assert abs(gaps.iloc[1, 2] - 0.01388) < 1e-5


def test_twelve_by_24():
    df = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)
    graph, table12x24 = bw.twelve_by_24(df.Spd40mN, var_name_label='wind speed', return_data=True)
    graph = bw.twelve_by_24(df.PrcpTot, aggregation_method='sum')
    def custom_agg(x):
        return x.mean() + (2 * x.std())
    graph, table12x24 = bw.twelve_by_24(df.PrcpTot, aggregation_method=custom_agg, return_data=True)

    assert True


def test_TI_twelve_by_24():
    df = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)
    bw.TI.twelve_by_24(df.Spd60mN, df.Spd60mNStd)
    bw.TI.twelve_by_24(df.Spd60mN, df.Spd60mNStd, return_data=True)
    bw.TI.twelve_by_24(df.Spd60mN, df.Spd60mNStd, return_data=True, var_name_label='Speed 60 m N m/s')
    bw.TI.twelve_by_24(df.Spd60mN, df.Spd60mNStd, var_name_label='Speed 60 m N m/s')
    bw.TI.twelve_by_24(df.Spd40mN, df.Spd40mNStd)
    assert 1 == 1



def test_coverage():
    data = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)

    # hourly coverage
    data_hourly = bw.coverage(data.Spd80mN, period='1H')
    # monthly_coverage
    data_hourly = bw.coverage(data.Spd80mN, period='1M')
    # monthly_coverage of variance
    data_hourly = bw.coverage(data.Spd80mN, period='1M', aggregation_method='var')
    assert True


def test_distribution_by_dir_sector():
    df = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)

    rose = bw.distribution_by_dir_sector(df.Spd40mN, df.Dir38mS)

    rose, distribution = bw.distribution_by_dir_sector(df.Spd40mN, df.Dir38mS,
                                                       direction_bin_array=[0, 90, 130, 200, 360],
                                                       direction_bin_labels=['northerly', 'easterly', 'southerly',
                                                                             'westerly'],
                                                       return_data=True)


    rose, distribution = bw.distribution_by_dir_sector(df.Spd40mN, df.Dir38mS, aggregation_method='std',
                                                       return_data=True)


def test_freq_table():
    df = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)

    graph, tab = bw.freq_table(df.Spd40mN, df.Dir38mS, return_data=True)

    # Calling with user defined dir_bin labels BUGGY
    graph, tab = bw.freq_table(df.Spd40mN, df.Dir38mS, direction_bin_array=[0, 90, 160, 210, 360],
                           direction_bin_labels=['lowest','lower','mid','high'], return_data=True)
    assert (tab.columns==['lowest','lower','mid','high']).all()

    tab = bw.freq_table(df.Spd40mN, df.Dir38mS, plot_bins=[0, 3, 6, 9, 12, 15, 41],
                        plot_labels=['0-3 m/s', '4-6 m/s', '7-9 m/s', '10-12 m/s', '13-15 m/s', '15+ m/s'],
                        return_data=True)
    #Calling with user defined var_bin labels
    graph, tab = bw.freq_table(df.Spd40mN, df.Dir38mS, var_bin_array=[0, 10, 15, 50],
                               var_bin_labels=['low', 'mid', 'high'], plot_bins=None, plot_labels=None,
                               return_data=True)

    tab = bw.freq_table(df.Spd40mN, df.Dir38mS, var_bin_array=[0, 8, 14, 41], var_bin_labels=['low', 'mid', 'high'],
                        direction_bin_array=[0, 90, 130, 200, 360],
                        direction_bin_labels=['northerly', 'easterly', 'southerly', 'westerly'],
                        plot_bins=None, plot_labels=None, return_data=True)
                        # var_bin_labels=['operating','shutdow','dangerous'],

def test_distribution():
    df = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)

    # For distribution of %frequency of wind speeds
    dist = bw.distribution(df.Spd40mN, bins=[0, 8, 12, 21], bin_labels=['normal', 'gale', 'storm'])

    # For distribution of mean temperature
    temp_dist = bw.distribution(df.T2m, bins=[-10, 4, 12, 18, 30], aggregation_method='mean')

    # For custom aggregation function
    def custom_agg(x):
        return x.mean() + (2 * x.std())

    temp_dist = bw.distribution(df.T2m, bins=[-10, 4, 12, 18, 30], aggregation_method=custom_agg)

    # For distribution of mean wind speeds with respect to temperature
    spd_dist = bw.distribution(df.Spd40mN, var_to_bin_against=df.T2m,
                               bins=[-10, 4, 12, 18, 30],
                               bin_labels=['freezing', 'cold', 'mild', 'hot'], aggregation_method='mean')

def test_TI_by_speed():
    df = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)
    TI_by_speed = bw.TI.by_speed(df.Spd80mN, df.Spd80mNStd)

    #60 percentile
    TI_by_speed_60 = bw.TI.by_speed(df.Spd80mN, df.Spd80mNStd, percentile=60, return_data=True)

    #bin_array
    TI_by_speed = bw.TI.by_speed(df.Spd80mN, df.Spd80mNStd, speed_bin_array=[0, 10, 14, 51],
                                      speed_bin_labels=['low', 'mid', 'high'], return_data=True)
    # assert TI_by_speed.index == ['low', 'mid', 'high']
    assert True


def test_calc_air_density():
    data = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)
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

