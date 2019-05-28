import pytest
import brightwind as bw

def test_monthly_means():
    #Load data
    bw.monthly_means(bw.load_csv(bw.datasets.shell_flats_80m_csv))
    bw.monthly_means(bw.load_csv(bw.datasets.shell_flats_80m_csv)[['WS70mA100NW_Avg','WS70mA100SE_Avg',
                                                                          'WS50mA100NW_Avg','WS50mA100SE_Avg',
                                                                          'WS20mA100CB1_Avg','WS20mA100CB2_Avg']],
                        return_data=True)
    # monthly_means(load_csv(brightwind.dathttps://github.com/brightwind-dev/brightwind/pull/51/conflict?name=brightwind%252Ftests%252Ftest_analyse.py&ancestor_oid=327cb3af2fe06cb245669ae2d57c2484eed00991&base_oid=baa2a6ac676b4b08cce0fa48dc7903946a8ba49f&head_oid=4ec3529abbc2e330c2ba35e05a8227690d956273asets.shell_flats_80m_csv).WS80mWS425NW_Avg)
    bw.monthly_means(bw.load_csv(bw.datasets.shell_flats_80m_csv).WS80mWS425NW_Avg, return_data=True)
    assert True

def test_sector_ratio_by_sector():
    data = bw.load_csv(bw.datasets.shell_flats_80m_csv)
    bw.SectorRatio.by_sector(data['WS70mA100NW_Avg'], data['WS70mA100SE_Avg'], data['WD50mW200PNW_VAvg'],
                          sectors = 72, boom_dir_1 = 315, boom_dir_2 = 135,return_data=True)[1]
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
    assert 1==1

def test_coverage():
    data = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)

    # hourly coverage
    data_hourly = bw.coverage(data.Spd80mN, period='1H')
    # monthly_coverage
    data_hourly = bw.coverage(data.Spd80mN, period='1M')
    # monthly_coverage of variance
    data_hourly = bw.coverage(data.Spd80mN, period='1M', aggregation_method='var')
