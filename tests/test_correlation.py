import pytest
import brightwind as bw
import pandas as pd
import numpy as np
import warnings

wndspd = 8
wndspd_df = pd.DataFrame([2, 13, np.NaN, 5, 8])
wndspd_series = pd.Series([2, 13, np.NaN, 5, 8])
current_slope = 0.045
current_offset = 0.235
new_slope = 0.046
new_offset = 0.236
wndspd_adj = 8.173555555555556
wndspd_adj_df = pd.DataFrame([2.0402222222222224, 13.284666666666668, np.NaN, 5.106888888888888, 8.173555555555556])
wndspd_adj_series = pd.Series([2.0402222222222224, 13.284666666666668, np.NaN, 5.106888888888888, 8.173555555555556])

DATA = bw.load_campbell_scientific(bw.demo_datasets.demo_campbell_scientific_data)
DATA_CLND = bw.apply_cleaning(DATA, bw.demo_datasets.demo_cleaning_file)
WSPD_COLS = ['Spd80mN', 'Spd80mS', 'Spd60mN', 'Spd60mS', 'Spd40mN', 'Spd40mS']
WDIR_COLS = ['Dir78mS', 'Dir58mS', 'Dir38mS']
MERRA2_NE = bw.load_csv(bw.demo_datasets.demo_merra2_NE)
MERRA2_NW = bw.load_csv(bw.demo_datasets.demo_merra2_NW)


def test_ordinary_least_squares():
    spd80mn_monthly_mean_list = [9.25346307, 8.90438194, 6.43050216, 6.59887454, 8.72965727,
                                 5.10815648, 6.96853427, 7.09395587, 8.18052477, 6.66944556,
                                 6.74182714, 8.90077755, 7.83337582, 9.13450868, 7.48893795,
                                 7.78338958, 6.49058893, 8.52524884, 6.78224843, 6.7158853,
                                 7.08256829, 9.47901579, 7.35934137]
    data_monthly_index_list = ['2016-01-01', '2016-02-01', '2016-03-01', '2016-04-01',
                               '2016-05-01', '2016-06-01', '2016-07-01', '2016-08-01',
                               '2016-09-01', '2016-10-01', '2016-11-01', '2016-12-01',
                               '2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01',
                               '2017-05-01', '2017-06-01', '2017-07-01', '2017-08-01',
                               '2017-09-01', '2017-10-01', '2017-11-01']
    spd80mn_monthly_cov_list = [0.71886201, 1., 0.98454301, 1., 0.36536738, 1., 1., 1., 1., 1., 0.93472222, 1.,
                                0.9858871, 1., 1., 1., 1., 1., 1., 1., 1., 0.99283154, 0.74861111]
    m2_monthly_mean_list = [9.62391129, 9.01344253, 6.85649462, 6.66197639, 6.99338038,
                            5.29984306, 6.73991667, 7.11679032, 8.39015556, 6.83381317,
                            6.84408889, 9.0631707, 8.28869355, 9.2853869, 7.62800806,
                            7.73957917, 6.63575403, 7.81355417]
    correl_monthly_results = {'slope': 0.91963, 'offset': 0.61137, 'r2': 0.8192, 'num_data_points': 18}
    correl_monthly_results_90 = {'slope': 0.99357, 'offset': -0.03654, 'r2': 0.9433, 'num_data_points': 16}
    correl_hourly_results = {'slope': 0.98922, 'offset': -0.03616, 'r2': 0.7379, 'num_data_points': 12369}
    correl_monthly_by_sector_results = [{'num_data_points': 54, 'offset': -1.2344603199476245, 'r2': 0.8783996355797528,
                                         'sector_max': 45.0, 'sector_min': 315.0, 'sector_number': 1,
                                         'slope': 1.1638279733401182},
                                        {'num_data_points': 92, 'offset': -0.055627810879849644,
                                         'r2': 0.7757314640946426, 'sector_max': 135.0, 'sector_min': 45.0,
                                         'sector_number': 2, 'slope': 0.9021112054525899},
                                        {'num_data_points': 157, 'offset': -0.3483428410889726,
                                         'r2': 0.8956396609535162, 'sector_max': 225.0, 'sector_min': 135.0,
                                         'sector_number': 3, 'slope': 1.0287715500843069},
                                        {'num_data_points': 206, 'offset': -0.08618395596626072,
                                         'r2': 0.9304631940882327, 'sector_max': 315.0, 'sector_min': 225.0,
                                         'sector_number': 4, 'slope': 1.028330085357324}]

    correl = bw.Correl.OrdinaryLeastSquares(MERRA2_NE['WS50m_m/s'], DATA_CLND['Spd80mN'], averaging_prd='1M',
                                            coverage_threshold=0)
    correl.run()
    assert round(correl.params['slope'], 5) == correl_monthly_results['slope']
    assert round(correl.params['offset'], 5) == correl_monthly_results['offset']
    assert round(correl.params['r2'], 4) == correl_monthly_results['r2']
    assert round(correl.params['num_data_points'], 5) == correl_monthly_results['num_data_points']
    for idx, row in enumerate(correl.data.iterrows()):
        assert data_monthly_index_list[idx] == str(row[0].date())
        assert round(spd80mn_monthly_mean_list[idx], 5) == round(row[1]['Spd80mN'], 5)
        assert round(spd80mn_monthly_cov_list[idx], 5) == round(row[1]['Spd80mN_Coverage'], 5)
        assert round(m2_monthly_mean_list[idx], 5) == round(row[1]['WS50m_m/s'], 5)

    # check 90% coverage, checked against Excel
    correl = bw.Correl.OrdinaryLeastSquares(MERRA2_NE['WS50m_m/s'], DATA_CLND['Spd80mN'], averaging_prd='1M',
                                            coverage_threshold=0.9)
    correl.run()
    assert round(correl.params['slope'], 5) == correl_monthly_results_90['slope']
    assert round(correl.params['offset'], 5) == correl_monthly_results_90['offset']
    assert round(correl.params['r2'], 4) == correl_monthly_results_90['r2']
    assert round(correl.params['num_data_points'], 5) == correl_monthly_results_90['num_data_points']

    # check hourly, checked against Excel
    correl = bw.Correl.OrdinaryLeastSquares(MERRA2_NE['WS50m_m/s'], DATA_CLND['Spd80mN'], averaging_prd='1H',
                                            coverage_threshold=1)
    correl.run()
    assert round(correl.params['slope'], 5) == correl_hourly_results['slope']
    assert round(correl.params['offset'], 5) == correl_hourly_results['offset']
    assert round(correl.params['r2'], 4) == correl_hourly_results['r2']
    assert round(correl.params['num_data_points'], 5) == correl_hourly_results['num_data_points']

    # check aggregation method
    correl_aggregation_results = {'slope': 5.98789, 'offset': -9.32585, 'r2': 0.9304, 'num_data_points': 12445}
    correl = bw.Correl.OrdinaryLeastSquares(MERRA2_NE['T2M_degC'], DATA_CLND['T2m'],
                                            averaging_prd='1H', coverage_threshold=1,
                                            ref_aggregation_method='sum', target_aggregation_method='sum')
    correl.run()
    correl.plot()
    assert round(correl.params['slope'], 5) == correl_aggregation_results['slope']
    assert round(correl.params['offset'], 5) == correl_aggregation_results['offset']
    assert round(correl.params['r2'], 4) == correl_aggregation_results['r2']
    assert round(correl.params['num_data_points'], 5) == correl_aggregation_results['num_data_points']

    # check correlation by sector
    correl = bw.Correl.OrdinaryLeastSquares(MERRA2_NE['WS50m_m/s'], DATA_CLND['Spd80mN'],
                                            ref_dir=MERRA2_NE['WD50m_deg'], averaging_prd='1D',
                                            coverage_threshold=0.9, sectors=4)
    correl.run()
    correl.plot()
    assert len(correl_monthly_by_sector_results) == len(correl.params)

    for test_param, param in zip(correl_monthly_by_sector_results, correl.params):
        assert round(test_param['slope'], 5) == round(param['slope'], 5)
        assert round(test_param['offset'], 5) == round(param['offset'], 5)
        assert round(test_param['r2'], 5) == round(param['r2'], 5)
        assert test_param['num_data_points'] == param['num_data_points']
        assert test_param['sector_min'] == param['sector_min']
        assert test_param['sector_max'] == param['sector_max']
        assert test_param['sector_number'] == param['sector_number']


def test_synthesize():
    result_ord_lst_sq = {'Spd80mN_Synthesized': {'2016-02-11 00:00:00': 8.63072777777779,
                                                 '2016-02-21 00:00:00': 6.086831018518519,
                                                 '2016-03-02 00:00:00': 7.086503567981888,
                                                 '2016-03-12 00:00:00': 4.897541212360702,
                                                 '2016-03-22 00:00:00': 7.877298272421463,
                                                 '2016-04-01 00:00:00': 6.656276388888899,
                                                 '2016-04-11 00:00:00': 6.580104166666677,
                                                 '2016-04-21 00:00:00': 6.560243055555553,
                                                 '2016-05-01 00:00:00': 9.291759676888642,
                                                 '2016-05-11 00:00:00': 6.9470544917454085,
                                                 '2016-05-21 00:00:00': 4.797701636789135,
                                                 '2016-05-31 00:00:00': 3.7033950617284,
                                                 '2016-06-10 00:00:00': 4.693395138888888,
                                                 '2016-06-20 00:00:00': 6.504886805555568,
                                                 '2016-06-30 00:00:00': 7.931319444444444,
                                                 '2016-07-10 00:00:00': 7.283383011447513,
                                                 '2016-07-20 00:00:00': 5.323442197823473,
                                                 '2016-07-30 00:00:00': 8.539987847222225,
                                                 '2016-08-09 00:00:00': 6.592088888888887,
                                                 '2016-08-19 00:00:00': 6.016320833333339,
                                                 '2016-08-29 00:00:00': 8.502877314814816,
                                                 '2016-09-08 00:00:00': 8.54020632973431,
                                                 '2016-09-18 00:00:00': 8.501560364828546,
                                                 '2016-09-28 00:00:00': 7.811264880952389,
                                                 '2016-10-08 00:00:00': 6.903861805555553,
                                                 '2016-10-18 00:00:00': 6.173363194444445,
                                                 '2016-10-28 00:00:00': 5.325427083333329,
                                                 '2016-11-07 00:00:00': 7.738058265264794,
                                                 '2016-11-17 00:00:00': 5.2015049295114775,
                                                 '2016-11-27 00:00:00': 4.868479166666669,
                                                 '2016-12-07 00:00:00': 8.218102083333324,
                                                 '2016-12-17 00:00:00': 11.669419444444442,
                                                 '2016-12-27 00:00:00': 9.56760277777779,
                                                 '2017-01-06 00:00:00': 10.327056125191662,
                                                 '2017-01-16 00:00:00': 6.409178834121811,
                                                 '2017-01-26 00:00:00': 11.439151041666667,
                                                 '2017-02-05 00:00:00': 7.978196527777766,
                                                 '2017-02-15 00:00:00': 9.767215972222225,
                                                 '2017-02-25 00:00:00': 8.138878472222219,
                                                 '2017-03-07 00:00:00': 8.615134869947957,
                                                 '2017-03-17 00:00:00': 8.324815182745093,
                                                 '2017-03-27 00:00:00': 8.092620833333331,
                                                 '2017-04-06 00:00:00': 8.858566666666677,
                                                 '2017-04-16 00:00:00': 6.926072916666661,
                                                 '2017-04-26 00:00:00': 7.038437500000004,
                                                 '2017-05-06 00:00:00': 6.628228690007795,
                                                 '2017-05-16 00:00:00': 5.694735248634614,
                                                 '2017-05-26 00:00:00': 6.418529513888895,
                                                 '2017-06-05 00:00:00': 9.88184374999999,
                                                 '2017-06-15 00:00:00': 7.639711111111101,
                                                 '2017-06-25 00:00:00': 9.144633101851865,
                                                 '2017-07-05 00:00:00': np.NaN,
                                                 '2017-07-15 00:00:00': np.NaN,
                                                 '2017-07-25 00:00:00': 7.127115740740741,
                                                 '2017-08-04 00:00:00': 6.4158555555555505,
                                                 '2017-08-14 00:00:00': 7.510521527777763,
                                                 '2017-08-24 00:00:00': 5.943415798611104,
                                                 '2017-09-03 00:00:00': np.NaN,
                                                 '2017-09-13 00:00:00': np.NaN,
                                                 '2017-09-23 00:00:00': 12.601222222222226,
                                                 '2017-10-03 00:00:00': 9.425352777777784,
                                                 '2017-10-13 00:00:00': 9.466307638888878,
                                                 '2017-10-23 00:00:00': 8.84323971518988}}
    result_ord_lst_sq_dir = {'Spd80mN_Synthesized': {'2016-03-01': np.NaN,
                                                     '2016-04-01': 6.598875,
                                                     '2016-05-01': np.NaN,
                                                     '2016-06-01': 5.108156,
                                                     '2016-07-01': 6.319782,
                                                     '2016-08-01': 7.093956,
                                                     '2016-09-01': np.NaN,
                                                     '2016-10-01': 6.669446,
                                                     '2016-11-01': np.NaN,
                                                     '2016-12-01': 8.900778,
                                                     '2017-01-01': 9.501281,
                                                     '2017-02-01': 9.134509,
                                                     '2017-03-01': np.NaN,
                                                     '2017-04-01': 7.783390,
                                                     '2017-05-01': np.NaN,
                                                     '2017-06-01': 8.525249,
                                                     '2017-08-01': 6.715885,
                                                     '2017-10-01': 9.479016}}
    result_speed_sort = {'Spd80mN_Synthesized': {'2016-03-01': np.NaN,
                                                 '2016-04-01': 6.598875,
                                                 '2016-05-01': np.NaN,
                                                 '2016-06-01': 5.108156,
                                                 '2016-07-01': np.NaN,
                                                 '2016-08-01': 7.093956,
                                                 '2016-09-01': np.NaN,
                                                 '2016-10-01': 6.669446,
                                                 '2016-11-01': np.NaN,
                                                 '2016-12-01': 8.900778,
                                                 '2017-01-01': np.NaN,
                                                 '2017-02-01': 9.134509,
                                                 '2017-03-01': np.NaN},
                         'Dir78mS_Synthesized': {'2016-03-01': np.NaN,
                                                 '2016-04-01': 318.639591,
                                                 '2016-05-01': np.NaN,
                                                 '2016-06-01': 129.720897,
                                                 '2016-07-01': np.NaN,
                                                 '2016-08-01': 235.377543,
                                                 '2016-09-01': 223.965222,
                                                 '2016-10-01': 114.451746,
                                                 '2016-11-01': np.NaN,
                                                 '2016-12-01': 219.815444,
                                                 '2017-01-01': 237.667123,
                                                 '2017-02-01': 197.538174,
                                                 '2017-03-01': np.NaN}}

    data_spd80mn_even_months = DATA_CLND['Spd80mN'][DATA_CLND.index.month.isin([2, 4, 6, 8, 10, 12])]
    # Test the synthesise for when the target data starts before the reference data.
    correl = bw.Correl.OrdinaryLeastSquares(MERRA2_NE['WS50m_m/s']['2016-03-02 00:00:00':],
                                            data_spd80mn_even_months['2016-02-09 11:20:00':],
                                            averaging_prd='10D',
                                            coverage_threshold=1)
    correl.run(show_params=False)
    synth = correl.synthesize(target_coverage_threshold=0)

    for idx, row in pd.DataFrame(result_ord_lst_sq).iterrows():
        assert str(row[0]) == str(synth.loc[idx][0])

    # Test the synthesise for when the ref_dir is given as input.
    correl = bw.Correl.OrdinaryLeastSquares(MERRA2_NE['WS50m_m/s']['2016-03-02 00:00:00':],
                                            data_spd80mn_even_months['2016-02-09 11:20:00':],
                                            ref_dir=MERRA2_NE['WD50m_deg'], averaging_prd='1M',
                                            coverage_threshold=0.9, sectors=12)
    correl.run(show_params=False)
    synth = correl.synthesize()

    for idx, row in pd.DataFrame(result_ord_lst_sq_dir).iterrows():
        assert str(row[0]) == str(round(synth.loc[idx][0], 6))

    # Test the synthesise when SpeedSort correlation is used.
    correl = bw.Correl.SpeedSort(MERRA2_NE['WS50m_m/s']['2016-03-02 00:00:00':'2017-03-02 00:00:00'],
                                 MERRA2_NE['WD50m_deg']['2016-03-02 00:00:00':'2017-03-02 00:00:00'],
                                 DATA_CLND['Spd80mN'][DATA_CLND.index.month.isin([2, 4, 6, 8, 10, 12])],
                                 DATA_CLND['Dir78mS'][DATA_CLND.index.month.isin([2, 4, 6, 8, 10, 12])],
                                 averaging_prd='1M', coverage_threshold=0.9, sectors=12)
    correl.run(show_params=False)
    synth = correl.synthesize()

    for idx, row in pd.DataFrame(result_speed_sort).iterrows():
        print(idx)
        assert str(row[0]) == str(round(synth.loc[idx][0], 6))

    # Test the synthesise when SpeedSort correlation is used using 10 min averaging period.
    data_test = DATA_CLND[['Spd80mN', 'Spd60mN', 'Dir78mS', 'Dir58mS']].copy()
    data_test['Dir78mS']['2016-01-09 17:10:00':'2016-01-09 17:50:00'] = np.NaN
    data_test['Spd80mN']['2016-01-09 17:10:00':'2016-01-09 17:50:00'] = np.NaN
    data_test['Dir58mS']['2016-01-09 17:50:00':'2016-01-10 19:10:00'] = np.NaN
    data_test['Spd60mN']['2016-01-09 17:50:00':'2016-01-10 19:10:00'] = np.NaN
    ss_cor = bw.Correl.SpeedSort(data_test['Spd80mN'], data_test['Dir78mS'], data_test['Spd60mN'], data_test['Dir58mS'],
                                 averaging_prd='10min')
    ss_cor.run()
    data_synt = ss_cor.synthesize()
    assert (~data_synt['Dir58mS_Synthesized']['2016-01-09 18:00:00':'2016-01-09 19:10:00'].isnull()
            ).all() and data_test['Dir58mS']['2016-01-09 18:00:00':'2016-01-09 19:10:00'].isnull().all()
    assert (~data_synt['Spd60mN_Synthesized']['2016-01-09 18:00:00':'2016-01-09 19:10:00'].isnull()
            ).all() and data_test['Spd60mN']['2016-01-09 18:00:00':'2016-01-09 19:10:00'].isnull().all()
    assert ((data_synt['Spd60mN_Synthesized']['2016-01-09 18:00:00':'2016-01-09 18:30:00'] /
            [7.46707, 8.12027, 9.29721, 9.77779] - 1) < 1e-6).all()
    assert ((data_synt['Dir58mS_Synthesized']['2016-01-09 18:00:00':'2016-01-09 18:30:00'] /
            [116.34621159, 115.74488287, 120.15381789, 115.54469266] - 1) < 1e-6).all()
    assert (data_synt['Spd60mN_Synthesized']['2016-01-09 17:10:00':'2016-01-09 17:40:00']
            == data_test['Spd60mN']['2016-01-09 17:10:00':'2016-01-09 17:40:00']).all()
    assert (data_synt['Dir58mS_Synthesized']['2016-01-09 17:10:00':'2016-01-09 17:40:00']
            == data_test['Dir58mS']['2016-01-09 17:10:00':'2016-01-09 17:40:00']).all()

    # Test the synthesise when SpeedSort correlation is used and ref_dir and target_dir are the same
    ss_cor = bw.Correl.SpeedSort(data_test['Spd80mN'], data_test['Dir78mS'], data_test['Spd60mN'], data_test['Dir78mS'],
                                 averaging_prd='10min')
    ss_cor.run(show_params=False)
    data_synt = ss_cor.synthesize()
    assert (data_synt['Dir78mS_Synthesized'].dropna() == data_test['Dir78mS'].dropna()).all()
    assert ss_cor._ref_dir_col_name == 'Dir78mS_ref'
    assert ss_cor._tar_dir_col_name == 'Dir78mS'

    # Test the synthesise when SpeedSort correlation is used and ref_spd and target_spd are the same
    ss_cor = bw.Correl.SpeedSort(data_test['Spd80mN'], data_test['Dir78mS'], data_test['Spd80mN'], data_test['Dir58mS'],
                                 averaging_prd='10min')
    ss_cor.run(show_params=False)
    data_synt = ss_cor.synthesize()
    assert (data_synt['Spd80mN_Synthesized'].dropna() == data_test['Spd80mN'].dropna()).all()
    assert ss_cor._ref_spd_col_name == 'Spd80mN_ref'
    assert ss_cor._tar_spd_col_name == 'Spd80mN'


def test_orthogonal_least_squares():
    correl_monthly_results = {'slope': 1.01778, 'offset': -0.13473, 'r2': 0.8098, 'num_data_points': 18}
    correl_hourly_results = {'slope': 1.17829, 'offset': -1.48193, 'r2': 0.711, 'num_data_points': 12369}

    correl = bw.Correl.OrthogonalLeastSquares(MERRA2_NE['WS50m_m/s'], DATA_CLND['Spd80mN'], averaging_prd='1M',
                                              coverage_threshold=0)
    correl.run()
    assert round(correl.params['slope'], 3) == round(correl_monthly_results['slope'], 3)
    assert round(correl.params['offset'], 3) == round(correl_monthly_results['offset'], 3)
    assert round(correl.params['r2'], 4) == correl_monthly_results['r2']
    assert round(correl.params['num_data_points'], 5) == correl_monthly_results['num_data_points']

    # check hourly
    correl = bw.Correl.OrthogonalLeastSquares(MERRA2_NE['WS50m_m/s'], DATA_CLND['Spd80mN'], averaging_prd='1H',
                                              coverage_threshold=1)
    correl.run()
    assert round(correl.params['slope'], 5) == correl_hourly_results['slope']
    assert round(correl.params['offset'], 5) == correl_hourly_results['offset']
    assert round(correl.params['r2'], 4) == correl_hourly_results['r2']
    assert round(correl.params['num_data_points'], 5) == correl_hourly_results['num_data_points']

    # check aggregation method
    correl_aggregation_results = {'slope': 6.42434, 'offset': -12.8301, 'r2': 0.9255, 'num_data_points': 12445}
    correl = bw.Correl.OrthogonalLeastSquares(MERRA2_NE['T2M_degC'], DATA_CLND['T2m'],
                                              averaging_prd='1H', coverage_threshold=1,
                                              ref_aggregation_method='sum', target_aggregation_method='sum')
    correl.run()
    assert round(correl.params['slope'], 5) == correl_aggregation_results['slope']
    assert round(correl.params['offset'], 5) == correl_aggregation_results['offset']
    assert round(correl.params['r2'], 4) == correl_aggregation_results['r2']
    assert round(correl.params['num_data_points'], 5) == correl_aggregation_results['num_data_points']


def test_multiple_linear_regression():
    correl_monthly_results = {'slope': [2.03723, -0.93837], 'offset': -0.51343}
    correl = bw.Correl.MultipleLinearRegression([MERRA2_NE['WS50m_m/s'], MERRA2_NW['WS50m_m/s']],
                                                DATA_CLND['Spd80mN'], averaging_prd='1M',
                                                coverage_threshold=0.95)
    correl.run()
    for idx, slope in enumerate(correl.params['slope']):
        assert round(slope, 5) == correl_monthly_results['slope'][idx]
    assert round(correl.params['offset'], 5) == correl_monthly_results['offset']

    # check aggregation method
    correl_aggregation_results = {'slope': [5.51666, 0.54769], 'offset': -10.44818, 'num_data_pts': 12445}
    correl = bw.Correl.MultipleLinearRegression([MERRA2_NE['T2M_degC'], MERRA2_NW['T2M_degC']], DATA_CLND['T2m'],
                                                averaging_prd='1H', coverage_threshold=1,
                                                ref_aggregation_method='sum', target_aggregation_method='sum')
    correl.run()
    for idx, slope in enumerate(correl.params['slope']):
        assert round(slope, 5) == correl_aggregation_results['slope'][idx]
    assert round(correl.params['offset'], 5) == correl_aggregation_results['offset']
    assert correl.num_data_pts == correl_aggregation_results['num_data_pts']


def test_simple_speed_ratio():
    result = {'simple_speed_ratio': 0.99519,
              'ref_long_term_momm': 7.70707,
              'target_long_term': 7.67001,
              'target_overlap_coverage': 0.95812}
    ssr = bw.Correl.SimpleSpeedRatio(MERRA2_NE['WS50m_m/s'], DATA_CLND['Spd80mN'])
    ssr.run()
    assert round(ssr.params['simple_speed_ratio'], 5) == result['simple_speed_ratio']
    assert round(ssr.params['ref_long_term_momm'], 5) == result['ref_long_term_momm']
    assert round(ssr.params['target_long_term'], 5) == result['target_long_term']
    assert round(ssr.params['target_overlap_coverage'], 5) == result['target_overlap_coverage']

    # test monthly values
    result = {'simple_speed_ratio': 1.00127,
              'ref_long_term_momm': 7.70707,
              'target_long_term': 7.71684,
              'target_overlap_coverage': 1.0}
    ssr = bw.Correl.SimpleSpeedRatio(MERRA2_NE['WS50m_m/s'], bw.monthly_means(DATA_CLND['Spd80mN'],
                                                                              return_data=True)[1])
    ssr.run()
    assert round(ssr.params['simple_speed_ratio'], 5) == result['simple_speed_ratio']
    assert round(ssr.params['ref_long_term_momm'], 5) == result['ref_long_term_momm']
    assert round(ssr.params['target_long_term'], 5) == result['target_long_term']
    assert round(ssr.params['target_overlap_coverage'], 5) == result['target_overlap_coverage']

    # test with loads of data missing within the overlapping period which will throw a warning
    result = {'simple_speed_ratio': 1.01997,
              'ref_long_term_momm': 7.70707,
              'target_long_term': 7.86098,
              'target_overlap_coverage': 0.52323}
    with warnings.catch_warnings(record=True) as w:
        spd80m_even_months = DATA_CLND['Spd80mN'][DATA_CLND.index.month.isin([2, 4, 6, 8, 10, 12])]
        ssr = bw.Correl.SimpleSpeedRatio(MERRA2_NE['WS50m_m/s'], spd80m_even_months)
        ssr.run()
        assert round(ssr.params['simple_speed_ratio'], 5) == result['simple_speed_ratio']
        assert round(ssr.params['ref_long_term_momm'], 5) == result['ref_long_term_momm']
        assert round(ssr.params['target_long_term'], 5) == result['target_long_term']
        assert round(ssr.params['target_overlap_coverage'], 5) == result['target_overlap_coverage']
        assert UserWarning in [warning.category for warning in w]


def test_speed_sort():
    result = {
        1: {'average_veer': 11.08645,
            'num_pts_for_speed_fit': 464,
            'num_pts_for_veer': 317,
            'num_total_pts': 556,
            'offset': -4.15875,
            'slope': 1.59577,
            'target_speed_cutoff': 3.01367},
        2: {'average_veer': 5.37546,
            'num_pts_for_speed_fit': 236,
            'num_pts_for_veer': 157,
            'num_total_pts': 325,
            'offset': -0.12061,
            'slope': 1.102,
            'target_speed_cutoff': 3.825},
        3: {'average_veer': 2.02807,
            'num_pts_for_speed_fit': 621,
            'num_pts_for_veer': 455,
            'num_total_pts': 751,
            'offset': -0.84763,
            'slope': 1.01131,
            'target_speed_cutoff': 3.4115},
        4: {'average_veer': 8.7018,
            'num_pts_for_speed_fit': 739,
            'num_pts_for_veer': 429,
            'num_total_pts': 864,
            'offset': -3.37205,
            'slope': 1.32232,
            'target_speed_cutoff': 2.19117},
        5: {'average_veer': 12.80534,
            'num_pts_for_speed_fit': 599,
            'num_pts_for_veer': 336,
            'num_total_pts': 764,
            'offset': -3.67068,
            'slope': 1.46735,
            'target_speed_cutoff': 1.98667},
        6: {'average_veer': 21.84119,
            'num_pts_for_speed_fit': 685,
            'num_pts_for_veer': 460,
            'num_total_pts': 845,
            'offset': -2.34562,
            'slope': 1.16502,
            'target_speed_cutoff': 2.74767},
        7: {'average_veer': 12.00857,
            'num_pts_for_speed_fit': 1300,
            'num_pts_for_veer': 1144,
            'num_total_pts': 1435,
            'offset': -0.68645,
            'slope': 1.10294,
            'target_speed_cutoff': 3.88083},
        8: {'average_veer': 1.59793,
            'num_pts_for_speed_fit': 1400,
            'num_pts_for_veer': 1175,
            'num_total_pts': 1550,
            'offset': -0.02292,
            'slope': 1.02148,
            'target_speed_cutoff': 3.80183},
        9: {'average_veer': -2.56125,
            'num_pts_for_speed_fit': 1468,
            'num_pts_for_veer': 1239,
            'num_total_pts': 1617,
            'offset': -0.62316,
            'slope': 1.07478,
            'target_speed_cutoff': 3.55967},
        10: {'average_veer': -1.06416,
             'num_pts_for_speed_fit': 1714,
             'num_pts_for_veer': 1473,
             'num_total_pts': 1889,
             'offset': -1.68963,
             'slope': 1.2476,
             'target_speed_cutoff': 3.71267},
        11: {'average_veer': -3.56724,
             'num_pts_for_speed_fit': 1039,
             'num_pts_for_veer': 773,
             'num_total_pts': 1173,
             'offset': -3.26636,
             'slope': 1.41472,
             'target_speed_cutoff': 2.84083},
        12: {'average_veer': -1.6001,
             'num_pts_for_speed_fit': 514,
             'num_pts_for_veer': 323,
             'num_total_pts': 600,
             'offset': -3.31562,
             'slope': 1.37569,
             'target_speed_cutoff': 2.27833},
        'overall_average_veer': 3.73915,
        'ref_speed_cutoff': 3.81325,
        'ref_veer_cutoff': 4.91168,
        'target_veer_cutoff': 4.88203
    }
    ss_cor = bw.Correl.SpeedSort(MERRA2_NE['WS50m_m/s'], MERRA2_NE['WD50m_deg'],
                                 DATA_CLND['Spd80mN'], DATA_CLND['Dir78mS'],
                                 averaging_prd='1H')
    ss_cor.run()

    assert ss_cor.params['overall_average_veer'] == result['overall_average_veer']
    assert ss_cor.params['ref_speed_cutoff'] == result['ref_speed_cutoff']
    assert ss_cor.params['ref_veer_cutoff'] == result['ref_veer_cutoff']
    assert ss_cor.params['target_veer_cutoff'] == result['target_veer_cutoff']
    for key in result:
        if isinstance(key, int):
            assert ss_cor.params[key]['average_veer'] == result[key]['average_veer']
            assert ss_cor.params[key]['num_pts_for_speed_fit'] == result[key]['num_pts_for_speed_fit']
            assert ss_cor.params[key]['num_pts_for_veer'] == result[key]['num_pts_for_veer']
            # assert ss_cor.params[key]['num_total_pts'] == result[key]['num_total_pts'] # comes out different each time
            assert round(ss_cor.params[key]['offset'], 0) == round(result[key]['offset'], 0)  # comes out different
            assert round(ss_cor.params[key]['slope'], 1) == round(result[key]['slope'], 1)  # comes out different
            assert round(ss_cor.params[key]['target_speed_cutoff'], 0) == round(result[key]['target_speed_cutoff'], 0)
