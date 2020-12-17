import pytest
import brightwind as bw
import pandas as pd
import numpy as np
import datetime


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
MERRA2 = bw.load_csv(bw.demo_datasets.demo_merra2_NE)


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

    correl = bw.Correl.OrdinaryLeastSquares(MERRA2['WS50m_m/s'], DATA_CLND['Spd80mN'], averaging_prd='1M',
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
    correl = bw.Correl.OrdinaryLeastSquares(MERRA2['WS50m_m/s'], DATA_CLND['Spd80mN'], averaging_prd='1M',
                                            coverage_threshold=0.9)
    correl.run()
    assert round(correl.params['slope'], 5) == correl_monthly_results_90['slope']
    assert round(correl.params['offset'], 5) == correl_monthly_results_90['offset']
    assert round(correl.params['r2'], 4) == correl_monthly_results_90['r2']
    assert round(correl.params['num_data_points'], 5) == correl_monthly_results_90['num_data_points']

    # check hourly, checked against Excel
    correl = bw.Correl.OrdinaryLeastSquares(MERRA2['WS50m_m/s'], DATA_CLND['Spd80mN'], averaging_prd='1H',
                                            coverage_threshold=1)
    correl.run()
    assert round(correl.params['slope'], 5) == correl_hourly_results['slope']
    assert round(correl.params['offset'], 5) == correl_hourly_results['offset']
    assert round(correl.params['r2'], 4) == correl_hourly_results['r2']
    assert round(correl.params['num_data_points'], 5) == correl_hourly_results['num_data_points']
