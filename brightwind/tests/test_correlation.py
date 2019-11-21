import pytest
import brightwind as bw
import pandas as pd
import numpy as np
import datetime


data = bw.datasets.demo_data


def test_CorrelBase():
    data = bw.load_csv(bw.datasets.demo_data)
    ref_spd = data['Spd80mN']
    target_spd = data['Spd80mS']
    ref_dir = data['Dir78mS']
    target_dir = data['Dir78mS']

    # test with no directions
    CB = bw.Correl.CorrelBase(ref_spd=ref_spd, target_spd=target_spd, averaging_prd='10min', coverage_threshold=.9)

    # test attributes
    assert CB.num_data_pts == 95629
    assert CB.coverage_threshold == .9
    assert CB.ref_spd.all() == ref_spd.all()
    assert CB.target_spd.all() == target_spd.all()
    assert CB.get_error_metrics() == 0
    assert CB.data['ref_spd'][0] == ref_spd[0]
    assert CB.data['target_spd'][0] == target_spd[0]

    # test with directions
    CB_directions = bw.Correl.CorrelBase(ref_spd=ref_spd, target_spd=target_spd, averaging_prd='10min',
                                         coverage_threshold=.9,
                                         ref_dir=ref_dir, target_dir=target_dir)

    # test with preprocess=False and directions
    CB_directions_preprocess_false = bw.Correl.CorrelBase(ref_spd=ref_spd, target_spd=target_spd, averaging_prd='10min',
                                                          coverage_threshold=.9,
                                                          ref_dir=ref_dir, target_dir=target_dir, preprocess=False)

    # test attributes
    assert CB.num_data_pts == 95629
    assert CB.coverage_threshold == .9
    assert CB.ref_spd.all() == ref_spd.all()
    assert CB.target_spd.all() == target_spd.all()
    assert CB_directions.ref_dir.all() == ref_spd.all()
    assert CB_directions.target_dir.all() == target_dir.all()
    assert CB.data['ref_dir'][0] == ref_dir[0]
    assert CB.data['target_dir'][0] == target_dir[0]

    # test error metrics
    assert CB.get_error_metrics() == 0


def test_OrdinaryLeastSquares():
    data = bw.load_csv(bw.datasets.demo_data)
    ref_spd = data['Spd80mN']
    target_spd = data['Spd80mS']

    OLS = bw.Correl.OrdinaryLeastSquares(ref_spd=ref_spd, target_spd=target_spd, averaging_prd='10min',
                                         coverage_threshold=.9)

    # test with preprocess=False
    OLS_preprocess_false = bw.Correl.OrdinaryLeastSquares(ref_spd=ref_spd, target_spd=target_spd, averaging_prd='10min',
                                                          coverage_threshold=.9, preprocess=False)
    OLS.run()

    # test params
    assert OLS.params['slope'] == 0.8471316221748566
    assert OLS.params['offset'] == 0.1219418144494032
    assert OLS.params['r2'] == 0.5773700175122298
    assert OLS.params['Num data points'] == 95629

    # test attributes
    assert OLS.num_data_pts == 95629
    assert OLS.coverage_threshold == .9
    assert OLS.ref_spd.all() == ref_spd.all()
    assert OLS.target_spd.all() == target_spd.all()
    assert OLS.data['ref_spd'][0] == ref_spd[0]
    assert OLS.data['target_spd'][0] == target_spd[0]

    # test error metrics
    assert OLS.get_error_metrics() == 0

    # test linear function
    p = list()
    p.append(OLS.params['slope'])
    p.append(OLS.params['offset'])
    linear_func_OLS = OLS.linear_func(p, target_spd)
    assert linear_func_OLS[0] == 6.823600077474694

    # test get_r2()
    assert OLS.get_r2() ==  OLS.params['r2']


def test_OrthogonalLeastSquares():
    data = bw.load_csv(bw.datasets.demo_data)
    ref_spd = data['Spd80mN']
    target_spd = data['Spd80mS']

    ORGLS = bw.Correl.OrthogonalLeastSquares(ref_spd=ref_spd, target_spd=target_spd, averaging_prd='10min',
                                             coverage_threshold=.9, preprocess=True)
    ORGLS_preprocess_false = bw.Correl.OrthogonalLeastSquares(ref_spd=ref_spd, target_spd=target_spd,
                                                              averaging_prd='10min',
                                                              coverage_threshold=.9, preprocess=False)
    ORGLS.run()

    #test params
    assert ORGLS.params['slope'] == 1.1535970704590985
    assert ORGLS.params['offset'] == -2.1761398200777227
    assert ORGLS.params['r2'] == 0.5018059240561804
    assert ORGLS.params['Num data points'] == 95629

    # test attributes
    assert ORGLS.num_data_pts == 95629
    assert ORGLS.coverage_threshold == .9
    assert ORGLS.ref_spd.all() == ref_spd.all()
    assert ORGLS.target_spd.all() == target_spd.all()
    assert ORGLS.data['ref_spd'][0] == ref_spd[0]
    assert ORGLS.data['target_spd'][0] == target_spd[0]

    # test get_error_metrics()
    assert ORGLS.get_error_metrics() == 0

    # test linear function -  BUG
    # p = list()
    # p.append(ORGLS.params['slope'])
    # p.append(ORGLS.params['offset'])
    # linear_func_ORGLS = ORGLS.linear_func(p, target_spd)
    # assert linear_func_ORGLS[0] == 6.823600077474694


def test_MultipleLinearRegression():
    data = bw.load_csv(bw.datasets.demo_data)
    ref_spd = [data['Spd80mN'], data['Spd40mN']]
    target_spd = data['Spd80mS']

    # test basic
    MLR = bw.Correl.MultipleLinearRegression(ref_spd=ref_spd, target_spd=target_spd)
    MLR.run()

    # test show_params
    MLR.show_params()
    assert MLR.params['offset'] == 0.1335730899262883
    assert MLR.params['slope'][0] == 0.808772620245024
    assert MLR.params['slope'][1] == 0.04092208540564364

    # test attributes
    assert MLR.coverage_threshold == .9
    assert MLR.data['ref_spd_1'].all() == ref_spd[0].all()
    assert MLR.data['ref_spd_2'].all() == ref_spd[1].all()
    assert MLR.data['target_spd'].all() == target_spd.all()
    assert MLR.ref_spd.all() == pd.concat([ref_spd[0], ref_spd[1]], axis=1).all()
    assert MLR.target_spd.all() == target_spd.all()

    # test get_error_metrics()
    assert MLR.get_error_metrics() == 0

    # test get_r2()
    assert MLR.get_r2() == 0.5685111334796743

    # test synthesis()
    assert len(MLR.synthesize()) == 15940
    assert MLR.synthesize()[0] == 7.177942188251347

    # test plot
    assert MLR.plot() == "Cannot plot Multiple Linear Regression"


def test_SimpleSpeedRatio():
    data = bw.load_csv(bw.datasets.demo_data)
    ref_spd = data['Spd80mN']
    target_spd = data['Spd80mS']

    # test basic
    SSR = bw.Correl.SimpleSpeedRatio(ref_spd=ref_spd,target_spd=target_spd)
    SSR.run()

    # test show_params()
    SSR.show_params()
    assert SSR.params['ratio'] == 0.8633934258290723

    # test attributes
    assert SSR.num_data_pts == 98469
    assert SSR.data['ref_spd'][0] == ref_spd[0]
    assert SSR.data['target_spd'][SSR.num_data_pts-1] == target_spd.tail(1).values[0]


    # test syntheesise()
    assert len(SSR.synthesize()) == 98469
    assert SSR.synthesize().iloc[0, 0] == 7.9110000000000005

    # test get_error_metrics(
    assert SSR.get_error_metrics() == 0

    # test plot
    SSR.plot()


def test_SpeedSort():
    data = bw.load_csv(bw.datasets.demo_data)
    ref_spd = data['Spd80mN']
    target_spd = data['Spd80mS']
    target_dir = data['Dir78mS']
    ref_dir = data['Dir78mS']

    # test_basic
    SS_1AS = bw.Correl.SpeedSort(ref_spd=ref_spd, target_spd=target_spd, ref_dir=ref_dir, target_dir=target_dir,
                                 averaging_prd='1AS')
    SS_1M = bw.Correl.SpeedSort(ref_spd=ref_spd, target_spd=target_spd, ref_dir=ref_dir, target_dir=target_dir,
                                averaging_prd='1M')
    SS_1AS.run()
    SS_1M.run()

    # test params()
    SS_1AS.show_params()
    SS_1M.show_params()
    assert SS_1AS.params['Ref_cutoff_for_speed'] == 3.6607786770604434
    assert SS_1AS.params[9]['target_cutoff'] == 7.271502457886856
    assert SS_1M.params['Ref_cutoff_for_speed'] == 3.712444306380053
    assert SS_1M.params[9]['target_cutoff'] == 6.316606182795701

    # test attributes
    assert SS_1AS.num_data_pts == 1
    assert SS_1M.num_data_pts == 20
    assert SS_1AS.coverage_threshold == .9
    assert SS_1M.coverage_threshold == .9
    assert SS_1AS.lt_ref_speed == 7.321557354120887
    assert SS_1M.lt_ref_speed == 7.424888612760106
    assert SS_1AS.averaging_prd =='1AS'
    assert SS_1M.averaging_prd == '1M'
    assert SS_1M.ref_spd.all() == ref_spd.all()
    assert SS_1AS.ref_spd.all() == ref_spd.all()
    assert SS_1M.target_spd.all( )== target_spd.all()
    assert SS_1AS.target_spd.all() == target_spd.all()
    assert SS_1M.ref_dir.all() == ref_dir.all()
    assert SS_1AS.ref_dir.all() == ref_dir.all()
    assert SS_1M.target_dir.all() == target_dir.all()
    assert SS_1AS.target_dir.all() == target_dir.all()

    # test get_error_metrics()
    assert SS_1AS.get_error_metrics() == 0
    assert SS_1M.get_error_metrics() == 0

    # test get_result_table
    SS_1AS.get_result_table()
    SS_1M.get_result_table()

    # test synthesise with 10 min averaging period
    SS_10min = bw.Correl.SpeedSort(ref_spd=ref_spd, target_spd=target_spd, ref_dir=ref_dir, target_dir=target_dir,
                                   averaging_prd='10min')
    SS_10min.run()
    SS_10min_synthesised = SS_10min.synthesize(input_spd=target_spd, input_dir=target_dir)
    assert len(SS_10min_synthesised) == 95629

    # check that if wind speed value is 0 for a given timestamp, its synthesised speed is 0
    assert SS_10min_synthesised.iloc[(len(SS_10min_synthesised)-1), 0] == target_spd.iloc[(len(target_spd)-1)]

    # test plots
    SS_10min.plot()
    SS_10min.plot_wind_vane()
    SS_1AS.plot()
    SS_1AS.plot_wind_vane()

    # test SectorSpeedModel
    SSM = bw.Correl.SpeedSort.SectorSpeedModel(ref_spd=ref_spd,target_spd=target_spd,lt_ref_speed=bw.momm(ref_spd))

    # test param()
    assert SSM.params['slope'] == 1.0913560550604777
    assert SSM.params['offset'] == -1.6183937026706863

    # test attributes
    assert SSM.target_cutoff == 2.026
    assert SSM.data_pts == 77835
    assert SSM.cutoff == 3.7782940972797765
    assert SSM.sector_ref.all() == ref_spd.all()
    assert SSM.sector_target.all() == target_spd.all()

    # test predict()
    assert SSM.sector_predict(ref_spd)[0] == 7.516256478185512


def test_SVR():
    import numpy as np
    data = bw.load_csv(bw.datasets.demo_data)
    ref_spd = data['Spd80mN'][:100]
    target_spd = data['Spd80mS'][:100]

    SVR_model_0 = bw.Correl.SVR(ref_spd=ref_spd, target_spd=target_spd, averaging_prd='10min', coverage_threshold=.9)
    SVR_model_1 = bw.Correl.SVR(ref_spd=ref_spd, target_spd=target_spd, averaging_prd='10min', bw_model=1,
                                coverage_threshold=.9)
    SVR_model_0.run()
    SVR_model_1.run()

    # test params()
    assert SVR_model_0.params['RMSE'].all() == np.array([0.76246471, 0.28490021, 1.94454812]).all()
    assert SVR_model_0.params['MAE'].all() == np.array([0.82772912, 0.43492475, 1.29813051]).all()
    assert SVR_model_0.params['Explained Variance'].all() == np.array([-0.98498249, -0.94907944, -0.94004678]).all()
    assert SVR_model_1.params['RMSE'].all() == np.array([-0.9863871 , -0.90654105, -0.94750595]).all()
    assert SVR_model_1.params['MAE'].all() == np.array([1.00416624, 0.46770968, 1.3029321]).all()
    assert SVR_model_1.params['Explained Variance'].all() == np.array([1.07844633, 0.34123735, 1.92476305]).all()

    # test attributes
    assert SVR_model_0.num_data_pts == 100
    assert SVR_model_0.data.all() == pd.concat([ref_spd, target_spd], axis=1).all()
    assert SVR_model_1.num_data_pts == 100
    assert SVR_model_1.data.all() == pd.concat([ref_spd, target_spd], axis=1).all()

    # test plot
    assert SVR_model_0.plot()
    assert SVR_model_1.plot()

