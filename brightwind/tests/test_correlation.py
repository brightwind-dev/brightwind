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
    result = bw.Correl.CorrelBase(ref_spd=ref_spd, target_spd=target_spd, averaging_prd='10min', coverage_threshold=.9)
    assert result.coverage_threshold == .9
    assert result.ref_spd.all() == ref_spd.all()
    assert result.target_spd.all() == target_spd.all()
    assert result.get_error_metrics() == 0

    # test with directions
    result = bw.Correl.CorrelBase(ref_spd=ref_spd, target_spd=target_spd, averaging_prd='10min', coverage_threshold=.9,
                                  ref_dir=ref_dir, target_dir=target_dir)

    # test with preprocess=False and no directions
    result = bw.Correl.CorrelBase(ref_spd=ref_spd, target_spd=target_spd, averaging_prd='10min', coverage_threshold=.9,
                                  ref_dir=ref_dir, target_dir=target_dir, preprocess=False)
    result = bw.Correl.CorrelBase(ref_spd=ref_spd, target_spd=target_spd, averaging_prd='10min', coverage_threshold=.9,
                                  ref_dir=ref_dir, target_dir=target_dir, preprocess=False)


def test_OrdinaryLeastSquares():
    data = bw.load_csv(bw.datasets.demo_data)
    ref_spd = data['Spd80mN']
    target_spd = data['Spd80mS']

    result = bw.Correl.OrdinaryLeastSquares(ref_spd=ref_spd, target_spd=target_spd, averaging_prd='10min',
                                            coverage_threshold=.9)
    result.run()
    # test params
    assert result.params['slope'] == 0.8471316221748566
    assert result.params['offset'] == 0.1219418144494032
    assert result.params['r2'] == 0.5773700175122298
    assert result.params['Num data points'] == 95629

    # test linear function
    p = list()
    p.append(result.params['slope'])
    p.append(result.params['offset'])
    linear_func_result = result.linear_func(p, target_spd)
    assert linear_func_result[0] == 6.823600077474694

    # test get_r2()
    assert result.get_r2() ==  result.params['r2']

    # test with preprocess=False
    result = bw.Correl.OrdinaryLeastSquares(ref_spd=ref_spd, target_spd=target_spd, averaging_prd='10min',
                                            coverage_threshold=.9, preprocess=True)


def test_OrthogonalLeastSquares():
    data = bw.load_csv(bw.datasets.demo_data)
    ref_spd = data['Spd80mN']
    target_spd = data['Spd80mS']

    result = bw.Correl.OrthogonalLeastSquares(ref_spd=ref_spd, target_spd=target_spd, averaging_prd='10min',
                                              coverage_threshold=.9, preprocess=True)
    result.run()
    #test params
    assert result.params['slope'] == 1.1535970704590985
    assert result.params['offset'] == -2.1761398200777227
    assert result.params['r2'] == 0.5018059240561804
    assert result.params['Num data points'] == 95629

    # p = list()
    # p.append(result.params['slope'])
    # p.append(result.params['offset'])
    # linear_func_result = result.linear_func(p, target_spd)
    # assert linear_func_result[0] == 6.823600077474694


def test_MultipleLinearRegression():
    data = bw.load_csv(bw.datasets.demo_data)
    ref_spd = [data['Spd80mN'], data['Spd40mN']]
    target_spd = data['Spd80mS']

    # test basic
    MLR = bw.Correl.MultipleLinearRegression(ref_spd=ref_spd, target_spd=target_spd)
    MLR.run()

    #test show_params(
    MLR.show_params()

    assert MLR.params['offset'] == 0.1335730899262883
    assert MLR.params['slope'][0] == 0.808772620245024
    assert MLR.params['slope'][1] == 0.04092208540564364

    # test get_r2()
    assert MLR.get_r2() == 0.5685111334796743

    # test synthesis()
    assert len(MLR.synthesize()) == 15940
    assert MLR.synthesize()[0] == 7.177942188251347

    # test plot
    assert MLR.plot() == "Cannot plot Multiple Linear Regression"