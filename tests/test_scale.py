import pytest
import brightwind as bw
import pandas as pd
import numpy as np

DATA = bw.load_csv(bw.demo_datasets.demo_data)
DATA = bw.apply_cleaning(DATA, bw.demo_datasets.demo_cleaning_file)


def test_scale_air_pressure_to_height():
    assert round(bw.scale_air_pressure_to_height(ref_air_pressure_hPa=1000, ref_air_temp_degC=12,
                                                 ref_height_m=10, target_height_m=200), 2) == 977.45
    assert round(bw.scale_air_pressure_to_height(ref_air_pressure_hPa=1000, ref_air_temp_degC=12,
                                                 ref_height_m=10, target_height_m=15), 2) == 999.4
    pd.testing.assert_series_equal(bw.scale_air_pressure_to_height(
        ref_air_pressure_hPa=DATA['P2m'].loc['2016-01-09 17:10':'2016-01-09 18:00'],
        ref_air_temp_degC=DATA['T2m'].loc['2016-01-09 17:10':'2016-01-09 18:00'],
        ref_height_m=2, target_height_m=10),
        pd.Series(data=[933.07, 933.07, 933.07, 932.07, 932.07, 932.07],
                  index=pd.to_datetime(['2016-01-09 17:10:00', '2016-01-09 17:20:00', '2016-01-09 17:30:00',
                                        '2016-01-09 17:40:00', '2016-01-09 17:50:00', '2016-01-09 18:00:00'])),
        check_names=False)

    # test error raising for invalid input types
    with pytest.raises(TypeError):
        bw.scale_air_pressure_to_height(
            ref_air_pressure_hPa='invalid_type',
            ref_air_temp_degC=12,
            ref_height_m=2,
            target_height_m=10)

    # test error raising for invalid slope type
    with pytest.raises(TypeError):
        bw.scale_air_pressure_to_height(
            ref_air_pressure_hPa=1000,
            ref_air_temp_degC=12,
            ref_height_m=2,
            target_height_m='invalid_type')

    # test error raising for mismatched dimensions with Series
    with pytest.raises(ValueError):
        bw.scale_air_pressure_to_height(
            ref_air_pressure_hPa=pd.Series([1, 2, 3]),
            ref_air_temp_degC=pd.Series([1, 2]),
            ref_height_m=2,
            target_height_m=10)


def test_scale_air_density_to_height():
    assert bw.scale_air_density_to_height(ref_air_density_kg_m3=1.224, ref_height_m=80, target_height_m=100) == 1.22174

    assert bw.scale_air_density_to_height(ref_air_density_kg_m3=1.224,
                                          ref_height_m=80,
                                          target_height_m=100,
                                          lapse_rate_kg_m3_m=-0.0002) == 1.22

    test_density = bw.calc_air_density(DATA.T2m, DATA.P2m)
    pd.testing.assert_series_equal(bw.scale_air_density_to_height(
        ref_air_density_kg_m3=test_density.loc['2016-01-09 17:10':'2016-01-09 18:00'],
        ref_height_m=2, target_height_m=10),
        pd.Series(data=[1.186780, 1.187175, 1.187747, 1.185950, 1.186301, 1.185686],
                  index=pd.to_datetime(['2016-01-09 17:10:00', '2016-01-09 17:20:00', '2016-01-09 17:30:00',
                                        '2016-01-09 17:40:00', '2016-01-09 17:50:00', '2016-01-09 18:00:00'])),
        check_names=False)


def test_scale_air_temperature_to_height():
    assert bw.scale_air_temperature_to_height(10.0065, 10, 11) == 10.0

    assert bw.scale_air_temperature_to_height(ref_air_temperature=10,
                                              ref_height_m=12,
                                              target_height_m=10,
                                              lapse_rate_deg_m=-0.001) == 10.002

    pd.testing.assert_series_equal(
        bw.scale_air_temperature_to_height(
            ref_air_temperature=DATA.T2m.loc['2016-01-09 17:10':'2016-01-09 18:00'],
            ref_height_m=2,
            target_height_m=20,
            lapse_rate_deg_m=-0.001),
        pd.Series(data=[0.936, 0.845, 0.713, 0.834, 0.753, 0.895],
                  index=pd.to_datetime(['2016-01-09 17:10:00', '2016-01-09 17:20:00', '2016-01-09 17:30:00',
                                        '2016-01-09 17:40:00', '2016-01-09 17:50:00', '2016-01-09 18:00:00'])),
        check_names=False)

    pd.testing.assert_series_equal(
        bw.scale_air_temperature_to_height(
            ref_air_temperature=DATA.T2m.loc['2016-01-09 17:10':'2016-01-09 18:00'],
            ref_height_m=2,
            target_height_m=20),
        pd.Series(data=[0.837, 0.746, 0.614, 0.735, 0.654, 0.796],
                  index=pd.to_datetime(['2016-01-09 17:10:00', '2016-01-09 17:20:00', '2016-01-09 17:30:00',
                                        '2016-01-09 17:40:00', '2016-01-09 17:50:00', '2016-01-09 18:00:00'])),
        check_names=False)


def test_linear_transform():
    # test with float and int inputs
    assert bw.linear_transform(x_target=10, x_ref=5, y_ref=10, slope=-0.5) == 7.5


# test with array input for y_ref
assert (bw.linear_transform(
    x_target=20,
    x_ref=2,
    y_ref=DATA.T2m.loc['2016-01-09 17:10':'2016-01-09 18:00'].values,
    slope=-0.0065) == np.array([0.837, 0.746, 0.614, 0.735, 0.654, 0.796])).all()

# test with pandas Series input for y_ref
pd.testing.assert_series_equal(
    bw.linear_transform(
        x_target=DATA.T2m.loc['2016-01-09 17:10':'2016-01-09 18:00'],
        x_ref=2,
        y_ref=20,
        slope=-0.0065).round(3),
    pd.Series(data=[20.007, 20.007, 20.008, 20.007, 20.008, 20.007],
              index=pd.to_datetime(['2016-01-09 17:10:00', '2016-01-09 17:20:00', '2016-01-09 17:30:00',
                                    '2016-01-09 17:40:00', '2016-01-09 17:50:00', '2016-01-09 18:00:00'])),
    check_names=False)

# test error raising for invalid input types
with pytest.raises(TypeError):
    bw.linear_transform(
        x_target='invalid_type',
        x_ref=2,
        y_ref=20,
        slope=-0.0065)

# test error raising for invalid slope type
with pytest.raises(TypeError):
    bw.linear_transform(
        x_target=10,
        x_ref=2,
        y_ref=20,
        slope='invalid_type')

# test error raising for mismatched dimensions
with pytest.raises(ValueError):
    bw.linear_transform(
        x_target=np.array([1, 2, 3]),
        x_ref=np.array([1, 2]),
        y_ref=20,
        slope=-0.5)

# test error raising for mismatched dimensions with Series
with pytest.raises(ValueError):
    bw.linear_transform(
        x_target=pd.Series([1, 2, 3]),
        x_ref=2,
        y_ref=pd.Series([1, 2]),
        slope=-0.5)


def test_apply_scale_factor():
    assert bw.apply_scale_factor(3, 0.5) == 1.5
    assert (bw.apply_scale_factor(np.array([0, 1, 2]), 0.5) == [0, 0.5, 1]).all()
    assert (bw.apply_scale_factor(pd.Series([10, 20, 30, 40]), -10) == [-100, -200, -300, -400]).all()
    df = pd.DataFrame({'a': [0.5, 1.2], 'b': [3, 4], 'c': ['a', 'b']})
    result_df = pd.DataFrame({'a': [1.0, 2.4], 'b': [6, 8], 'c': ['a', 'b']})
    assert result_df.equals(bw.apply_scale_factor(df, 2))
