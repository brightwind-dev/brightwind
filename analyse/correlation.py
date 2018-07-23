import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List, Dict
from transform.transform import _preprocess_data_for_correlations, _common_idxs
from plot.plot import _scatter_plot


def linear_regression(ref: pd.Series, target: pd.Series, averaging_prd: str, coverage_threshold: float, plot:bool=False):
    """Accepts two dataframes with timestamps as indexes and averaging period.
    :param: ref_speed : Dataframe containing reference speed as a column, timestamp as the index.
    :param: target_speed: Dataframe containing target speed as a column, timestamp as the index.
    :averaging_prd: Groups data by the period specified by period.
        2T, 2 min for minutely average
        Set period to 1D for a daily average, 3D for three hourly average, similarly 5D, 7D, 15D etc.
        Set period to 1H for hourly average, 3H for three hourly average and so on for 5H, 6H etc.
        week
        Set period to 1MS for monthly average
        Set period to 1AS fo annual average
    :return :A dictionary containing the following keys, r2, slope, offset and num_data_points
    """
    ref = ref.sort_index().dropna()
    target = target.sort_index().dropna()

    ref_processed, target_processed = _preprocess_data_for_correlations(ref, target, averaging_prd, coverage_threshold)
    common_idxs, data_pts = _common_idxs(ref_processed, target_processed)
    if plot:
        _scatter_plot(ref_processed.loc[common_idxs].values,target_processed.loc[common_idxs].values, "Reference Data", "Target Data  ")

    # Linear Regression
    model = LinearRegression()
    model.fit(ref_processed.loc[common_idxs].values,
              target_processed.loc[common_idxs].values)
    r2 = model.score(ref_processed.loc[common_idxs].values,
                     target_processed.loc[common_idxs].values)
    prediction = model.predict(ref_processed.loc[common_idxs].values)
    slope = model.coef_
    offset = model.intercept_
    rmse = mean_squared_error(prediction,target_processed.loc[common_idxs].values) **0.5
    mae = mean_absolute_error(prediction,target_processed.loc[common_idxs].values)
    # lt_ref_speed = mean_of_monthly_means(ref_speed).mean()
    # predicted_lt_speed  = calc_target_value_by_linear_model(lt_ref_speed, slope[0], offset[0])
    return {'num_data_points': data_pts, 'slope': slope, 'offset': offset, 'r2': r2, 'RMSE':rmse, 'MAE':mae}