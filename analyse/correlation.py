import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List, Dict
from transform.transform import _get_overlapping_data, _average_data_by_period, _filter_by_coverage_threshold, _common_idxs
from plot.plot import _scatter_plot
from scipy.odr import ODR, RealData, Model


def _preprocess_data_for_correlations(ref: pd.DataFrame, target:pd.DataFrame, averaging_prd, coverage_threshold):
    """A wrapper function that calls other functions necessary for pre-processing the data"""
    ref = ref.sort_index().dropna()
    target = target.sort_index().dropna()
    ref_overlap, target_overlap = _get_overlapping_data(ref, target, averaging_prd)
    ref_overlap_avgd = _average_data_by_period(ref_overlap, averaging_prd)
    target_overlap_avgd = _average_data_by_period(target_overlap, averaging_prd)
    ref_filtered_for_coverage = _filter_by_coverage_threshold(ref, ref_overlap_avgd, coverage_threshold)
    target_filtered_for_coverage = _filter_by_coverage_threshold(target, target_overlap_avgd, coverage_threshold)
    common_idxs, data_pts = _common_idxs(ref_filtered_for_coverage, target_filtered_for_coverage)
    return ref_filtered_for_coverage.drop(['Count','Coverage'], axis=1).loc[common_idxs], \
                    target_filtered_for_coverage.drop(['Count','Coverage'], axis=1).loc[common_idxs]


def linear_func(p, x):
    return p[0]*x +p[1]


class orthogonal_least_squares:
    def __init__(self, ref, target, averaging_prd, coverage_threshold, plot=False, function=Model(linear_func)):
        self.ref = ref
        self.target = target
        self.averaging_prd = averaging_prd
        self.coverage_threshold = coverage_threshold
        self.plot = plot
        self.function = function
        self.fit()

    def fit(self):
        self._ref_processed, self._target_processed = _preprocess_data_for_correlations(self.ref, self.target, self.averaging_prd,
                                                                            self.coverage_threshold)
        data = RealData(self._ref_processed.values.flatten(), self._target_processed.values.flatten())
        self._model = ODR(data, self.function, beta0=[1.0,0.0])
        self.out = self._model.run()
        self.params = self.out.beta

    def output(self):
        print(self.out.pprint())

    def show_params(self):
        print("Parameters:", self.params)

    def plot_model(self):
        self._prediction = [linear_func(self.params, i) for i in self._ref_processed.values.flatten()]
        _scatter_plot(self._ref_processed.values.flatten(), self._target_processed.values.flatten(), self._prediction)

    def synthesize(self):
        def linear_func_inverted(x, p):
            return linear_func(p,x)
        return self.ref.transform(linear_func_inverted, p=self.params)


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

    ref_processed, target_processed = _preprocess_data_for_correlations(ref, target, averaging_prd, coverage_threshold)
    common_idxs, data_pts = _common_idxs(ref_processed, target_processed)
    if plot:
        _scatter_plot(ref_processed.loc[common_idxs].values,target_processed.loc[common_idxs].values, predicted_y=0, x_label="Reference Data", y_label="Target Data  ")


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