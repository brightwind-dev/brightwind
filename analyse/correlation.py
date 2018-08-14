import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List, Dict
from transform.transform import calc_lt_ref_speed, _get_overlapping_data, _average_data_by_period, \
    _filter_by_coverage_threshold, _common_idxs
from plot.plot import _scatter_plot
from scipy.odr import ODR, RealData, Model
from scipy.linalg import lstsq
from .frequency_analysis import get_binned_direction_series


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


def _compute_wind_vector(wspd, wdir):
    """Returns north and east component of wind-vector"""
    return wspd*np.cos(wdir), wspd*np.sin(wdir)


def _range_0_to_360(dir):
    if dir < 0:
        return dir+360
    elif dir > 360:
        return dir % 360
    else:
        return dir


def degree_to_radian(degree):
    return (degree /180.0) * np.pi


def radian_to_degree(radian):
    return (radian/np.pi) * 180.0


def _preprocess_dir_data_for_correlations(ref_spd: pd.DataFrame, ref_dir: pd.DataFrame, target_spd:pd.DataFrame,
                                          target_dir: pd.DataFrame, averaging_prd, coverage_threshold):
    ref_spd = ref_spd.sort_index().dropna()
    target_spd = target_spd.sort_index().dropna()
    ref_overlap, target_overlap = _get_overlapping_data(ref_spd, target_spd, averaging_prd)
    ref = pd.concat([ref_overlap, ref_dir.apply(degree_to_radian)],axis=1, join='inner')
    ref.columns = ['ref_spd','ref_dir']
    ref['N'], ref['E'] = _compute_wind_vector(ref['ref_spd'],ref['ref_dir'])
    target = pd.concat([target_overlap, target_dir.apply(degree_to_radian)],axis=1, join='inner')
    target.columns = ['target_spd', 'target_dir']
    target['N'], target['E'] = _compute_wind_vector(target['target_spd'], target['target_dir'])
    ref_N_avgd = _average_data_by_period(ref['N'], averaging_prd)
    ref_E_avgd = _average_data_by_period(ref['E'], averaging_prd)
    ref_dir_avgd = np.arctan2(ref_E_avgd, ref_N_avgd).applymap(radian_to_degree).applymap(_range_0_to_360)
    target_N_avgd = _average_data_by_period(target['N'], averaging_prd)
    target_E_avgd = _average_data_by_period(target['E'], averaging_prd)
    target_dir_avgd = np.arctan2(target_E_avgd, target_N_avgd).applymap(radian_to_degree).applymap(_range_0_to_360)
    ref_dir_filtered_for_coverage = _filter_by_coverage_threshold(ref_dir, ref_dir_avgd, coverage_threshold)
    target_dir_filtered_for_coverage = _filter_by_coverage_threshold(target_dir, target_dir_avgd, coverage_threshold)
    common_idxs, data_pts = _common_idxs(ref_dir_filtered_for_coverage, target_dir_filtered_for_coverage)
    return ref_dir_filtered_for_coverage.drop(['Count','Coverage'], axis=1).loc[common_idxs], \
                    target_dir_filtered_for_coverage.drop(['Count','Coverage'], axis=1).loc[common_idxs]


def linear_func(p, x):
    return p[0] * x + p[1]


class orthogonal_least_squares():
    """Accepts two dataframes with timestamps as indexes and averaging period.
    :param ref_speed : Dataframe containing reference speed as a column, timestamp as the index.
    :param target_speed: Dataframe containing target speed as a column, timestamp as the index.
    :param averaging_prd: Groups data by the period specified by period.
        2T, 2 min for minutely average
        Set period to 1D for a daily average, 3D for three hourly average, similarly 5D, 7D, 15D etc.
        Set period to 1H for hourly average, 3H for three hourly average and so on for 5H, 6H etc.
        week
        Set period to 1MS for monthly average
        Set period to 1AS fo annual average
    :return An object representing orthogonals least squares fit model
    """

    def __init__(self, ref, target, averaging_prd, coverage_threshold, function=Model(linear_func)):
        self.ref = ref
        self.target = target
        self.averaging_prd = averaging_prd
        self.coverage_threshold = coverage_threshold
        self.function = function
        self.fit()

    def fit(self):
        self._ref_processed, self._target_processed = _preprocess_data_for_correlations(self.ref, self.target,
                                                                        self.averaging_prd, self.coverage_threshold)
        data = RealData(self._ref_processed.values.flatten(), self._target_processed.values.flatten())
        self._model = ODR(data, self.function, beta0=[1.0,0.0])
        self.out = self._model.run()
        self.params = self.out.beta
        print(self.out.pprint())

    def show_params(self):
        print("Parameters:", self.params)

    def _predict(self, x):
        def linear_func_inverted(x, p):
            return linear_func(p, x)
        return x.transform(linear_func_inverted, p=self.params)

    def plot_model(self):
        _scatter_plot(self._ref_processed.values.flatten(), self._target_processed.values.flatten(),
                      self._predict(self._ref_processed).values.flatten())

    def synthesize(self, data=None):
        if data is None:
            data = self.ref
        return self._predict(data)


class oridnary_least_squares:
    """Accepts two dataframes with timestamps as indexes and averaging period.
    :param ref_speed : Dataframe containing reference speed as a column, timestamp as the index.
    :param target_speed: Dataframe containing target speed as a column, timestamp as the index.
    :param averaging_prd: Groups data by the period specified by period.
        2T, 2 min for minutely average
        Set period to 1D for a daily average, 3D for three hourly average, similarly 5D, 7D, 15D etc.
        Set period to 1H for hourly average, 3H for three hourly average and so on for 5H, 6H etc.
        week
        Set period to 1MS for monthly average
        Set period to 1AS fo annual average
    :return An object representing ordinary least squares fit model
    """

    def __init__(self, ref, target, averaging_prd, coverage_threshold):
        self.ref = ref
        self.target = target
        self.averaging_prd = averaging_prd
        self.coverage_threshold = coverage_threshold
        self.fit()

    def fit(self):
        self._ref_processed, self._target_processed = _preprocess_data_for_correlations(self.ref, self.target,
                                                                            self.averaging_prd, self.coverage_threshold)
        p, res = lstsq(self._ref_processed.values.flatten()[:, np.newaxis]**[1, 0],
                                             self._target_processed.values.flatten())[0:2]
        self.params = {'slope':p[0],'offset': p[1],'sum of residues': res}

    def show_params(self):
        print("Parameters:", self.params)

    def _predict(self, x):
        def linear_function(x, slope, offset):
            return x*slope + offset
        return x.transform(linear_function, slope=self.params['slope'], offset=self.params['offset'])

    def plot_model(self):
        _scatter_plot(self._ref_processed.values.flatten(), self._target_processed.values.flatten(),
                      self._predict(self._ref_processed).values.flatten())

    def synthesize(self, data=None):
        if data is None:
            data = self.ref
        return self._predict(data)


class speedsort_nondirectional:

    def __init__(self, ref, target, averaging_prd, coverage_threshold, lt_ref_speed=None, pre_processing=True):
        self.ref = ref
        self.target = target
        self.averaging_prd = averaging_prd
        self.coverage_threshold = coverage_threshold
        self.pre_processing = pre_processing
        if lt_ref_speed is None:
            self.lt_ref_speed = calc_lt_ref_speed(ref)
        else:
            self.lt_ref_speed = lt_ref_speed
        self.cutoff = min(0.5*self.lt_ref_speed, 4.0)
        self.fit()

    def fit(self):
        if self.pre_processing == True:
            self._ref_processed, self._target_processed = _preprocess_data_for_correlations(self.ref, self.target,
                                                                                        self.averaging_prd,
                                                                                        self.coverage_threshold)
        else:
            self._ref_processed, self._target_processed = self.ref, self.target
        x_data = sorted([wdspd for wdspd in self._ref_processed.values.flatten()])
        y_data = sorted([wdspd for wdspd in self._target_processed.values.flatten()])
        start_idx = 0
        for idx, wdspd in enumerate(x_data):
            if wdspd >= self.cutoff:
                start_idx = idx
                break
        x_data = x_data[start_idx:]
        y_data = y_data[start_idx:]
        # print(y_data)
        self.target_cutoff = y_data[0]
        # print("start idx:", start_idx)
        # print([i for i in y_data if i < self.target_cutoff])

        self.data_pts = min(len(x_data),len(y_data))
        # orthogonal least squares
        self._data = RealData(x_data[:self.data_pts], y_data[:self.data_pts])
        print((np.asarray(x_data)[:, np.newaxis]**[1, 0]).shape, (np.asarray(y_data)[:,np.newaxis]).shape)

        p, res = lstsq(np.nan_to_num(np.asarray(x_data)[:, np.newaxis]**[1, 0]), np.nan_to_num(np.asarray(y_data)
                                                                                               [:, np.newaxis]))[0:2]
        print("Linear regression:", p[0], p[1])
        """
        self._model = ODR(self._data, Model(linear_func), beta0=[p[0][0], p[1][0]])

        self.out = self._model.run()
        self.params = self.out.beta

        print(self.out.pprint())
        """
        # Karen's line fit
        mid_pnt = int(len(x_data)/2)
        xmean1 = np.mean(x_data[:mid_pnt])
        xmean2 = np.mean(x_data[mid_pnt:])
        ymean1 = np.mean(y_data[:mid_pnt])
        ymean2 = np.mean(y_data[mid_pnt:])
        self.params = [0 , 0]
        self.params[0] = (ymean2 - ymean1) /(xmean2 - xmean1)
        self.params[1] = ymean1 - (xmean1 * self.params[0])
        print("Karen's regression:", self.params)

    def show_params(self):
        print("Parameters:", self.params)

    def _predict(self, x):
        if isinstance(x, pd.Series):
            x = x.to_frame()

        def linear_func_inverted(x, p=self.params):
            if x > self.cutoff:
                return linear_func(p, x)
            else:
                return (self._data.y[0]/self._data.x[0])*x
        return x.applymap(linear_func_inverted)

    def plot_model(self):
        #_scatter_plot(self._data.x, self._data.y, self._predict(self._data.x))
        _scatter_plot(sorted(self._ref_processed.values.flatten()), sorted(self._target_processed.values.flatten()),
                      sorted(self._predict(self._ref_processed).values.flatten()))

    def synthesize(self, data=None):
        if data is None:
            data = self.ref
        return self._predict(data)


class speedsort:
    def __init__(self, ref_spd, ref_dir, target_spd, target_dir, averaging_prd, coverage_threshold, sectors=12,
                 direction_bin_array=None, lt_ref_speed=None):
        self.ref_spd = ref_spd
        self.target_spd = target_spd
        self.ref_dir = ref_dir
        self.target_dir = target_dir
        self.averaging_prd = averaging_prd
        self.sectors = sectors
        self.direction_bin_array = direction_bin_array
        self.coverage_threshold = coverage_threshold
        self.params = dict()
        #preprocess data
        self._ref_spd_processed, self._target_spd_processed = _preprocess_data_for_correlations(self.ref_spd,
                                                                                                self.target_spd,
                                                                                                self.averaging_prd,
                                                                                                self.coverage_threshold)
        self._ref_dir_processed, self._target_dir_processed = _preprocess_dir_data_for_correlations(self.ref_spd,
                                                                                                self.ref_dir,
                                                                                                self.target_spd,
                                                                                                self.target_dir,
                                                                                                self.averaging_prd,
                                                                                                self.coverage_threshold)
        # collect all the data in dataframe
        self.data = pd.concat([self._ref_spd_processed, self._target_spd_processed,self._ref_dir_processed,
                          self._target_dir_processed],axis=1, join='inner')
        self.data.columns = ['ref_spd', 'target_spd', 'ref_dir', 'target_dir']
        if lt_ref_speed is None:
            self.lt_ref_speed = calc_lt_ref_speed(self.data['ref_spd'])
        else:
            self.lt_ref_speed = lt_ref_speed
        self.cutoff = min(0.5 * self.lt_ref_speed, 4.0)
        self.ref_veer_cutoff = self._get_veer_cutoff(self.data['ref_spd'])
        self.target_veer_cutoff = self._get_veer_cutoff((self.data['target_spd']))
        self._randomize_calm_periods()
        self._get_overall_veer()
        # for low ref_speed and high target_speed recalculate direction sector
        self._adjust_low_reference_speed_dir()

        #round-off diections for checking with Ciaran's tool
        self._round_off_directions()
        # add direction sector
        self.ref_dir_bins = get_binned_direction_series(self.data['ref_dir'], sectors,
                                                    direction_bin_array=self.direction_bin_array).rename('ref_dir_bin')

        self.data = pd.concat([self.data, self.ref_dir_bins], axis=1, join='inner')

        self.data = self.data.dropna()
        self.fit()

    def _round_off_directions(self):
        self.data.loc[:, 'ref_dir'] = round(self.data.loc[:, 'ref_dir'])
        self.data.loc[:, 'target_dir'] = round(self.data.loc[:, 'target_dir'])

    def _randomize_calm_periods(self):
        idxs = self.data[self.data['ref_spd'] < 1].index
        self.data.loc[idxs, 'ref_dir'] = 360.0 * np.random.random(size=len(idxs))
        idxs = self.data[self.data['target_spd'] < 1].index
        self.data.loc[idxs, 'target_dir'] = 360.0 * np.random.random(size=len(idxs))

    def _get_overall_veer(self):
        idxs = self.data[(self.data['ref_spd'] >= self.ref_veer_cutoff) & (self.data['target_spd'] >=
                                                                           self.target_veer_cutoff)].index
        self.overall_veer = self._get_veer(self.data.loc[idxs, 'ref_dir'], self.data.loc[idxs, 'target_dir']).mean()

    def _adjust_low_reference_speed_dir(self):
        idxs = self.data[(self.data['ref_spd'] < 2) & (self.data['target_spd'] > (self.data['ref_spd']+4))].index

        self.data.loc[idxs, 'ref_dir'] = (self.data.loc[idxs, 'target_dir'] - self.overall_veer).apply(_range_0_to_360)

    @staticmethod
    def _get_veer_cutoff(speed_col):
        return 0.5*(6.0 + (0.5*speed_col.mean()))

    @staticmethod
    def _get_veer(ref_d, target_d):
        def change_range(veer):
            if veer > 180:
                return veer - 360.0
            elif veer < -180:
                return veer + 360.0
            else:
                return veer
        v = target_d - ref_d
        return v.apply(change_range)

    def _avg_veer(self, sector_data):
        sector_data = sector_data[(sector_data['ref_spd'] >= self.ref_veer_cutoff) & (sector_data['target_spd'] >=
                                                                                      self.target_veer_cutoff)]
        return {'average_veer': self._get_veer(sector_data['ref_dir'], sector_data['target_dir']).mean(),
                'num_pts_for_veer': len(sector_data['ref_dir'])}

    def fit(self):
        self.params['Ref_cutoff_for_speed'] =self.cutoff
        self.params['Ref_veer_cutoff'] = self.ref_veer_cutoff
        self.params['Target_veer_cutoff'] = self.target_veer_cutoff
        self.params['Overall_average_veer'] = self.overall_veer
        print(self.params)
        for sector, group in self.data.groupby(['ref_dir_bin']):
            print('Processing sector:', sector)
            self.speed_model = speedsort_nondirectional(ref=group['ref_spd'], target=group['target_spd'],
                                averaging_prd=self.averaging_prd, coverage_threshold=0.0,
                                                        lt_ref_speed=self.lt_ref_speed, pre_processing=False)
            self.params[sector] = {'slope':self.speed_model.params[0],'offset':self.speed_model.params[1],
                                   'target_cutoff': self.speed_model.target_cutoff,
                                   'num_pts_for_speed_fit': self.speed_model.data_pts,
                                   'num_total_pts': min(group.count())}
            self.params[sector].update(self._avg_veer(group))
            if sector == 9 or sector == 10:
                group.to_csv("sector{0}_speedsort_test.csv".format(sector))

    def _predict(self, x, param):
        if isinstance(x, pd.Series):
            x = x.to_frame()

        def linear_func_inverted(x, p=param):
            if x > self.cutoff:
                return linear_func(p, x)
            else:
                return (self._data.y[0] / self._data.x[0]) * x

        return x.applymap(linear_func_inverted)

    def synthesize(self, data=None):
        if data is None:
            data = self.ref
        return self._predict(data)


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
        _scatter_plot(ref_processed.loc[common_idxs].values,target_processed.loc[common_idxs].values, predicted_y=0,
                      x_label="Reference Data", y_label="Target Data  ")


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