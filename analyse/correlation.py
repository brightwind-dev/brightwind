import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List, Dict
from transform.transform import calc_lt_ref_speed, _get_overlapping_data, _average_data_by_period, _filter_by_coverage_threshold, _common_idxs

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
    return ref_filtered_for_coverage.drop(['Count', 'Coverage'], axis=1).loc[common_idxs], \
                    target_filtered_for_coverage.drop(['Count', 'Coverage'], axis=1).loc[common_idxs]


def _preprocess_power_data_for_correlations(ref: pd.DataFrame, target:pd.DataFrame, averaging_prd, coverage_threshold):
    """A wrapper function that calls other functions necessary for pre-processing the data"""
    ref = ref.sort_index().dropna()
    target = target.sort_index().dropna()
    ref_overlap, target_overlap = _get_overlapping_data(ref, target, averaging_prd)
    ref_overlap_avgd = _average_data_by_period(ref_overlap, averaging_prd)
    target_overlap_avgd = _average_data_by_period(target_overlap, averaging_prd, aggregation_method='sum')
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


def _dir_averager(spd_overlap, dir, averaging_prd, coverage_threshold):
    vec = pd.concat([spd_overlap, dir.apply(degree_to_radian)], axis=1, join='inner')
    vec.columns = ['spd', 'dir']
    vec['N'], vec['E'] = _compute_wind_vector(vec['spd'], vec['dir'])
    vec_N_avgd = _average_data_by_period(vec['N'], averaging_prd)
    vec_E_avgd = _average_data_by_period(vec['E'], averaging_prd)
    vec_dir_avgd = np.arctan2(vec_E_avgd.loc[:,vec_E_avgd.columns != 'Count'], vec_N_avgd.loc[:,vec_N_avgd.columns !=
                                                    'Count']).applymap(radian_to_degree).applymap(_range_0_to_360)
    vec_dir_avgd.loc[:] = round(vec_dir_avgd.loc[:])
    vec_dir_avgd = pd.concat([vec_dir_avgd,vec_E_avgd['Count']], axis=1, join='inner')
    return vec_dir_avgd


def _preprocess_dir_data_for_correlations(ref_spd: pd.DataFrame, ref_dir: pd.DataFrame, target_spd:pd.DataFrame,
                                          target_dir: pd.DataFrame, averaging_prd, coverage_threshold):
    ref_spd = ref_spd.sort_index().dropna()
    target_spd = target_spd.sort_index().dropna()
    ref_overlap, target_overlap = _get_overlapping_data(ref_spd, target_spd, averaging_prd)
    ref_dir_avgd = _dir_averager(ref_overlap, ref_dir, averaging_prd, coverage_threshold)
    target_dir_avgd = _dir_averager(target_overlap, target_dir, averaging_prd, coverage_threshold)
    ref_dir_filtered_for_coverage = _filter_by_coverage_threshold(ref_dir, ref_dir_avgd, coverage_threshold)
    target_dir_filtered_for_coverage = _filter_by_coverage_threshold(target_dir, target_dir_avgd, coverage_threshold)
    common_idxs, data_pts = _common_idxs(ref_dir_filtered_for_coverage, target_dir_filtered_for_coverage)
    return ref_dir_filtered_for_coverage.drop(['Count','Coverage'], axis=1).loc[common_idxs], \
                    target_dir_filtered_for_coverage.drop(['Count','Coverage'], axis=1).loc[common_idxs]


class CorrelBase:
    def __init__(self, ref, target, averaging_prd, coverage_threshold, ref_dir=None, target_dir=None, preprocess=True):
        self.ref = ref
        self.ref_dir = ref_dir
        self.target = target
        self.averaging_prd = averaging_prd
        self.coverage_threshold = coverage_threshold
        self.preprocess = preprocess
        if preprocess:
            self.data = CorrelBase._averager(ref, target, averaging_prd, coverage_threshold, ref_dir, target_dir)
        else:
            self.data = pd.concat([ref, target, ref_dir, target_dir], axis=1, join='inner')
        if ref_dir is None and target_dir is None:
            self.data.columns = ['ref_spd', 'target_spd']
        else:
            self.data.columns = ['ref_spd', 'target_spd', 'ref_dir', 'target_dir']
        self.num_data_pts = len(self.data)

    @staticmethod
    def _averager(ref, target, averaging_prd, coverage_threshold, ref_dir, target_dir):
        data = pd.concat(list(_preprocess_data_for_correlations(ref, target, averaging_prd,
                            coverage_threshold)), axis=1, join='inner')
        if ref_dir is not None and target_dir is not None:
            data = pd.concat([data]+list(_preprocess_dir_data_for_correlations(ref, ref_dir, target,target_dir
                                                , averaging_prd, coverage_threshold)), axis=1, join='inner')
        return data

    def show_params(self):
        print(self.params)

    def plot(self, title=""):
        _scatter_plot(self.data['ref_spd'].values.flatten(), self.data['target_spd'].values.flatten(),
                   self._predict(self.data['ref_spd']).values.flatten(), title=title)

    def synthesize(self, ext_input=None):
        if input is None:
            return pd.concat([self._predict(_average_data_by_period(self.ref.loc[:min(self.data.index)],
                                        self.averaging_prd, drop_count=True)),self.data['target_spd']],axis=0)
        else:
            return self._predict(ext_input)

    def get_r2(self):
        """Returns the r2 score of the model"""
        return 1.0 - (sum((self.data['target_spd'] - self._predict(self.data['ref_spd'])) ** 2) / (
            sum((self.data['target_spd'] - self.data['target_spd'].mean()) ** 2)))

    def get_coverage(self):
        return 0

    def get_error_metrics(self):
        return 0


class OrthogonalLeastSquares(CorrelBase):
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
    # @staticmethod
    def linear_func(p, x):
        return p[0] * x + p[1]

    def __init__(self, ref, target, averaging_prd, coverage_threshold, preprocess=True):
        CorrelBase.__init__(self,ref, target, averaging_prd, coverage_threshold, preprocess=preprocess)
        self.params = 'not run yet'

    def __repr__(self):
        return 'Orthogonal Least Squares Model '+str(self.params)

    def run(self):
        fit_data = RealData(self.data['ref_spd'].values.flatten(), self.data['target_spd'].values.flatten())
        p, res = lstsq(np.nan_to_num(fit_data.x[:, np.newaxis] ** [1, 0]), np.nan_to_num(np.asarray(fit_data.y)
                                                                                                 [:, np.newaxis]))[0:2]
        self._model = ODR(fit_data, Model(OrthogonalLeastSquares.linear_func), beta0=[p[0][0], p[1][0]])
        self.out = self._model.run()
        self.params = {'slope':self.out.beta[0], 'offset':self.out.beta[1]}
        self.params['r2'] = self.get_r2()
        print("Model output:", self.out.pprint())

    def _predict(self, x):
        def linear_func_inverted(x, p):
            return OrthogonalLeastSquares.linear_func(p, x)
        return x.transform(linear_func_inverted, p=[self.params['slope'],self.params['offset']])


class OrdinaryLeastSquares(CorrelBase):
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
    @staticmethod
    def linear_func(p, x):
        return (p[0] * x) + p[1]

    def __init__(self, ref, target, averaging_prd='1H', coverage_threshold=0.9, preprocess=True):
        CorrelBase.__init__(self, ref, target, averaging_prd, coverage_threshold, preprocess=preprocess)
        self.params = 'not run yet'

    def __repr(self):
        return 'Ordinary Least Squares Model' + str(self.params)

    def run(self):
        p, res = lstsq(self.data['ref_spd'].values.flatten()[:, np.newaxis]**[1, 0],
                                             self.data['target_spd'].values.flatten())[0:2]


        self.params = {'slope':p[0],'offset': p[1]}
        self.params['r2'] = self.get_r2()

    def _predict(self, x):
        def linear_function(x, slope, offset):
            return (x*slope) + offset
        return x.transform(linear_function, slope=self.params['slope'], offset=self.params['offset'])


class BulkSpeedRatio(CorrelBase):
    def __init__(self, ref, target, averaging_prd, coverage_threshold, cutoff=None, preprocess=True):
        CorrelBase.__init__(self, ref, target, averaging_prd, coverage_threshold, preprocess=preprocess)
        self.params = 'not run yet'
        self.cutoff = cutoff
        self._filter()      #Filter low wind speeds

    def _filter(self):
        if self.cutoff is not None:
            self.data = self.data[(self.data['ref_spd'] >= self.cutoff) & (self.data['target_spd'] >= self.cutoff)]

    def run(self):
        self.params = dict()
        self.params["slope"] = self.data['target_spd'].mean()/self.data['ref_spd'].mean()

    def _predict(self, x):
        def linear_function(x, slope):
            return (x*slope)
        return x.transform(linear_function, slope=self.params['slope'])


class SpeedSort(CorrelBase):

    class SectorSpeedModel:
        def __init__(self, ref, target, lt_ref_speed=None):
            self.sector_ref = ref
            self.sector_target = target
            self.cutoff = min(0.5 * lt_ref_speed, 4.0)
            x_data = sorted([wdspd for wdspd in self.sector_ref.values.flatten()])
            y_data = sorted([wdspd for wdspd in self.sector_target.values.flatten()])
            start_idx = 0
            for idx, wdspd in enumerate(x_data):
                if wdspd >= self.cutoff:
                    start_idx = idx
                    break
            x_data = x_data[start_idx:]
            y_data = y_data[start_idx:]
            self.target_cutoff = y_data[0]
            self.data_pts = min(len(x_data), len(y_data))
            # Line fit
            mid_pnt = int(len(x_data) / 2)
            xmean1 = np.mean(x_data[:mid_pnt])
            xmean2 = np.mean(x_data[mid_pnt:])
            ymean1 = np.mean(y_data[:mid_pnt])
            ymean2 = np.mean(y_data[mid_pnt:])
            self.params = dict()
            self.params['slope'] = (ymean2 - ymean1) / (xmean2 - xmean1)
            self.params['offset'] = ymean1 - (xmean1 * self.params['slope'])
            print(self.params)

        def sector_predict(self, x):
            def linear_function(x, slope, offset):
                return x * slope + offset
            return x.transform(linear_function, slope=self.params['slope'], offset=self.params['offset'])

        def plot_model(self, title):
            _scatter_plot(sorted(self.sector_ref.values.flatten()), sorted(self.sector_target.values.flatten()),
                          sorted(self.sector_predict(self.sector_ref).values.flatten()),title=title, size=(7,7))

    def __init__(self, ref_spd, ref_dir, target_spd, target_dir, averaging_prd, coverage_threshold, sectors=12,
                 direction_bin_array=None, lt_ref_speed=None, preprocess=True):
        CorrelBase.__init__(self, ref_spd, target_spd, averaging_prd, coverage_threshold, ref_dir=ref_dir,
                                                                        target_dir=target_dir, preprocess=preprocess)
        self.sectors = sectors
        self.direction_bin_array = direction_bin_array
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
        #round-off directions
        self._round_off_directions()
        # add direction sector
        self.ref_dir_bins = get_binned_direction_series(self.data['ref_dir'], sectors,
                                                    direction_bin_array=self.direction_bin_array).rename('ref_dir_bin')
        self.data = pd.concat([self.data, self.ref_dir_bins], axis=1, join='inner')
        self.data = self.data.dropna()
        self.params = 'not run yet'

    def __repr__(self):
        return 'Speed Sort Model '+str(self.params)

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

    def run(self):
        self.params = dict()
        self.params['Ref_cutoff_for_speed'] =self.cutoff
        self.params['Ref_veer_cutoff'] = self.ref_veer_cutoff
        self.params['Target_veer_cutoff'] = self.target_veer_cutoff
        self.params['Overall_average_veer'] = self.overall_veer
        print(self.params)
        self.speed_model = dict()
        for sector, group in self.data.groupby(['ref_dir_bin']):
            print('Processing sector:', sector)
            self.speed_model[sector] = SpeedSort.SectorSpeedModel(ref=group['ref_spd'], target=group['target_spd'],
                                                        lt_ref_speed=self.lt_ref_speed)
            self.params[sector] = {'slope':self.speed_model[sector].params['slope'],
                                   'offset':self.speed_model[sector].params['offset'],
                                   'target_cutoff': self.speed_model[sector].target_cutoff,
                                   'num_pts_for_speed_fit': self.speed_model[sector].data_pts,
                                   'num_total_pts': min(group.count())}
            self.params[sector].update(self._avg_veer(group))

    def get_result_table(self):
        result = pd.DataFrame()
        for key in self.params:
            if not isinstance(key, str):
                result = pd.concat([pd.DataFrame.from_records(self.params[key], index=[key]), result], axis=0)
        result = result.sort_index()
        return result

    def plot(self):
        for model in self.speed_model:
            self.speed_model[model].plot_model('Sector '+str(model))

    def _predict(self, x_spd, x_dir):
        x = pd.concat([x_spd.rename('spd'), get_binned_direction_series(x_dir, self.sectors, direction_bin_array=
                                                self.direction_bin_array).rename('ref_dir_bin')], axis=1, join='inner')
        prediction = pd.DataFrame()
        for sector, data in x.groupby(['ref_dir_bin']):
            prediction.append(self.speed_model[sector].sector_predict(data['spd']))
        return prediction.sort_index()

    def synthesize(self, input_spd=None, input_dir=None):
        if input_spd is None and input_dir is None:
            return pd.concat([self._predict(_average_data_by_period(self.ref.loc[:min(self.data.index)],
                                        self.averaging_prd, drop_count=True),
                                _average_data_by_period(self.ref_dir.loc[:min(self.data.index)],self.averaging_prd,
                                                        drop_count=True)),self.data['target_spd']], axis=0)
        else:
            return self._predict(input_spd, input_dir)
