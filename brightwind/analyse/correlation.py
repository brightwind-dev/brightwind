#     brightwind is a library that provides wind analysts with easy to use tools for working with meteorological data.
#     Copyright (C) 2018 Stephen Holleran, Inder Preet
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Lesser General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pandas as pd
import numpy as np
from typing import List
from brightwind.transform import transform as tf
from brightwind.analyse.plot import _scatter_plot
from scipy.odr import ODR, RealData, Model
from scipy.linalg import lstsq
from brightwind.analyse.analyse import momm, _binned_direction_series
from sklearn.svm import SVR as sklearn_SVR
from sklearn.model_selection import cross_val_score as sklearn_cross_val_score
from brightwind.utils import utils


__all__ = ['']


# def _preprocess_data_for_correlations(ref: pd.DataFrame, target: pd.DataFrame, averaging_prd, coverage_threshold):
#     """A wrapper function that calls other functions necessary for pre-processing the data"""
#     ref = ref.sort_index().dropna()
#     target = target.sort_index().dropna()
#     ref_overlap, target_overlap = tf._get_overlapping_data(ref, target, averaging_prd)
#     ref_overlap_avgd = tf.average_data_by_period(ref_overlap, averaging_prd)
#     target_overlap_avgd = tf.average_data_by_period(target_overlap, averaging_prd)
#     ref_filtered_for_coverage = tf._filter_by_coverage_threshold(ref, ref_overlap_avgd, coverage_threshold)
#     target_filtered_for_coverage = tf._filter_by_coverage_threshold(target, target_overlap_avgd, coverage_threshold)
#     common_idxs, data_pts = tf._common_idxs(ref_filtered_for_coverage, target_filtered_for_coverage)
#     return ref_filtered_for_coverage.drop(['Count', 'Coverage'], axis=1).loc[common_idxs], \
#                     target_filtered_for_coverage.drop(['Count', 'Coverage'], axis=1).loc[common_idxs]


# def _preprocess_power_data_for_correlations(ref: pd.DataFrame, target:pd.DataFrame, averaging_prd, coverage_threshold):
#     """A wrapper function that calls other functions necessary for pre-processing the data"""
#     ref = ref.sort_index().dropna()
#     target = target.sort_index().dropna()
#     ref_overlap, target_overlap = tf._get_overlapping_data(ref, target, averaging_prd)
#     ref_overlap_avgd = tf.average_data_by_period(ref_overlap, averaging_prd)
#     target_overlap_avgd = tf.average_data_by_period(target_overlap, averaging_prd, aggregation_method='sum')
#     ref_filtered_for_coverage = tf._filter_by_coverage_threshold(ref, ref_overlap_avgd, coverage_threshold)
#     target_filtered_for_coverage = t
# f._filter_by_coverage_threshold(target, target_overlap_avgd, coverage_threshold)
#     common_idxs, data_pts = tf._common_idxs(ref_filtered_for_coverage, target_filtered_for_coverage)
#     return ref_filtered_for_coverage.drop(['Count','Coverage'], axis=1).loc[common_idxs], \
#                     target_filtered_for_coverage.drop(['Count','Coverage'], axis=1).loc[common_idxs]


class CorrelBase:
    def __init__(self, ref_spd, target_spd, averaging_prd, coverage_threshold, ref_dir=None,
                 target_dir=None, preprocess=True):
        self.ref_spd = ref_spd
        self.ref_dir = ref_dir
        self.target_spd = target_spd
        self.target_dir = target_dir
        self.averaging_prd = averaging_prd
        self.coverage_threshold = coverage_threshold
        self.preprocess = preprocess
        if preprocess:
            self.data = CorrelBase._averager(ref_spd, target_spd, averaging_prd, coverage_threshold,
                                             ref_dir, target_dir)
        else:
            self.data = pd.concat([ref_spd, target_spd, ref_dir, target_dir], axis=1, join='inner')
        if ref_dir is None and target_dir is None:
            self.data.columns = ['ref_spd', 'target_spd']
        else:
            self.data.columns = ['ref_spd', 'target_spd', 'ref_dir', 'target_dir']
        self.num_data_pts = len(self.data)

    @staticmethod
    def _averager(ref_spd, target_spd, averaging_prd, coverage_threshold, ref_dir, target_dir):
        data = pd.concat(list(tf._preprocess_data_for_correlations(
            ref_spd, target_spd, averaging_prd, coverage_threshold)),
            axis=1, join='inner')
        if ref_dir is not None and target_dir is not None:
            data = pd.concat([data] + list(tf._preprocess_dir_data_for_correlations(
                ref_spd, ref_dir, target_spd, target_dir, averaging_prd, coverage_threshold)),
                             axis=1, join='inner')
        return data

    def show_params(self):
        """Show the dictionary of parameters"""
        import pprint
        pprint.pprint(self.params)

    def plot(self, title=""):
        """For plotting"""
        return _scatter_plot(self.data['ref_spd'].values.flatten(), self.data['target_spd'].values.flatten(),
                             self._predict(self.data['ref_spd']).values.flatten())

    def synthesize(self, ext_input=None):
        # This will give erroneous result when the averaging period is not a whole number such that ref and target does
        # bot get aligned -Inder
        if ext_input is None:
            output = self._predict(tf.average_data_by_period(self.ref_spd, self.averaging_prd,
                                                             return_coverage=False))
            output = tf.average_data_by_period(self.target_spd, self.averaging_prd,
                                               return_coverage=False).combine_first(output)

        else:
            output = self._predict(ext_input)
        if isinstance(output, pd.Series):
            return output.to_frame(name=self.target_spd.name + "_Synthesized")
        else:
            output.columns = [self.target_spd.name + "_Synthesized"]
            return output

    def get_r2(self):
        """Returns the r2 score of the model"""
        return 1.0 - (sum((self.data['target_spd'] - self._predict(self.data['ref_spd'])) ** 2) / (
            sum((self.data['target_spd'] - self.data['target_spd'].mean()) ** 2)))

    def get_error_metrics(self):
        return 0


class OrdinaryLeastSquares(CorrelBase):
    """Accepts two DataFrames with timestamps as indexes and averaging period.

    :param ref_spd: Series containing reference speed as a column, timestamp as the index.
    :type ref_spd: pandas.Series
    :param target_spd: DataFrame containing target speed as a column, timestamp as the index.
    :type target_spd: pandas.Series
    :param averaging_prd: Groups data by the period specified by period.

            - 2T, 2 min for minutely average
            - Set period to 1D for a daily average, 3D for three hourly average, similarly 5D, 7D, 15D etc.
            - Set period to 1H for hourly average, 3H for three hourly average and so on for 5H, 6H etc.
            - Set period to 1MS for monthly average
            - Set period to 1AS fo annual average

    :type averaging_prd: string or pandas.DateOffset
    :param coverage_threshold: Minimum coverage to include for correlation
    :type coverage_threshold: float
    :param preprocess: To average and check for coverage before correlating
    :type preprocess: bool
    :returns: An object representing ordinary least squares fit model


    """

    @staticmethod
    def linear_func(p, x):
        return (p[0] * x) + p[1]

    def __init__(self, ref_spd, target_spd, averaging_prd='1H', coverage_threshold=0.9, preprocess=True):
        CorrelBase.__init__(self, ref_spd, target_spd, averaging_prd, coverage_threshold, preprocess=preprocess)
        self.params = 'not run yet'

    def __repr(self):
        return 'Ordinary Least Squares Model' + str(self.params)

    def run(self, show_params=True):
        p, res = lstsq(np.nan_to_num(self.data['ref_spd'].values.flatten()[:, np.newaxis] ** [1, 0]),
                       np.nan_to_num(self.data['target_spd'].values.flatten()))[0:2]

        self.params = {'slope': p[0], 'offset': p[1]}
        self.params['r2'] = self.get_r2()
        self.params['Num data points'] = self.num_data_pts
        if show_params:
            self.show_params()

    def _predict(self, x):
        def linear_function(x, slope, offset):
            return (x * slope) + offset

        return x.transform(linear_function, slope=self.params['slope'], offset=self.params['offset'])


class OrthogonalLeastSquares(CorrelBase):
    """
    Accepts two series with timestamps as indexes and averaging period.

    :param ref_spd: Series containing reference speed as a column, timestamp as the index
    :param target_spd: Series containing target speed as a column, timestamp as the index
    :param averaging_prd: Groups data by the period specified by period.

            * 2T, 2 min for minutely average
            * Set period to 1D for a daily average, 3D for three hourly average, similarly 5D, 7D, 15D etc.
            * Set period to 1H for hourly average, 3H for three hourly average and so on for 5H, 6H etc.
            * Set period to 1MS for monthly average
            * Set period to 1AS fo annual average

    :param coverage_threshold: Minimum coverage to include for correlation
    :param preprocess: To average and check for coverage before correlating
    :returns: Returns an object representing the model

    """

    @staticmethod
    def linear_func(p, x):
        return p[0] * x + p[1]

    def __init__(self, ref_spd, target_spd, averaging_prd, coverage_threshold=0.9, preprocess=True):
        CorrelBase.__init__(self, ref_spd, target_spd, averaging_prd, coverage_threshold, preprocess=preprocess)
        self.params = 'not run yet'

    def __repr__(self):
        return 'Orthogonal Least Squares Model ' + str(self.params)

    def run(self, show_params=True):
        fit_data = RealData(self.data['ref_spd'].values.flatten(), self.data['target_spd'].values.flatten())
        p, res = lstsq(np.nan_to_num(fit_data.x[:, np.newaxis] ** [1, 0]), np.nan_to_num(np.asarray(fit_data.y)
                                                                                         [:, np.newaxis]))[0:2]
        self._model = ODR(fit_data, Model(OrthogonalLeastSquares.linear_func), beta0=[p[0][0], p[1][0]])
        self.out = self._model.run()
        self.params = {'slope': self.out.beta[0], 'offset': self.out.beta[1]}
        self.params['r2'] = self.get_r2()
        self.params['Num data points'] = self.num_data_pts
        # print("Model output:", self.out.pprint())
        if show_params:
            self.show_params()

    def _predict(self, x):
        def linear_func_inverted(x, p):
            return OrthogonalLeastSquares.linear_func(p, x)

        return x.transform(linear_func_inverted, p=[self.params['slope'], self.params['offset']])


class MultipleLinearRegression(CorrelBase):
    def __init__(self, ref_spd: List, target_spd, averaging_prd='1H', coverage_threshold=0.9, preprocess=True):
        self.ref_spd = pd.concat(ref_spd, axis=1, join='inner')
        self.target_spd = target_spd
        self.averaging_prd = averaging_prd
        self.coverage_threshold = coverage_threshold
        self.preprocess = preprocess
        if preprocess:
            self.data = pd.concat(list(tf._preprocess_data_for_correlations(
                self.ref_spd, self.target_spd, averaging_prd, coverage_threshold)),
                axis=1, join='inner')
        else:
            self.data = pd.concat(list(self.ref_spd, self.target_spd), axis=1, join='inner')
        self.data.columns = ['ref_spd_' + str(i + 1) for i in range(0, len(self.ref_spd.columns))] + ['target_spd']
        self.data = self.data.dropna()
        self.params = 'not run yet'

    def __repr__(self):
        return 'Multiple Linear Regression Model ' + str(self.params)

    def run(self, show_params=True):
        p, res = lstsq(np.column_stack((self.data.iloc[:, :len(self.data.columns) - 1].values,
                                        np.ones(len(self.data)))), self.data['target_spd'].values.flatten())[0:2]
        self.params = {'slope': p[:-1], 'offset': p[-1]}
        if show_params:
            self.show_params()

    def show_params(self):
        import pprint
        pprint.pprint(self.params)

    def _predict(self, x):
        def linear_function(x, slope, offset):
            return sum(x * slope) + offset

        return x.apply(linear_function, axis=1, slope=self.params['slope'], offset=self.params['offset'])

    def synthesize(self, ext_input=None):
        if ext_input is None:
            return pd.concat([self._predict(tf.average_data_by_period(self.ref_spd.loc[:min(self.data.index)],
                                                                      self.averaging_prd,
                                                                      return_coverage=False)),
                              self.data['target_spd']], axis=0)
        else:
            return self._predict(ext_input)

    def get_r2(self):
        return 1.0 - (sum((self.data['target_spd'] - self._predict(self.data.drop(['target_spd'], axis=1))) ** 2) / (
            sum((self.data['target_spd'] - self.data['target_spd'].mean()) ** 2)))

    def plot(self):
        return "Cannot plot Multiple Linear Regression"


class SimpleSpeedRatio(CorrelBase):
    def __init__(self, ref_spd, target_spd, preprocess=True):
        from pandas.tseries.frequencies import to_offset
        ref_resolution, target_resolution = tf._get_data_resolution(ref_spd.index), \
                                            tf._get_data_resolution(target_spd.index)
        if ref_resolution > target_resolution:
            averaging_prd = to_offset(ref_resolution)
        else:
            averaging_prd = to_offset(target_resolution)
        CorrelBase.__init__(self, ref_spd, target_spd, averaging_prd, coverage_threshold=0, preprocess=preprocess)
        self.params = 'not run yet'
        # self.cutoff = cutoff
        # self._filter()      #Filter low wind speeds

    def __repr__(self):
        return 'Simple Speed Ratio Model ' + str(self.params)

    # def _filter(self):
    #     if self.cutoff is not None:
    #         self.data = self.data[(self.data['ref_spd'] >= self.cutoff) & (self.data['target_spd'] >= self.cutoff)]

    def run(self, show_params=True):
        self.params = dict()
        self.params["ratio"] = self.data['target_spd'].mean() / self.data['ref_spd'].mean()
        if show_params:
            self.show_params()

    def _predict(self, x):
        def linear_function(x, slope):
            return x * slope

        return x.transform(linear_function, slope=self.params['ratio'])


class SpeedSort(CorrelBase):
    class SectorSpeedModel:
        def __init__(self, ref_spd, target_spd, lt_ref_speed=None):
            self.sector_ref = ref_spd
            self.sector_target = target_spd
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
            # print(self.params)

        def sector_predict(self, x):
            def linear_function(x, slope, offset):
                return x * slope + offset

            return x.transform(linear_function, slope=self.params['slope'], offset=self.params['offset'])

        def plot_model(self, title):
            _scatter_plot(sorted(self.sector_ref.values.flatten()), sorted(self.sector_target.values.flatten()),
                          sorted(self.sector_predict(self.sector_ref).values.flatten()))

    def __init__(self, ref_spd, ref_dir, target_spd, target_dir, averaging_prd, coverage_threshold=0.9, sectors=12,
                 direction_bin_array=None, lt_ref_speed=None, preprocess=True):
        CorrelBase.__init__(self, ref_spd, target_spd, averaging_prd, coverage_threshold, ref_dir=ref_dir,
                            target_dir=target_dir, preprocess=preprocess)
        self.sectors = sectors
        self.direction_bin_array = direction_bin_array
        if direction_bin_array is not None:
            self.sectors = len(direction_bin_array)-1
        if lt_ref_speed is None:
            self.lt_ref_speed = momm(self.data['ref_spd'])
        else:
            self.lt_ref_speed = lt_ref_speed
        self.cutoff = min(0.5 * self.lt_ref_speed, 4.0)
        self.ref_veer_cutoff = self._get_veer_cutoff(self.data['ref_spd'])
        self.target_veer_cutoff = self._get_veer_cutoff((self.data['target_spd']))
        self._randomize_calm_periods()
        self._get_overall_veer()
        # for low ref_speed and high target_speed recalculate direction sector
        self._adjust_low_reference_speed_dir()

        self.ref_dir_bins = _binned_direction_series(self.data['ref_dir'], sectors,
                                                     direction_bin_array=self.direction_bin_array).rename('ref_dir_bin')
        self.data = pd.concat([self.data, self.ref_dir_bins], axis=1, join='inner')
        self.data = self.data.dropna()
        self.params = 'not run yet'

    def __repr__(self):
        return 'Speed Sort Model ' + str(self.params)

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
        idxs = self.data[(self.data['ref_spd'] < 2) & (self.data['target_spd'] > (self.data['ref_spd'] + 4))].index

        self.data.loc[idxs, 'ref_dir'] = (self.data.loc[idxs, 'target_dir'] - self.overall_veer).apply(
            utils._range_0_to_360)

    @staticmethod
    def _get_veer_cutoff(speed_col):
        return 0.5 * (6.0 + (0.5 * speed_col.mean()))

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

    def run(self, show_params=True):
        self.params = dict()
        self.params['Ref_cutoff_for_speed'] = self.cutoff
        self.params['Ref_veer_cutoff'] = self.ref_veer_cutoff
        self.params['Target_veer_cutoff'] = self.target_veer_cutoff
        self.params['Overall_average_veer'] = self.overall_veer
        # print(self.params)
        self.speed_model = dict()
        for sector, group in self.data.groupby(['ref_dir_bin']):
            # print('Processing sector:', sector)
            self.speed_model[sector] = SpeedSort.SectorSpeedModel(ref_spd=group['ref_spd'],
                                                                  target_spd=group['target_spd'],
                                                                  lt_ref_speed=self.lt_ref_speed)
            self.params[sector] = {'slope': self.speed_model[sector].params['slope'],
                                   'offset': self.speed_model[sector].params['offset'],
                                   'target_cutoff': self.speed_model[sector].target_cutoff,
                                   'num_pts_for_speed_fit': self.speed_model[sector].data_pts,
                                   'num_total_pts': min(group.count())}
            self.params[sector].update(self._avg_veer(group))
        if show_params:
            self.show_params()

    def get_result_table(self):
        result = pd.DataFrame()
        for key in self.params:
            if not isinstance(key, str):
                result = pd.concat([pd.DataFrame.from_records(self.params[key], index=[key]), result], axis=0)
        result = result.sort_index()
        return result

    def plot(self):
        for model in self.speed_model:
            self.speed_model[model].plot_model('Sector ' + str(model))
        self.plot_wind_vane()

    def _predict_dir(self, x_dir):
        sec_veer = []
        for i in range(1, self.sectors+1):
            sec_veer.append(self.params[i]['average_veer'])
        # Add additional entry for first sector
        sec_veer.append(self.params[1]['average_veer'])
        if self.direction_bin_array is None:
            veer_bins = [i*(360/self.sectors) for i in range(0, self.sectors+1)]
        else:
            veer_bins = [self.direction_bin_array[i]+self.direction_bin_array[i+1]/2.0
                         for i in range(0, len(self.direction_bin_array)-1)]
        x = pd.concat([x_dir.dropna().rename('dir'), _binned_direction_series(x_dir.dropna(), self.sectors,
                       direction_bin_array=veer_bins).rename('veer_bin')], axis=1, join='inner')
        x['sec_mid_pt'] = [veer_bins[i-1] for i in x['veer_bin']]
        x['ratio'] = (x['dir'] - x['sec_mid_pt'])/(360.0/self.sectors)
        x['sec_veer'] = [sec_veer[i - 1] for i in x['veer_bin']]
        x['multiply_factor'] = [sec_veer[i]-sec_veer[i-1] for i in x['veer_bin']]
        x['adjustment'] = x['sec_veer'] + (x['ratio']*x['multiply_factor'])
        return (x['dir']+x['adjustment']).sort_index().apply(utils._range_0_to_360)

    def _predict(self, x_spd, x_dir):
        x = pd.concat([x_spd.dropna().rename('spd'),
                       _binned_direction_series(x_dir.dropna(), self.sectors,
                                                direction_bin_array=self.direction_bin_array).rename('ref_dir_bin')],
                      axis=1, join='inner')
        prediction = pd.DataFrame()
        first = True
        for sector, data in x.groupby(['ref_dir_bin']):
            if first is True:
                first = False
                prediction = self.speed_model[sector].sector_predict(data['spd'])
            else:
                prediction = pd.concat([prediction, self.speed_model[sector].sector_predict(data['spd'])], axis=0)

        return prediction.sort_index()

    def synthesize(self, input_spd=None, input_dir=None):
        # This will give erroneous result when the averaging period is not a whole number such that ref and target does
        # bot get aligned -Inder
        if input_spd is None and input_dir is None:
            output = self._predict(tf.average_data_by_period(self.ref_spd, self.averaging_prd,
                                                             return_coverage=False),
                                   tf.average_data_by_period(self.ref_dir, self.averaging_prd,
                                                             return_coverage=False))
            output = tf.average_data_by_period(self.target_spd, self.averaging_prd,
                                               return_coverage=False).combine_first(output)
            dir_output = self._predict_dir(tf.average_data_by_period(self.ref_dir, self.averaging_prd,
                                                        filter_by_coverage_threshold=False, return_coverage=False))

        else:
            output = self._predict(input_spd, input_dir)
            dir_output = self._predict_dir(input_dir)
        output[output < 0] = 0
        return pd.concat([output.rename(self.target_spd.name + "_Synthesized"),
                          dir_output.rename(self.target_dir.name+"_Synthesized")], axis=1, join='inner')

    def plot_wind_vane(self):
        """
        Plots reference and target directions in a scatter plot
        """

        # _scatter_plot(self.ref_dir, self.target_dir,title='original data')
        _scatter_plot(
            self.data['ref_dir'][(self.data['ref_spd'] > self.cutoff) & (self.data['target_spd'] > self.cutoff)],
            self.data['target_dir'][(self.data['ref_spd'] > self.cutoff) & (self.data['target_spd'] > self.cutoff)],
            x_label='Reference direction', y_label="Target direction")


class SVR(CorrelBase):
    def __init__(self, ref_spd, target_spd, averaging_prd, coverage_threshold, bw_model=0, preprocess=True,
                 **sklearn_args):
        CorrelBase.__init__(self, ref_spd, target_spd, averaging_prd, coverage_threshold, preprocess=preprocess)
        bw_models = [{'kernel': 'rbf', 'C': 30, 'gamma': 0.01}, {'kernel': 'linear', 'C': 10}]
        self.model = sklearn_SVR(**{**bw_models[bw_model], **sklearn_args})
        self.params = 'not run yet'

    def __repr__(self):
        return 'Support Vector Regression Model ' + str(self.params)

    def run(self, show_params=True):
        if len(self.data['ref_spd'].values.shape) == 1:
            x = self.data['ref_spd'].values.reshape(-1, 1)
        else:
            x = self.data['ref_spd'].values
        self.model.fit(x, self.data['target_spd'].values.flatten())
        self.params = dict()
        self.params['RMSE'] = -1 * sklearn_cross_val_score(self.model, x, self.data['target_spd'].values.flatten(),
                                                           scoring='neg_mean_squared_error', cv=3)
        self.params['MAE'] = -1 * sklearn_cross_val_score(self.model, x, self.data['target_spd'].values.flatten(),
                                                          scoring='neg_mean_absolute_error', cv=3)
        self.params['Explained Variance'] = -1 * sklearn_cross_val_score(self.model, x,
                                                                         self.data['target_spd'].values.flatten(),
                                                                         scoring='explained_variance', cv=3)
        if show_params:
            self.show_params()

    def _predict(self, x):
        if isinstance(x, pd.Series):
            X = x.values.reshape(-1, 1)
            return pd.DataFrame(data=self.model.predict(X), index=x.index)
        elif isinstance(x, pd.DataFrame):
            X = x.values
            return pd.DataFrame(data=self.model.predict(X), index=x.index)
        else:
            if not len(x.shape) == 2:
                raise ValueError("Expected shape of input data (num of data points, number of reference datasets), "
                                 "but found ", x.shape)
            else:
                return self.model.predict(x)

    def plot(self, title=""):
        """For plotting"""
        _scatter_plot(self.data['ref_spd'].values.flatten(), self.data['target_spd'].values.flatten(),
                      self._predict(self.data['ref_spd']).values.flatten(), prediction_marker='.')
