import numpy as np
import pandas as pd
from typing import List
from brightwind.transform import transform as tf
from brightwind.analyse.plot import plot_scatter, plot_scatter_by_sector, plot_scatter_wdir
from scipy.odr import ODR, RealData, Model
from scipy.linalg import lstsq
from brightwind.analyse.analyse import momm, _binned_direction_series
from brightwind.transform.transform import offset_wind_direction
# from sklearn.svm import SVR as sklearn_SVR
# from sklearn.model_selection import cross_val_score as sklearn_cross_val_score
from brightwind.utils import utils
import pprint
import warnings


__all__ = ['']


class CorrelBase:
    def __init__(self, ref_spd, target_spd, averaging_prd, coverage_threshold=None, ref_dir=None, target_dir=None,
                 sectors=12, direction_bin_array=None, ref_aggregation_method='mean', target_aggregation_method='mean'):
        self.ref_spd = ref_spd
        self.ref_dir = ref_dir
        self.target_spd = target_spd
        self.target_dir = target_dir
        self.averaging_prd = averaging_prd
        self.coverage_threshold = coverage_threshold
        self.ref_aggregation_method = ref_aggregation_method
        self.target_aggregation_method = target_aggregation_method
        # Get the name of the columns so they can be passed around
        self._ref_spd_col_name = ref_spd.name if ref_spd is not None and isinstance(ref_spd, pd.Series) else None
        self._ref_spd_col_names = ref_spd.columns if ref_spd is not None and isinstance(ref_spd, pd.DataFrame) else None
        self._ref_dir_col_name = ref_dir.name if ref_dir is not None else None
        self._tar_spd_col_name = target_spd.name if target_spd is not None else None
        self._tar_dir_col_name = target_dir.name if target_dir is not None else None
        # Average and merge datasets into one df
        self.data = CorrelBase._averager(self, ref_spd, target_spd, averaging_prd, coverage_threshold,
                                         ref_dir, target_dir, ref_aggregation_method, target_aggregation_method)
        self.num_data_pts = len(self.data)
        self.params = {'status': 'not yet run'}

        # The self variables defined below are defined for OrdinaryLeastSquares, OrthogonalLeastSquares and SpeedSort
        if ref_dir is not None:
            self.sectors = sectors
            self.direction_bin_array = direction_bin_array

            if direction_bin_array is None:
                sector_direction_bins = utils.get_direction_bin_array(sectors)
                step = float(max(np.unique(np.diff(sector_direction_bins))))
                self._dir_sector_max = [angle for i, angle in enumerate(sector_direction_bins)
                                        if offset_wind_direction(float(angle), step/2) > sector_direction_bins[i-1]]
                self._dir_sector_min = self._dir_sector_max.copy()
                self._dir_sector_min.insert(0, self._dir_sector_min.pop())
            else:
                raise NotImplementedError("Analysis using direction_bin_array input not implemented yet.")
                # self.sectors = len(direction_bin_array) - 1
                # self._dir_sector_max = direction_bin_array[1:]
                # self._dir_sector_min = direction_bin_array[:-1]

            self._ref_dir_bins = _binned_direction_series(self.data[self._ref_dir_col_name], sectors,
                                                          direction_bin_array=self.direction_bin_array
                                                          ).rename('ref_dir_bin')
            self._predict_ref_spd = pd.Series()

    def _averager(self, ref_spd, target_spd, averaging_prd, coverage_threshold, ref_dir, target_dir,
                  ref_aggregation_method, target_aggregation_method):
        # If directions sent, concat speed and direction first
        if ref_dir is not None:
            ref_spd = pd.concat([ref_spd, ref_dir], axis=1)
        if target_dir is not None:
            target_spd = pd.concat([target_spd, target_dir], axis=1)
        data = tf.merge_datasets_by_period(data_1=ref_spd, data_2=target_spd, period=averaging_prd,
                                           coverage_threshold_1=coverage_threshold,
                                           coverage_threshold_2=coverage_threshold,
                                           wdir_column_names_1=self._ref_dir_col_name,
                                           wdir_column_names_2=self._tar_dir_col_name,
                                           aggregation_method_1=ref_aggregation_method,
                                           aggregation_method_2=target_aggregation_method)
        if len(data.index) <= 1:
            raise ValueError("Not enough overlapping data points to perform correlation.")
        return data

    def show_params(self):
        """Show the dictionary of parameters"""
        pprint.pprint(self.params)

    def plot(self, figure_size=(10, 10.2)):
        """
        Plots scatter plot of reference versus target speed data. If ref_dir is given as input to the correlation then
        the plot is showing scatter subplots for each sector. The regression line and the line of slope 1 passing
        through the origin are also shown on each plot.

        :param figure_size: Figure size in tuple format (width, height)
        :type figure_size:  tuple
        :returns:           A matplotlib figure
        :rtype:             matplotlib.figure.Figure

        **Example usage**
        ::
            import brightwind as bw
            data = bw.load_csv(bw.demo_datasets.demo_data)
            m2_ne = bw.load_csv(bw.demo_datasets.demo_merra2_NE)

            # Correlate by directional sector, using 36 sectors.
            ols_cor = bw.Correl.OrdinaryLeastSquares(m2_ne['WS50m_m/s'], data['Spd80mN'],
                                                     ref_dir=m2_ne['WD50m_deg'], averaging_prd='1D',
                                                     coverage_threshold=0.9, sectors=36)
            ols_cor.run()

            # To plot the scatter subplots by directional sectors, the regression line and the line of
            # slope 1 passing through the origin
            ols_cor.plot()

            # To set the figure size
            ols_cor.plot(figure_size=(20, 20.2))

        """
        if self.ref_dir is None:
            return plot_scatter(self.data[self._ref_spd_col_name],
                                self.data[self._tar_spd_col_name],
                                self._predict(self.data[self._ref_spd_col_name]),
                                x_label=self._ref_spd_col_name, y_label=self._tar_spd_col_name,
                                line_of_slope_1=True, figure_size=figure_size)
        else:
            """For plotting scatter by sector"""
            return plot_scatter_by_sector(self.data[self._ref_spd_col_name],
                                          self.data[self._tar_spd_col_name],
                                          self.data[self._ref_dir_col_name],
                                          trendline_y=self._predict_ref_spd, sectors=self.sectors,
                                          line_of_slope_1=True, figure_size=figure_size)

    @staticmethod
    def _get_r2(target_spd, predict_spd):
        """Returns the r2 score of the model"""
        return 1.0 - (sum((target_spd - predict_spd) ** 2) /
                      (sum((target_spd - target_spd.mean()) ** 2)))

    @staticmethod
    def _get_logic_dir_sector(ref_dir, sector_min, sector_max):
        if sector_max > sector_min:
            logic_sector = ((ref_dir >= sector_min) & (ref_dir < sector_max))
        else:
            logic_sector = ((ref_dir >= sector_min) & (ref_dir <= 360)) | \
                           ((ref_dir < sector_max) & (ref_dir >= 0))
        return logic_sector

    def _get_synth_start_dates(self):
        none_even_freq = ['5H', '7H', '9H', '10H', '11H', '13H', '14H', '15H', '16H', '17H', '18H', '19H',
                          '20H', '21H', '22H', '23H', 'D', 'W']
        if any(freq in self.averaging_prd for freq in none_even_freq):
            ref_time_array = pd.date_range(start=self.data.index[0], freq='-' + self.averaging_prd,
                                           end=self.ref_spd.index[0])
            if ref_time_array.empty:
                ref_start_date = self.data.index[0]
            else:
                ref_start_date = ref_time_array[-1]

            tar_time_array = pd.date_range(start=self.data.index[0], freq='-' + self.averaging_prd,
                                           end=self.target_spd.index[0])
            if tar_time_array.empty:
                tar_start_date = self.data.index[0]
            else:
                tar_start_date = tar_time_array[-1]
        else:
            ref_start_date = self.ref_spd.index[0]
            tar_start_date = self.target_spd.index[0]
        return ref_start_date, tar_start_date

    def synthesize(self, ext_input=None, ref_coverage_threshold=None, target_coverage_threshold=None):
        """
        Apply the derived correlation model to the reference dataset used to create the model. The resulting synthesized
        dataset is spliced with the target dataset. That is, where a target value is available, it is used instead of
        the synthesized value.

        :param ext_input:                 Optional external dataset to apply the derived correlation model to instead
                                          of the original reference. If this is used, the resulting synthesized
                                          dataset is not spliced with the target dataset.
        :type ext_input:                  pd.Series or pd.DataFrame
        :param ref_coverage_threshold:    Minimum coverage required when aggregating the reference data to calculate
                                          the synthesised data. If None, it uses the coverage_threshold supplied to the
                                          correlation model.
        :type ref_coverage_threshold:     float
        :param target_coverage_threshold: Minimum coverage required when aggregating the target data to splice with
                                          the calculated synthesised data. If None, it uses the coverage_threshold
                                          supplied to the correlation model.
        :type target_coverage_threshold:  float
        :return:                          The synthesized dataset.
        :rtype:                           pd.Series or pd.DataFrame
        """
        if ref_coverage_threshold is None:
            ref_coverage_threshold = self.coverage_threshold
        if target_coverage_threshold is None:
            target_coverage_threshold = self.coverage_threshold

        if ext_input is None:
            ref_start_date, target_start_date = self._get_synth_start_dates()

            target_spd_averaged = tf.average_data_by_period(self.target_spd[target_start_date:], self.averaging_prd,
                                                            coverage_threshold=target_coverage_threshold,
                                                            return_coverage=False)
            if self.ref_dir is None:
                ref_spd_averaged = tf.average_data_by_period(self.ref_spd[ref_start_date:], self.averaging_prd,
                                                             coverage_threshold=ref_coverage_threshold,
                                                             return_coverage=False)
                synth_data = self._predict(ref_spd_averaged)
            else:
                ref_df = pd.concat([self.ref_spd, self.ref_dir], axis=1, join='inner')
                ref_averaged = tf.average_data_by_period(ref_df[ref_start_date:], self.averaging_prd,
                                                         wdir_column_names=self._ref_dir_col_name,
                                                         coverage_threshold=ref_coverage_threshold,
                                                         return_coverage=False)
                synth_data = ref_averaged[self._ref_spd_col_name].copy() * np.nan
                for params_dict in self.params:
                    if params_dict['num_data_points'] > 1:
                        logic_sect = self._get_logic_dir_sector(ref_dir=ref_averaged[self._ref_dir_col_name],
                                                                sector_min=params_dict['sector_min'],
                                                                sector_max=params_dict['sector_max'])

                        synth_data[logic_sect] = self._predict(ref_spd=ref_averaged[self._ref_spd_col_name][logic_sect],
                                                               slope=params_dict['slope'], offset=params_dict['offset'])
            output = target_spd_averaged.combine_first(synth_data)
        else:
            if self.ref_dir is None:
                output = self._predict(ext_input)
            else:
                raise NotImplementedError

        if isinstance(output, pd.Series):
            return output.to_frame(name=self.target_spd.name + "_Synthesized")
        else:
            output.columns = [self.target_spd.name + "_Synthesized"]
            return output

    # def get_error_metrics(self):
    #     raise NotImplementedError


class OrdinaryLeastSquares(CorrelBase):
    """
    Correlate two datasets against each other using the Ordinary Least Squares method. This accepts two wind speed
    Series with timestamps as indexes and an averaging period which merges the datasets by this time period before
    performing the correlation.

    :param ref_spd:                   Series containing reference wind speed as a column, timestamp as the index.
    :type ref_spd:                    pd.Series
    :param target_spd:                Series containing target wind speed as a column, timestamp as the index.
    :type target_spd:                 pd.Series
    :param averaging_prd:             Groups data by the time period specified here. The following formats are supported

            - Set period to '10min' for 10 minute average, '30min' for 30 minute average.
            - Set period to '1H' for hourly average, '3H' for three hourly average and so on for '4H', '6H' etc.
            - Set period to '1D' for a daily average, '3D' for three day average, similarly '5D', '7D', '15D' etc.
            - Set period to '1W' for a weekly average, '3W' for three week average, similarly '2W', '4W' etc.
            - Set period to '1M' for monthly average with the timestamp at the start of the month.
            - Set period to '1A' for annual average with the timestamp at the start of the year.

    :type averaging_prd:              str
    :param coverage_threshold:        Minimum coverage required when aggregating the data to the averaging_prd.
    :type coverage_threshold:         float
    :param ref_dir:                   Series containing reference wind direction as a column, timestamp as the index.
    :type ref_dir:                    pd.Series
    :param sectors:                   Number of direction sectors to bin in to. The first sector is centered at 0 by
                                      default. To change that behaviour specify 'direction_bin_array' which overwrites
                                      'sectors'.
    :type sectors:                    int
    :param direction_bin_array:       An optional parameter where if you want custom direction bins, pass an array
                                      of the bins. To add custom bins for direction sectors, overwrites sectors. For
                                      instance, for direction bins [0,120), [120, 215), [215, 360) the list would
                                      be [0, 120, 215, 360]
    :type direction_bin_array:        List()
    :param ref_aggregation_method:    Default `mean`, returns the mean of the data for the specified period. Can also
                                      use `median`, `prod`, `sum`, `std`,`var`, `max`, `min` which are shorthands for
                                      median, product, summation, standard deviation, variance, maximum and minimum
                                      respectively.
    :type ref_aggregation_method:     str
    :param target_aggregation_method: Default `mean`, returns the mean of the data for the specified period. Can also
                                      use `median`, `prod`, `sum`, `std`,`var`, `max`, `min` which are shorthands for
                                      median, product, summation, standard deviation, variance, maximum and minimum
                                      respectively.
    :type target_aggregation_method:  str
    :returns:                         An object representing ordinary least squares fit model

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)
        m2_ne = bw.load_csv(bw.demo_datasets.demo_merra2_NE)
        m2_nw = bw.load_csv(bw.demo_datasets.demo_merra2_NW)

        # Correlate wind speeds on a monthly basis.
        ols_cor = bw.Correl.OrdinaryLeastSquares(m2_ne['WS50m_m/s'], data['Spd80mN'], averaging_prd='1M',
                                                 coverage_threshold=0.95)
        ols_cor.run()

        # To plot the scatter plot and regression line.
        ols_cor.plot()

        # To change the plot's size.
        ols_cor.plot(figure_size=(12,15))

        # To show the resulting parameters.
        ols_cor.params
        # or
        ols_cor.show_params()

        # To synthesize data at the target site.
        ols_cor.synthesize()

        # To synthesize data at the target site using a different external reference dataset.
        ols_cor.synthesize(ext_input=m2_nw['WS50m_m/s'])

        # To run the correlation without immediately showing results.
        ols_cor.run(show_params=False)

        # To retrieve the merged and aggregated data used in the correlation.
        ols_cor.data

        # To retrieve the number of data points used for the correlation
        ols_cor.num_data_pts

        # To retrieve the input parameters.
        ols_cor.averaging_prd
        ols_cor.coverage_threshold
        ols_cor.ref_spd
        ols_cor.ref_aggregation_method
        ols_cor.target_spd
        ols_cor.target_aggregation_method

        # Correlate temperature on an hourly basis using a different aggregation method.
        ols_cor = bw.Correl.OrdinaryLeastSquares(m2_ne['T2M_degC'], data['T2m'],
                                                 averaging_prd='1H', coverage_threshold=0,
                                                 ref_aggregation_method='min', target_aggregation_method='min')

        # Correlate by directional sector, using 36 sectors.
        ols_cor = bw.Correl.OrdinaryLeastSquares(m2_ne['WS50m_m/s'], data['Spd80mN'],
                                                ref_dir=m2_ne['WD50m_deg'], averaging_prd='1D',
                                                coverage_threshold=0.9, sectors=36)

    """
    def __init__(self, ref_spd, target_spd, averaging_prd, coverage_threshold=0.9, ref_dir=None, sectors=12,
                 direction_bin_array=None, ref_aggregation_method='mean', target_aggregation_method='mean'):
        CorrelBase.__init__(self, ref_spd, target_spd, averaging_prd, coverage_threshold, ref_dir=ref_dir,
                            sectors=sectors, direction_bin_array=direction_bin_array,
                            ref_aggregation_method=ref_aggregation_method,
                            target_aggregation_method=target_aggregation_method)

    def __repr__(self):
        return 'Ordinary Least Squares Model ' + str(self.params)

    @staticmethod
    def _leastsquare(ref_spd, target_spd):
        p, res = lstsq(np.nan_to_num(ref_spd.values.flatten()[:, np.newaxis] ** [1, 0]),
                       np.nan_to_num(target_spd.values.flatten()))[0:2]
        return p[0], p[1]

    def run(self, show_params=True):
        if self.ref_dir is None:
            slope, offset = self._leastsquare(ref_spd=self.data[self._ref_spd_col_name],
                                              target_spd=self.data[self._tar_spd_col_name])
            self.params = dict([('slope', slope), ('offset', offset)])
            self.params['r2'] = self._get_r2(target_spd=self.data[self._tar_spd_col_name],
                                             predict_spd=self._predict(ref_spd=self.data[self._ref_spd_col_name]))
            self.params['num_data_points'] = self.num_data_pts
        elif type(self.ref_dir) is pd.Series:
            self.params = []
            for sector, group in pd.concat([self.data, self._ref_dir_bins],
                                           axis=1, join='inner').dropna().groupby(['ref_dir_bin']):
                # print('Processing sector:', sector)
                if len(group) > 1:
                    slope, offset = self._leastsquare(ref_spd=group[self._ref_spd_col_name],
                                                      target_spd=group[self._tar_spd_col_name])
                    predict_ref_spd_sector = self._predict(ref_spd=group[self._ref_spd_col_name],
                                                           slope=slope, offset=offset)
                    r2 = self._get_r2(target_spd=group[self._tar_spd_col_name],
                                      predict_spd=predict_ref_spd_sector)
                else:
                    slope = np.nan
                    offset = np.nan
                    r2 = np.nan
                    predict_ref_spd_sector = self._predict(ref_spd=group[self._ref_spd_col_name],
                                                           slope=slope, offset=offset)

                self._predict_ref_spd = pd.concat([self._predict_ref_spd, predict_ref_spd_sector])
                self.params.append({'slope': slope,
                                    'offset': offset,
                                    'r2': r2,
                                    'num_data_points': len(group[self._tar_spd_col_name]),
                                    'sector_min': self._dir_sector_min[sector-1],
                                    'sector_max': self._dir_sector_max[sector-1],
                                    'sector_number': sector})
            self._predict_ref_spd.sort_index(ascending=True, inplace=True)

        if show_params:
            self.show_params()

    def _predict(self, ref_spd, slope=None, offset=None):
        if slope is None:
            slope = self.params['slope']
        if offset is None:
            offset = self.params['offset']
        return ref_spd * slope + offset


class OrthogonalLeastSquares(CorrelBase):
    """
    Correlate two datasets against each other using the Orthogonal Least Squares method. This accepts two wind speed
    Series with timestamps as indexes and an averaging period which merges the datasets by this time period before
    performing the correlation.

    :param ref_spd:                   Series containing reference wind speed as a column, timestamp as the index.
    :type ref_spd:                    pd.Series
    :param target_spd:                Series containing target wind speed as a column, timestamp as the index.
    :type target_spd:                 pd.Series
    :param averaging_prd:             Groups data by the time period specified here. The following formats are supported

            - Set period to '10min' for 10 minute average, '30min' for 30 minute average.
            - Set period to '1H' for hourly average, '3H' for three hourly average and so on for '4H', '6H' etc.
            - Set period to '1D' for a daily average, '3D' for three day average, similarly '5D', '7D', '15D' etc.
            - Set period to '1W' for a weekly average, '3W' for three week average, similarly '2W', '4W' etc.
            - Set period to '1M' for monthly average with the timestamp at the start of the month.
            - Set period to '1A' for annual average with the timestamp at the start of the year.

    :type averaging_prd:              str
    :param coverage_threshold:        Minimum coverage required when aggregating the data to the averaging_prd.
    :type coverage_threshold:         float
    :param ref_aggregation_method:    Default `mean`, returns the mean of the data for the specified period. Can also
                                      use `median`, `prod`, `sum`, `std`,`var`, `max`, `min` which are shorthands for
                                      median, product, summation, standard deviation, variance, maximum and minimum
                                      respectively.
    :type ref_aggregation_method:     str
    :param target_aggregation_method: Default `mean`, returns the mean of the data for the specified period. Can also
                                      use `median`, `prod`, `sum`, `std`,`var`, `max`, `min` which are shorthands for
                                      median, product, summation, standard deviation, variance, maximum and minimum
                                      respectively.
    :type target_aggregation_method:  str
    :returns:                         An object representing orthogonal least squares fit model

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)
        m2_ne = bw.load_csv(bw.demo_datasets.demo_merra2_NE)
        m2_nw = bw.load_csv(bw.demo_datasets.demo_merra2_NW)

        # Correlate wind speeds on a monthly basis.
        orthog_cor = bw.Correl.OrthogonalLeastSquares(m2_ne['WS50m_m/s'], data['Spd80mN'], averaging_prd='1M',
                                                      coverage_threshold=0.95)
        orthog_cor.run()

        # To plot the scatter plot and regression line.
        ols_cor.plot()

        # To change the plot's size.
        ols_cor.plot(figure_size=(12,15))

        # To show the resulting parameters.
        orthog_cor.params
        # or
        orthog_cor.show_params()

        # To synthesize data at the target site.
        orthog_cor.synthesize()

        # To synthesize data at the target site using a different external reference dataset.
        orthog_cor.synthesize(ext_input=m2_nw['WS50m_m/s'])

        # To run the correlation without immediately showing results.
        orthog_cor.run(show_params=False)

        # To retrieve the merged and aggregated data used in the correlation.
        orthog_cor.data

        # To retrieve the number of data points used for the correlation
        orthog_cor.num_data_pts

        # To retrieve the input parameters.
        orthog_cor.averaging_prd
        orthog_cor.coverage_threshold
        orthog_cor.ref_spd
        orthog_cor.ref_aggregation_method
        orthog_cor.target_spd
        orthog_cor.target_aggregation_method

        # Correlate temperature on an hourly basis using a different aggregation method.
        orthog_cor = bw.Correl.OrthogonalLeastSquares(m2_ne['T2M_degC'], data['T2m'],
                                                      averaging_prd='1H', coverage_threshold=0,
                                                      ref_aggregation_method='min', target_aggregation_method='min')

    """
    @staticmethod
    def linear_func(p, x):
        return p[0] * x + p[1]

    def __init__(self, ref_spd, target_spd, averaging_prd, coverage_threshold=0.9,
                 ref_aggregation_method='mean', target_aggregation_method='mean'):
        CorrelBase.__init__(self, ref_spd, target_spd, averaging_prd, coverage_threshold,
                            ref_aggregation_method=ref_aggregation_method,
                            target_aggregation_method=target_aggregation_method)

    def __repr__(self):
        return 'Orthogonal Least Squares Model ' + str(self.params)

    def run(self, show_params=True):
        fit_data = RealData(self.data[self._ref_spd_col_name].values.flatten(), 
                            self.data[self._tar_spd_col_name].values.flatten())
        p, res = lstsq(np.nan_to_num(fit_data.x[:, np.newaxis] ** [1, 0]), 
                       np.nan_to_num(np.asarray(fit_data.y)[:, np.newaxis]))[0:2]
        model = ODR(fit_data, Model(OrthogonalLeastSquares.linear_func), beta0=[p[0][0], p[1][0]])
        output = model.run()
        self.params = dict([('slope', output.beta[0]), ('offset', output.beta[1])])
        self.params['r2'] = self._get_r2(target_spd=self.data[self._tar_spd_col_name],
                                         predict_spd=self._predict(ref_spd=self.data[self._ref_spd_col_name]))
        self.params['num_data_points'] = self.num_data_pts
        # print("Model output:", output.pprint())
        if show_params:
            self.show_params()

    def _predict(self, ref_spd):
        def linear_func_inverted(x, p):
            return OrthogonalLeastSquares.linear_func(p, x)

        return ref_spd.transform(linear_func_inverted, p=[self.params['slope'], self.params['offset']])


class MultipleLinearRegression(CorrelBase):
    """
    Correlate multiple reference datasets against a target dataset using ordinary least squares. This accepts a
    list of multiple reference wind speeds and a single target wind speed. The wind speed datasets are Pandas
    Series with timestamps as indexes. Also sen is an averaging period which merges the datasets by this time period
    before performing the correlation.

    :param ref_spd:                   A list of Series containing reference wind speed as a column, timestamp as the index.
    :type ref_spd:                    List(pd.Series)
    :param target_spd:                Series containing target wind speed as a column, timestamp as the index.
    :type target_spd:                 pd.Series
    :param averaging_prd:             Groups data by the time period specified here. The following formats are supported

            - Set period to '10min' for 10 minute average, '30min' for 30 minute average.
            - Set period to '1H' for hourly average, '3H' for three hourly average and so on for '4H', '6H' etc.
            - Set period to '1D' for a daily average, '3D' for three day average, similarly '5D', '7D', '15D' etc.
            - Set period to '1W' for a weekly average, '3W' for three week average, similarly '2W', '4W' etc.
            - Set period to '1M' for monthly average with the timestamp at the start of the month.
            - Set period to '1A' for annual average with the timestamp at the start of the year.

    :type averaging_prd:              str
    :param coverage_threshold:        Minimum coverage required when aggregating the data to the averaging_prd.
    :type coverage_threshold:         float
    :param ref_aggregation_method:    Default `mean`, returns the mean of the data for the specified period. Can also
                                      use `median`, `prod`, `sum`, `std`,`var`, `max`, `min` which are shorthands for
                                      median, product, summation, standard deviation, variance, maximum and minimum
                                      respectively.
    :type ref_aggregation_method:     str
    :param target_aggregation_method: Default `mean`, returns the mean of the data for the specified period. Can also
                                      use `median`, `prod`, `sum`, `std`,`var`, `max`, `min` which are shorthands for
                                      median, product, summation, standard deviation, variance, maximum and minimum
                                      respectively.
    :type target_aggregation_method:  str
    :returns:                         An object representing Multiple Linear Regression fit model

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)
        m2_ne = bw.load_csv(bw.demo_datasets.demo_merra2_NE)
        m2_nw = bw.load_csv(bw.demo_datasets.demo_merra2_NW)

        # Correlate on a monthly basis
        mul_cor = bw.Correl.MultipleLinearRegression([m2_ne['WS50m_m/s'], m2_ne['WS50m_m/s']], data['Spd80mN'],
                                                     averaging_prd='1M',
                                                     coverage_threshold=0.95)
        mul_cor.run()

        # To plot the scatter plot and line fit.
        mul_cor.plot()

        # To show the resulting parameters.
        mul_cor.params
        # or
        mul_cor.show_params()

        # To calculate the correlation coefficient R^2.
        mul_cor.get_r2()

        # To synthesize data at the target site.
        mul_cor.synthesize()

        # To run the correlation without immediately showing results.
        mul_cor.run(show_params=False)

        # To retrieve the merged and aggregated data used in the correlation.
        mul_cor.data

        # To retrieve the number of data points used for the correlation
        mul_cor.num_data_pts

        # To retrieve the input parameters.
        mul_cor.averaging_prd
        mul_cor.coverage_threshold
        mul_cor.ref_spd
        mul_cor.ref_aggregation_method
        mul_cor.target_spd
        mul_cor.target_aggregation_method

        # Correlate temperature on an hourly basis using a different aggregation method.
        mul_cor = bw.Correl.MultipleLinearRegression([m2_ne['T2M_degC'], m2_nw['T2M_degC']], data['T2m'],
                                                     averaging_prd='1H', coverage_threshold=0,
                                                     ref_aggregation_method='min', target_aggregation_method='min')

    """
    def __init__(self, ref_spd: List, target_spd, averaging_prd, coverage_threshold=0.9,
                 ref_aggregation_method='mean', target_aggregation_method='mean'):
        self.ref_spd = self._merge_ref_spds(ref_spd)
        CorrelBase.__init__(self, self.ref_spd, target_spd, averaging_prd, coverage_threshold,
                            ref_aggregation_method=ref_aggregation_method,
                            target_aggregation_method=target_aggregation_method)

    def __repr__(self):
        return 'Multiple Linear Regression Model ' + str(self.params)

    @staticmethod
    def _merge_ref_spds(ref_spds):
        # ref_spds is a list of pd.Series that may have the same names.
        for idx, ref_spd in enumerate(ref_spds):
            ref_spd.name = ref_spd.name + '_' + str(idx + 1)
        return pd.concat(ref_spds, axis=1, join='inner')

    def run(self, show_params=True):
        p, res = lstsq(np.column_stack((self.data[self._ref_spd_col_names].values, np.ones(len(self.data)))),
                       self.data[self._tar_spd_col_name].values.flatten())[0:2]
        self.params = {'slope': p[:-1], 'offset': p[-1]}
        if show_params:
            self.show_params()

    def show_params(self):
        pprint.pprint(self.params)

    def _predict(self, x):
        def linear_function(x, slope, offset):
            return sum(x * slope) + offset

        return x.apply(linear_function, axis=1, slope=self.params['slope'], offset=self.params['offset'])

    def synthesize(self):
        # def synthesize(self, ext_input=None):     # REMOVE UNTIL FIXED
        ext_input = None
        # CorrelBase.synthesize(self.data ???????? Why not??????????????????????????????????????
        if ext_input is None:
            return pd.concat([self._predict(tf.average_data_by_period(self.ref_spd.loc[:min(self.data.index)],
                                                                      self.averaging_prd,
                                                                      return_coverage=False)),
                              self.data[self._tar_spd_col_name]], axis=0)
        else:
            return self._predict(ext_input)

    def get_r2(self):
        return 1.0 - (sum((self.data[self._tar_spd_col_name] - 
                           self._predict(self.data[self._ref_spd_col_names])) ** 2) /
                      (sum((self.data[self._tar_spd_col_name] - self.data[self._tar_spd_col_name].mean()) ** 2)))

    def plot(self, figure_size=(10, 10.2)):
        raise NotImplementedError


class SimpleSpeedRatio:
    """
    Calculate the simple speed ratio between overlapping datasets and apply to the MOMM of the reference.

    The simple speed ratio is calculated by finding the limits of the overlapping period between the target and
    reference datasets. The ratio of the mean wind speed of these two datasets for the overlapping period is
    calculated i.e. target_overlap_mean / ref_overlap_mean. This ratio is then applied to the Mean of Monthly
    Means (MOMM) of the complete reference dataset resulting in a long term wind speed for the target dataset.

    This is a "back of the envelope" style long term calculation and is intended to be used as a guide and not
    to be used in a robust wind resource assessment.

    A warning message will be raised if the data coverage of either the target or the reference overlapping
    period is poor.

    :param ref_spd:    Series containing reference wind speed as a column, timestamp as the index.
    :type ref_spd:     pd.Series
    :param target_spd: Series containing target wind speed as a column, timestamp as the index.
    :type target_spd:  pd.Series
    :return:           An object representing the simple speed ratio model

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)
        m2 = bw.load_csv(bw.demo_datasets.demo_merra2_NE)

        # Calculate the simple speed ratio between overlapping datasets
        simple_ratio = bw.Correl.SimpleSpeedRatio(m2['WS50m_m/s'], data['Spd80mN'])
        simple_ratio.run()

    """
    def __init__(self, ref_spd, target_spd):
        self.ref_spd = ref_spd
        self.target_spd = target_spd
        self._start_ts = tf._get_min_overlap_timestamp(ref_spd.dropna().index, target_spd.dropna().index)
        self._end_ts = min(ref_spd.dropna().index.max(), ref_spd.dropna().index.max())
        self.data = ref_spd[self._start_ts:self._end_ts], target_spd[self._start_ts:self._end_ts]
        self.params = {'status': 'not yet run'}

    def __repr__(self):
        return 'Simple Speed Ratio Model ' + str(self.params)

    def run(self, show_params=True):
        self.params = dict()
        simple_speed_ratio = self.data[1].mean() / self.data[0].mean()  # target / ref
        ref_long_term_momm = momm(self.ref_spd)

        # calculate the coverage of the target data to raise warning if poor
        tar_count = self.data[1].dropna().count()
        tar_res = tf._get_data_resolution(self.data[1].index)
        max_pts = (self._end_ts - self._start_ts) / tar_res
        if tar_res == pd.Timedelta(1, unit='M'):  # if is monthly
            # round the result to 0 decimal to make whole months.
            max_pts = np.round(max_pts, 0)
        target_overlap_coverage = tar_count / max_pts

        self.params["simple_speed_ratio"] = simple_speed_ratio
        self.params["ref_long_term_momm"] = ref_long_term_momm
        self.params["target_long_term"] = simple_speed_ratio * ref_long_term_momm
        self.params["target_overlap_coverage"] = target_overlap_coverage
        if show_params:
            self.show_params()

        if target_overlap_coverage < 0.9:
            warnings.warn('\nThe target data overlapping coverage is poor at {}. '
                          'Please use this calculation with caution.'.format(round(target_overlap_coverage, 3)))

    def show_params(self):
        """Show the dictionary of parameters"""
        pprint.pprint(self.params)


class SpeedSort(CorrelBase):
    class SectorSpeedModel:
        def __init__(self, ref_spd, target_spd, cutoff):
            self.sector_ref = ref_spd
            self.sector_target = target_spd
            x_data = sorted([wdspd for wdspd in self.sector_ref.values.flatten()])
            y_data = sorted([wdspd for wdspd in self.sector_target.values.flatten()])
            start_idx = 0
            for idx, wdspd in enumerate(x_data):
                if wdspd >= cutoff:
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

        def plot_model(self):
            return plot_scatter(self.sector_ref,
                                self.sector_target,
                                self.sector_predict(self.sector_ref),
                                x_label=self.sector_ref.name, y_label=self.sector_target.name)

    def __init__(self, ref_spd, ref_dir, target_spd, target_dir, averaging_prd, coverage_threshold=0.9, sectors=12,
                 direction_bin_array=None, lt_ref_speed=None):
        """
        Correlate two datasets against each other using the SpeedSort method as outlined in 'The SpeedSort, DynaSort
        and Scatter Wind Correlation Methods, Wind Engineering 29(3):217-242, Ciaran King, Brian Hurley, May 2005'.

        This accepts two wind speed and direction Series with timestamps as indexes and an averaging period which
        merges the datasets by this time period before performing the correlation.

        :param ref_spd:             Series containing reference wind speed as a column, timestamp as the index.
        :type ref_spd:              pd.Series
        :param target_spd:          Series containing target wind speed as a column, timestamp as the index.
        :type target_spd:           pd.Series
        :param ref_dir:             Series containing reference wind direction as a column, timestamp as the index.
        :type ref_dir:              pd.Series
        :param target_dir:          Series containing target wind direction as a column, timestamp as the index.
        :type target_dir:           pd.Series
        :param averaging_prd:       Groups data by the time period specified here. The following formats are supported

                - Set period to '10min' for 10 minute average, '30min' for 30 minute average.
                - Set period to '1H' for hourly average, '3H' for three hourly average and so on for '4H', '6H' etc.
                - Set period to '1D' for a daily average, '3D' for three day average, similarly '5D', '7D', '15D' etc.
                - Set period to '1W' for a weekly average, '3W' for three week average, similarly '2W', '4W' etc.
                - Set period to '1M' for monthly average with the timestamp at the start of the month.
                - Set period to '1A' for annual average with the timestamp at the start of the year.

        :type averaging_prd:        str
        :param coverage_threshold:  Minimum coverage required when aggregating the data to the averaging_prd.
        :type coverage_threshold:   float
        :param sectors:             Number of direction sectors to bin in to. The first sector is centered at 0 by
                                    default. To change that behaviour specify 'direction_bin_array' which overwrites
                                    'sectors'.
        :type sectors:              int
        :param direction_bin_array: An optional parameter where if you want custom direction bins, pass an array
                                    of the bins. To add custom bins for direction sectors, overwrites sectors. For
                                    instance, for direction bins [0,120), [120, 215), [215, 360) the list would
                                    be [0, 120, 215, 360]
        :type direction_bin_array:  List()
        :param lt_ref_speed:        An alternative to the long term wind speed for the reference dataset calculated
                                    using mean of monthly means (MOMM).
        :type lt_ref_speed:         float or int
        :returns:                   An object representing the SpeedSort fit model

        **Example usage**
        ::
            import brightwind as bw
            data = bw.load_csv(bw.demo_datasets.demo_data)
            m2 = bw.load_csv(bw.demo_datasets.demo_merra2_NE)

            # Basic usage on an hourly basis
            ss_cor = bw.Correl.SpeedSort(m2['WS50m_m/s'], m2['WD50m_deg'], data['Spd80mN'], data['Dir78mS'],
                                         averaging_prd='1H')
            ss_cor.run()
            ss_cor.plot_wind_directions()
            ss_cor.get_result_table()
            ss_cor.synthesize()

            # Sending an array of direction sectors
            ss_cor = bw.Correl.SpeedSort(m2['WS50m_m/s'], m2['WD50m_deg'], data['Spd80mN'], data['Dir78mS'],
                                         averaging_prd='1H', direction_bin_array=[0,90,130,200,360])
            ss_cor.run()

        """
        CorrelBase.__init__(self, ref_spd, target_spd, averaging_prd, coverage_threshold, ref_dir=ref_dir,
                            target_dir=target_dir, sectors=sectors, direction_bin_array=direction_bin_array)

        if lt_ref_speed is None:
            self.lt_ref_speed = momm(self.data[self._ref_spd_col_name])
        else:
            self.lt_ref_speed = lt_ref_speed
        self.cutoff = min(0.5 * self.lt_ref_speed, 4.0)
        self.ref_veer_cutoff = self._get_veer_cutoff(self.data[self._ref_spd_col_name])
        self.target_veer_cutoff = self._get_veer_cutoff((self.data[self._tar_spd_col_name]))
        self._randomize_calm_periods()
        self._get_overall_veer()
        # for low ref_speed and high target_speed recalculate direction sector
        self._adjust_low_reference_speed_dir()

        self.speed_model = dict()

    def __repr__(self):
        return 'SpeedSort Model ' + str(self.params)

    def _randomize_calm_periods(self):
        idxs = self.data[self.data[self._ref_spd_col_name] < 1].index
        self.data.loc[idxs, self._ref_dir_col_name] = 360.0 * np.random.random(size=len(idxs))
        idxs = self.data[self.data[self._tar_spd_col_name] < 1].index
        self.data.loc[idxs, self._tar_dir_col_name] = 360.0 * np.random.random(size=len(idxs))

    def _get_overall_veer(self):
        idxs = self.data[(self.data[self._ref_spd_col_name] >= self.ref_veer_cutoff) &
                         (self.data[self._tar_spd_col_name] >= self.target_veer_cutoff)].index
        self.overall_veer = self._get_veer(self.data.loc[idxs, self._ref_dir_col_name],
                                           self.data.loc[idxs, self._tar_dir_col_name]).mean()

    def _adjust_low_reference_speed_dir(self):
        idxs = self.data[(self.data[self._ref_spd_col_name] < 2) &
                         (self.data[self._tar_spd_col_name] > (self.data[self._ref_spd_col_name] + 4))].index

        self.data.loc[idxs, self._ref_dir_col_name] = (self.data.loc[idxs, self._tar_dir_col_name] -
                                                       self.overall_veer).apply(utils._range_0_to_360)

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
        sector_data = sector_data[(sector_data[self._ref_spd_col_name] >= self.ref_veer_cutoff) &
                                  (sector_data[self._tar_spd_col_name] >= self.target_veer_cutoff)]
        return {'average_veer': round(self._get_veer(sector_data[self._ref_dir_col_name],
                                                     sector_data[self._tar_dir_col_name]).mean(), 5),
                'num_pts_for_veer': len(sector_data[self._ref_dir_col_name])}

    def run(self, show_params=True):
        self.params = dict()
        self.params['ref_speed_cutoff'] = round(self.cutoff, 5)
        self.params['ref_veer_cutoff'] = round(self.ref_veer_cutoff, 5)
        self.params['target_veer_cutoff'] = round(self.target_veer_cutoff, 5)
        self.params['overall_average_veer'] = round(self.overall_veer, 5)
        for sector, group in pd.concat([self.data, self._ref_dir_bins],
                                       axis=1, join='inner').dropna().groupby(['ref_dir_bin']):
            # print('Processing sector:', sector)
            self.speed_model[sector] = SpeedSort.SectorSpeedModel(ref_spd=group[self._ref_spd_col_name],
                                                                  target_spd=group[self._tar_spd_col_name],
                                                                  cutoff=self.cutoff)
            self.params[sector] = {'slope': round(self.speed_model[sector].params['slope'], 5),
                                   'offset': round(self.speed_model[sector].params['offset'], 5),
                                   'target_speed_cutoff': round(self.speed_model[sector].target_cutoff, 5),
                                   'num_pts_for_speed_fit': self.speed_model[sector].data_pts,
                                   'num_total_pts': min(group.count()),
                                   'sector_min': self._dir_sector_min[sector - 1],
                                   'sector_max': self._dir_sector_max[sector - 1],
                                   }
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
        return self.plot_wind_directions()

    @staticmethod
    def _linear_interpolation(xa, xb, ya, yb, xc):
        m = (xc - xa) / (xb - xa)
        yc = (yb - ya) * m + ya
        return yc

    def _predict_dir(self, x_dir):

        x_dir = x_dir.dropna().rename('dir')

        sector_min = []
        sector_max = []
        if self.direction_bin_array is None:
            # First sector is centered at 0.
            step = 360/self.sectors
            veer_bins = list(map(float, np.arange(0, 360 + step, step)))
            for veer_bin in veer_bins:
                sector_min.append(offset_wind_direction(veer_bin, -float(step/2)))
                sector_max.append(offset_wind_direction(veer_bin, float(step/2)))

            sec_veers = np.empty(np.shape(veer_bins))
            sec_veers[:] = np.nan
            sec_veers = list(sec_veers)
            for key in self.params.keys():
                if type(key) is int:
                    if self.params[key]['sector_min'] in sector_min:
                        sec_veers[sector_min.index(self.params[key]['sector_min'])] = self.params[key]['average_veer']
            if (0 in veer_bins) and (360 in veer_bins):
                sec_veers[-1] = sec_veers[0]

        else:
            veer_bins = []
            sec_veers = []
            # Calculate middle point of each sector, as each sectoral veer is applied at the mid-point of the sector.
            for key in self.params.keys():
                if type(key) is int:
                    sec_veers.append(self.params[key]['average_veer'])
                    sector_min.append(self.params[key]['sector_min'])
                    sector_max.append(self.params[key]['sector_max'])
                    if self.params[key]['sector_min'] < self.params[key]['sector_max']:
                        veer_bins.append((self.params[key]['sector_min'] + self.params[key]['sector_max']) / 2)
                    else:
                        veer_bins.append(offset_wind_direction(self.params[key]['sector_max'],
                                                               float(360 - self.params[key]['sector_max']
                                                                     + self.params[key]['sector_min']) / 2))

            # If first sector is not centered at 0 and 0 and 360 are the extremes of the direction_bin_array
            # then the first and the last sectors are taken into account for deriving the veer as for code below.
            if (0 in self.direction_bin_array) and (360 in self.direction_bin_array):
                sec_veers.insert(0, self._linear_interpolation(0 - (360 - veer_bins[-1]), veer_bins[0],
                                                               sec_veers[-1], sec_veers[0], 0))
                sec_veers.append(self._linear_interpolation(veer_bins[-1], 360 + veer_bins[0],
                                                            sec_veers[-1], sec_veers[1], 360))
                veer_bins.insert(0, 0)
                veer_bins.append(360)
                sector_min.insert(0, sector_min[-1])
                sector_min.append(sector_min[0])
                sector_max.insert(0, sector_max[0])
                sector_max.append(sector_max[0])

        # The veer correction is derived linear interpolating the veer between two mid-points of near sectors.
        adjustment = x_dir.rename('adjustment').copy() * np.nan
        for i in range(1, len(veer_bins)):

            if np.isnan(sec_veers[i - 1]) and not np.isnan(sec_veers[i]):
                logic_sect_mid_min_sector = self._get_logic_dir_sector(ref_dir=x_dir,
                                                                       sector_min=sector_min[i],
                                                                       sector_max=veer_bins[i])
                if logic_sect_mid_min_sector.sum() > 0:
                    adjustment[logic_sect_mid_min_sector] = offset_wind_direction(
                                x_dir[logic_sect_mid_min_sector] * 0, sec_veers[i])

            if np.isnan(sec_veers[i]) and not np.isnan(sec_veers[i - 1]):
                logic_sect_mid_max_sector = self._get_logic_dir_sector(ref_dir=x_dir,
                                                                       sector_min=veer_bins[i],
                                                                       sector_max=sector_max[i])
                if logic_sect_mid_max_sector.sum() > 0:
                    adjustment[logic_sect_mid_max_sector] = offset_wind_direction(
                                x_dir[logic_sect_mid_max_sector] * 0, sec_veers[i])

            if i < len(veer_bins) - 1:
                if not np.isnan(sec_veers[i]) and (np.isnan(sec_veers[i - 1]) and np.isnan(sec_veers[i + 1])):
                    logic_sect_min_max_sector = self._get_logic_dir_sector(ref_dir=x_dir,
                                                                           sector_min=sector_min[i],
                                                                           sector_max=sector_max[i])
                    if logic_sect_min_max_sector.sum() > 0:
                        adjustment[logic_sect_min_max_sector] = offset_wind_direction(
                            x_dir[logic_sect_min_max_sector] * 0, sec_veers[i])

                elif not np.isnan(sec_veers[i]) and np.isnan(sec_veers[i + 1]):
                    logic_sect_mid_max_sector = self._get_logic_dir_sector(ref_dir=x_dir,
                                                                           sector_min=veer_bins[i],
                                                                           sector_max=sector_max[i])
                    if logic_sect_mid_max_sector.sum() > 0:
                        adjustment[logic_sect_mid_max_sector] = offset_wind_direction(
                            x_dir[logic_sect_mid_max_sector] * 0, sec_veers[i])

            elif (sector_min[i] == sector_min[0]) and np.isnan(sec_veers[1]) and not np.isnan(sec_veers[i]):
                logic_sect_min_max_sector = self._get_logic_dir_sector(ref_dir=x_dir,
                                                                       sector_min=veer_bins[i],
                                                                       sector_max=sector_max[i])
                if logic_sect_min_max_sector.sum() > 0:
                    adjustment[logic_sect_min_max_sector] = offset_wind_direction(
                        x_dir[logic_sect_min_max_sector] * 0, sec_veers[i])

            logic_sect_mid_point = self._get_logic_dir_sector(ref_dir=x_dir,
                                                              sector_min=veer_bins[i - 1],
                                                              sector_max=veer_bins[i])

            if logic_sect_mid_point.sum() != 0 and not (np.isnan(sec_veers[i]) or np.isnan(sec_veers[i - 1])):
                adjustment[logic_sect_mid_point] = self._linear_interpolation(veer_bins[i - 1], veer_bins[i],
                                                                              sec_veers[i - 1], sec_veers[i],
                                                                              x_dir[logic_sect_mid_point])

        return offset_wind_direction(x_dir, adjustment).sort_index()

    def _predict(self, x_spd, x_dir):
        x = pd.concat([x_spd.rename('spd'),
                       _binned_direction_series(x_dir, self.sectors,
                                                direction_bin_array=self.direction_bin_array).rename('ref_dir_bin')],
                      axis=1, join='inner').dropna()
        prediction = pd.Series().rename('spd')
        for sector, data in x.groupby(['ref_dir_bin']):
            if sector in list(self.speed_model.keys()):
                prediction_spd = self.speed_model[sector].sector_predict(data['spd'])
            else:
                prediction_spd = data['spd'] * np.nan

            prediction = pd.concat([prediction, prediction_spd], axis=0)

        return prediction.sort_index()

    def synthesize(self, input_spd=None, input_dir=None):

        if input_spd is None and input_dir is None:
            ref_start_date, target_start_date = self._get_synth_start_dates()

            output = self._predict(tf.average_data_by_period(self.ref_spd[ref_start_date:], self.averaging_prd,
                                                             return_coverage=False),
                                   tf.average_data_by_period(self.ref_dir[ref_start_date:], self.averaging_prd,
                                                             wdir_column_names=self._ref_dir_col_name,
                                                             return_coverage=False))
            output = tf.average_data_by_period(self.target_spd[target_start_date:], self.averaging_prd,
                                               return_coverage=False).combine_first(output)
            dir_output = self._predict_dir(tf.average_data_by_period(self.ref_dir[ref_start_date:], self.averaging_prd,
                                                                     wdir_column_names=self._ref_dir_col_name,
                                                                     return_coverage=False))

        else:
            output = self._predict(input_spd, input_dir)
            dir_output = self._predict_dir(input_dir)
        output[output < 0] = 0
        return pd.concat([output.rename(self._tar_spd_col_name + "_Synthesized"),
                          dir_output.rename(self._tar_dir_col_name + "_Synthesized")], axis=1, join='inner')

    def plot_wind_directions(self):
        """
        Plots reference and target directions in a scatter plot
        """
        return plot_scatter_wdir(
            self.data[self._ref_dir_col_name][(self.data[self._ref_spd_col_name] > self.cutoff) &
                                              (self.data[self._tar_spd_col_name] > self.cutoff)],
            self.data[self._tar_dir_col_name][(self.data[self._ref_spd_col_name] > self.cutoff) &
                                              (self.data[self._tar_spd_col_name] > self.cutoff)],
            x_label=self._ref_dir_col_name, y_label=self._tar_dir_col_name)


class SVR:
    def __init__(self, ref_spd, target_spd, averaging_prd, coverage_threshold, bw_model=0, **sklearn_args):
        raise NotImplementedError
    #     CorrelBase.__init__(self, ref_spd, target_spd, averaging_prd, coverage_threshold)
    #     bw_models = [{'kernel': 'rbf', 'C': 30, 'gamma': 0.01}, {'kernel': 'linear', 'C': 10}]
    #     self.model = sklearn_SVR(**{**bw_models[bw_model], **sklearn_args})
    #
    # def __repr__(self):
    #     return 'Support Vector Regression Model ' + str(self.params)
    #
    # def run(self, show_params=True):
    #     if len(self.data[self._ref_spd_col_name].values.shape) == 1:
    #         x = self.data[self._ref_spd_col_name].values.reshape(-1, 1)
    #     else:
    #         x = self.data[self._ref_spd_col_name].values
    #     self.model.fit(x, self.data[self._tar_spd_col_name].values.flatten())
    #     self.params = dict()
    #     self.params['RMSE'] = -1 * sklearn_cross_val_score(self.model, x,
    #                                                        self.data[self._tar_spd_col_name].values.flatten(),
    #                                                        scoring='neg_mean_squared_error', cv=3)
    #     self.params['MAE'] = -1 * sklearn_cross_val_score(self.model, x,
    #                                                       self.data[self._tar_spd_col_name].values.flatten(),
    #                                                       scoring='neg_mean_absolute_error', cv=3)
    #     self.params['Explained Variance'] = -1 * sklearn_cross_val_score(self.model, x,
    #                                                                      self.data[self._tar_spd_col_name].values.flatten(),
    #                                                                      scoring='explained_variance', cv=3)
    #     if show_params:
    #         self.show_params()
    #
    # def _predict(self, x):
    #     if isinstance(x, pd.Series):
    #         X = x.values.reshape(-1, 1)
    #         return pd.DataFrame(data=self.model.predict(X), index=x.index)
    #     elif isinstance(x, pd.DataFrame):
    #         X = x.values
    #         return pd.DataFrame(data=self.model.predict(X), index=x.index)
    #     else:
    #         if not len(x.shape) == 2:
    #             raise ValueError("Expected shape of input data (num of data points, number of reference datasets), "
    #                              "but found ", x.shape)
    #         else:
    #             return self.model.predict(x)
    #
    # def plot(self, title=""):
    #     """For plotting"""
    #     plot_scatter(self.data[self._ref_spd_col_name],
    #                  self.data[self._tar_spd_col_name],
    #                  self._predict(self.data[self._ref_spd_col_name]), trendline_dots=True,
    #                  x_label=self._ref_spd_col_name, y_label=self._tar_spd_col_name)
