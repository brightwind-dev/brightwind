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

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import calendar
import numpy as np
import pandas as pd
import os
from brightwind.utils import utils
from pandas.plotting import register_matplotlib_converters
from brightwind.utils.utils import _convert_df_to_series
import re
import six

register_matplotlib_converters()

__all__ = ['plot_timeseries',
           'plot_scatter',
           'plot_scatter_wspd',
           'plot_scatter_wdir']
#
# try:
#     if 'Gotham Rounded' in \
#             [mpl.font_manager.FontProperties(fname=i).get_name() for i in mpl.font_manager.findSystemFonts()]:
#         mpl.rcParams['font.family'] = 'Gotham Rounded'
# except Exception as ex:
#     raise 'Found exception when checking installed fonts. {}'.format(str(ex))
#
plt.style.use(os.path.join(os.path.dirname(__file__), 'bw.mplstyle'))


class _ColorPalette:

    def __init__(self):
        """
        Color palette to be used for plotting graphs and tables. Colors can be reset by using

        ::
            import brightwind as bw
            bw.analyse.plot.COLOR_PALETTE.primary = '#3366CC'

        Color are called 'primary', secondary', 'tertiary', etc. Lighter and darker shades of primary are called
        'primary_10' for 10% of primary. Gradient goes from 0% (darkest) to 100% (lightest). See
        https://www.w3schools.com/colors/colors_picker.asp for more info.
        """
        self.primary = '#9CC537'        # slightly darker than YellowGreen #9acd32, rgb[156/255, 197/255, 55/255]
        self.secondary = '#2E3743'      # asphalt, rgb[46/255, 55/255, 67/255]
        self.tertiary = '#9B2B2C'       # red'ish, rgb(155, 43, 44)
        self.fourth = '#E57925'         # orange'ish, rgb(229, 121, 37)
        self.fifth = '#F2D869'          # yellow'ish, rgb(242, 216, 105)
        self.sixth = '#AB8D60'
        self.seventh = '#A4D29F'
        self.eighth = '#6E807B'
        self.ninth = '#3D636F'          # blue grey
        self.tenth = '#A49E9D'
        self.eleventh = '#DA9BA6'
        self.primary_10 = '#1F290A'     # darkest green, 10% of primary
        self.primary_35 = '#6C9023'     # dark green, 35% of primary
        self.primary_80 = '#D7EBAD'     # light green, 80% of primary
        self.primary_90 = '#ebf5d6'     # light green, 90% of primary
        self.primary_95 = '#F5FAEA'     # lightest green, 95% of primary
        self.secondary_70 = '#6d737b'   # light asphalt

        _col_map_colors = [self.primary_95,  # lightest primary
                           self.primary,     # primary
                           self.primary_10]  # darkest primary
        self._color_map = self._set_col_map(_col_map_colors)

        self.color_list = [self.primary, self.secondary, self.tertiary, self.fourth, self.fifth, self.sixth,
                           self.seventh, self.eighth, self.ninth, self.tenth, self.eleventh, self.primary_35]

        # set the mpl color cycler to our colors. It has 10 colors
        # mpl.rcParams['axes.prop_cycle']

    @staticmethod
    def _set_col_map(col_map_colors):
        return LinearSegmentedColormap.from_list('color_map', col_map_colors, N=256)

    @property
    def color_map(self):
        return self._color_map

    @color_map.setter
    def color_map(self, col_map_colors):
        self._color_map = self._set_col_map(col_map_colors)


COLOR_PALETTE = _ColorPalette()


def plot_monthly_means(data, coverage=None, ylbl=''):
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    if len(data.shape) > 1:
        ax.set_prop_cycle(color=COLOR_PALETTE.color_list)
        ax.plot(data, '-o')
        ax.legend(list(data.columns))
    else:
        ax.plot(data, '-o', color=COLOR_PALETTE.primary)
        ax.legend([data.name])
    ax.set_ylabel(ylbl)

    ax.set_xticks(data.index)
    ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=20, ha='center')

    if coverage is not None:
        plot_coverage = True
        if len(coverage.shape) > 1:
            if coverage.shape[1] > 1:
                plot_coverage = False
        if plot_coverage:
            ax.clear()
            ax.plot(data, '-o', color=COLOR_PALETTE.secondary)
            ax2 = ax.twinx()

            for month, coverage in zip(coverage.index, coverage.values):
                ax2.imshow(np.array([[mpl.colors.to_rgb(COLOR_PALETTE.primary)],
                                     [mpl.colors.to_rgb(COLOR_PALETTE.primary_80)]]),
                           interpolation='gaussian', extent=(mdates.date2num(month - pd.Timedelta('10days')),
                                                             mdates.date2num(month + pd.Timedelta('10days')),
                                                             0, coverage), aspect='auto', zorder=1)
                ax2.bar(mdates.date2num(month), coverage, edgecolor=COLOR_PALETTE.secondary, linewidth=0.3,
                        fill=False, zorder=0)

            ax2.set_ylim(0, 1)
            ax.set_ylim(bottom=0)
            ax.set_xlim(data.index[0] - pd.Timedelta('20days'), data.index[-1] + pd.Timedelta('20days'))
            ax.set_zorder(3)
            ax2.yaxis.grid(True)
            ax2.set_axisbelow(True)
            ax.patch.set_visible(False)
            ax2.set_ylabel('Coverage [-]')
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")
            plt.close()
            return ax2.get_figure()
    plt.close()
    return ax.get_figure()


def plot_timeseries(data, date_from='', date_to='', y_limits=(None, None)):
    """
    Plot a timeseries of data.

    :param data: Data in the form of a Pandas DataFrame/Series to plot.
    :type data: pd.DataFrame, pd.Series
    :param date_from: Start date used for plotting, if not specified the first timestamp of data is considered. Should
        be in yyyy-mm-dd format
    :type date_from: str
    :param date_to: End date used for plotting, if not specified last timestamp of data is considered. Should
        be in yyyy-mm-dd format
    :type date_to: str
    :param y_limits: y-axis min and max limits. Default is (None, None).
    :type y_limits: tuple, None
    :return: Timeseries plot
    :rtype: matplotlib.figure.Figure

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)

        # To plot few variables
        bw.plot_timeseries(data[['Spd40mN', 'Spd60mS', 'T2m']])

        # To set a start date
        bw.plot_timeseries(data.Spd40mN, date_from='2017-09-01')

        # To set an end date
        bw.plot_timeseries(data.Spd40mN, date_to='2017-10-01')

        # For specifying a slice
        bw.plot_timeseries(data.Spd40mN, date_from='2017-09-01', date_to='2017-10-01')

        # To set the y-axis minimum to 0
        bw.plot_timeseries(data.Spd40mN, date_from='2017-09-01', date_to='2017-10-01', y_limits=(0, None))

        # To set the y-axis maximum to 25
        bw.plot_timeseries(data.Spd40mN, date_from='2017-09-01', date_to='2017-10-01', y_limits=(0, 25))

    """
    plt.rcParams['figure.figsize'] = (15, 8)    # ** this might be setting the global size which isn't good practice ***
    if isinstance(data, pd.Series):
        data_to_slice = data.copy(deep=False).to_frame()
    else:
        data_to_slice = data.copy()
    sliced_data = utils.slice_data(data_to_slice, date_from, date_to)
    figure = sliced_data.plot(color=COLOR_PALETTE.color_list).get_figure()
    if y_limits is not None:
        figure.axes[0].set_ylim(y_limits)
    plt.close()
    return figure


def _scatter_plot(x, y, predicted_y=None, x_label="Reference", y_label="Target", prediction_marker='-'):
    """
    Plots a scatter plot.
    :param x:
    :param y:
    :param predicted_y: A series of predicted y values after applying the correlation to the x series.
    :param x_label:
    :param y_label:
    :param prediction_marker:
    :return:
    """
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    no_dots = len(x)

    marker_size_max = 216
    marker_size_min = 18
    marker_size = -0.2 * no_dots + marker_size_max  # y=mx+c, m = (216 - 18) / (1000 - 0) i.e. slope changes up to 1000
    marker_size = marker_size_min if marker_size < marker_size_min else marker_size

    max_alpha = 0.7
    min_alpha = 0.3
    alpha = -0.0004 * no_dots + max_alpha  # y=mx+c, m = (0.7 - 0.3) / (1000 - 0) i.e. alpha changes up to 1000 dots
    alpha = min_alpha if alpha < min_alpha else alpha

    ax.scatter(x, y, marker='o', color=COLOR_PALETTE.primary, s=marker_size, alpha=alpha, edgecolors='none')
    fig.set_figwidth(10)
    fig.set_figheight(10.2)
    if predicted_y is not None:
        ax.plot(x, predicted_y, prediction_marker, color=COLOR_PALETTE.secondary)
        ax.legend(['Predicted', 'Original'])
    plt.close()
    return ax.get_figure()


def plot_scatter(x_series, y_series, x_axis_title=None, y_axis_title=None,
                 x_limits=None, y_limits=None):
    """
    Plots a scatter plot of two variable's timeseries.

    :param x_series: The x-axis values or reference variable.
    :type x_series: pd.Series
    :param y_series: The y-axis values or target variable.
    :type y_series: pd.Series
    :param x_axis_title: Title for the x-axis. If None, title will be taken from x_series name.
    :type x_axis_title: str, None
    :param y_axis_title: Title for the y-axis. If None, title will be taken from y_series name.
    :type y_axis_title: str, None
    :param x_limits: x-axis min and max limits.
    :type x_limits: tuple, None
    :param y_limits: y-axis min and max limits.
    :type y_limits: tuple, None
    :return: scatter plot
    :rtype: matplotlib.figure.Figure

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)

        # To plot two variables against each other
        bw.plot_scatter(data.Spd80mN, data.Spd80mS)

        # To overwrite the default axis titles.
        bw.plot_scatter(data.Dir78mS, data.Dir58mS, x_axis_title='Dir78mS', y_axis_title='Dir58mS')

        # To set the x and y axis limits by using a tuple.
        bw.plot_scatter(data.Dir78mS, data.Dir58mS, x_axis_title='Dir78mS', y_axis_title='Dir58mS',
                        x_limits=(50,300), y_limits=(250,300))

    """
    x_series = _convert_df_to_series(x_series)
    y_series = _convert_df_to_series(y_series)
    if x_axis_title is None:
        x_axis_title = x_series.name
    if y_axis_title is None:
        y_axis_title = y_series.name

    merged_df = pd.concat([x_series, y_series], join='inner', axis=1)
    scat_plot = _scatter_plot(merged_df[x_series.name], merged_df[y_series.name],
                              x_label=x_axis_title, y_label=y_axis_title)

    if x_limits is None:
        x_limits = (round(x_series.min() - 0.5), -(-x_series.max() // 1))
    if y_limits is None:
        y_limits = (round(y_series.min() - 0.5), -(-y_series.max() // 1))
    scat_plot.axes[0].set_xlim(x_limits[0], x_limits[1])
    scat_plot.axes[0].set_ylim(y_limits[0], y_limits[1])
    return scat_plot


def plot_scatter_wdir(x_wdir_series, y_wdir_series, x_axis_title=None, y_axis_title=None,
                      x_limits=(0, 360), y_limits=(0, 360)):
    """
    Plots a scatter plot of two wind direction timeseries and adds a line from 0,0 to 360,360.

    :param x_wdir_series: The x-axis values or reference wind directions.
    :type x_wdir_series: pd.Series
    :param y_wdir_series: The y-axis values or target wind directions.
    :type y_wdir_series: pd.Series
    :param x_axis_title: Title for the x-axis. If None, title will be taken from x_wdir_series name.
    :type x_axis_title: str, None
    :param y_axis_title: Title for the y-axis. If None, title will be taken from y_wdir_series name.
    :type y_axis_title: str, None
    :param x_limits: x-axis min and max limits.
    :type x_limits: tuple
    :param y_limits: y-axis min and max limits.
    :type y_limits: tuple
    :return: scatter plot
    :rtype: matplotlib.figure.Figure

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)

        #To plot few variables
        bw.plot_scatter_wdir(data.Dir78mS, data.Dir58mS)

        #To overwrite the default axis titles.
        bw.plot_scatter_wdir(data.Dir78mS, data.Dir58mS, x_axis_title='Wind direction at 78m',
                             y_axis_title='Wind direction at 58m')

        #To set the x and y axis limits by using a tuple.
        bw.plot_scatter_wdir(data.Dir78mS, data.Dir58mS, x_axis_title='Reference', y_axis_title='Target',
                             x_limits=(50,300), y_limits=(250,300))

    """
    if x_axis_title is None:
        x_axis_title = x_wdir_series.name + ' [°]'
    if y_axis_title is None:
        y_axis_title = y_wdir_series.name + ' [°]'
    scat_plot = plot_scatter(x_wdir_series, y_wdir_series, x_axis_title=x_axis_title, y_axis_title=y_axis_title,
                             x_limits=x_limits, y_limits=y_limits)
    x = [0, 360]
    y = [0, 360]
    scat_plot.axes[0].plot(x, y, '-', color=COLOR_PALETTE.secondary)
    scat_plot.axes[0].legend(['Reference line', 'Data points'])
    return scat_plot


def plot_scatter_wspd(x_wspd_series, y_wspd_series, x_axis_title=None, y_axis_title=None,
                      x_limits=(0, 30), y_limits=(0, 30)):
    """
    Plots a scatter plot of two wind speed timeseries and adds a reference line from 0,0 to 40,40. This should
    only be used for wind speeds in m/s and not when one of the wind speed series is normalised. Please use the
    basic 'plot_scatter()' function when using normalised wind speeds.

    :param x_wspd_series: The x-axis values or reference wind speeds.
    :type x_wspd_series: pd.Series
    :param y_wspd_series: The y-axis values or target wind speeds.
    :type y_wspd_series: pd.Series
    :param x_axis_title: Title for the x-axis. If None, title will be taken from x_wspd_series name.
    :type x_axis_title: str, None
    :param y_axis_title: Title for the y-axis. If None, title will be taken from y_wspd_series name.
    :type y_axis_title: str, None
    :param x_limits: x-axis min and max limits. Can be set to None to let the code derive the min and max from
                     the x_wspd_series.
    :type x_limits: tuple, None
    :param y_limits: y-axis min and max limits. Can be set to None to let the code derive the min and max from
                     the y_wspd_series.
    :type y_limits: tuple, None
    :return: scatter plot
    :rtype: matplotlib.figure.Figure

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)

        # To plot two wind speeds against each other
        bw.plot_scatter_wspd(data.Spd80mN, data.Spd80mS)

        # To overwrite the default axis titles.
        bw.plot_scatter_wspd(data.Spd80mN, data.Spd80mS, x_axis_title='Speed at 80m North',
                             y_axis_title='Speed at 80m South')

        # To set the x and y axis limits by using a tuple.
        bw.plot_scatter_wspd(data.Spd80mN, data.Spd80mS, x_axis_title='Speed at 80m North',
                             y_axis_title='Speed at 80m South', x_limits=(0,25), y_limits=(0,25))

    """
    if x_axis_title is None:
        x_axis_title = x_wspd_series.name + ' [m/s]'
    if y_axis_title is None:
        y_axis_title = y_wspd_series.name + ' [m/s]'
    scat_plot = plot_scatter(x_wspd_series, y_wspd_series, x_axis_title=x_axis_title, y_axis_title=y_axis_title,
                             x_limits=x_limits, y_limits=y_limits)
    x = [0, 40]
    y = [0, 40]
    scat_plot.axes[0].plot(x, y, '-', color=COLOR_PALETTE.secondary)
    scat_plot.axes[0].legend(['Reference line', 'Data points'])
    return scat_plot


def plot_freq_distribution(data, max_y_value=None, x_tick_labels=None, x_label=None, y_label=None):
    from matplotlib.ticker import PercentFormatter
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if isinstance(data.index[0], pd.Interval):
        x_data = [i.mid for i in data.index]
    else:
        x_data = data.index
    ax.set_xticks(x_data)
    ax.set_xlim(x_data[0] - 0.5, x_data[-1] + 0.5)
    if x_tick_labels is not None:
        ax.set_xticklabels(x_tick_labels)
    if max_y_value is None:
        ax.set_ylim(0, data.max() * 1.1)
    else:
        ax.set_ylim(0, max_y_value)
    if y_label[0] == '%':
        ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(b=True, axis='y', zorder=0)
    for frequency, ws_bin in zip(data, x_data):
        ax.imshow(np.array([[mpl.colors.to_rgb(COLOR_PALETTE.primary)], [mpl.colors.to_rgb(COLOR_PALETTE.primary_80)]]),
                  interpolation='gaussian', extent=(ws_bin - 0.4, ws_bin + 0.4, 0, frequency), aspect='auto', zorder=3)
        ax.bar(ws_bin, frequency, edgecolor=COLOR_PALETTE.primary_35, linewidth=0.3, fill=False, zorder=5)
    plt.close()
    return ax.get_figure()


def plot_rose(ext_data, plot_label=None):
    """
    Plot a wind rose from data by dist_by_dir_sector
    """
    result = ext_data.copy(deep=False)
    sectors = len(result)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 360.0/sectors))
    sector_mid_points = []
    widths = []
    for i in result.index:
        angular_pos_start = (np.pi / 180.0) * float(i.split('-')[0])
        angular_pos_end = (np.pi / 180.0) * float(i.split('-')[-1])
        if angular_pos_start < angular_pos_end:
            sector_mid_points.append((angular_pos_start+angular_pos_end)/2.0)
            widths.append(angular_pos_end-angular_pos_start - (np.pi / 180))
        else:
            sector_mid_points.append((np.pi + (angular_pos_start + angular_pos_end)/2.0) % 360)
            widths.append(2*np.pi - angular_pos_start + angular_pos_end - (np.pi / 180))
    max_contour = (ext_data.max() + ext_data.std())
    contour_spacing = max_contour / 10
    num_digits_to_round = 0
    while contour_spacing*(10**num_digits_to_round) <= 1:
        num_digits_to_round += 1
    if 0.5 < contour_spacing < 1:
        contour_spacing = 1
    levels = np.arange(0, max_contour, round(contour_spacing, num_digits_to_round))
    ax.set_rgrids(levels, labels=[str(i) for i in levels], angle=0)
    ax.bar(sector_mid_points, result, width=widths, bottom=0.0, color=COLOR_PALETTE.primary,
           edgecolor=[COLOR_PALETTE.primary_35 for i in range(len(result))], alpha=0.8)
    ax.legend([plot_label])
    plt.close()
    return ax.get_figure()


def plot_rose_with_gradient(freq_table, percent_symbol=True, plot_bins=None, plot_labels=None):
    table = freq_table.copy()
    sectors = len(table.columns)
    table_trans = table.T
    if plot_bins is not None:
        rows_to_sum = []
        intervals = [pd.Interval(plot_bins[i], plot_bins[i + 1], closed=table.index[0].closed)
                     for i in range(len(plot_bins) - 1)]
        bin_assigned = []
        for interval in intervals:
            row_group = []
            for var_bin, pos in zip(table.index, range(len(table.index))):
                if var_bin.overlaps(interval) and not (pos in bin_assigned):
                    bin_assigned.append(pos)
                    row_group.append(pos)
            rows_to_sum.append(row_group)
    else:
        if len(table.index) > 6:
            rows_to_sum = []
            num_rows = len(table.index) // 6
            ctr = 0
            while ctr < len(table.index) - (len(table.index) % 6):
                rows_to_sum.append(list(range(ctr, ctr + num_rows)))
                ctr += num_rows
            rows_to_sum[-1].extend(list(range(len(table.index) - (len(table.index) % 6), len(table.index))))

        else:
            rows_to_sum = [[i] for i in range(len(table.index))]

    table_binned = pd.DataFrame()
    bin_labels = []
    group = 0
    for i in rows_to_sum:
        bin_labels.append(str(table.index[i[0]].left) + ' - ' + str(table.index[i[-1]].right))
        to_concat = table_trans.iloc[:, i].sum(axis=1).rename(group)
        group += 1
        table_binned = pd.concat([table_binned, to_concat], axis=1, sort=True)
    table_binned = table_binned.T
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 360.0 / sectors), zorder=2)

    if percent_symbol:
        symbol = '%'
    else:
        symbol = ' '
    max_contour = max(table.sum(axis=0)) + table.sum(axis=0).std()
    contour_spacing = max_contour / 10
    num_digits_to_round = 0
    while contour_spacing * (10 ** num_digits_to_round) < 1:
        num_digits_to_round += 1
    if 0.5 < contour_spacing < 1:
        contour_spacing = 1
    levels = np.arange(0, max_contour, round(contour_spacing, num_digits_to_round))
    ax.set_rgrids(levels,
                  labels=[str(i) + symbol for i in levels],
                  angle=0, zorder=2)
    ax.set_ylim(0, max(table.sum(axis=0)) + 3.0)
    ax.bar(0, 1, alpha=0)
    norm = mpl.colors.Normalize(vmin=min(table_binned.index), vmax=max(table_binned.index), clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=COLOR_PALETTE.color_map)
    for column in table_binned:
        radial_pos = 0.0
        angular_pos_start = (np.pi / 180.0) * float(column.split('-')[0])
        angular_pos_end = (np.pi / 180.0) * float(column.split('-')[-1])
        # Check for sectors with 0 degrees within the sector
        if angular_pos_end > angular_pos_start:
            angular_width = angular_pos_end - angular_pos_start - (np.pi / 180)  # Leaving 1 degree gap
        else:
            angular_width = 2 * np.pi - angular_pos_start + angular_pos_end - (np.pi / 180)
        for speed_bin, frequency in zip(table_binned.index, table_binned[column]):
            patch = mpl.patches.Rectangle((angular_pos_start, radial_pos), angular_width,
                                          frequency, facecolor=mapper.to_rgba(speed_bin),
                                          edgecolor=COLOR_PALETTE.primary_35,
                                          linewidth=0.3, zorder=3)
            ax.add_patch(patch)
            radial_pos += frequency

    if plot_labels is None:
        plot_labels = [mpl.patches.Patch(color=mapper.to_rgba(table_binned.index[i]), label=bin_labels[i]) for i in
                       range(len(bin_labels))]
    else:
        plot_labels = [mpl.patches.Patch(color=mapper.to_rgba(table_binned.index[i]), label=plot_labels[i]) for i in
                       range(len(plot_labels))]
    ax.legend(handles=plot_labels)
    plt.close()
    return ax.get_figure()


def plot_TI_by_speed(wspd, wspd_std, ti, IEC_class=None):
    """
    Plot turbulence intensity graphs alongside with IEC standards
    :param wspd:
    :param wspd_std:
    :param ti: DataFrame returned from TI.by_speed() in analyse
    :param IEC_class: By default IEC class 2005 is used for custom class pass a DataFrame. Note we have removed
        option to include IEC Class 1999 as no longer appropriate.
        This may need to be placed in a separate function when updated IEC standard is released
    :return: Plots turbulence intensity distribution by wind speed
    """

    # IEC Class 2005

    if IEC_class is None:
        IEC_class = pd.DataFrame(np.zeros([26, 4]), columns=['Windspeed', 'IEC Class A', 'IEC Class B', 'IEC Class C'])
        for n in range(1, 26):
            IEC_class.iloc[n, 0] = n
            IEC_class.iloc[n, 1] = 0.16 * (0.75 + (5.6 / n))
            IEC_class.iloc[n, 2] = 0.14 * (0.75 + (5.6 / n))
            IEC_class.iloc[n, 3] = 0.12 * (0.75 + (5.6 / n))
    common_idxs = wspd.index.intersection(wspd_std.index)
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.scatter(wspd.loc[common_idxs], wspd_std.loc[common_idxs] / wspd.loc[common_idxs],
               color=COLOR_PALETTE.primary, alpha=0.3, marker='.')
    ax.plot(ti.index.values, ti.loc[:, 'Mean_TI'].values, color=COLOR_PALETTE.secondary, label='Mean_TI')
    ax.plot(ti.index.values, ti.loc[:, 'Rep_TI'].values, color=COLOR_PALETTE.primary_35, label='Rep_TI')
    ax.plot(IEC_class.iloc[:, 0], IEC_class.iloc[:, 1], color=COLOR_PALETTE.tertiary, linestyle='dashed',
            label=IEC_class.columns[1])
    ax.plot(IEC_class.iloc[:, 0], IEC_class.iloc[:, 2], color=COLOR_PALETTE.fourth, linestyle='dashed',
            label=IEC_class.columns[2])
    ax.plot(IEC_class.iloc[:, 0], IEC_class.iloc[:, 3], color=COLOR_PALETTE.fifth, linestyle='dashed',
            label=IEC_class.columns[3])
    ax.set_xlim(3, 25)
    ax.set_ylim(0, 0.6)
    ax.set_xticks(np.arange(3, 26, 1))
    ax.set_xlabel('Wind speed [m/s]')
    ax.set_ylabel('Turbulence Intensity')
    ax.grid(True)
    ax.legend()
    plt.close()
    return ax.get_figure()


def plot_TI_by_sector(turbulence, wdir, ti):
    radians = np.radians(utils._get_dir_sector_mid_pts(ti.index))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(utils._get_dir_sector_mid_pts(ti.index))
    ax.plot(np.append(radians, radians[0]), ti.append(ti.iloc[0])['Mean_TI'], color=COLOR_PALETTE.primary, linewidth=4,
            figure=fig)
    maxlevel = ti['Mean_TI'].max() + 0.1
    ax.set_ylim(0, maxlevel)
    ax.scatter(np.radians(wdir), turbulence, color=COLOR_PALETTE.secondary, alpha=0.3, s=1)
    ax.legend(loc=8, framealpha=1)
    plt.close()
    return ax.get_figure()


def plot_shear_by_sector(scale_variable, wind_rose_data, calc_method='power_law'):

    result = wind_rose_data.copy(deep=False)
    radians = np.radians(utils._get_dir_sector_mid_pts(scale_variable.index))
    sectors = len(result)
    fig = plt.figure(figsize=(12, 12),)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    bin_edges = pd.Series([])
    for i in range(sectors):
        bin_edges[i] = float(re.findall(r"[-+]?\d*\.\d+|\d+", wind_rose_data.index[i])[0])
        if i == sectors - 1:
            bin_edges[i + 1] = abs(float(re.findall(r"[-+]?\d*\.\d+|\d+", wind_rose_data.index[i])[1]))
    label = ''
    if calc_method == 'power_law':
        label = 'Mean_Shear'
    if calc_method == 'log_law':
        label = 'Mean_Roughness_Coefficient'

    scale_variable_y = np.append(scale_variable, scale_variable[0])
    plot_x = np.append(radians, radians[0])
    scale_to_fit = max(scale_variable) / max(result/100)
    wind_rose_r = (result/100) * scale_to_fit
    bin_edges = np.array(bin_edges)
    width = pd.Series([])

    for i in range(len(bin_edges) - 1):
        if bin_edges[i + 1] == 0:
            width[i] = 2 * np.pi * (360 - bin_edges[i]) / 360 - (np.pi / 180)
        elif bin_edges[i + 1] > bin_edges[i]:
            width[i] = 2 * np.pi * ((bin_edges[i + 1] - bin_edges[i]) / 360) - (np.pi / 180)
        else:
            width[i] = 2 * np.pi * (((360 + bin_edges[i + 1]) - bin_edges[i]) / 360) - (np.pi / 180)

    ax.bar(radians, wind_rose_r, width=width, color=COLOR_PALETTE.secondary, align='center',
           edgecolor=[COLOR_PALETTE.secondary for i in range(len(result))],
           alpha=0.8, label='Wind_Directional_Frequency')

    maxlevel = (max(scale_variable_y)) + max(scale_variable_y) * .1
    ax.set_thetagrids(radians*180/np.pi)
    ax.plot(plot_x, scale_variable_y, color=COLOR_PALETTE.primary, linewidth=4, label=label)
    ax.set_ylim(0, top=maxlevel)
    ax.legend(loc=8, framealpha=1)

    return ax.get_figure()


def plot_12x24_contours(tab_12x24, label=('Variable', 'mean'), plot=None):
    """
    Get Contour Plot of 12 month x 24 hour matrix of variable
    :param tab_12x24: DataFrame returned from get_12x24() in analyse
    :param label: Label of the colour bar on the plot.
    :return: 12x24 figure
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    x = ax.contourf(tab_12x24.columns, tab_12x24.index, tab_12x24.values, cmap=COLOR_PALETTE.color_map)
    cbar = fig.colorbar(x)
    cbar.ax.set_ylabel(label[1].capitalize() + " of " + label[0])
    ax.set_xlabel('Month of Year')
    ax.set_ylabel('Hour of Day')
    month_names = calendar.month_abbr[1:13]
    ax.set_xticks(tab_12x24.columns)
    ax.set_xticklabels([month_names[i - 1] for i in tab_12x24.columns])
    ax.set_yticks(np.arange(0, 24, 1))
    if plot is None:
        plt.close()
    return ax.get_figure()


def plot_sector_ratio(sec_ratio, wdir, sec_ratio_dist, col_names, boom_dir_1=-1, boom_dir_2=-1):
    """
    Accepts a DataFrame table, along with 2 anemometer names, and one wind vane name and plots the speed ratio
    by sector. Optionally can include anemometer boom directions also.
    :param sec_ratio: Series of sector_ratios
    :type sec_ratio: pandas.Series
    :param wdir: Direction series
    :type wdir: pandas.Series
    :param sec_ratio_dist: DataFrame from SectorRatio.by_sector()
    :type sec_ratio_dist; pandas.Series
    :param boom_dir_1: Boom direction in degrees of speed_col_name_1. Defaults to -1.
    :type boom_dir_1: float
    :param boom_dir_2: Boom direction in degrees of speed_col_name_2. Defaults to -1.
    :type boom_dir_2: float
    :param col_names: A list of strings containing column names of wind speeds, first string is divisor and second is
        dividend
    :type col_names: list(str)
    :returns A speed ratio plot showing average speed ratio by sector and scatter of individual data points.

    """
    radians = np.radians(utils._get_dir_sector_mid_pts(sec_ratio_dist.index))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(utils._get_dir_sector_mid_pts(sec_ratio_dist.index))
    ax.plot(np.append(radians, radians[0]), sec_ratio_dist['Mean_Sector_Ratio'].append(sec_ratio_dist.iloc[0]),
            color=COLOR_PALETTE.primary, linewidth=4)
    # Get max and min levels and set chart axes
    max_level = sec_ratio_dist['Mean_Sector_Ratio'].max() + 0.05
    min_level = sec_ratio_dist['Mean_Sector_Ratio'].min() - 0.1
    ax.set_ylim(min_level, max_level)
    # Add boom dimensions to chart, if required
    width = np.pi / 108
    radii = max_level
    annotate = False
    annotation_text = '* Plot generated using '
    if boom_dir_1 >= 0:
        boom_dir_1_rad = np.radians(boom_dir_1)
        ax.bar(boom_dir_1_rad, radii, width=width, bottom=min_level, color=COLOR_PALETTE.fourth)
        if boom_dir_2 == -1:
            annotation_text += '{} (top mounted) divided by {} ({}° boom)'.format(col_names[1], col_names[0],
                                                                                  boom_dir_1)
            annotate = True
    if boom_dir_2 >= 0:
        boom_dir_2_rad = np.radians(boom_dir_2)
        ax.bar(boom_dir_2_rad, radii, width=width, bottom=min_level, color=COLOR_PALETTE.fifth)
        if boom_dir_1 == -1:
            annotation_text += '{} ({}° boom) divided by {} (top mounted)'.format(col_names[1], boom_dir_2,
                                                                                  col_names[0])
            annotate = True
    if boom_dir_2 >= 0 and boom_dir_1 >= 0:
        annotation_text += '{} ({}° boom) divided by {} ({}° boom)'.format(col_names[1], boom_dir_2,
                                                                           col_names[0], boom_dir_1)
        annotate = True
    if annotate:
        ax.annotate(annotation_text, xy=(0.5, 0.035), xycoords='figure fraction', horizontalalignment='center')
    ax.scatter(np.radians(wdir), sec_ratio, color=COLOR_PALETTE.secondary, alpha=0.3, s=1)
    plt.close()
    return ax.get_figure()


def plot_power_law(avg_alpha, avg_c, wspds, heights, max_plot_height=None, avg_slope=None, avg_intercept=None,
                   plot_both=False):
    if max_plot_height is None:
        max_plot_height = max(heights)

    plot_heights = np.arange(1, max_plot_height+1, 1)
    speeds = avg_c * (plot_heights ** avg_alpha)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('Speed [m/s]')
    ax.set_ylabel('Elevation [m]')
    ax.plot(speeds, plot_heights, '-', color=COLOR_PALETTE.primary, label='power_law')
    ax.scatter(wspds, heights, marker='o', color=COLOR_PALETTE.secondary)
    if plot_both is True:
        plot_heights = np.arange(1, max_plot_height+1, 1)
        speeds = avg_slope * np.log(plot_heights) + avg_intercept
        ax.plot(speeds, plot_heights, '-', color=COLOR_PALETTE.secondary, label='log_law')
        ax.scatter(wspds, heights, marker='o', color=COLOR_PALETTE.secondary)
        plt.legend(loc='upper left')

    ax.grid()
    ax.set_xlim(0, max(speeds) + 1)
    ax.set_ylim(0, max(plot_heights) + 10)
    return ax.get_figure()


def plot_log_law(avg_slope, avg_intercept, wspds, heights, max_plot_height=None):
    if max_plot_height is None:
        max_plot_height = max(heights)

    plot_heights = np.arange(1, max_plot_height + 1, 1)
    speeds = avg_slope * np.log(plot_heights) + avg_intercept
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('Speed [m/s]')
    ax.set_ylabel('Elevation [m]')
    ax.plot(speeds, plot_heights, '-', color=COLOR_PALETTE.primary)
    ax.scatter(wspds, heights, marker='o', color=COLOR_PALETTE.secondary)
    ax.grid()
    ax.set_xlim(0, max(speeds) + 1)
    ax.set_ylim(0, max(plot_heights) + 10)
    return ax.get_figure()


def plot_shear_time_of_day(df, calc_method, plot_type='step'):

    df_copy = df.copy()
    # colours in use
    colors = [(0.6313725490196078, 0.6470588235294118, 0.6705882352941176, 1.0),  # Jan
              (0.1568627450980392, .19215686274509805, 0.6705882352941176, 1.0),  # Feb
              (0.06666666666666667, 0.4196078431372549, 0.6901960784313725, 1.0),  # March
              (0.22745098039215686, 0.7294117647058823, 0.9803921568627451, 1.0),  # April
              (0.2392156862745098, 0.5666666666666667, 0.42745098039215684, 1.0),  # May
              (0.4117647058823529, 0.7137254901960784, 0.16470588235294117, 1.0),  # June
              (0.611764705882353, 0.7725490196078432, 0.21568627450980393, 1.0),  # July
              (0.6823529411764706, 0.403921568627451, 0.1607843137254902, 1.0),  # Aug
              (0.7901960784313726, 0.48627450980392156, 0.1843137254901961, 1.0),  # Sep
              (1, 0.7019607843, .4, 1),  # Oct
              (0, 0, 0, 1.0),  # Nov
              (0.40588235294117647, 0.43137254901960786, 0.4666666666666667, 1.0)]  # Dec

    if len(df.columns) == 1:
        colors[0] = colors[5]
    if calc_method == 'power_law':
        label = 'Average Shear'

    if calc_method == 'log_law':
        label = 'Roughness Coefficient'

    if plot_type == '12x24':
        df.columns = np.arange(1, 13, 1)
        df.index = np.arange(0, 24, 1)
        df[df.columns[::-1]]
        return plot_12x24_contours(df, label=(label, 'mean'), plot='tod')

    else:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlabel('Time of Day')
        ax.set_ylabel(label)
        import matplotlib.dates as mdates

        # create x values for plot
        idx = pd.date_range('2017-01-01 00:00', '2017-01-01 23:00', freq='1H').time

        if plot_type == 'step':
            df = df.shift(+1, axis=0)
            df.iloc[0, :] = df_copy.tail(1).values
            for i in range(0, len(df.columns)):
                ax.step(idx, df.iloc[:, i], label=df.iloc[:, i].name, color=colors[i])

        if plot_type == 'line':
            for i in range(0, len(df.columns)):
                ax.plot(idx, df.iloc[:, i], label=df.iloc[:, i].name, color=colors[i])

        ax.legend(bbox_to_anchor=(1.1, 1.05))
        ax.set_xticks(df.index)
        ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter("%H-%M"))
        _ = plt.xticks(rotation=90)
        return ax.get_figure()


def plot_dist_matrix(matrix, colorbar_label=None, xticklabels=None, yticklabels=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    cm = ax.pcolormesh(matrix, cmap=COLOR_PALETTE.color_map)
    ax.set(xlim=(0, matrix.shape[1]), ylim=(0, matrix.shape[0]))
    ax.set(xticks=np.array(range(0, matrix.shape[1]))+0.5, yticks=np.array(range(0, matrix.shape[0])) + 0.5)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    ax.set_xlabel(matrix.columns.names[-1])
    ax.set_ylabel(matrix.index.name)
    cbar = ax.figure.colorbar(cm, ax=ax)
    if colorbar_label is not None:
        cbar.ax.set_ylabel(colorbar_label)
    plt.close()
    return ax.get_figure()


def render_table(data, col_width=3.0, row_height=0.625, font_size=16, header_color=COLOR_PALETTE.primary,
                 row_colors=[COLOR_PALETTE.primary_90, 'w'], edge_color='w', bbox=[0, 0, 1, 1],
                 header_columns=0, show_col_head=1,
                 ax=None, cellLoc='center', padding=0.01, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    if show_col_head == 1:
        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, cellLoc=cellLoc, **kwargs)
    else:
        mpl_table = ax.table(cellText=data.values, bbox=bbox, cellLoc=cellLoc, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if show_col_head == 1:
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
                cell.PAD = padding
            else:
                cell.set_facecolor(row_colors[k[0] % len(row_colors)])
                cell.PAD = padding
        else:
            if k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
                cell.PAD = padding
            else:
                cell.set_facecolor(row_colors[k[0] % len(row_colors)])
                cell.PAD = padding
                # if k[1]==1:
                #   cell.set_width(0.03)
    return ax

# def plot_3d_rose(matrix, colorbar_label=None):
