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
import calendar
import numpy as np
import pandas as pd
import os
from brightwind.utils import utils
import matplotlib as mpl
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


__all__ = ['plot_timeseries',
           'plot_scatter',
           'plot_scatter_wspd',
           'plot_scatter_wdir']


try:
    if 'Gotham Rounded' in \
            [mpl.font_manager.FontProperties(fname=i).get_name() for i in mpl.font_manager.findSystemFonts()]:
        mpl.rcParams['font.family'] = 'Gotham Rounded'
except Exception as ex:
    raise 'Found exception when checking installed fonts. {}'.format(str(ex))
    

plt.style.use(os.path.join(os.path.dirname(__file__), 'bw.mplstyle'))


def bw_colors(bw_color):
    # Define color scheme to be used across graphs, and tables.
    if bw_color == 'green':
        bw_color = [156, 197, 55]
    elif bw_color == 'wind_rose_gradient':
        bw_color = []
    elif bw_color == 'light_green_for_gradient':
        bw_color = [154, 205, 50]
    elif bw_color == 'dark_green_for_gradient':
        bw_color = [215, 235, 173]
    elif bw_color == 'asphault':
        bw_color = [46, 55, 67]
    elif bw_color == 'greyline':
        bw_color = [108, 120, 134]
    elif bw_color == 'darkgreen':
        bw_color = [108, 144, 35]
    elif bw_color == 'redline':
        bw_color = [255, 0, 0]
    else:
        bw_color = [156, 197, 55]
    bw_color[:] = [x / 255.0 for x in bw_color]
    return bw_color


def plot_monthly_means(data, coverage=None, ylbl=''):
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    if len(data.shape) > 1:
        ax.plot(data, '-D')
        ax.legend(list(data.columns))
    else:
        ax.plot(data, '-D', color=bw_colors('asphault'))
        ax.legend([data.name])
    ax.set_ylabel(ylbl)

    from matplotlib.dates import DateFormatter
    ax.set_xticks(data.index)
    ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=20, ha='center')

    if coverage is not None:
        plot_coverage = True
        if len(coverage.shape) > 1:
            if coverage.shape[1] > 1:
                plot_coverage = False
        if plot_coverage:
            import matplotlib.dates as mdates
            ax2 = ax.twinx()

            plot_colors = [bw_colors('light_green_for_gradient'), bw_colors('dark_green_for_gradient'),
                           bw_colors('darkgreen')]
            for month, coverage in zip(coverage.index, coverage.values):
                ax2.imshow(np.array([[plot_colors[0]], [plot_colors[1]]]),
                           interpolation='gaussian', extent=(mdates.date2num(month - pd.Timedelta('10days')),
                                                             mdates.date2num(month + pd.Timedelta('10days')),
                                                             0, coverage), aspect='auto', zorder=1)
                ax2.bar(mdates.date2num(month), coverage, edgecolor=plot_colors[2], linewidth=0.3, fill=False, zorder=0)

            ax2.set_ylim(0, 1)
            ax.set_ylim(bottom=0)
            ax.set_xlim(data.index[0]-pd.Timedelta('20days'), data.index[-1]+pd.Timedelta('20days'))
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
        data = bw.load_csv(bw.datasets.demo_data)

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
    plt.rcParams['figure.figsize'] = (15, 8)
    if isinstance(data, pd.Series):
        data_to_slice = data.copy(deep=False).to_frame()
    else:
        data_to_slice = data.copy()
    sliced_data = utils.slice_data(data_to_slice, date_from, date_to)
    figure = sliced_data.plot().get_figure()
    if y_limits is not None:
        figure.axes[0].set_ylim(y_limits)
    plt.close()
    return figure


def _scatter_plot(x, y, predicted_y=None, x_label="Reference", y_label="Target", title="", prediction_marker='k-'):
    """
    Plots a scatter plot.
    :param x:
    :param y:
    :param predicted_y: A series of predicted y values after applying the correlation to the x series.
    :param x_label:
    :param y_label:
    :param title:
    :param prediction_marker:
    :return:
    """
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.scatter(x, y, marker='.', color='#9ACD32', alpha=0.5)
    fig.set_figwidth(10)
    fig.set_figheight(10.2)
    # ax.set_title(title)
    if predicted_y is not None:
        ax.plot(x, predicted_y, prediction_marker)
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
        data = bw.load_csv(bw.datasets.demo_data)

        # To plot two variables against each other
        bw.plot_scatter(data.Spd80mN, data.Spd80mS)

        # To overwrite the default axis titles.
        bw.plot_scatter(data.Dir78mS, data.Dir58mS, x_axis_title='Dir78mS', y_axis_title='Dir58mS')

        # To set the x and y axis limits by using a tuple.
        bw.plot_scatter(data.Dir78mS, data.Dir58mS, x_axis_title='Dir78mS', y_axis_title='Dir58mS',
                        x_limits=(50,300), y_limits=(250,300))

    """
    if x_axis_title is None:
        x_axis_title = x_series.name
    if y_axis_title is None:
        y_axis_title = y_series.name

    merged_df = pd.concat([x_series, y_series], join='inner', axis=1)
    scat_plot = _scatter_plot(merged_df[x_series.name], merged_df[y_series.name],
                              x_label=x_axis_title, y_label=y_axis_title)

    if x_limits is None:
        x_limits = (round(x_series.min()-0.5), -(-x_series.max()//1))
    if y_limits is None:
        y_limits = (round(y_series.min()-0.5), -(-y_series.max()//1))
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
        data = bw.load_csv(bw.datasets.demo_data)

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
    scat_plot.axes[0].plot(x, y, 'k-')
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
        data = bw.load_csv(bw.datasets.demo_data)

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
    scat_plot.axes[0].plot(x, y, 'k-')
    scat_plot.axes[0].legend(['Reference line', 'Data points'])
    return scat_plot


def plot_freq_distribution(data, max_y_value=None, labels=None, y_label=None,
                           plot_colors=[bw_colors('light_green_for_gradient'),
                                        bw_colors('dark_green_for_gradient'),
                                        bw_colors('darkgreen')]):
    from matplotlib.ticker import PercentFormatter
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlabel('Speed [m/s]')
    ax.set_ylabel(y_label)
    if isinstance(data.index[0], pd.Interval):
        x_data = [i.mid for i in data.index]
    else:
        x_data = data.index
    ax.set_xticks(x_data)
    ax.set_xlim(x_data[0]-0.5, x_data[-1]+0.5)
    if labels is not None:
        ax.set_xticklabels(labels)
    if max_y_value is None:
        ax.set_ylim(0, data.max()*1.1)
    else:
        ax.set_ylim(0, max_y_value)
    if y_label[0] == '%':
        ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(b=True, axis='y', zorder=0)
    for frequency, ws_bin in zip(data, x_data):
        ax.imshow(np.array([[plot_colors[0]], [plot_colors[1]]]),
                  interpolation='gaussian', extent=(ws_bin-0.4, ws_bin+0.4, 0, frequency), aspect='auto', zorder=3)
        ax.bar(ws_bin, frequency, edgecolor=plot_colors[2], linewidth=0.3, fill=False, zorder=5)
    plt.close()
    return ax.get_figure()


def plot_rose(ext_data):
    """
    Plot a wind rose from data by distribution_by_dir_sector
    """
    result = ext_data.copy(deep=False)
    sectors = len(result)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 360.0/sectors))
    ax.set_rgrids(np.arange(0, 101, 10), labels=[str(i)+'%' for i in np.arange(0, 101, 10)], angle=0)
    ax.bar(np.arange(0, 2.0*np.pi, 2.0*np.pi/sectors), result, width=2.0*np.pi/sectors, bottom=0.0, color='#9ACD32',
           edgecolor=['#6C9023' for i in range(len(result))], alpha=0.8)
    # ax.set_title('Wind Rose', loc='center')
    plt.close()
    return ax.get_figure()


def plot_rose_with_gradient(freq_table, percent_symbol=True, plot_bins=None, plot_labels=None,
                            gradient_colors=['#f5faea', '#d6ebad', '#b8dc6f', '#9acd32', '#7ba428', '#5c7b1e']):
    table = freq_table.copy()
    sectors = len(table.columns)
    table_trans = table.T
    if plot_bins is not None:
        rows_to_sum = []
        intervals = [pd.Interval(plot_bins[i], plot_bins[i+1], closed=table.index[0].closed)
                     for i in range(len(plot_bins)-1)]
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
            num_rows = len(table.index)//6
            ctr = 0
            while ctr < len(table.index)-(len(table.index) % 6):
                rows_to_sum.append(list(range(ctr, ctr+num_rows)))
                ctr += num_rows
            rows_to_sum[-1].extend(list(range(len(table.index)-(len(table.index) % 6), len(table.index))))

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
    # print(rows_to_sum)
    # return table_binned
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 360.0 / sectors), zorder=2)

    if percent_symbol:
        symbol = '%'
    else:
        symbol = ' '
    ax.set_rgrids(np.linspace(0.1, max(table.sum(axis=0)) + 2.0, 10),
                  labels=['%.0f' % round(i) + symbol for i in np.linspace(0.1, max(table.sum(axis=0)) + 2.0, 10)],
                  angle=0, zorder=2)
    ax.set_ylim(0, max(table.sum(axis=0)) + 3.0)
    ax.bar(0, 1, alpha=0)
    for column in table_binned:
        radial_pos = 0.0
        angular_pos_start = (np.pi / 180.0) * float(column.split('-')[0])
        angular_pos_end = (np.pi / 180.0) * float(column.split('-')[-1])
        # Check for sectors with 0 degrees within the sector
        if angular_pos_end > angular_pos_start:
            angular_width = angular_pos_end - angular_pos_start - (np.pi / 180)  # Leaving 1 degree gap
        else:
            angular_width = 2*np.pi - angular_pos_start + angular_pos_end - (np.pi / 180)
        for speed_bin, frequency in zip(table_binned.index, table_binned[column]):
            patch = mpl.patches.Rectangle((angular_pos_start, radial_pos), angular_width,
                                          frequency, facecolor=gradient_colors[speed_bin], edgecolor='#5c7b1e',
                                          linewidth=0.3, zorder=3)
            ax.add_patch(patch)
            radial_pos += frequency

    if plot_labels is None:
        plot_labels = [mpl.patches.Patch(color=gradient_colors[i], label=bin_labels[i]) for i in range(len(bin_labels))]
    else:
        plot_labels = [mpl.patches.Patch(color=gradient_colors[i], label=plot_labels[i]) for i in range(len(plot_labels))]
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
    ax.scatter(wspd.loc[common_idxs], wspd_std.loc[common_idxs]/wspd.loc[common_idxs],
               color=bw_colors('green'), alpha=0.3, marker='.')
    ax.plot(ti.index.values, ti.loc[:, 'Mean_TI'].values, color=bw_colors('darkgreen'), label='Mean_TI')
    ax.plot(ti.index.values, ti.loc[:, 'Rep_TI'].values, color=bw_colors('redline'), label='Rep_TI')
    ax.plot(IEC_class.iloc[:, 0], IEC_class.iloc[:, 1], color=bw_colors('greyline'), linestyle='dashed')
    ax.plot(IEC_class.iloc[:, 0], IEC_class.iloc[:, 2], color=bw_colors('greyline'), linestyle='dashdot')
    ax.plot(IEC_class.iloc[:, 0], IEC_class.iloc[:, 3], color=bw_colors('greyline'), linestyle='dotted')
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
    ax.plot(np.append(radians, radians[0]), ti.append(ti.iloc[0])['Mean_TI'], color=bw_colors('green'), linewidth=4,
            figure=fig)
    maxlevel = ti['Mean_TI'].max() + 0.1
    ax.set_ylim(0, maxlevel)
    ax.scatter(np.radians(wdir), turbulence, color=bw_colors('asphault'), alpha=0.3, s=1)
    ax.legend(loc=8, framealpha=1)
    plt.close()
    return ax.get_figure()


def plot_shear_by_sector(shear, wdir, shear_dist):
    radians = np.radians(utils._get_dir_sector_mid_pts(shear_dist.index))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(utils._get_dir_sector_mid_pts(shear_dist.index))
    ax.plot(np.append(radians, radians[0]), shear_dist.append(shear_dist.iloc[0])['Mean_Shear'],
            color=bw_colors('green'), linewidth=4)
    maxlevel = shear_dist['Mean_Shear'].max() + 0.1
    ax.set_ylim(0, maxlevel)
    ax.scatter(np.radians(wdir), shear, color=bw_colors('asphault'), alpha=0.3, s=1)
    ax.legend(loc=8, framealpha=1)
    plt.close()
    return ax.get_figure()


def plot_12x24_contours(tab_12x24, label=('Variable', 'mean')):
    """
    Get Contour Plot of 12 month x 24 hour matrix of variable
    :param tab_12x24: DataFrame returned from get_12x24() in analyse
    :param label: Label of the colour bar on the plot.
    :return: 12x24 figure
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    levels = np.linspace(tab_12x24.min().min(), tab_12x24.max().max(), num=9)
    x = ax.contourf(tab_12x24.columns, tab_12x24.index, tab_12x24.values, levels=levels,
                    colors=['#e1f0c1', '#d6ebad', '#c2e184', '#aed75b', '#9acd32', '#8ab92d', '#7ba428', '#6b9023'])
    cbar = plt.colorbar(x)
    cbar.ax.set_ylabel(label[1].capitalize() + " of " + label[0])
    ax.set_xlabel('Month of Year')
    ax.set_ylabel('Hour of Day')
    month_names= calendar.month_abbr[1:13]
    ax.set_xticks(tab_12x24.columns)
    ax.set_xticklabels([month_names[i-1] for i in tab_12x24.columns])
    ax.set_yticks(np.arange(0, 24, 1))
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
            color=bw_colors('green'), linewidth=4)
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
        boom_dir_1 = np.radians(boom_dir_1)
        ax.bar(boom_dir_1, radii, width=width, bottom=min_level, color='#659CEF')
        if boom_dir_2 == -1:
            annotation_text += col_names[1] + ' (top mounted) divided by ' + col_names[0] + ' (blue boom)'
            annotate = True
    if boom_dir_2 >= 0:
        boom_dir_2 = np.radians(boom_dir_2)
        ax.bar(boom_dir_2, radii, width=width, bottom=min_level, color='#DCF600')
        if boom_dir_1 == -1:
            annotation_text += col_names[1] + ' (yellow green boom) divided by ' + col_names[0] + ' (top mounted)'
            annotate = True
    if boom_dir_2 >= 0 and boom_dir_1 >= 0:
        annotation_text += col_names[1] + ' (yellow green boom) divided by ' + col_names[0] + ' (blue boom)'
        annotate = True
    if annotate:
        ax.annotate(annotation_text, xy=(0.5, 0.035), xycoords='figure fraction', horizontalalignment='center')
    ax.scatter(np.radians(wdir), sec_ratio, color=bw_colors('asphault'), alpha=0.3, s=1)
    plt.close()
    return ax.get_figure()


def plot_shear(avg_alpha, avg_c, wspds, heights):
    plot_heights = np.linspace(0, max(heights), num=100)
    speeds = avg_c*(plot_heights**avg_alpha)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('Speed (m/s)')
    ax.set_ylabel('Elevation (m)')
    ax.plot(speeds, plot_heights, '-', color='#9ACD32')
    ax.scatter(wspds, heights, marker='o', color=bw_colors('asphault'))
    ax.grid()
    ax.set_xlim(0, max(speeds)+1)
    ax.set_ylim(0, max(plot_heights)+10)
    plt.close()
    return ax.get_figure()
