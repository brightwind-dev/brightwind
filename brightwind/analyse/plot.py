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
import math
from brightwind.utils import utils
import os
import matplotlib as mpl

__all__ = ['plot_timeseries', 'plot_freq_distribution', 'plot_rose']


try:
    if 'Gotham Rounded' in \
            [mpl.font_manager.FontProperties(fname=i).get_name() for i in mpl.font_manager.findSystemFonts()]:
        mpl.rcParams['font.family'] = 'Gotham Rounded'
except Exception as ex:
    raise 'Found exception when checking installed fonts. {}'.format(str(ex))
    

#plt.style.use(os.path.join(os.path.dirname(__file__), 'bw.mplstyle'))


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
            return ax2.get_figure()
    return ax.get_figure()


def plot_timeseries(data, date_from='', date_to=''):
    """
    Plots timeseries data

    :param data: DataFrame to plot
    :param date_from: Start date used for plotting in yyyy-mm-dd format
    :type date_from: str
    :param date_to: End date used for plotting in yyyy-mm-dd format
    :type date_to: str
    :return: Timeseries plot

    """
    if isinstance(data, pd.Series):
        data_to_slice = data.copy(deep=False).to_frame()
    else:
        data_to_slice = data.copy()
    sliced_data = utils._slice_data(data_to_slice, date_from, date_to)
    return sliced_data.plot().get_figure()


def _scatter_plot(x, y, predicted_y=None, x_label="Reference", y_label="Target", title="", prediction_marker='k-'):
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.scatter(x, y, marker='.', color='#9ACD32', alpha=0.5)
    # fig.set_figwidth(size[0])
    # fig.set_figheight(size[1])
    # ax.set_title(title)
    if predicted_y is not None:
        ax.plot(x, predicted_y, prediction_marker)
        ax.legend(['Predicted', 'Original'])
    return ax.get_figure()


def plot_freq_distribution(data, max_y_value=None, plot_colors=[bw_colors('light_green_for_gradient'),
                                                        bw_colors('dark_green_for_gradient'),
                                                        bw_colors('darkgreen')]):
    from matplotlib.ticker import PercentFormatter
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlabel('Speed [m/s]')
    ax.set_ylabel('Frequency [%]')
    if isinstance(data.index[0], pd.Interval):
        x_data = [i.mid for i in data.index]
    else:
        x_data = data.index
    ax.set_xticks(x_data)
    ax.set_xlim(x_data[0]-0.5, x_data[-1]+0.5)
    if max_y_value is None:
        ax.set_ylim(0, max(data)*1.1)
    else:
        ax.set_ylim(0, max_y_value)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(b=True, axis='y', zorder=0)
    for frequency, ws_bin in zip(data, x_data):
        ax.imshow(np.array([[plot_colors[0]], [plot_colors[1]]]),
                  interpolation='gaussian', extent=(ws_bin-0.4, ws_bin+0.4, 0, frequency), aspect='auto', zorder=3)
        ax.bar(ws_bin, frequency, edgecolor=plot_colors[2], linewidth=0.3, fill=False, zorder=5)
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
    return ax.get_figure()


def plot_rose_with_gradient(freq_table, percent_symbol=True, plot_bins=None, plot_labels=None,
                            gradient_colors=['#f5faea', '#d6ebad', '#b8dc6f', '#9acd32', '#7ba428', '#5c7b1e']):
    # ['0-3 m/s', '4-6 m/s', '7-9 m/s', '10-12 m/s', '13-15 m/s', '15+ m/s']):
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
            # print(bin_assigned)
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
        #Check for sectors with 0 degrees within the sector
        if angular_pos_end > angular_pos_start:
            angular_width = angular_pos_end - angular_pos_start - (np.pi / 180) # Leaving 1 degree gap
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
    return ax.get_figure()


def plot_TI_by_speed(wspd, wspd_std, ti, IEC_class=None):
    """
    Plot turbulence intensity graphs alongside with IEC standards
    :param wdspd:
    :param wdspd_std:
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
    fig, ax = plt.subplots()
    ax.scatter(wspd.loc[common_idxs], wspd_std.loc[common_idxs]/wspd.loc[common_idxs],
               color=bw_colors('green'), alpha=0.3, marker='.')
    ax.plot(ti.index.__array__(), ti.loc[:, 'Mean_TI'].values, color=bw_colors('darkgreen'))[0].set_label('Mean_TI')
    ax.plot(ti.index.__array__(), ti.loc[:, 'Rep_TI'].values, color=bw_colors('redline'))[0].set_label('Rep_TI')
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
    return ax.get_figure()


def plot_TI_by_sector(turbulence, wdir, ti):
    radians = np.radians(utils._get_dir_sector_mid_pts(ti.index))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(utils._get_dir_sector_mid_pts(ti.index))
    ax.plot(np.append(radians, radians[0]), ti.append(ti.iloc[0])['Mean_TI'], c=bw_colors('green'), linewidth=4,
            figure=fig)
    maxlevel = ti['Mean_TI'].max() + 0.1
    ax.set_ylim(0, maxlevel)
    ax.scatter(np.radians(wdir), turbulence, c=bw_colors('asphault'), alpha=0.3, s=1)
    ax.legend(loc=8, framealpha=1)
    return ax.get_figure()


def plot_shear_by_sector(shear, wdir, shear_dist):
    radians = np.radians(utils._get_dir_sector_mid_pts(shear_dist.index))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(utils._get_dir_sector_mid_pts(shear_dist.index))
    ax.plot(np.append(radians, radians[0]), shear_dist.append(shear_dist.iloc[0])['Mean_Shear'],
            c=bw_colors('green'), linewidth=4)
    # ax.set_title('Shear by Direction')
    maxlevel = shear_dist['Mean_Shear'].max() + 0.1
    ax.set_ylim(0, maxlevel)
    ax.scatter(np.radians(wdir), shear, c=bw_colors('asphault'), alpha=0.3, s=1)
    ax.legend(loc=8, framealpha=1)
    return ax.get_figure()


def plot_12x24_contours(tab_12x24, title='Variable'):
    """
    Get Contour Plot of 12 month x 24 hour matrix of turbulence intensity
    :param tab_12x24: DataFrame returned from get_12x24() in analyse
    :param title: Title of the plot
    :return: 12x24 figure
    """

    max_v = math.ceil(tab_12x24.max().max() * 100) / 100
    min_v = math.floor(tab_12x24.min().min() * 100) / 100
    step = (max_v - min_v) / 8
    levels = np.arange(min_v, max_v + step, step)#.round(2)
    fig, ax = plt.subplots()
    x = ax.contourf(tab_12x24, colors=['#e1f0c1', '#d6ebad', '#c2e184', '#aed75b', '#9acd32',
                                       '#8ab92d', '#7ba428', '#6b9023'],
                    levels=levels)
    cbar = plt.colorbar(x)
    cbar.ax.set_ylabel(title)
    ax.set_xlabel('Month of Year')
    ax.set_ylabel('Hour of Day')
    ax.set_xticks(np.arange(12), calendar.month_name[1:13])
    ax.set_yticks(np.arange(0, 24, 1))
    # ax.set_title('Hourly Mean '+title+' Calendar Month')
    return ax.get_figure()


def plot_sector_ratio(sec_ratio, wdir, sec_ratio_dist, col_names, boom_dir_1=-1, boom_dir_2=-1):
    """
    Accepts a DataFrame table, along with 2 anemometer names, and one wind vane name and plots the speed ratio
    by sector. Optionally can include anemometer boom directions also.
    :param sec_ratio:
    :param wdir:
    :param sec_ratio_dist: DataFrame from SectorRation.by_speed()
    :param boom_dir_1: Boom direction in degrees of speed_col_name_1. Defaults to 0.
    :param boom_dir_2: Boom direction in degrees of speed_col_name_2. Defaults to 0.
    :param col_names: A list of strings containing column names of wind speeds
    :returns A speed ratio plot showing average speed ratio by sector and scatter of individual datapoints.
    """
    radians = np.radians(utils._get_dir_sector_mid_pts(sec_ratio_dist.index))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(utils._get_dir_sector_mid_pts(sec_ratio_dist.index))
    ax.plot(np.append(radians, radians[0]), sec_ratio_dist['Mean_Sector_Ratio'].append(sec_ratio_dist.iloc[0]),
            c=bw_colors('green'), linewidth=4)
    # plt.title('Speed Ratio by Direction')
    # Get max and min levels and set chart axes
    maxlevel = sec_ratio_dist['Mean_Sector_Ratio'].max() + 0.05
    minlevel = sec_ratio_dist['Mean_Sector_Ratio'].min() - 0.1
    ax.set_ylim(minlevel, maxlevel)
    # Add boom dimensions to chart, if required
    width = np.pi / 108
    radii = maxlevel
    ctr = 0
    if boom_dir_1 >= 0:
        boom_dir_1 = np.radians(boom_dir_1)
        ax.bar(boom_dir_1, radii, width=width, bottom=minlevel, color='#659CEF')
        ctr += 1
    if boom_dir_2 >= 0:
        boom_dir_2 = np.radians(boom_dir_2)
        ax.bar(boom_dir_2, radii, width=width, bottom=minlevel, color='#DCF600')
        ctr += 1

    if ctr == 2:
        ax.annotate('*Plot generated using ' + col_names[1] + ' (yellow green boom) divided by' + col_names[0] +
                    ' (blue boom)', xy=(20, 10), xycoords='figure pixels')
    ax.scatter(np.radians(wdir), sec_ratio, c=bw_colors('asphault'), alpha=0.3, s=1)
    return ax.get_figure()


def plot_shear(avg_alpha, avg_c, wspds, heights):
    plot_heights = np.linspace(0, max(heights), num=100)
    speeds = avg_c*(plot_heights**avg_alpha)
    fig, ax = plt.subplots()
    ax.set_xlabel('Speed (m/s)')
    ax.set_ylabel('Elevation (m)')
    ax.plot(speeds, plot_heights, '-', color='#9ACD32')
    ax.scatter(wspds, heights, marker='o', color=bw_colors('asphault'))
    ax.grid()
    ax.set_xlim(0, max(speeds)+1)
    ax.set_ylim(0, max(plot_heights)+10)
    # ax.set_title("Shear Profile")
    return ax.get_figure()


# def plot_shear(wind_speeds, heights):
#     """
#     Show derivation and output (alpha value) of shear calculation function for a given timestep.
#     :param wind_speeds: List of wind speeds [m/s]
#     :param heights: List of heights [m above ground]. The position of the height in the list must be the same
# position in the list as its
#     corresponding wind speed value.
#     :return:
#         1) Log-log plot of speed and elevation data, including linear fit.
#         2) Speed and elevation data plotted on regular scale, showing power law fit resulting from alpha value
#     """
#
#     alpha, wind_speedsfit = sh.calc_shear(wind_speeds, heights, plot=True)
#
#     # PLOT INPUT AND MODELLED DATA ON LOG-LOG SCALE
#     heightstrend = np.linspace(0.1, max(heights) + 2, 100)  # create variable to define (interpolated) power law trend
#     plt.loglog(wind_speeds, heights, 'bo')  # plot input data on log log scale
#     plt.loglog(wind_speedsfit(heightstrend), heightstrend, 'k--')  # Show interpolated power law trend
#     plt.xlabel('Speed (m/s)')
#     plt.ylabel('Elevation (m)')
#     plt.legend(['Input data', 'Best fit line for power law (' r'$\alpha$ = %i)' % int(round(alpha))])
#     plt.grid(True)
#     plt.show()
#
#     # PLOT INPUT AND MODELLED DATA ON REGULAR SCALE
#     plt.plot(wind_speeds, heights, 'bo')  # plot input data
#     plt.plot(wind_speedsfit(heightstrend), heightstrend, 'k--')  # Show interpolated power law trend
#     plt.xlabel('Speed (m/s)')
#     plt.ylabel('Elevation (m)')
#     plt.legend(['Input data', 'Power law trend (' r'$\alpha$ = %i)' % int(round(alpha))])
#     plt.ylim(0, max(heights) + 2)
#     plt.xlim(0, max([max(wind_speeds), max(wind_speedsfit(heights))]) + 2)
#     plt.grid(True)
#     plt.show()
