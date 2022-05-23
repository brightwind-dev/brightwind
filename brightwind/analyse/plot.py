import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import calendar
import numpy as np
import pandas as pd
import os
from brightwind.utils import utils
import brightwind as bw
from pandas.plotting import register_matplotlib_converters
from brightwind.utils.utils import _convert_df_to_series
import re
import six

register_matplotlib_converters()

__all__ = ['plot_timeseries',
           'plot_scatter',
           'plot_scatter_wspd',
           'plot_scatter_wdir',
           'plot_scatter_by_sector',
           'plot_sector_ratio']
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


def _derive_axes_limits_for_scatter_plot(x, y):
    x_min, x_max = (round(np.nanmin(x) - 0.5), -(-np.nanmax(x) // 1))
    y_min, y_max = (round(np.nanmin(y) - 0.5), -(-np.nanmax(y) // 1))
    return x_min, x_max, y_min, y_max


def _scatter_subplot(x, y, trendline_y=None, trendline_x=None, line_of_slope_1=False,
                     x_label=None, y_label=None, x_limits=None, y_limits=None, axes_equal=True, subplot_title=None,
                     trendline_dots=False, scatter_color=COLOR_PALETTE.primary,
                     trendline_color=COLOR_PALETTE.secondary, legend=True, scatter_name=None,
                     trendline_name=None, ax=None):
    """
    Plots a scatter subplot between the inputs x and y. The trendline_y data and the line of slope 1 passing through
    the origin are also shown if provided as input of the function.

    :param x:                   The x-axis values or reference variable.
    :type x:                    pd.Series or list np.array
    :param y:                   The y-axis values or target variable.
    :type y:                    pd.Series or list or np.array
    :param trendline_y:         Series or list or np.array of trendline y values.
    :type trendline_y:          pd.Series or list or np.array or None
    :param trendline_x:         X values to plot with trendline_y. If None then the x variable is used.
    :type trendline_x:          pd.Series or list or np.array or None
    :param line_of_slope_1:     Boolean to choose to plot the line with slope one and passing through the origin.
    :type line_of_slope_1:      Bool
    :param x_label:             Label for the x axis
    :type x_label:              str or None
    :param y_label:             Label for the y axis
    :type y_label:              str or None
    :param x_limits:            x-axis min and max limits.
    :type x_limits:             tuple, None
    :param y_limits:            y-axis min and max limits.
    :type y_limits:             tuple, None
    :param axes_equal:          Boolean to set the units for the x and y axes to be equal. If x_limits and y_limits are
                                both None then the two axes limits are set to be the same.
    :type axes_equal:           Bool
    :param subplot_title:       Title show on top of the subplot
    :type subplot_title:        str or None
    :param trendline_dots:      Boolean to chose if marker to use for the trendline is dot-line or a line
    :type trendline_dots:       Bool
    :param scatter_color:       Color to assign to scatter data. Default is COLOR_PALETTE.primary
    :type scatter_color:        str or Hex or Rgb
    :param trendline_color:     Color to assign to trendline data. Default is COLOR_PALETTE.secondary
    :type trendline_color:      str or Hex or Rgb
    :param legend:              Boolean to choose if legend is shown.
    :type legend:               Bool
    :param scatter_name:        Label to assign to scatter data in legend if legend is True. If None then the label
                                assigned is 'Data points'
    :type scatter_name:         str or None
    :param trendline_name:      Label to assign to trendline data in legend if legend is True. If None then the label
                                assigned is 'Regression line'
    :type trendline_name:       str or None
    :param ax:                  Subplot axes to which assign the subplot to in a plot. If
    :type ax:                   matplotlib.axes._subplots.AxesSubplot or None
    :return:                    A scatter subplot
    :rtype:                     matplotlib.axes._subplots.AxesSubplot

     **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)

        # To plot only one subplot in a figure without axis equal
        fig, axes = plt.subplots(1, 1)
        bw.analyse.plot._scatter_subplot(data.Spd80mN, data.Spd80mS, axes_equal=False, ax=axes)

        # To plot multiple subplots in a figure without legend and with x and y labels
        fig, axes = plt.subplots(1, 2)
        bw.analyse.plot._scatter_subplot(data.Spd80mN, data.Spd80mS, ax=axes[0], legend=False,
                                         x_label='Spd80mN', y_label='Spd80mS')
        bw.analyse.plot._scatter_subplot(data.Dir78mS, data.Dir58mS, ax=axes[1], legend=False,
                                         x_label='Dir78mS', y_label='Dir58mS')

        # To plot multiple scatters in the same subplot in a figure and set the legend
        fig, axes = plt.subplots(1, 1)
        scat_plot = bw.analyse.plot._scatter_subplot(data.Spd80mN, data.Spd80mS, ax=axes)
        scat_plot = bw.analyse.plot._scatter_subplot(data.Spd60mN, data.Spd60mS, ax=axes,
                                                     scatter_color=bw.analyse.plot.COLOR_PALETTE.tertiary)
        scat_plot.axes.legend(['Data1', 'Data2'])

        # To set the x and y axis limits by using a tuple.
        fig, axes = plt.subplots(1, 1)
        bw.analyse.plot._scatter_subplot(data.Dir78mS, data.Dir58mS, x_label='Dir78mS', y_label='Dir58mS',
                        x_limits=(50,300), y_limits=(250,300), ax=axes)

        # To show a trendline giving trendline_x and trendline_y data, assign the color of the trendline and
        # set the x and y labels
        fig, axes = plt.subplots(1, 1)
        bw.analyse.plot._scatter_subplot(data.Spd80mN, data.Spd80mS, trendline_y=[0, 15], trendline_x=[0, 10],
                        trendline_color = 'r', x_label="Reference", y_label="Target", ax=axes)

        # To show the line with slope 1 passing through the origin.
        fig, axes = plt.subplots(1, 1)
        bw.analyse.plot._scatter_subplot(data.Spd80mN, data.Spd80mS, line_of_slope_1=True, ax=axes)

        # To set the name of the scatter data and of the trendline.
        fig, axes = plt.subplots(1, 1)
        bw.analyse.plot._scatter_subplot(data.Spd80mN, data.Spd80mS, trendline_y=[0, 15], trendline_x=[0, 10],
                                         scatter_name='Data scatter', trendline_name='Regression line', ax=axes)
    """
    if ax is None:
        ax = plt.gca()

    if scatter_name is None:
        scatter_name = 'Data points'

    if trendline_name is None:
        trendline_name = 'Regression line'

    if trendline_dots is True:
        trendline_marker = 'o-'
    else:
        trendline_marker = '-'

    if x_limits is None or y_limits is None:
        x_min, x_max, y_min, y_max = _derive_axes_limits_for_scatter_plot(x, y)

    if axes_equal:
        ax.set_aspect('equal')
        if x_limits is None and y_limits is None:
            axes_min = min(x_min, y_min)
            axes_max = max(x_max, y_max)
            x_limits = (axes_min, axes_max)
            y_limits = (axes_min, axes_max)

    if x_limits is None:
        x_limits = (x_min, x_max)
    if y_limits is None:
        y_limits = (y_min, y_max)

    ax.set_xlim(x_limits[0], x_limits[1])
    ax.set_ylim(y_limits[0], y_limits[1])

    no_dots = len(x)

    marker_size_max = 216
    marker_size_min = 18
    marker_size = -0.2 * no_dots + marker_size_max  # y=mx+c, m = (216 - 18) / (1000 - 0) i.e. slope changes up to 1000
    marker_size = marker_size_min if marker_size < marker_size_min else marker_size

    max_alpha = 0.7
    min_alpha = 0.3
    alpha = -0.0004 * no_dots + max_alpha  # y=mx+c, m = (0.7 - 0.3) / (1000 - 0) i.e. alpha changes up to 1000 dots
    alpha = min_alpha if alpha < min_alpha else alpha

    ax.scatter(x, y, marker='o', color=scatter_color, s=marker_size, alpha=alpha,
               edgecolors='none', label=scatter_name)

    if trendline_y is not None:
        if trendline_x is None:
            trendline_x = x

        ax.plot(trendline_x, trendline_y, trendline_marker, color=trendline_color, label=trendline_name)

    if line_of_slope_1:
        low_x, high_x = ax.get_xlim()
        low_y, high_y = ax.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        ax.plot([low, high], [low, high], color=COLOR_PALETTE.secondary_70, label='1:1 line')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if legend:
        ax.legend()

    if subplot_title is not None:
        ax.set_title(subplot_title, fontsize=mpl.rcParams['ytick.labelsize'])

    return ax


def _get_best_row_col_number_for_subplot(number_subplots):
    # get all divisors of total number of subplots
    divs = {1, number_subplots}
    for i in range(2, int(np.sqrt(number_subplots)) + 1):
        if number_subplots % i == 0:
            divs.update((i, number_subplots // i))

    divs_list = sorted(list(divs))
    # get divisor number closer to sqrt of number_subplots
    diff_to_sqrt = [np.abs(np.sqrt(number_subplots) - i) for i in divs_list]
    div_closer_to_sqrt = divs_list[np.argmin(diff_to_sqrt)]
    other_divider = number_subplots / div_closer_to_sqrt

    best_row = min(div_closer_to_sqrt, other_divider)
    best_col = max(div_closer_to_sqrt, other_divider)

    return int(best_row), int(best_col)


def plot_scatter(x, y, trendline_y=None, trendline_x=None, line_of_slope_1=False,
                 x_label=None, y_label=None, x_limits=None, y_limits=None, axes_equal=True, figure_size=(10, 10.2),
                 trendline_dots=False, **kwargs):
    """
    Plots a scatter plot of x and y data. The trendline_y data is also shown if provided as input of the function.

    :param x:                   The x-axis values or reference variable.
    :type x:                    pd.Series or list or np.array
    :param y:                   The y-axis values or target variable.
    :type y:                    pd.Series or list or np.array
    :param trendline_y:         Series or list of trendline y values.
    :type trendline_y:          pd.Series or list or np.array or None
    :param trendline_x:         X values to plot with trendline_y. If None then the x variable is used.
    :type trendline_x:          pd.Series or list or np.array or None
    :param line_of_slope_1:     Boolean to choose to plot the line with slope one and passing through the origin.
    :type line_of_slope_1:      Bool
    :param x_label:             Label for the x-axis. If None, label will be taken from x_series name.
    :type x_label:              str, None
    :param y_label:             Label for the y-axis. If None, label will be taken from y_series name.
    :type y_label:              str, None
    :param x_limits:            x-axis min and max limits.
    :type x_limits:             tuple, None
    :param y_limits:            y-axis min and max limits.
    :type y_limits:             tuple, None
    :param axes_equal:          Boolean to set the units for the x and y axes to be equal. If x_limits and y_limits are
                                both None then the two axes limits are set to be the same.
    :type axes_equal:           Bool
    :param trendline_dots:      Boolean to chose if marker to use for the trendline is dot-line or a line
    :type trendline_dots:       Bool
    :param figure_size:         Figure size in tuple format (width, height)
    :type figure_size:          tuple
    :param kwargs:              Additional keyword arguments for matplotlib.pyplot.subplot
    :return:                    A scatter plot
    :rtype:                     matplotlib.figure.Figure

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)

        # To plot two variables against each other
        bw.plot_scatter(data.Spd80mN, data.Spd80mS)

        # To overwrite the default axis titles.
        bw.plot_scatter(data.Dir78mS, data.Dir58mS, x_label='Dir78mS', y_label='Dir58mS')

        # To set the x and y axis limits by using a tuple.
        bw.plot_scatter(data.Dir78mS, data.Dir58mS, x_label='Dir78mS', y_label='Dir58mS',
                        x_limits=(50,300), y_limits=(250,300))

        # To show a trendline line giving trendline_x and trendline_y data.
        bw.plot_scatter(data.Spd80mN, data.Spd80mS, trendline_y=[0, 15],trendline_x=[0, 10],
                        x_label="Reference", y_label="Target")

        # To show a line with slope 1 and passing through the origin.
        bw.plot_scatter(data.Spd80mN, data.Spd80mS, line_of_slope_1=True)

         # To set the plot axes as not equal.
         bw.plot_scatter(data.Spd80mN, data.Spd80mS, axes_equal=False)
    """
    if type(x) is pd.DataFrame:
        x = _convert_df_to_series(x)
    elif type(x) is np.ndarray or type(x) is list:
        x = pd.Series(x).rename('x')

    if type(y) is pd.DataFrame:
        y = _convert_df_to_series(y)
    elif type(y) is np.ndarray or type(y) is list:
        y = pd.Series(y).rename('y')

    if x_label is None:
        x_label = x.name
    if y_label is None:
        y_label = y.name

    merged_df = pd.concat([x, y], join='inner', axis=1)
    x = merged_df[x.name]
    y = merged_df[y.name]

    if trendline_y is not None:
        legend = True
        if trendline_x is None:
            trendline_x = merged_df[x.name]
    else:
        legend = False

    if line_of_slope_1 is True:
        legend = True

    fig, axes = plt.subplots(figsize=figure_size, **kwargs)
    _scatter_subplot(x, y, trendline_y=trendline_y, trendline_x=trendline_x, line_of_slope_1=line_of_slope_1,
                     x_label=x_label, y_label=y_label, x_limits=x_limits, y_limits=y_limits, axes_equal=axes_equal,
                     trendline_dots=trendline_dots, legend=legend, ax=axes)

    plt.close()
    return fig


def plot_scatter_wdir(x_wdir_series, y_wdir_series, x_label=None, y_label=None,
                      x_limits=(0, 360), y_limits=(0, 360)):
    """
    Plots a scatter plot of two wind direction timeseries and adds a line from 0,0 to 360,360.

    :param x_wdir_series:            The x-axis values or reference wind directions.
    :type x_wdir_series:             pd.Series
    :param y_wdir_series:            The y-axis values or target wind directions.
    :type y_wdir_series:             pd.Series
    :param x_label:                  Title for the x-axis. If None, title will be taken from x_wdir_series name.
    :type x_label:                   str, None
    :param y_label:                  Title for the y-axis. If None, title will be taken from y_wdir_series name.
    :type y_label:                   str, None
    :param x_limits:                 x-axis min and max limits.
    :type x_limits:                  tuple
    :param y_limits:                 y-axis min and max limits.
    :type y_limits:                  tuple
    :return:                         A scatter plot
    :rtype:                          matplotlib.figure.Figure

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)

        #To plot few variables
        bw.plot_scatter_wdir(data.Dir78mS, data.Dir58mS)

        #To overwrite the default axis titles.
        bw.plot_scatter_wdir(data.Dir78mS, data.Dir58mS, x_label='Wind direction at 78m',
                             y_label='Wind direction at 58m')

        #To set the x and y axis limits by using a tuple.
        bw.plot_scatter_wdir(data.Dir78mS, data.Dir58mS, x_label='Reference', y_label='Target',
                             x_limits=(50,300), y_limits=(250,300))

    """
    if x_label is None:
        x_label = x_wdir_series.name + ' [°]'
    if y_label is None:
        y_label = y_wdir_series.name + ' [°]'
    scat_plot = plot_scatter(x_wdir_series, y_wdir_series, x_label=x_label, y_label=y_label,
                             x_limits=x_limits, y_limits=y_limits, line_of_slope_1=True)

    scat_plot.axes[0].legend(['1:1 line', 'Data points'])
    return scat_plot


def plot_scatter_wspd(x_wspd_series, y_wspd_series, x_label=None, y_label=None,
                      x_limits=(0, 30), y_limits=(0, 30)):
    """
    Plots a scatter plot of two wind speed timeseries and adds a reference line from 0,0 to 40,40. This should
    only be used for wind speeds in m/s and not when one of the wind speed series is normalised. Please use the
    basic 'plot_scatter()' function when using normalised wind speeds.

    :param x_wspd_series: The x-axis values or reference wind speeds.
    :type x_wspd_series:  pd.Series
    :param y_wspd_series: The y-axis values or target wind speeds.
    :type y_wspd_series:  pd.Series
    :param x_label:       Title for the x-axis. If None, title will be taken from x_wspd_series name.
    :type x_label:        str, None
    :param y_label:       Title for the y-axis. If None, title will be taken from y_wspd_series name.
    :type y_label:        str, None
    :param x_limits:      x-axis min and max limits. Can be set to None to let the code derive the min and max from
                          the x_wspd_series.
    :type x_limits:       tuple, None
    :param y_limits:      y-axis min and max limits. Can be set to None to let the code derive the min and max from
                          the y_wspd_series.
    :type y_limits:       tuple, None
    :return:              A scatter plot
    :rtype:               matplotlib.figure.Figure

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)

        # To plot two wind speeds against each other
        bw.plot_scatter_wspd(data.Spd80mN, data.Spd80mS)

        # To overwrite the default axis titles.
        bw.plot_scatter_wspd(data.Spd80mN, data.Spd80mS, x_label='Speed at 80m North',
                             y_label='Speed at 80m South')

        # To set the x and y axis limits by using a tuple.
        bw.plot_scatter_wspd(data.Spd80mN, data.Spd80mS, x_label='Speed at 80m North',
                             y_label='Speed at 80m South', x_limits=(0,25), y_limits=(0,25))

    """
    if x_label is None:
        x_label = x_wspd_series.name + ' [m/s]'
    if y_label is None:
        y_label = y_wspd_series.name + ' [m/s]'
    scat_plot = plot_scatter(x_wspd_series, y_wspd_series, x_label=x_label, y_label=y_label,
                             x_limits=x_limits, y_limits=y_limits, line_of_slope_1=True)

    scat_plot.axes[0].legend(['1:1 line', 'Data points'])
    return scat_plot


def plot_scatter_by_sector(x, y, wdir, trendline_y=None, line_of_slope_1=True, sectors=12,
                           x_limits=None, y_limits=None, axes_equal=True, figure_size=(10, 10.2), **kwargs):
    """
    Plot scatter subplots (with shared x and y axis) of x versus y for each directional sector. If a trendline
    timeseries is given as input then this is also plotted in the graph. The line with slope 1 and passing
    through the origin is shown if line_of_slope_1=True

    :param x:               The x-axis values or reference variable.
    :type x:                pd.Series
    :param y:               The y-axis values or target variable.
    :type y:                pd.Series
    :param wdir:            Timeseries of wind directions.
    :type wdir:             pd.Series
    :param trendline_y:     Series of trendline y values.
    :type trendline_y:      pd.Series
    :param line_of_slope_1: Boolean to choose to plot the line with slope one and passing through the origin.
    :type line_of_slope_1:  Bool
    :param sectors:         Number of directional sectors
    :type sectors:          int
    :param x_limits:        x-axis min and max limits. Can be set to None to let the code derive the min and max from
                            the x_wspd_series.
    :type x_limits:         tuple, None
    :param y_limits:        y-axis min and max limits. Can be set to None to let the code derive the min and max from
                            the y_wspd_series.
    :type y_limits:         tuple, None
    :param axes_equal:      Boolean to set the units for the x and y axes to be equal. If x_limits and y_limits are
                            both None then the two axes limits are set to be the same.
    :type axes_equal:       Bool
    :param figure_size:     Figure size in tuple format (width, height)
    :type figure_size:      tuple
    :param kwargs:          Additional keyword arguments for matplotlib.pyplot.subplot
    :returns:               matplotlib.figure.Figure

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)

        # To plot scatter plots by 36 sectors, with the slope 1 line passing through the origin, without trendline
        # and with axes not equal
        bw.plot_scatter_by_sector(data.Spd80mN, data.Spd80mS, data.Dir78mS, trendline_y=None,
                                  line_of_slope_1=True, sectors=36, axes_equal=False)

        # To plot scatter plots by 12 sectors, with the slope 1 line passing through the origin, with trendline data
        # given as input as a pd.Series (trendline_y) with same index than x data. The input trendline series must
        # be derived previously for the same number of sectors used in the plot_scatter_by_sector function for
        # example from a directional correlation where trendline_y is the synthesised data.
        bw.plot_scatter_by_sector(data.Spd80mN, data.Spd80mS, data.Dir78mS, trendline_y=trendline_y,
                                  line_of_slope_1=False, sectors=12)

        # To plot scatter plots by 12 sectors, and set the figure size.
        bw.plot_scatter_by_sector(data.Spd80mN, data.Spd80mS, data.Dir78mS, sectors=12, figure_size=(15, 10.2))

    """
    if type(x) is pd.DataFrame:
        x = _convert_df_to_series(x)
    elif type(x) is np.ndarray or type(x) is list:
        x = pd.Series(x).rename('x')

    if type(y) is pd.DataFrame:
        y = _convert_df_to_series(y)
    elif type(y) is np.ndarray or type(y) is list:
        y = pd.Series(y).rename('y')

    sector = 360 / sectors

    rows, cols = _get_best_row_col_number_for_subplot(sectors)

    x_min, x_max, y_min, y_max = _derive_axes_limits_for_scatter_plot(x, y)

    if axes_equal:
        if x_limits is None and y_limits is None:
            axes_min = min(x_min, y_min)
            axes_max = max(x_max, y_max)
            x_limits = (axes_min, axes_max)
            y_limits = (axes_min, axes_max)

    if x_limits is None:
        x_limits = (x_min, x_max)
    if y_limits is None:
        y_limits = (y_min, y_max)

    fig, axes = plt.subplots(rows, cols, squeeze=False, sharex=True, sharey=True, figsize=figure_size, **kwargs)

    for i_angle, ax_subplot in zip(np.arange(0, 360, sector), axes.flatten()):

        ratio_min = bw.offset_wind_direction(float(i_angle), - sector / 2)
        ratio_max = bw.offset_wind_direction(float(i_angle), + sector / 2)
        if ratio_max > ratio_min:
            logic_sect = ((wdir >= ratio_min) & (wdir < ratio_max))
        else:
            logic_sect = ((wdir >= ratio_min) & (wdir <= 360)) | ((wdir < ratio_max) & (wdir >= 0))

        if trendline_y is not None:
            trendline_y_input = trendline_y[logic_sect]
        else:
            trendline_y_input = trendline_y

        _scatter_subplot(x[logic_sect], y[logic_sect], trendline_y_input, trendline_x=None,
                         line_of_slope_1=line_of_slope_1, x_label=None, y_label=None,
                         x_limits=x_limits, y_limits=y_limits, axes_equal=axes_equal,
                         subplot_title=str(round(ratio_min)) + '-' + str(round(ratio_max)),
                         legend=False, ax=ax_subplot)

    fig.text(0.5, 0.06, x.name, va='center', ha='center', fontsize=mpl.rcParams['axes.labelsize'])
    fig.text(0.06, 0.5, y.name, va='center', ha='center', rotation='vertical',
             fontsize=mpl.rcParams['axes.labelsize'])
    plt.close()
    return fig


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


def plot_sector_ratio(sec_ratio, wdir, sec_ratio_dist, col_names, boom_dir_1=-1, boom_dir_2=-1,
                      radial_limits=None, annotate=True, figure_size=(10, 10), **kwargs):
    """
    Accepts a DataFrame table or a dictionary with multiple ratio of anemometer pairs per sector, a wind direction,
    multiple distributions of anemometer ratio pairs per sector, along with 2 anemometer names,
    and plots the speed ratio by sector. Optionally can include anemometer boom directions also.

    :param sec_ratio:         Sector_ratios
    :type sec_ratio:          pandas.Series or dict
    :param wdir:              Direction series
    :type wdir:               pandas.Series or dict
    :param sec_ratio_dist:    DataFrames from SectorRatio.by_sector()
    :type sec_ratio_dist:     pandas.Series or dict
    :param col_names:         A list of strings containing column names of wind speeds, first string is divisor and
                              second is dividend.
    :type col_names:          list(str)
    :param boom_dir_1:        Boom orientation in degrees of speed_col_name_1. Defaults to -1. One or more boom
                              orientations can be accepted. If multiple orientations, number of orientations must equal
                              number of anemometer pairs.
    :type boom_dir_1:         float or list
    :param boom_dir_2:        Boom orientation in degrees of speed_col_name_2. Defaults to -1. One or more boom
                              orientations can be accepted. If multiple orientations, number of orientations must equal
                              number of anemometer pairs.
    :type boom_dir_2:         float or list
    :param radial_limits:     the min and max values of the radial axis. Defaults to +0.05 of max ratio and -0.1 of min.
    :type radial_limits:      tuple or list
    :param annotate:          Set to True to show annotations on plot.
    :type annotate:           bool
    :param figure_size:       Figure size in tuple format (width, height)
    :type figure_size:        tuple
    :param kwargs:            Additional keyword arguments for matplotlib.pyplot.subplot

    :returns:                 A speed ratio plot showing average speed ratio by sector and scatter of individual data
                              points.

    """

    if type(sec_ratio) == pd.core.series.Series:
        sec_ratio = {0: sec_ratio}

    wdir = pd.DataFrame(wdir)

    if len(wdir.columns) != 1:
        if len(wdir.columns) != len(sec_ratio):
            raise ValueError('Number of anemometers does not match number of wind vanes. Please ensure there is one ' +
                             'direction vane per anemometer pair or include one direction vane only to be used for ' +
                             'all anemometer pairs.')

    if type(boom_dir_1) is list:
        if (len(boom_dir_1) != len(sec_ratio)) & (len(boom_dir_1) != 1):
            raise ValueError('Number of boom orientations must be 1 or equal to number of ' +
                             'anemometer pairs.')

    if type(boom_dir_2) is list:
        if (len(boom_dir_2) != len(sec_ratio)) & (len(boom_dir_2) != 1):
            raise ValueError('Number of boom orientations must be 1 or equal to number of ' +
                             'anemometer pairs.')

    row, col = _get_best_row_col_number_for_subplot(len(sec_ratio))
    fig, axes = plt.subplots(row, col, figsize=figure_size, subplot_kw={'projection': 'polar'}, **kwargs)
    font_size = min(figure_size)/row/col+2.5

    if (len(sec_ratio)) > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    if type(boom_dir_1) is not list:
        boom_dir_1 = [boom_dir_1] * len(sec_ratio)
    elif len(boom_dir_1) == 1:
        boom_dir_1 = boom_dir_1 * len(sec_ratio)

    if type(boom_dir_2) is not list:
        boom_dir_2 = [boom_dir_2] * len(sec_ratio)
    elif len(boom_dir_2) == 1:
        boom_dir_2 = boom_dir_2 * len(sec_ratio)

    for pair, boom1, boom2 in zip(sec_ratio, boom_dir_1, boom_dir_2):
        if len(wdir.columns) == 1:
            wd = _convert_df_to_series(wdir).dropna()
        else:
            wd = _convert_df_to_series(wdir.iloc[:, pair]).dropna()

        common_idx = sec_ratio[pair].index.intersection(wd.index)

        _plot_sector_ratio_subplot(sec_ratio[pair].loc[common_idx], wd.loc[common_idx], sec_ratio_dist[pair],
                                   col_names[pair], boom_dir_1=boom1, boom_dir_2=boom2,
                                   radial_limits=radial_limits, annotate=annotate, font_size=font_size, ax=axes[pair])
    plt.close()

    return fig


def _plot_sector_ratio_subplot(sec_ratio, wdir, sec_ratio_dist, col_names, boom_dir_1=-1, boom_dir_2=-1,
                               radial_limits=None, annotate=True, font_size=10, ax=None):
    """
    Accepts a ratio of anemometers per sector, a wind direction, a distribution of anemometer ratios per sector,
    along with 2 anemometer names, and returns an axis object to plot the speed ratio by sector. Optionally can
    include anemometer boom directions also.

    :param sec_ratio:         Series of sector_ratios
    :type sec_ratio:          pandas.Series
    :param wdir:              Direction series
    :type wdir:               pandas.Series
    :param sec_ratio_dist:    DataFrame from SectorRatio.by_sector()
    :type sec_ratio_dist:     pandas.Series
    :param col_names:         A list of strings containing column names of wind speeds, first string is divisor and
                              second is dividend.
    :type col_names:          list(str)
    :param boom_dir_1:        Boom orientation in degrees of speed_col_name_1. Defaults to -1.
    :type boom_dir_1:         float
    :param boom_dir_2:        Boom orientation in degrees of speed_col_name_2. Defaults to -1.
    :type boom_dir_2:         float
    :param radial_limits:     The min and max values of the radial axis. Defaults to +0.05 of max ratio and -0.1 of min.
    :type radial_limits:      tuple or list
    :param annotate:          Set to True to show annotations on plot.
    :type annotate:           bool
    :param font_size:         Size of font in plot annotation. Defaults to 10.
    :type font_size:          int
    :param ax:                Subplot axes to which the subplot is assigned. If None subplot is displayed on its own.
    :type ax:                 matplotlib.axes._subplots.AxesSubplot or None

    :returns:                 A speed ratio plot showing average speed ratio by sector and scatter of individual
                              data points.

    """

    if ax is None:
        ax = plt.gca(polar=True)

    if radial_limits is None:
        max_level = sec_ratio_dist['Mean_Sector_Ratio'].max() + 0.05
        min_level = sec_ratio_dist['Mean_Sector_Ratio'].min() - 0.1
    else:
        max_level = max(radial_limits)
        min_level = min(radial_limits)
    ax.set_ylim(min_level, max_level)

    radians = np.radians(utils._get_dir_sector_mid_pts(sec_ratio_dist.index))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(utils._get_dir_sector_mid_pts(sec_ratio_dist.index))
    ax.plot(np.append(radians, radians[0]), sec_ratio_dist['Mean_Sector_Ratio'].append(sec_ratio_dist.iloc[0]),
            color=COLOR_PALETTE.primary, linewidth=4)

    # Add boom dimensions to chart, if required
    width = np.pi / 108
    radii = max_level
    annotation_text = '* Plot generated using '
    if boom_dir_1 >= 0:
        boom_dir_1_rad = np.radians(boom_dir_1)
        ax.bar(boom_dir_1_rad, radii, width=width, bottom=min_level, color=COLOR_PALETTE.fourth)
        if boom_dir_2 == -1:
            annotation_text += '{} (top mounted) divided by {} ({}° boom)'.format(col_names[1], col_names[0],
                                                                                  boom_dir_1)
    if boom_dir_2 >= 0:
        boom_dir_2_rad = np.radians(boom_dir_2)
        ax.bar(boom_dir_2_rad, radii, width=width, bottom=min_level, color=COLOR_PALETTE.fifth)
        if boom_dir_1 == -1:
            annotation_text += '{} ({}° boom) divided by {} (top mounted)'.format(col_names[1], boom_dir_2,
                                                                                  col_names[0])
    if boom_dir_2 >= 0 and boom_dir_1 >= 0:
        annotation_text += '{} ({}° boom) divided by {} ({}° boom)'.format(col_names[1], boom_dir_2,
                                                                           col_names[0], boom_dir_1)
    if boom_dir_1 == -1 and boom_dir_2 == -1:
        annotation_text += '{} divided by {}'.format(col_names[1], col_names[0])
    if annotate:
        ax.set_title(annotation_text, y=0.004*(font_size-2.5)-0.15)
    else: ax.axes.set_xticks(ax.get_xticks(), "")
    ax.scatter(np.radians(wdir), sec_ratio, color=COLOR_PALETTE.secondary, alpha=0.3, s=1)

    for item in ([ax.title] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)

    return ax


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
