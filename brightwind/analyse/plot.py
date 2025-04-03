import matplotlib.pyplot as plt
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
from colormap import rgb2hex, rgb2hls, hls2rgb
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import ListedColormap, to_hex, LinearSegmentedColormap
import warnings
from matplotlib.patches import Patch

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
        Color palette to be used for plotting graphs and tables. This Class generates also color_list, color_map,
        color_map_cyclical and some adjusted lightness color variables that are created from the main colors defined
        below and used by several brightwind functions. The color_map, color_map_cyclical and the adjusted lightness
        color variables can also be set independently from the main colors as for examples below.

        1) The main colors used to define the color palette are:
            self.primary = '#9CC537'        # slightly darker than YellowGreen #9acd32, rgb(156/255, 197/255, 55/255)
            self.secondary = '#2E3743'      # asphalt, rgb(46/255, 55/255, 67/255)
            self.tertiary = '#9B2B2C'       # red'ish, rgb(155/255, 43/255, 44/255)
            self.fourth = '#E57925'         # Vivid Tangelo, rgb(229/255, 121/255, 37/255)
            self.fifth = '#ffc008'          # Mikado Yellow, rgb(255/255, 192/255, 8/255)
            self.sixth = '#AB8D60'          # Bronze (Metallic), rgb(171/255, 141/255, 96/255)
            self.seventh = '#A4D29F'        # green'ish, rgb(164/255, 210/255, 159/255)
            self.eighth = '#01958a'         # Dark Cyan, rgb(1/255, 149/255, 138/255)
            self.ninth = '#3D636F'          # blue grey, rgb(61/255, 99/255, 111/255)
            self.tenth = '#A49E9D'          # Quick Silver, rgb(164/255, 158/255, 157/255)
            self.eleventh = '#DA9BA6'       # Parrot Pink, rgb(218/255, 155/255, 166/255)

           Some of these colors are then used to set other color lists and maps. For example the primary color is
           used to define the color_map.

        2) The adjusted lightness color variables derived from the main colors above are as below.
           Gradient goes from 0% (darkest) to 100% (lightest).
           See https://www.w3schools.com/colors/colors_picker.asp for more info.

            self.primary_10  # 10% lightness of primary
            self.primary_35  # lightness of primary
            self.primary_80  # lightness of primary
            self.primary_90  # 90% lightness of primary
            self.primary_95  # 95% lightness of primary
            self.tenth_40    # 40% lightness of tenth

        3) The color sequence used for defining the color_list variable is as below:
            color_list = [self.primary, self.secondary, self.tertiary, self.fourth, self.fifth, self.sixth,
                          self.seventh, self.eighth, self.ninth, self.tenth, self.eleventh, self.primary_35]

           This color list is used primarily in line and scatter plots.

        4) The color sequence used for defining the color_map variable is as below. This variable is a
           color map having a linear color pattern from the first color to the last color of the color_map_colors list.
            self.color_map_colors = [self.primary_95,  # lightest primary
                                      self.primary,     # primary
                                      self.primary_10]  # darkest primary

           This color map is used in the wind rose plot and also the 12x24 heat map plot from a shear by time of day.

        5) The color sequence used for defining the color_map_cyclical variable is as below. This variable is a
           color map having a cyclical color pattern from/to the first color of the color_map_cyclical_colors.
            self.color_map_cyclical_colors = [self.secondary, self.fifth, self.primary, self.tertiary, self.secondary]

           This sequence of colors is for plots where the pattern is cyclical such as a seasons. This is also used in
           12x24 plot from a shear by time of day.

        **Example usage**
        ::
            import brightwind as bw

            # The color palette used by brightwind library by default based on main colors can be visualised using
            # code below.

            import matplotlib.pyplot as plt
            fig, axes = plt.subplots()
            for i,c in enumerate(bw.analyse.plot.COLOR_PALETTE.color_list):
                axes.bar([i],1,color=c)

            # Main colors can be reset by using the code below for each color of the defined color palette. When doing
            # this the linked color_list, color_map, color_map_cyclical and adjusted lightness color variables are
            # automatically updated as a consequence.

            bw.analyse.plot.COLOR_PALETTE.primary = '#3366CC'

            # If required, the individual adjusted lightness colors can also be reset by using the example code below:

            bw.analyse.plot.COLOR_PALETTE.primary_10 = '#0a1429'

            # The colors used for defining the color_map can also be reset by using the example code below:

            bw.analyse.plot.COLOR_PALETTE.color_map_colors = ['#ccfffc',   # lightest primary
                                                              '#00b4aa',   # vert-dark
                                                              '#008079']   # darkest primary

            # The colors used for defining the color_map_cyclical can also be reset by using the example code below:

            bw.analyse.plot.COLOR_PALETTE.color_map_cyclical_colors = ['#ccfffc',   # lightest primary
                                                                       '#00b4aa',   # vert-dark
                                                                       '#008079',   # darkest primary
                                                                       '#ccfffc']   # lightest primary

            # The hex color value corresponding to an input color adjusted by a percentage lightness (e.g 0.5 %)
            # can be derived as below:

            bw.analyse.plot.COLOR_PALETTE._adjust_color_lightness('#171a28', 0.5)

        """
        self.primary = '#9CC537'        # slightly darker than YellowGreen #9acd32, rgb(156/255, 197/255, 55/255)
        self.secondary = '#2E3743'      # asphalt, rgb(46/255, 55/255, 67/255)
        self.tertiary = '#9B2B2C'       # red'ish, rgb(155/255, 43/255, 44/255)
        self.fourth = '#E57925'         # Vivid Tangelo, rgb(229/255, 121/255, 37/255)
        self.fifth = '#ffc008'          # Mikado Yellow, rgb(255/255, 192/255, 8/255)
        self.sixth = '#AB8D60'          # Bronze (Metallic), rgb(171/255, 141/255, 96/255)
        self.seventh = '#A4D29F'        # green'ish, rgb(164/255, 210/255, 159/255)
        self.eighth = '#01958a'         # Dark Cyan, rgb(1/255, 149/255, 138/255)
        self.ninth = '#3D636F'          # blue grey, rgb(61/255, 99/255, 111/255)
        self.tenth = '#A49E9D'          # Quick Silver, rgb(164/255, 158/255, 157/255)
        self.eleventh = '#DA9BA6'       # Parrot Pink, rgb(218/255, 155/255, 166/255)

        self._color_map_colors = None
        self._color_map_cyclical_colors = None

        # set the mpl color cycler to our colors. It has 10 colors
        # mpl.rcParams['axes.prop_cycle']

    @property
    def primary(self):
        return self._primary

    @primary.setter
    def primary(self, val):
        self._primary = val
        self.primary_10 = self._adjust_color_lightness(self._primary, 0.1)  # darkest green, 10% lightness of primary
        self.primary_35 = self._adjust_color_lightness(self._primary, 0.35)  # dark green, 35% lightness of primary
        self.primary_80 = self._adjust_color_lightness(self._primary, 0.80)  # light green, 80% lightness of primary
        self.primary_90 = self._adjust_color_lightness(self._primary, 0.90)  # light green, 90% lightness of primary
        self.primary_95 = self._adjust_color_lightness(self._primary, 0.95)  # lightest green, 95% lightness of primary

    @property
    def tenth(self):
        return self._tenth

    @tenth.setter
    def tenth(self, val):
        self._tenth = val
        self.tenth_40 = self._adjust_color_lightness(self._tenth, 0.4)  # darker grayish red, 40% lightness of tenth

    @property
    def color_list(self):
        return [self.primary, self.secondary, self.tertiary, self.fourth, self.fifth, self.sixth,
                self.seventh, self.eighth, self.ninth, self.tenth, self.eleventh, self.primary_35]

    @property
    def color_map(self):
        return self._set_col_map('color_map', self._get_color_map_colors())

    @property
    def color_map_bw(self):
        return self._set_col_map('bw_color_map', [COLOR_PALETTE.primary, COLOR_PALETTE.eighth, COLOR_PALETTE.ninth]) # #COLOR_PALETTE.sixth,COLOR_PALETTE.tenth

    @staticmethod
    def _set_col_map(color_map_name, col_map_colors):
        return LinearSegmentedColormap.from_list(color_map_name, col_map_colors, N=256)

    @property
    def color_map_colors(self):
        return self._get_color_map_colors()

    @color_map_colors.setter
    def color_map_colors(self, val):
        self._color_map_colors = val

    def _get_color_map_colors(self):
        if self._color_map_colors:
            # if the user has set a new color_map_color, use them.
            color_map_colors = self._color_map_colors
        else:
            # if the user has not set a new one, use our default colors.
            color_map_colors = [self.primary_95,  # lightest primary
                                self.primary,     # primary
                                self.primary_10]  # darkest primary
        return color_map_colors

    @property
    def color_map_cyclical(self):
        return self._set_col_map('color_map_cyclical', self._get_color_map_cyclical_colors())

    @property
    def color_map_cyclical_colors(self):
        return self._get_color_map_cyclical_colors()

    @color_map_cyclical_colors.setter
    def color_map_cyclical_colors(self, val):
        self._color_map_cyclical_colors = val

    def _get_color_map_cyclical_colors(self):
        if self._color_map_cyclical_colors:
            # if the user has set a new color_map_cyclical_color, use them.
            color_map_cyclical_colors = self._color_map_cyclical_colors
        else:
            # if the user has not set a new one, use our default colors.
            color_map_cyclical_colors = [self.secondary, self.fifth, self.primary, self.tertiary, self.secondary]
        return color_map_cyclical_colors

    @staticmethod
    def _adjust_color_lightness(input_color, lightness_factor):
        """
        Generate the color corresponding to the input primary color corrected by a lightness factor indicating the
        percentage lightness. Lighter colors are obtained with a factor > 0.5 and darker colors with a factor < 0.5.
        This function is converting the input color to hue, saturation, lightness (hsl) format and adjusting only the
        lightness value. See https://www.w3schools.com/colors/colors_picker.asp for more info.

        :param input_color:         Input base color to adjust. It can accept any matplotlib recognised color inputs.
                                    (see https://matplotlib.org/stable/tutorials/colors/colors.html) or `numpy.ma.masked`
        :type input_color:          str, tuple, list
        :param lightness_factor:    Percentage of lightness (>0.5) or darkness (<0.5). Value should be between 0 and 1.
        :type lightness_factor:     float
        :return:                    color in hex format
        :rtype:                     hex
        """
        
        lightness_factor = 0 if lightness_factor is None else lightness_factor
        if lightness_factor < 0 or lightness_factor > 1:
            raise TypeError("Invalid lightness_factor, this should be between or equal to 0 and 1.")

        r, g, b = tuple(255 * np.array(mpl.colors.to_rgb(input_color)))  # convert to rgb format
        hue, lightness, saturation = rgb2hls(r / 255.0, g / 255.0, b / 255.0)
        r, g, b = hls2rgb(hue, lightness_factor, saturation)
        return mpl.colors.to_hex([r, g, b])


COLOR_PALETTE = _ColorPalette()


def _colormap_to_colorscale(cmap, n_colors):
    """
    Function that transforms a matplotlib colormap to a list of colors
    """
    return [to_hex(cmap(k*1/(n_colors-1))) for k in range(n_colors)]


def plot_monthly_means(data, coverage=None, ylbl='', legend=True, external_legend=False,  show_grid=True,
                       xtick_delta='1MS'):
    """
    Plots the monthly means where x is the time axis. Monthly mean data and coverage can be derived using brightwind
    function average_data_by_period(data, period='1MS', return_coverage=True).

    :param data:            A Series or DataFrame with data aggregated as the mean for each month.
    :type data:             pd.Series or pd.DataFrame
    :param coverage:        A Series with data coverage for each month. Optional.
                            The coverage is plotted along with the data only if a single series is passed to data.
    :type coverage:         None or pd.Series, optional
    :param ylbl:            y axis label used on the left hand axis, defaults to ''.
    :type ylbl:             str, optional
    :param legend:          Flag to show a legend (True) or not (False). Default is True.
    :type legend:           bool, optional
    :param external_legend: Flag to show legend outside and above the plot area (True) or show it inside
                            the plot (False). Default is False.
    :type external_legend:  bool, optional
    :param show_grid:       Flag to show a grid in the plot area (True) or not (False) when 'coverage' is None.
                            Default True.
                            When 'coverage' is not None then the grid is always shown.
    :type show_grid:        bool, optional
    :param xtick_delta:     String to give the frequency of x ticks and their associated labels. Given as a pandas
                            frequency string, remembering that S at the end is required for months starting on the first
                            day of the month. Default '1MS'.
    :type xtick_delta:      str, optional
    :return:                A plot of monthly means for the input data.
    :rtype:                 matplotlib.figure.Figure

     **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)
        monthly_means, coverage = bw.average_data_by_period(data[['Spd80mN','Spd80mS']], period='1MS',
                                                            return_coverage=True)
        
        # to plot monthly mean speed for all speeds without coverage and legend outside and above the plot area.
        bw.analyse.plot.plot_monthly_means(monthly_means, ylbl='Speed [m/s]', legend=True, external_legend=True,
                                           xtick_delta='1MS')

        # to plot monthly mean speed for all speeds without coverage, legend inside and not to show the grid.
        bw.analyse.plot.plot_monthly_means(monthly_means, ylbl='Speed [m/s]', legend=True, external_legend=True, 
                                           show_grid=False)

        # to plot monthly mean speed with coverage and xticks every 3 months.
        bw.analyse.plot.plot_monthly_means(monthly_means.Spd80mN, coverage.Spd80mN_Coverage, 
                                           ylbl='Speed [m/s]', legend=True, external_legend=False, xtick_delta='3MS')

    """
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    is_dataframe = len(data.shape) > 1
    legend_items = list(data.columns) if is_dataframe else [data.name]

    if is_dataframe:
        ax.set_prop_cycle(color=COLOR_PALETTE.color_list)
        ax.plot(data, '-o')
    else:
        ax.plot(data, '-o', color=COLOR_PALETTE.primary)   

    if legend:
        ncol_legend = min((len(legend_items)+1)//2, 6) if len(legend_items) > 6 else 6
        legend_kwargs = {
            'bbox_to_anchor': (0.5, 1), 'loc': 'lower center', 'ncol': ncol_legend,
            } if external_legend else {}
        ax.legend(legend_items, **legend_kwargs)

    ax.set_ylabel(ylbl)

    start_date = data.index[0]
    end_date = data.index[-1]
    xticks = pd.date_range(
        start=pd.Timestamp(year=start_date.year, month=start_date.month, day=1), end=end_date, freq=xtick_delta
        )
    ax.set_xticks(xticks[(xticks >= start_date) & (xticks <= end_date)])
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=20, ha='center')
    ax.set_xlim(data.index[0] - pd.Timedelta('20days'), data.index[-1] + pd.Timedelta('20days'))

    if show_grid:
        ax.grid(linestyle='-', color='lightgrey')

    if coverage is not None:
        plot_coverage = not (len(coverage.shape) > 1 and coverage.shape[1] > 1)

        if plot_coverage:
            ax.clear()
            line = ax.plot(data, '-o', color=COLOR_PALETTE.secondary, label=data.name)
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
            ax2.yaxis.grid(True, color='lightgrey')
            ax2.set_axisbelow(True)
            ax.patch.set_visible(False)
            ax2.set_ylabel('Coverage [-]')
            ax.set_ylabel(ylbl)
            if legend:
                # Patch needed purely for legend
                coverage_patch = Patch(facecolor=COLOR_PALETTE.primary, 
                                       edgecolor=COLOR_PALETTE.secondary,
                                       linewidth=0.3,
                                       label='Data coverage')
                handles = [line[0], coverage_patch]
                labels = [data.name, 'Data coverage']
                fig = plt.gcf()  # Get current figure
                legend_kwargs = {
                    'bbox_to_anchor': (0.5, 0.96), 'loc': 'upper center', 'ncol': 2
                    } if external_legend else {'bbox_to_anchor': (0.1, 0.1), 'loc': 'lower left'}
                fig.legend(handles, labels, **legend_kwargs)
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")
            ax.set_xticks(xticks[(xticks >= start_date) & (xticks <= end_date)])
            ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
            for tick in ax.get_xticklabels():
                tick.set_rotation(20)
                tick.set_horizontalalignment('center')
            plt.close()
            return ax2.get_figure()
    plt.close()
    return ax.get_figure()


def _timeseries_subplot(x, y, x_label=None, y_label=None, x_limits=None, y_limits=None, x_tick_label_angle=25,
                        line_marker_types=None, line_colors=None, subplot_title=None,
                        legend=True, ax=None, external_legend=False, legend_fontsize=12, show_grid=True, 
                        colourmap=None):
    """
    Plots a timeseries subplot where x is the time axis.

    :param x:                       The x-axis values in a time format.
    :type x:                        pandas.core.indexes.datetimes.DatetimeIndex,
                                    list(pandas._libs.tslibs.timestamps.Timestamp) or list(numpy.datetime64)
                                    or np.array(numpy.datetime64)
    :param y:                       The y-axis values.
    :type y:                        pd.DataFrame, pd.Series or list or np.array
    :param x_label:                 Label for the x axis. Default is None.
    :type x_label:                  str or None
    :param y_label:                 Label for the y axis. Default is None.
    :type y_label:                  str or None
    :param x_limits:                x-axis min and max limits. Default is None.
    :type x_limits:                 tuple, None
    :param y_limits:                y-axis min and max limits. Default is None.
    :type y_limits:                 tuple, None
    :param x_tick_label_angle:      The angle to rotate the x-axis tick labels by.
                                    Default is 25, i.e. the tick labels will be horizontal.
    :type x_tick_label_angle:       float or int
    :param line_marker_types:       String or list of marker type(s) to use for the timeseries plot. Default is None.
                                    If only one marker type is provided then all timeseries will use the same marker,
                                    otherwise the number of marker types provided will need to be equal to the number
                                    of columns in the y input. Marker type options are as for
                                    https://matplotlib.org/stable/api/markers_api.html
    :type line_marker_types:        list or str or None
    :param line_colors:             Line colors used for the timeseries plot. Colors input can be given as:
                                        1) Single str (https://matplotlib.org/stable/gallery/color/named_colors.html)
                                           or Hex (https://www.w3schools.com/colors/colors_picker.asp) or tuple (Rgb):
                                           all plotted timeseries will use the same color.
                                        2) List of str or Hex or Rgb: the number of colors provided needs to be
                                           at least equal to the number of columns in the y input.
                                        3) None: the default COLOR_PALETTE.color_list will be used for plotting.
    :type line_colors:              str or list or tuple or None
    :param subplot_title:           Title show on top of the subplot. Default is None.
    :type subplot_title:            str or None
    :param legend:                  Boolean to choose if legend is shown. Default is True.
    :type legend:                   Bool
    :param ax:                      Subplot axes to which assign the subplot to in a plot. If None then a single plot is
                                    generated
    :type ax:                       matplotlib.axes._subplots.AxesSubplot or None
    :param external_legend:         Flag to show legend outside and above the plot area (True) or show it inside
                                    the plot (False). Default is False.
    :type external_legend:          bool
    :param legend_fontsize:         Font size for legend. Default 12.
    :type legend_fontsize:          int
    :param show_grid:               Flag to show a grid in the plot area (True) or not (False). Default True.
    :type show_grid:                bool
    :param colourmap:               Optional argument to choose line colours equally spaced from a default colour map. 
                                    Default None.
    :type colourmap:                bool
    :return:                        A timeseries subplot
    :rtype:                         matplotlib.axes._subplots.AxesSubplot

     **Example usage**
    ::
        import brightwind as bw
        import matplotlib.pyplot as plt
        data = bw.load_csv(bw.demo_datasets.demo_data)
        wspd_cols = ['Spd80mN', 'Spd80mS', 'Spd60mN', 'Spd60mS', 'Spd40mN', 'Spd40mS']

        # To plot only one subplot in a figure and set different marker types for each line
        fig, axes = plt.subplots(1, 1)
        bw.analyse.plot._timeseries_subplot(data.index, data[wspd_cols],
                                            line_marker_types=['.', 'o', 'v', '^', '<', None], ax=axes)

        # To plot multiple subplots in a figure without legend and with x and y labels and assigning subplot_title
        fig, axes = plt.subplots(2, 1)
        bw.analyse.plot._timeseries_subplot(data.index, data[wspd_cols], ax=axes[0], legend=False,
                                         x_label=None, y_label='Spd80mS', subplot_title='Speed [m/s]')
        bw.analyse.plot._timeseries_subplot(data.index, data.Dir78mS, ax=axes[1], legend=False,
                                         x_label='Time', y_label='Dir78mS', subplot_title='Direction [deg]')

        # To plot multiple timeseries with different x values/length in the same subplot,
        # only for 1st timeseries set marker type and color different than default, set legend for all timeseries
        fig, axes = plt.subplots(1, 1)
        ts_plot = bw.analyse.plot._timeseries_subplot(data['2016-02-10':'2016-03-10'].index,
                                                      data['2016-02-10':'2016-03-10'].Spd60mS, line_marker_types='.',
                                                      line_colors=bw.analyse.plot.COLOR_PALETTE.tertiary, ax=axes)
        ts_plot = bw.analyse.plot._timeseries_subplot(data.index, data[['Spd80mS','Spd60mN']], ax=axes)

        # To set the x and y axis limits by using a tuple, set x_tick_label_angle to 45 deg and change x_ticks major
        # label format to "W%W" and location to be weekly and the first day of the week (monday).
        import pandas as pd
        import matplotlib
        fig, axes = plt.subplots(1, 1)
        sub_plot = bw.analyse.plot._timeseries_subplot(data.index, data.Dir58mS, x_label='Dir78mS', y_label='Dir58mS',
                                               x_limits=(pd.datetime(2016,2,1),pd.datetime(2016,7,10)),
                                               y_limits=(250,300), x_tick_label_angle=45, ax=axes)
        sub_plot.axes.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(byweekday=0))
        sub_plot.axes.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("W%W"))

        # To set the matplotlib default color list
        import matplotlib.pyplot as plt
        prop_cycle = plt.rcParams['axes.prop_cycle']
        mpl_colors = prop_cycle.by_key()['color']
        fig, axes = plt.subplots(1, 1)
        sub_plot = bw.analyse.plot._timeseries_subplot(data.index, data.Dir58mS, line_colors=mpl_colors)

        # To use an external legend without a grid displayed and size 8 font
        fig, axes = plt.subplots(1, 1)
        bw.analyse.plot._timeseries_subplot(data.index, data[wspd_cols],
                                            line_marker_types=['.', 'o', 'v', '^', '<', None], ax=axes, 
                                            external_legend=True, legend_fontsize=8, show_grid=False)

        # To use the default colourmap
        fig, axes = plt.subplots(1, 1)
        columns_to_plot = ['Spd_Met_2m', 'Spd_20m', 'Spd_40m', 'Spd_60m', 'Spd_80m', 'Spd_100m','Spd_124m','Spd_150m',
                            'Spd_175m','Spd_200m']
        bw.analyse.plot._timeseries_subplot(data.index, data[columns_to_plot], ax=axes, colourmap=True, 
                                             external_legend=True, legend_fontsize=8)

    """

    if ax is None:
        ax = plt.gca()

    if type(y) is pd.Series:
        y = pd.DataFrame(y)
    elif type(y) is list:
        y = pd.DataFrame(y, columns=['y'])
    elif type(y) is dict:
        y = pd.DataFrame.from_dict(y)
    elif type(y) is np.ndarray:
        y = pd.DataFrame(y, columns=['y']) 
    if len(x) != len(y):
        ValueError("Length of x input is different than length of y input. Length of these must be the same.")

    if type(line_marker_types) is list:
        if len(y.columns) != len(line_marker_types):
            ValueError("You have provided 'line_markers_type' input as a list but length is different than the number "
                    "of columns provided for y input. Please make sure that length is the same.")
    elif (type(line_marker_types) is str) or (line_marker_types is None):
        line_marker_types = [line_marker_types] * len(y.columns)

    if colourmap:
        cmap = COLOR_PALETTE.color_map_bw
        line_colors = bw.analyse.plot._colormap_to_colorscale(cmap, len(y.columns))
    else:
        if line_colors is None:
            line_colors = COLOR_PALETTE.color_list

        if type(line_colors) is list:
            if len(y.columns) > len(line_colors):
                ValueError("You have provided 'line_colors' input as a list but length is smaller than the number "
                        "of columns provided for y input. Please make sure that length is the same.")
        else:
            line_colors = [line_colors] * len(y.columns)

    alpha = [1 for _ in line_colors]
    num_colour_sets = int(np.ceil(len(y.columns) / len(line_colors)))
    alpha_delta = 0.8 / num_colour_sets
    if len(line_colors) < len(y.columns):
        while len(line_colors) < len(y.columns):
            for i in range(num_colour_sets - 1):
                for j in range(len(line_colors)):
                    line_colors.append(line_colors[j])
                    alpha.append(1 - alpha_delta * (i + 1))

    for i_col, (col, marker_type) in enumerate(zip(y.columns, line_marker_types)):
        ax.plot(x, y.iloc[:, i_col], marker=marker_type, color=line_colors[i_col], label=col, alpha=alpha[i_col])

    if x_limits is None:
        if type(x) == pd.DatetimeIndex:
            x_min = x.min()
            x_max = x.max()
        else:
            x_min = np.min(x)
            x_max = np.max(x)
        x_limits = (x_min, x_max)
    ax.set_xlim(x_limits[0], x_limits[1])

    if y_limits is not None:
        ax.set_ylim(y_limits[0], y_limits[1])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.tick_params(axis="x", rotation=x_tick_label_angle)

    if legend:
        ncol_legend = min((len(y.columns)+1)//2, 6) if len(y.columns) > 6 else 6
        legend_kwargs = {
            'bbox_to_anchor': (0.5, 1), 'loc': 'lower center', 'ncol': ncol_legend,
            } if external_legend else {}
        if legend_fontsize:
            legend_kwargs['fontsize'] = legend_fontsize
        ax.legend(**legend_kwargs)
    if show_grid:
        ax.grid(linestyle='-', color='lightgrey')

    if subplot_title is not None:
        ax.set_title(subplot_title, fontsize=mpl.rcParams['axes.labelsize'])

    return ax


def plot_timeseries(data, date_from=None, date_to=None, x_label=None, y_label=None, y_limits=None,
                    x_tick_label_angle=25, line_colors=None, legend=True, figure_size=(15, 8),
                    external_legend=False, legend_fontsize=12, show_grid=True, colourmap=None):
    """
    Plot a timeseries of data.

    :param data:                    Data in the form of a Pandas DataFrame/Series to plot.
    :type data:                     pd.DataFrame, pd.Series
    :param date_from:               Start date as string in format YYYY-MM-DD or YYYY-MM-DD hh:mm. Start date is
                                    included in the sliced data. If format of date_from is YYYY-MM-DD, then the first
                                    timestamp of the date is used (e.g if date_from=2023-01-01 then 2023-01-01 00:00
                                    is the first timestamp of the sliced data). If date_from is not given then the
                                    sliced data are taken from the first timestamp of the dataset.
    :type date_from:                str
    :param date_to:                 End date as string in format YYYY-MM-DD or YYYY-MM-DD hh:mm. End date is not
                                    included in the sliced data. If format date_to is YYYY-MM-DD, then the last
                                    timestamp of the previous day is used (e.g if date_to=2023-02-01 then
                                    2023-01-31 23:50 is the last timestamp of the sliced data). If date_to is not given
                                    then the sliced data are taken up to the  last timestamp of the dataset.
    :type date_to:                  str
    :param x_label:                 Label for the x-axis. Default is None.
    :type x_label:                  str, None
    :param y_label:                 Label for the y-axis. Default is None.
    :type y_label:                  str, None
    :param y_limits:                y-axis min and max limits. Default is None.
    :type y_limits:                 tuple, None
    :param x_tick_label_angle:      The angle to rotate the x-axis tick labels by.
                                    Default is 25, i.e. the tick labels will be horizontal.
    :type x_tick_label_angle:       float or int
    :param line_colors:             Line colors used for the timeseries plot. Colors input can be given as:
                                        1) Single str (https://matplotlib.org/stable/gallery/color/named_colors.html)
                                           or Hex (https://www.w3schools.com/colors/colors_picker.asp) or tuple (Rgb):
                                           all plotted timeseries will use the same color.
                                        2) List of str or Hex or Rgb: the number of colors provided needs to be
                                           at least equal to the number of columns in the y input.
                                        3) None: the default COLOR_PALETTE.color_list will be used for plotting.
    :type line_colors:              str or list or tuple or None
    :param legend:                  Boolean to choose if legend is shown. Default is True.
    :type legend:                   Bool
    :param figure_size:             Figure size in tuple format (width, height). Default is (15, 8).
    :type figure_size:              tuple
    :param external_legend:         Flag to show legend outside and above the plot area (True) or show it inside
                                    the plot (False). Default is False.
    :type external_legend:          bool
    :param legend_fontsize:         Font size for legend. Default 12.
    :type legend_fontsize:          int
    :param show_grid:               Flag to show a grid in the plot area (True) or not (False). Default True.
    :type show_grid:                bool
    :param colourmap:               Optional argument to choose line colours equally spaced from a colour map. The given 
                                    string should be the name of a matplotlib colormap. Default None.
    :type colourmap:                Optional[str | bool]
    :return:                        A timeseries plot
    :rtype:                         matplotlib.figure.Figure

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)

        # To plot few variables
        bw.plot_timeseries(data[['Spd40mN', 'Spd60mS', 'T2m']])

        # To set a start date, do not show legend, and set x and y labels
        bw.plot_timeseries(data.Spd40mN, date_from='2017-09-01', x_label='Time', y_label='Spd40mN', legend=False)

        # To set an end date
        bw.plot_timeseries(data.Spd40mN, date_to='2017-10-01')

        # For specifying a slice and set axis tilted by 25 deg
        bw.plot_timeseries(data.Spd40mN, date_from='2017-09-01', date_to='2017-10-01', x_tick_label_angle=25)

        # To set the y-axis minimum to 0
        bw.plot_timeseries(data.Spd40mN, date_from='2017-09-01', date_to='2017-10-01', y_limits=(0, None))

        # To set the y-axis maximum to 25
        bw.plot_timeseries(data.Spd40mN, date_from='2017-09-01', date_to='2017-10-01', y_limits=(0, 25))

        # To change line colors respect default and set figure size to (20, 4)
        bw.plot_timeseries(data[['Spd40mN', 'Spd60mS', 'T2m']], line_colors= ['#009991', '#171a28',  '#726e83'],
                           figure_size=(20, 4))

        # To set the matplotlib default color list
        import matplotlib.pyplot as plt
        prop_cycle = plt.rcParams['axes.prop_cycle']
        mpl_colors = prop_cycle.by_key()['color']
        bw.plot_timeseries(data[['Spd40mN', 'Spd60mS', 'T2m']], line_colors= mpl_colors)

        # To use an external legend with fontsize 14 without grid displayed
        bw.plot_timeseries(data[['Spd40mN', 'Spd60mS', 'T2m']], external_legend=True, legend_fontsize=14, 
                           show_grid=False)

    """
    if line_colors is None:
        line_colors = COLOR_PALETTE.color_list

    fig, axes = plt.subplots(figsize=figure_size)
    if isinstance(data, pd.Series):
        data_to_slice = data.copy(deep=False).to_frame()
    else:
        data_to_slice = data.copy()
    sliced_data = utils.slice_data(data_to_slice, date_from, date_to)
    _timeseries_subplot(sliced_data.index, sliced_data, x_label=x_label, y_label=y_label,
                        y_limits=y_limits, x_tick_label_angle=x_tick_label_angle,
                        line_colors=line_colors, legend=legend, ax=axes, external_legend=external_legend,
                        legend_fontsize=legend_fontsize, show_grid=show_grid, colourmap=colourmap)
    plt.close()
    return fig


def _derive_axes_limits_for_scatter_plot(x, y):
    x_min, x_max = (round(np.nanmin(x) - 0.5), -(-np.nanmax(x) // 1))
    y_min, y_max = (round(np.nanmin(y) - 0.5), -(-np.nanmax(y) // 1))
    return x_min, x_max, y_min, y_max


def _scatter_subplot(x, y, trendline_y=None, trendline_x=None, line_of_slope_1=False,
                     x_label=None, y_label=None, x_limits=None, y_limits=None, axes_equal=True, subplot_title=None,
                     trendline_dots=False, scatter_color=None,
                     trendline_color=None, legend=True, scatter_name=None,
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
    :param scatter_color:       Color to assign to scatter data. If None default is COLOR_PALETTE.primary
    :type scatter_color:        str or Hex or Rgb
    :param trendline_color:     Color to assign to trendline data. If None default is COLOR_PALETTE.secondary
    :type trendline_color:      str or Hex or Rgb
    :param legend:              Boolean to choose if legend is shown.
    :type legend:               Bool
    :param scatter_name:        Label to assign to scatter data in legend if legend is True. If None then the label
                                assigned is 'Data points'
    :type scatter_name:         str or None
    :param trendline_name:      Label to assign to trendline data in legend if legend is True. If None then the label
                                assigned is 'Regression line'
    :type trendline_name:       str or None
    :param ax:                  Subplot axes to which assign the subplot to in a plot. If None then a single plot is
                                generated
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

    if scatter_color is None:
        scatter_color = COLOR_PALETTE.primary

    if trendline_color is None:
        trendline_color = COLOR_PALETTE.secondary

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
        ax.plot([low, high], [low, high], color=COLOR_PALETTE.tenth_40, label='1:1 line')

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

    # if x and y names are the same then rename pd.Series names to be unique
    if x.name == y.name:
        x = x.rename(x.name + '_x')
        y = y.rename(y.name + '_y')

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

    return scat_plot


def plot_scatter_by_sector(x, y, wdir, trendline_y=None, sort_trendline_inputs=False, line_of_slope_1=True, sectors=12,
                           x_limits=None, y_limits=None, axes_equal=True, figure_size=(10, 10.2), **kwargs):
    """
    Plot scatter subplots (with shared x and y axis) of x versus y for each directional sector. If a trendline
    timeseries is given as input then this is also plotted in the graph. The line with slope 1 and passing
    through the origin is shown if line_of_slope_1=True

    :param x:                       The x-axis values or reference variable.
    :type x:                        pd.Series
    :param y:                       The y-axis values or target variable.
    :type y:                        pd.Series
    :param wdir:                    Timeseries of wind directions.
    :type wdir:                     pd.Series
    :param trendline_y:             Series of trendline y values. This needs to be derived using the x-axis timeseries
                                    values as it is plotted against x.
    :type trendline_y:              pd.Series
    :param sort_trendline_inputs:   Boolean to chose if trendline inputs should be sorted in ascending order. Default is
                                    False and trendline inputs are not sorted.
    :type sort_trendline_inputs:    Bool
    :param line_of_slope_1:         Boolean to choose to plot the line with slope one and passing through the origin.
    :type line_of_slope_1:          Bool
    :param sectors:                 Number of directional sectors
    :type sectors:                  int
    :param x_limits:                x-axis min and max limits. Can be set to None to let the code derive the min and max
                                    from the x_wspd_series.
    :type x_limits:                 tuple, None
    :param y_limits:                y-axis min and max limits. Can be set to None to let the code derive the min and max
                                    from  the y_wspd_series.
    :type y_limits:                 tuple, None
    :param axes_equal:              Boolean to set the units for the x and y axes to be equal. If x_limits and y_limits
                                    are both None then the two axes limits are set to be the same.
    :type axes_equal:               Bool
    :param figure_size:             Figure size in tuple format (width, height)
    :type figure_size:              tuple
    :param kwargs:                  Additional keyword arguments for matplotlib.pyplot.subplot
    :returns:                       matplotlib.figure.Figure

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

        if sort_trendline_inputs:
            trendline_x_input = sorted([x_data for x_data in x[logic_sect]])
            trendline_y_input = sorted([y_data for y_data in trendline_y_input])
        else:
            trendline_x_input = x[logic_sect]

        _scatter_subplot(x[logic_sect], y[logic_sect], trendline_y_input, trendline_x=trendline_x_input,
                         line_of_slope_1=line_of_slope_1, x_label=None, y_label=None,
                         x_limits=x_limits, y_limits=y_limits, axes_equal=axes_equal,
                         subplot_title=str(round(ratio_min)) + '-' + str(round(ratio_max)),
                         legend=False, ax=ax_subplot)

    fig.text(0.5, 0.06, x.name, va='center', ha='center', fontsize=mpl.rcParams['axes.labelsize'])
    fig.text(0.06, 0.5, y.name, va='center', ha='center', rotation='vertical',
             fontsize=mpl.rcParams['axes.labelsize'])
    plt.close()
    return fig


def _create_colormap(color_rgb_from, color_rgb_to):
    """
    Create colormap for gradient image.

    :param color_rgb_from:  The rgb color from which the gradient image starts.
    :type color_rgb_from:   tuple(float, float, float)
    :param color_rgb_from:  The rgb color to which the gradient image ends.
    :type color_rgb_from:   tuple(float, float, float)
    :return:                Colormap
    :rtype:                 matplotlib.colors.ListedColormap

     **Example usage**
    ::
        import brightwind as bw
        # Create colormap from asphalt to red giving as input rgb colors
        cmp = bw.analyse.plot._create_colormap((46/255, 55/255, 67/255),(155,43,44))

    """
    n_max = 256
    vals = np.ones((n_max, 4))
    for i, (a, b) in enumerate(zip(color_rgb_from, color_rgb_to)):
        vals[:, i] = np.linspace(a, b, n_max)
    cmp = ListedColormap(vals)
    return cmp


def _gradient_image(direction=0.3, cmap_range=(0, 1)):
    """
    Create a gradient based on a colormap.

    :param direction:   The direction of the gradient. This is a number in range 0 (=vertical) to 1 (=horizontal).
    :type direction:    float
    :param cmap_range:  The fraction (cmin, cmax) of the colormap that should be used for the gradient,
                        where the complete colormap is (0, 1).
    :type cmap_range:   tuple(float, float)
    :return:            gradient
    :rtype:             numpy.ndarray

     **Example usage**
    ::
        import brightwind as bw
        # Create a gradient based on a colormap with horizontal gradient.
        gradient = bw.analyse.plot._gradient_image(direction=1, cmap_range=(0, 1))

    """
    phi = direction * np.pi / 2
    v = np.array([np.cos(phi), np.sin(phi)])
    x_grad = np.array([[v @ [1, 0], v @ [1, 1]],
                       [v @ [0, 0], v @ [0, 1]]])
    a, b = cmap_range
    x_grad = a + (b - a) / x_grad.max() * x_grad

    return x_grad


def _bar_subplot(data, x_label=None, y_label=None, min_bar_axis_limit=None, max_bar_axis_limit=None,
                 min_bin_axis_limit=None, max_bin_axis_limit=None, bin_tick_labels=None, x_tick_label_angle=0,
                 bin_tick_label_format=None, bar_tick_label_format=None, subplot_title=None, legend=False,
                 total_width=0.8, line_width=0.3, vertical_bars=True, ax=None):
    """
    Plots a bar subplot, either vertical or horizontal bars, from a pd.Series or pd.Dataframe where the interval of the
    bars is the data.index and the height/length of the bars are the values.
    If the data input is a Dataframe then the bars are plotted for each column of the Dataframe and with a different
    colour for each dataset. The colours are defined as for the brightwind library standard `COLOR_PALETTE.color_list`.
    Colours can be changed only updating the `COLOR_PALETTE.color_list`.
    The user can chose if the bars are horizontal or vertical based on vertical_bars boolean user input. The function
    is handling data.index with format float, int, pd.DatetimeIndex and bin ranges (ie [-0.5, 0.5)).

    :param data:                    The data values used to define the index and the height/length of the bars to plot.
    :type data:                     pd.Series or pd.Dataframe
    :param x_label:                 Label for the x axis
    :type x_label:                  str or None
    :param y_label:                 Label for the y axis
    :type y_label:                  str or None
    :param min_bar_axis_limit:      min y or x-axis limit depending if bar plot is vertical or horizontal.
    :type min_bar_axis_limit:       float or None
    :param max_bar_axis_limit:      max y or x-axis limit depending if bar plot is vertical or horizontal.
    :type max_bar_axis_limit:       float or None
    :param min_bin_axis_limit:      min x or y-axis limit depending if bar plot is vertical or horizontal.
    :type min_bin_axis_limit:       float or None
    :param max_bin_axis_limit:      max x or y limit depending if bar plot is vertical or horizontal.
    :type max_bin_axis_limit:       float or None
    :param bin_tick_labels:         List of x or y tick labels depending if bar plot is vertical or horizontal. The list
                                    must have the same number of entries as the data index.
                                    If left as None, the tick labels will be taken from the data index.
    :type bin_tick_labels:          list or None
    :param x_tick_label_angle:      The angle to rotate the x-axis tick labels by.
                                    Default is 0, i.e. the tick labels will be horizontal.
    :type x_tick_label_angle:       float or int
    :param bin_tick_label_format:   Set the formatter of the major ticker for the bin axis.
                                    Default is None where the behaviour will be to use
                                    matplotlib.dates.DateFormatter("%Y-%m") (e.g. 2022-05) for pandas.DatetimeIndex tick
                                    labels and the default from matplotlib.axis.set_major_formatter for all other
                                    number types. To change the default format for a Datetime to be, for example
                                    '2022-05-20', use matplotlib.dates.DateFormatter("%Y-%m-%d").
    :type bin_tick_label_format:    matplotlib.ticker.Formatter or matplotlib.dates.DateFormatter
    :param bar_tick_label_format:   Set the formatter of the major ticker for the bar axis.
                                    Default is None where the tick label format will be the default from
                                    matplotlib.axis.set_major_formatter.
    :type bar_tick_label_format:    matplotlib.ticker.Formatter
    :param subplot_title:           Title to show on top of the subplot
    :type subplot_title:            str or None
    :param legend:                  Boolean to choose if legend is shown.
    :type legend:                   Bool
    :param total_width:             Width of each group of bars in percentage between 0 and 1. Default is 0.8, which is
                                    80% of the available space for the group of bars.
    :type total_width:              float or int
    :param line_width:              Width of the bar or group of bar's border/edge. Values from 0 to 5.
                                    If 0, don't draw edges. Default is 0.3.
    :type line_width:               float or int
    :param vertical_bars:           Boolean to choose for having horizontal or vertical bars. Default is True to plot
                                    vertical bars.
    :type vertical_bars:            Bool
    :param ax:                      Subplot axes to which assign the subplot to in a plot. If None then a single plot is
                                    generated
    :type ax:                       matplotlib.axes._subplots.AxesSubplot or None
    :return:                        A bar subplot
    :rtype:                         matplotlib.axes._subplots.AxesSubplot

     **Example usage**
    ::
        import brightwind as bw
        from matplotlib.dates import DateFormatter
        data = bw.load_csv(bw.demo_datasets.demo_data)

        # To plot data with pd.DatetimeIndex, multiple columns, with bars total width of 20 days, line_width=0.3 and
        # assigning bin_tick_label_format as "%Y-%m-%d"
        fig = plt.figure(figsize=(15, 8))
        average_data, coverage = bw.average_data_by_period(data[['Spd80mN', 'Spd80mS', 'Spd60mN']], period='1M',
                                                           return_coverage=True)
        bw.analyse.plot._bar_subplot(coverage, max_bar_axis_limit=1, total_width=20/31, line_width=0.3,
                                    bin_tick_label_format=DateFormatter("%Y-%m-%d"), vertical_bars=True)

        # To plot multiple subplots in a figure
        fig, axes = plt.subplots(1, 2)
        bw.analyse.plot._bar_subplot(coverage[['Spd80mN_Coverage', 'Spd80mS_Coverage']], max_bar_axis_limit=1,
                                     total_width=20/31, line_width=0.3,  vertical_bars=True, ax=axes[0])
        bw.analyse.plot._bar_subplot(coverage['Spd60mN_Coverage'], max_bar_axis_limit=1, total_width=20/31,
                                     line_width=0.3, vertical_bars=True, ax=axes[1])

        # To plot data with integer data.index, multiple columns, horizontal bars and
        # setting bin_tick_labels, subplot title, bar_tick_label_format as percentage and with legend
        from matplotlib.ticker import PercentFormatter
        test_data = pd.DataFrame.from_dict({'mast': [99.87, 99.87, 99.87],'lidar': [97.11, 92.66, 88.82]})
        test_data.index=[50, 65, 80]
        fig = plt.figure(figsize=(15, 8))
        bw.analyse.plot._bar_subplot(test_data, x_label='Data Availability [%]', y_label='Measurement heights [m]',
                                     max_bar_axis_limit=100, bin_tick_labels=['a','b','c'],
                                     bar_tick_label_format=PercentFormatter(), subplot_title='coverage',
                                     legend=True, vertical_bars=False)

        # To plot data with integer data.index, multiple columns, horizontal bars and
        # setting minimum and maximum bin axis limits
        bw.analyse.plot._bar_subplot(test_data, x_label='Data Availability [%]', y_label='Measurement heights [m]',
                                     max_bar_axis_limit=100, min_bin_axis_limit=0, max_bin_axis_limit=100,
                                     subplot_title='coverage', legend=True, vertical_bars=False)

        # To plot frequency distribution data with index as bin ranges (ie [-0.5, 0.5)), single column,
        # vertical bars, default total_width
        distribution = bw.analyse.analyse._derive_distribution(data['Spd80mN'].to_frame(),
                                                               var_to_bin_against=data['Spd80mN'].to_frame(),
                                                               aggregation_method = '%frequency')
        fig = plt.figure(figsize=(15, 8))
        bw.analyse.plot._bar_subplot(distribution.replace([np.inf, -np.inf], np.NAN).dropna(), y_label='%frequency')

    """
    
    if ax is None:
        ax = plt.gca()

    if type(data) is pd.Series:
        data = data.to_frame()

    bar_colors = COLOR_PALETTE.color_list
    if len(data.columns) > len(bar_colors):
        raise ValueError('The number of variables to plot is higher than the number of colors implemented in the '
                         'brightwind library standard `COLOR_PALETTE.color_list`. The number of variables should '
                         'be lower than {} or you should assign to the `COLOR_PALETTE.color_list` a list of colors '
                         'with same length than the number of variables to plot.'.format(len(bar_colors)))

    if (total_width < 0) or (total_width > 1):
        raise ValueError('The total_width value should be between 0 and 1.')

    if (line_width < 0) or (line_width > 5):
        raise ValueError('The line_width value should be between 0 and 5.')

    if (bin_tick_labels is not None) and (bin_tick_labels != []):
        if len(bin_tick_labels) != len(data.index):
            raise ValueError('The length of the input bin_tick_labels list is different than the number '
                             'of entries in the data index.')

    if min_bar_axis_limit is None:
        min_bar_axis_limit = 0
    if max_bar_axis_limit is None:
        max_bar_axis_limit = data.max().max()*1.1

    if vertical_bars:
        ax.set_ylim(min_bar_axis_limit, max_bar_axis_limit)
    else:
        ax.set_xlim(min_bar_axis_limit, max_bar_axis_limit)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    index_time = False
    if isinstance(data.index[0], pd.Interval):
        data_bins = [i.mid for i in data.index]
    elif type(data.index) == pd.DatetimeIndex:
        index_time = True
        data_bins = mdates.date2num(data.index)
    else:
        data_bins = data.index

    if bin_tick_label_format is None:
        if index_time:
            bin_tick_label_format = DateFormatter("%Y-%m")

    bin_min_step = 1 if len(data_bins) == 1 else np.diff(data_bins).min()
    total_width = bin_min_step * total_width

    if vertical_bars:
        ax.set_xticks(data_bins)
        ax.set_xlim(data_bins[0] - total_width, data_bins[-1] + total_width)
        if bin_tick_labels is not None:
            ax.set_xticklabels(bin_tick_labels)
        if index_time:
            ax.locator_params(axis='x', nbins=10)
        if bin_tick_label_format is not None:
            ax.xaxis.set_major_formatter(bin_tick_label_format)
        if bar_tick_label_format is not None:
            ax.yaxis.set_major_formatter(bar_tick_label_format)
        ax.grid(True, axis='y', zorder=0)
    else:
        ax.set_yticks(data_bins)
        ax.set_ylim(data_bins[0] - total_width, data_bins[-1] + total_width)
        if bin_tick_labels is not None:
            ax.set_yticklabels(bin_tick_labels)
        if index_time:
            ax.locator_params(axis='y', nbins=10)
        if bin_tick_label_format is not None:
            ax.yaxis.set_major_formatter(bin_tick_label_format)
        if bar_tick_label_format is not None:
            ax.xaxis.set_major_formatter(bar_tick_label_format)
        ax.grid(True, axis='x', zorder=0)

    ax.tick_params(axis="x", rotation=x_tick_label_angle)

    # Number of bars per group
    n_bars = len(data.columns)
    # Bars width
    bar_width = total_width / n_bars
    # List containing handles for the drawn bars, used for the legend
    bars = []
    # Iterate over all data
    for i, name in enumerate(data.columns):
        bar_color = bar_colors[i]
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
        r, g, b = tuple(255 * np.array(mpl.colors.to_rgb(bar_color)))  # hex to rgb format
        hue, lightness, saturation = rgb2hls(r / 255, g / 255, b / 255)
        lightness_factor = max(min(lightness * 1.8, 1.0), 0.0)

        for data_bar, data_bin in zip(data[name], data_bins):
            if not np.isnan(data_bar):
                if vertical_bars:
                    ax.imshow(np.array([[mpl.colors.to_rgb(bar_color)],
                                        [mpl.colors.to_rgb(COLOR_PALETTE._adjust_color_lightness(bar_color,
                                                           lightness_factor=lightness_factor))]]),
                              interpolation='gaussian', extent=(data_bin + x_offset - bar_width / 2,
                                                                data_bin + x_offset + bar_width / 2, 0,
                                                                data_bar),
                              aspect='auto', zorder=2)#3
                    bar = ax.bar(data_bin + x_offset, data_bar, width=bar_width,
                                 edgecolor=bar_color, linewidth=line_width, fill=False,
                                 zorder=1)#5
                else:
                    cmp = _create_colormap(mpl.colors.to_rgb(COLOR_PALETTE._adjust_color_lightness(bar_color,
                                                             lightness_factor=lightness_factor)),
                                           mpl.colors.to_rgb(bar_color))
                    ax.imshow(_gradient_image(direction=1, cmap_range=(0, 1)), cmap=cmp,
                              interpolation='gaussian',
                              extent=(0, data_bar, data_bin + x_offset - bar_width / 2,
                                      data_bin + x_offset + bar_width / 2),
                              aspect='auto', zorder=2, vmin=0, vmax=1)
                    bar = ax.barh(data_bin + x_offset, data_bar, height=bar_width,
                                  edgecolor=bar_color, linewidth=line_width, fill=False,
                                  zorder=1)#2
        # Add a handle to the last drawn bar, which we'll need for the legend
        bar[0].set_color(bar_color)
        bar[0].set_fill(True)
        bars.append(bar[0])

    ax.set_axisbelow(True)

    if vertical_bars:
        ax.set_xlim(min_bin_axis_limit, max_bin_axis_limit)
    else:
        ax.set_ylim(min_bin_axis_limit, max_bin_axis_limit)

    if legend:
        ax.legend(bars, data.keys())

    if subplot_title is not None:
        ax.set_title(subplot_title, fontsize=mpl.rcParams['ytick.labelsize'])

    return ax


def plot_freq_distribution(data, max_y_value=None, x_tick_labels=None, x_label=None, y_label=None, legend=False,
                           total_width=0.8):
    """
    Plot distribution given as input and derived using _derive_distribution() function. The input can be a Pandas Series
    or a Dataframe. If it is a Dataframe then the distribution is plotted for each column of the Dataframe and with
    a different colour for each dataset. The colours are defined as for the brightwind library standard
    `COLOR_PALETTE.color_list`. Colours can be changed only updating the `COLOR_PALETTE.color_list`.

    ** THIS FUNCTION WILL BE REMOVED IN A FUTURE VERSION OF BRIGHTWIND LIBRARY **

    :param data:            The input distribution derived using bw.analyse.analyse._derive_distribution().
    :type data:             pd.Series or pd.Dataframe
    :param max_y_value:     y-axis max limit. It can be set to None to let the code derive the max from
                            the data column values.
    :type max_y_value:      float or int, None
    :param x_tick_labels:   x-axis tick labels provided in a list. It can be set to None to let the code derive the
                            x tick labels from the frequency distribution index.
    :type x_tick_labels:    list, None
    :param x_label:         Title for the x-axis. If None, there won't be a title for the x axis.
    :type x_label:          str, None
    :param y_label:         Title for the y-axis. If None, there won't be a title for the y axis.
    :type y_label:          str, None
    :param legend:          Boolean to choose if legend is shown.
    :type legend:           Bool
    :param total_width:     Width of each group of bars in percentage between 0 and 1. Default is 0.8, which is
                            80% of the available space.
    :type total_width:      float or int
    :returns:               matplotlib.figure.Figure

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)

        # Plot frequency distribution of only one variable, without x tick labels
        distribution = bw.analyse.analyse._derive_distribution(data['Spd40mN'],
                                                               var_to_bin_against=data['Spd40mN'], bins=None,
                                                               aggregation_method = '%frequency').rename('Spd40mN')
        bw.analyse.plot.plot_freq_distribution(distribution.replace([np.inf, -np.inf], np.NAN).dropna(),
                                               max_y_value=None,x_tick_labels=[], x_label=None,
                                               y_label='%frequency')

        # Plot distribution of counts for multiple variables, having the bars to take the total_width
        distribution1 = bw.analyse.analyse._derive_distribution(data['Spd40mN'],
                                                                var_to_bin_against=data['Spd40mN'],
                                                                aggregation_method='count').rename('Spd40mN')
        distribution2 = bw.analyse.analyse._derive_distribution(data['Spd80mN'],
                                                                var_to_bin_against=data['Spd80mN'],
                                                                aggregation_method='count').rename('Spd80mN')

        bw.analyse.plot.plot_freq_distribution(pd.concat([distribution1, distribution2], axis=1
                                                        ).replace([np.inf, -np.inf], np.NAN).dropna(),
                                               max_y_value=None, x_tick_labels=None, x_label=None,
                                               y_label='count', total_width=1, legend=True)

    """

    warnings.warn("In a future version of brightwind, `plot_freq_distribution` will be removed. Please use "
                  "'dist()' or `_bar_subplot()` instead.", category=FutureWarning)

    bar_tick_label_format = None
    if y_label:
        if '%' in y_label:
            bar_tick_label_format = PercentFormatter()

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    _bar_subplot(data.replace([np.inf, -np.inf], np.NAN), x_label=x_label,
                 y_label=y_label, max_bar_axis_limit=max_y_value,
                 bin_tick_labels=x_tick_labels, bar_tick_label_format=bar_tick_label_format,
                 legend=legend, total_width=total_width, ax=ax)
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
    ax.set_thetagrids(np.arange(0, 360, 360.0 / sectors))
    sector_mid_points = []
    widths = []
    for i in result.index:
        angular_pos_start = (np.pi / 180.0) * float(i.split('-')[0])
        angular_pos_end = (np.pi / 180.0) * float(i.split('-')[-1])
        if angular_pos_start < angular_pos_end:
            sector_mid_points.append((angular_pos_start + angular_pos_end) / 2.0)
            widths.append(angular_pos_end - angular_pos_start - (np.pi / 180))
        else:
            sector_mid_points.append((np.pi + (angular_pos_start + angular_pos_end) / 2.0) % 360)
            widths.append(2 * np.pi - angular_pos_start + angular_pos_end - (np.pi / 180))
    max_contour = (ext_data.max() + ext_data.std())
    contour_spacing = max_contour / 10
    num_digits_to_round = 0
    while contour_spacing * (10 ** num_digits_to_round) <= 1:
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


def plot_TI_by_speed(wspd, wspd_std, ti, min_speed=3, percentile=90, IEC_class=None):
    """
    Plot turbulence intensity graphs alongside with IEC standards

    :param wspd:        Wind speed data series
    :type wspd:         pandas.Series
    :param wspd_std:    Wind speed standard deviation data series
    :type wspd_std:     pandas.Series
    :param ti:          DataFrame returned from bw.TI.by_speed()
    :type ti:           pandas.DataFrame
    :param IEC_class:   Default value is None, this means that default IEC class 2005 is used. Note: we have removed
                        option to include IEC Class 1999 as no longer appropriate. This may need to be placed in a
                        separate function when updated IEC standard is released. For custom class give as input
                        a pandas.DataFrame having first column name as 'windspeed' and other columns reporting the
                        results of applying the IEC class formula for a range of wind speeds. See format as shown in
                        example usage.
    :param min_speed:   Set the minimum wind speed. Default is 3 m/s.
    :type min_speed:    integer or float
    :param percentile:  The percentile used for deriving the representative TI. This should be the same value as used
                        when calling the bw.TI.by_speed() function that derives the representative TI.
                        Default is set to 90 percentile.
    :type percentile:   float, integer
    :type IEC_class:    None or pandas.DataFrame
    :return:            Plots scatter plot of turbulence intensity (TI) & distribution of TI by speed bins
                        derived as for statistics below and the IEC Class curves defined as for IEC_class input.

                             * Mean_TI (average TI for a speed bin),
                             * Rep_TI (representative TI set at a certain percentile and derived from bw.TI.by_speed())

    **Example usage**
        ::
            import brightwind as bw
            data = bw.load_csv(bw.demo_datasets.demo_data)

            # Plots scatter plot of turbulence intensity (TI) and distribution of TI by speed bins and
            # IEC Class curves
            _ , ti_dist = bw.TI.by_speed(data.Spd80mN, data.Spd80mNStd, return_data=True)
            bw.analyse.plot.plot_TI_by_speed(data.Spd80mN, data.Spd80mNStd, ti_dist, IEC_class=None)

            # set min speed for plot
            _ , ti_dist = bw.TI.by_speed(data.Spd80mN, data.Spd80mNStd, return_data=True)
            bw.analyse.plot.plot_TI_by_speed(data.Spd80mN, data.Spd80mNStd, ti_dist, min_speed=0, IEC_class=None)

            # Plot TI distribution by speed bins and give as input custom IEC_class pandas.DataFrame
            IEC_class = pd.DataFrame({'windspeed': list(range(0,26)),
                          'IEC Class A': list(0.16 * (0.75 + (5.6 / np.array(range(0,26)))))}
                          ).replace(np.inf, 0)
            bw.analyse.plot.plot_TI_by_speed(data.Spd80mN, data.Spd80mNStd, ti_dist, IEC_class=IEC_class)

    """

    # IEC Class 2005

    if IEC_class is None:
        IEC_class = pd.DataFrame(np.zeros([26, 4]), columns=['windspeed', 'IEC Class A', 'IEC Class B', 'IEC Class C'])
        for n in range(1, 26):
            IEC_class.iloc[n, 0] = n
            IEC_class.iloc[n, 1] = 0.16 * (0.75 + (5.6 / n))
            IEC_class.iloc[n, 2] = 0.14 * (0.75 + (5.6 / n))
            IEC_class.iloc[n, 3] = 0.12 * (0.75 + (5.6 / n))
    elif type(IEC_class) is not pd.DataFrame:
        raise ValueError("The IEC_class input must be a pandas.DataFrame with format as stated in function docstring.")
    elif not pd.api.types.is_numeric_dtype(IEC_class.iloc[:, 0]):
        raise ValueError("The IEC_class input must be a pandas.DataFrame where the first column is 'windspeed' and "
                         "this needs to have numeric values.")

    common_idxs = wspd.index.intersection(wspd_std.index)
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.scatter(wspd.loc[common_idxs], wspd_std.loc[common_idxs] / wspd.loc[common_idxs],
               color=COLOR_PALETTE.primary, alpha=0.3, marker='.')
    ax.plot(ti.index.values, ti.loc[:, 'Mean_TI'].values, color=COLOR_PALETTE.secondary, label='Mean_TI')
    ax.plot(ti.index.values, ti.loc[:, 'Rep_TI'].values, color=COLOR_PALETTE.primary_35,
            label='Rep_TI ({}th percentile)'.format(percentile))
    for icol in range(1, len(IEC_class.columns)):
        ax.plot(IEC_class.iloc[:, 0], IEC_class.iloc[:, icol], color=COLOR_PALETTE.color_list[1+icol],
                linestyle='dashed', label=IEC_class.columns[icol])

    ax.set_xlim(min_speed, 25)
    ax.set_ylim(0, 0.6)
    ax.set_xticks(np.arange(min_speed, 26, 1))
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
    ax.plot(np.append(radians, radians[0]), pd.concat([ti, pd.DataFrame(ti.iloc[0]).T])['Mean_TI'],
            color=COLOR_PALETTE.primary, linewidth=4, figure=fig, label='Mean_TI')
    maxlevel = ti['Mean_TI'].max() + 0.1
    ax.set_ylim(0, maxlevel)
    ax.scatter(np.radians(wdir), turbulence, color=COLOR_PALETTE.secondary, alpha=0.3, s=1, label='TI')
    ax.legend(loc=8, framealpha=1)
    plt.close()
    return ax.get_figure()


def plot_shear_by_sector(scale_variable, wind_rose_data, calc_method='power_law'):
    """
    Plot shear by directional sectors and wind rose.

    :param scale_variable:  Shear values by directional sectors derived with brightwind.Shear.BySector().
    :type scale_variable:   pandas.Series
    :param wind_rose_data:  Wind speed %frequency distribution by sectors, with wind direction sector as row indexes.
                            This distribution is derived using brightwind.dist_by_dir_sector() function.
    :type wind_rose_data:   pandas.Series
    :param calc_method:     Method to use for calculation, either 'power_law' (returns alpha) or 'log_law'
                            (returns the roughness coefficient).
    :type calc_method:      str
    :return:                Plots shear values by directional sectors & distribution of wind speed by directional bins.

    **Example usage**
        ::
            import brightwind as bw
            data = bw.load_csv(bw.demo_datasets.demo_data)

            alpha = pd.Series({'345.0-15.0': 0.216, '15.0-45.0': 0.196, '45.0-75.0': 0.170, '75.0-105.0': 0.182,
                     '105.0-135.0': 0.148, '135.0-165.0': 0.129, '165.0-195.0': 0.156, '195.0-225.0': 0.159,
                     '225.0-255.0': 0.160, '255.0-285.0': 0.169, '285.0-315.0': 0.187, '315.0-345.0': 0.188})

            roughness = pd.Series({'345.0-15.0': 0.537, '15.0-45.0': 0.342, '45.0-75.0': 0.156, '75.0-105.0': 0.231,
                         '105.0-135.0': 0.223, '135.0-165.0': 0.124, '165.0-195.0': 0.135,
                         '195.0-225.0': 0.145, '225.0-255.0': 0.108, '255.0-285.0': 0.149,
                         '285.0-315.0': 0.263, '315.0-345.0': 0.275})

            wind_rose_plot, wind_rose_dist = bw.analyse.analyse.dist_by_dir_sector(data.Spd80mS, data.Dir78mS,
                                                    direction_bin_array=None,
                                                    sectors=12,
                                                    direction_bin_labels=None,
                                                    return_data=True)

            # Plots shear by directional sectors with calculation method as 'power law'.
            bw.analyse.plot.plot_shear_by_sector(scale_variable=alpha, wind_rose_data=wind_rose_dist,
            calc_method='power_law')

            # Plots shear by directional sectors with calculation method as 'log law'.
            bw.analyse.plot.plot_shear_by_sector(scale_variable=roughness, wind_rose_data=wind_rose_dist,
            calc_method='log_law')


    """
    result = wind_rose_data.copy(deep=False)
    radians = np.radians(utils._get_dir_sector_mid_pts(scale_variable.index))
    sectors = len(result)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    bin_edges = pd.Series([], dtype='float64')
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
    scale_to_fit = max(scale_variable[np.isfinite(scale_variable)]) / max(result / 100)
    wind_rose_r = (result / 100) * scale_to_fit
    bin_edges = np.array(bin_edges)
    width = pd.Series([], dtype='float64')

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

    maxlevel = (max(scale_variable_y[np.isfinite(scale_variable_y)])) + max(
        scale_variable_y[np.isfinite(scale_variable_y)]) * .1
    ax.set_thetagrids(radians * 180 / np.pi)
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
                      radial_limits=None, annotate=True, figure_size=(10, 10)):
    """
    Accepts a DataFrame table or a dictionary with multiple ratio of anemometer pairs per sector, a wind direction,
    multiple distributions of anemometer ratio pairs per sector, along with 2 anemometer names,
    and plots the speed ratio by sector. Optionally can include anemometer boom directions also.

    :param sec_ratio:         Ratio of wind speed timeseries. One or more ratio timeseries can be input as a dict.
    :type sec_ratio:          pandas.Series or dict
    :param wdir:              Direction series. If multiple direction series entered in dict format, number of series
                              must equal number of sector ratios. The first direction series is references the first
                              sector ratio and so on.
    :type wdir:               pandas.Series or dict
    :param sec_ratio_dist:    DataFrames from SectorRatio.by_sector()
    :type sec_ratio_dist:     pandas.Series or dict
    :param col_names:         A list of strings containing column names of wind speeds, first string is divisor and
                              second is dividend.
    :type col_names:          list[float]
    :param boom_dir_1:        Boom orientation in degrees of speed_col_name_1. Defaults to -1. One or more boom
                              orientations can be accepted. If multiple orientations, number of orientations must equal
                              number of anemometer pairs.
    :type boom_dir_1:         float or list[float]
    :param boom_dir_2:        Boom orientation in degrees of speed_col_name_2. Defaults to -1. One or more boom
                              orientations can be accepted. If multiple orientations, number of orientations must equal
                              number of anemometer pairs.
    :type boom_dir_2:         float or list[float]
    :param radial_limits:     the min and max values of the radial axis. Defaults to +0.05 of max ratio and -0.1 of min.
    :type radial_limits:      tuple[float] or list[float]
    :param annotate:          Set to True to show annotations on plot. If False then the annotation at the bottom of
                              the plot and the radial labels indicating the sectors are not shown.
    :type annotate:           bool
    :type annotate:           bool
    :param figure_size:       Figure size in tuple format (width, height)
    :type figure_size:        tuple[int]
    :returns:                 A speed ratio plot showing average speed ratio by sector and scatter of individual data
                              points.

    **Example usage**
    ::

    import brightwind as bw
    data = bw.load_csv(bw.demo_datasets.demo_data)

    wspd1, wspd2 = data['Spd80mN'], data['Spd80mS']
    wdir = data['Dir78mS']

    # calculate the ratio between wind speeds
    min_spd = 3
    sec_rat = bw.analyse.analyse._calc_ratio(wspd1, wspd2, min_spd)

    # calculate mean wind speed ratio per sector
    sec_rat_plot, sec_rat_dist = bw.dist_by_dir_sector(sec_rat, wdir, aggregation_method='mean', return_data=True)
    sec_rat_dist = sec_rat_dist.rename('Mean_Sector_Ratio').to_frame()

    # find the common indices between wind speed and wind direction
    common_idx   = sec_rat.index.intersection(wdir.index)

    # plot the sector ratio
    bw.plot_sector_ratio(sec_rat, wdir, sec_rat_dist, [wspd1.name, wspd2.name])

    # plot the sector ratio with boom orientations, radial limits, and larger figure size
    bw.plot_sector_ratio(sec_rat, wdir, sec_rat_dist, [wspd1.name, wspd2.name],
                         boom_dir_1=0, boom_dir_2=180, radial_limits=(0.8, 1.2), figure_size=(15, 15))

    """

    if type(sec_ratio) == pd.core.series.Series:
        sec_ratio = {0: sec_ratio}

    if type(sec_ratio_dist) == pd.core.frame.DataFrame:
        sec_ratio_dist = {0: sec_ratio_dist}

    if type(col_names) is list:
        col_names = {0: col_names}

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
    fig, axes = plt.subplots(row, col, figsize=figure_size, subplot_kw={'projection': 'polar'})
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

    :param sec_ratio:         Series of wind speed timeseries.
    :type sec_ratio:          pandas.Series
    :param wdir:              Direction timeseries.
    :type wdir:               pandas.Series
    :param sec_ratio_dist:    DataFrame from SectorRatio.by_sector()
    :type sec_ratio_dist:     pandas.Series
    :param col_names:         A list of strings containing column names of wind speeds, first string is divisor and
                              second is dividend.
    :type col_names:          list[str]
    :param boom_dir_1:        Boom orientation in degrees of speed_col_name_1. Defaults to -1, hidden from the plot.
    :type boom_dir_1:         float
    :param boom_dir_2:        Boom orientation in degrees of speed_col_name_2. Defaults to -1, hidden from the plot.
    :type boom_dir_2:         float
    :param radial_limits:     The min and max values of the radial axis. Defaults to +0.05 of max ratio and -0.1 of min.
    :type radial_limits:      tuple[float] or list[float]
    :param annotate:          Set to True to show annotations on plot.
    :type annotate:           bool
    :param font_size:         Size of font in plot annotation. Defaults to 10.
    :type font_size:          int
    :param ax:                Subplot axes to which the subplot is assigned. If None subplot is displayed on its own.
    :type ax:                 matplotlib.axes._subplots.AxesSubplot or None
    :returns:                 A speed ratio plot showing average speed ratio by sector and scatter of individual
                              data points.

    **Example usage**
    ::
        import brightwind as bw
        data = bw.load_csv(bw.demo_datasets.demo_data)

        wspd1, wspd2 = data['Spd80mN'], data['Spd80mS']
        wdir = data['Dir78mS']

        # calculate the ratio between wind speeds
        min_spd = 3
        sec_rat = bw.analyse.analyse._calc_ratio(wspd1, wspd2, min_spd)

        # calculate mean wind speed ratio per sector
        sec_rat_plot, sec_rat_dist = bw.dist_by_dir_sector(sec_rat, wdir, aggregation_method='mean', return_data=True)
        sec_rat_dist = sec_rat_dist.rename('Mean_Sector_Ratio').to_frame()

        # find the common indices between wind speed and wind direction
        common_idx   = sec_rat.index.intersection(wdir.index)

        # plot the sector ratio
        bw.analyse.plot._plot_sector_ratio_subplot(sec_rat.loc[common_idx], wdir.loc[common_idx], sec_rat_dist,
                                                   [wspd1.name, wspd2.name])

        # plot the sector ratio with booms, radial limits, no annotation, and larger font size
        bw.analyse.plot._plot_sector_ratio_subplot(sec_rat.loc[common_idx], wdir.loc[common_idx], sec_rat_dist,
                                                   [wspd1.name, wspd2.name], boom_dir_1=0, boom_dir_2=180,
                                                   radial_limits=(0.8, 1.2), annotate=False, font_size=20)

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
    ax.plot(np.append(radians, radians[0]), pd.concat([sec_ratio_dist['Mean_Sector_Ratio'], sec_ratio_dist.iloc[0]]),
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
    else:
        ax.axes.xaxis.set_ticklabels([])
    ax.scatter(np.radians(wdir), sec_ratio, color=COLOR_PALETTE.secondary, alpha=0.3, s=1)

    for item in ([ax.title] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)

    return ax


def plot_power_law(avg_alpha, avg_c, wspds, heights, max_plot_height=None, avg_slope=None, avg_intercept=None,
                   plot_both=False):
    if max_plot_height is None:
        max_plot_height = max(heights)

    plot_heights = np.arange(1, max_plot_height + 1, 1)
    speeds = avg_c * (plot_heights ** avg_alpha)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('Speed [m/s]')
    ax.set_ylabel('Elevation [m]')
    ax.plot(speeds, plot_heights, '-', color=COLOR_PALETTE.primary, label='power_law')
    ax.scatter(wspds, heights, marker='o', color=COLOR_PALETTE.secondary)
    if plot_both is True:
        plot_heights = np.arange(1, max_plot_height + 1, 1)
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
    """
    Function used by Shear.TimeOfDay for plotting the hourly shear for each calendar month or an average of all months.

    The color map used for plotting the shear by time of day for each calendar month depends on the plot_type input:
        1) if 'line' 'step' the COLOR_PALETTE.color_map_cyclical is used
        2) if '12x24' the 'COLOR_PALETTE.color_map is used
    The color used for plotting the average of all months shear is COLOR_PALETTE.primary.

    :param df:          Series of average shear by time of day or DataFrame of shear by time of day for
                        each calendar month.
    :type df:           pandas.Series or pandas.DataFrame
    :param calc_method: Method used by Shear.TimeOfDay for shear calculation, either 'power_law' or 'log_law'.
                        Input used for defining label of y axis.
    :type calc_method:  str
    :param plot_type:   Type of plot to be generated. Options include 'line', 'step' and '12x24'. Default is 'step'.
    :type plot_type:    str
    :returns:           A shear by time of day plot

    """
    df_copy = df.copy()

    if calc_method == 'power_law':
        label = 'Average Shear'
    elif calc_method == 'log_law':
        label = 'Roughness Coefficient'
    else:
        label = calc_method

    if plot_type == '12x24':
        df.columns = np.arange(1, 13, 1)
        df.index = np.arange(0, 24, 1)
        return plot_12x24_contours(df, label=(label, 'mean'), plot='tod')

    else:
        colors = _colormap_to_colorscale(COLOR_PALETTE.color_map_cyclical, 13)
        colors = colors[:-1]
        if len(df.columns) == 1:
            colors[0] = COLOR_PALETTE.primary

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlabel('Time of Day')
        ax.set_ylabel(label)

        # create x values for plot
        idx = pd.date_range('2017-01-01 00:00', '2017-01-01 23:00', freq='1H').hour

        if plot_type == 'step':
            df = df.shift(+1, axis=0)
            df.iloc[0, :] = df_copy.tail(1).values
            for i in range(0, len(df.columns)):
                ax.step(idx, df.iloc[:, i], label=df.iloc[:, i].name, color=colors[i])

        if plot_type == 'line':
            for i in range(0, len(df.columns)):
                ax.plot(idx, df.iloc[:, i], label=df.iloc[:, i].name, color=colors[i])

        ax.legend(bbox_to_anchor=(1.1, 1.05))
        ax.set_xticks([ix.hour for ix in df.index])
        ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter("%H-%M"))
        _ = plt.xticks(rotation=90)
        return ax.get_figure()


def plot_dist_matrix(matrix, colorbar_label=None, xticklabels=None, yticklabels=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    cm = ax.pcolormesh(matrix, cmap=COLOR_PALETTE.color_map)
    ax.set(xlim=(0, matrix.shape[1]), ylim=(0, matrix.shape[0]))
    ax.set(xticks=np.array(range(0, matrix.shape[1])) + 0.5, yticks=np.array(range(0, matrix.shape[0])) + 0.5)
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


def render_table(data, col_width=3.0, row_height=0.625, font_size=16, header_color=None,
                 row_colors=None, edge_color='w', bbox=[0, 0, 1, 1],
                 header_columns=0, show_col_head=1,
                 ax=None, cellLoc='center', padding=0.01, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    if row_colors is None:
        row_colors = [COLOR_PALETTE.primary_90, 'w']

    if header_color is None:
        header_color = COLOR_PALETTE.primary

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
