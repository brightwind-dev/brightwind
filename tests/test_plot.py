import pytest
import brightwind as bw
from brightwind.analyse import plot as bw_plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import PercentFormatter
from matplotlib.dates import DateFormatter
import matplotlib as mpl
from colormap import rgb2hex, rgb2hls, hls2rgb

DATA = bw.load_csv(bw.demo_datasets.demo_data)
DATA = bw.apply_cleaning(DATA, bw.demo_datasets.demo_cleaning_file)
WSPD_COLS = ['Spd80mN', 'Spd80mS', 'Spd60mN', 'Spd60mS', 'Spd40mN', 'Spd40mS']
WDIR_COLS = ['Dir78mS', 'Dir58mS', 'Dir38mS']


def test_plot_sector_ratio():
    wspd_1 = bw.utils.utils._convert_df_to_series(DATA['Spd80mN']).dropna()
    wspd_2 = bw.utils.utils._convert_df_to_series(DATA['Spd80mS']).dropna()
    wdir = bw.utils.utils._convert_df_to_series(DATA['Dir78mS']).dropna()

    sec_rats = {}
    sec_rats_dists = {}

    sec_rat = bw.analyse.analyse._calc_ratio(wspd_1, wspd_2, 3)
    common_idx = sec_rat.index.intersection(wdir.index)
    sec_rat_plot, sec_rat_dist = bw.dist_by_dir_sector(sec_rat.loc[common_idx], wdir.loc[common_idx], return_data=True)
    sec_rat_dist = sec_rat_dist.rename('Mean_Sector_Ratio').to_frame()
    sec_rats[0] = sec_rat
    sec_rats_dists[0] = sec_rat_dist

    fig = bw.plot_sector_ratio(sec_rats, DATA['Dir78mS'], sec_rats_dists, [DATA['Spd80mN'].name, DATA['Spd80mS'].name],
                               boom_dir_1=-1, boom_dir_2=-1, radial_limits=None, figure_size=(10, 10))


def test_plot_timeseries():
    bw.plot_timeseries(DATA[['Spd40mN', 'Spd60mS', 'T2m']])
    bw.plot_timeseries(DATA[['Spd40mN']], date_from='2017-09-01')
    bw.plot_timeseries(DATA.Spd40mN, date_from='2017-09-01', x_label='Time', y_label='Spd40mN', legend=False)
    bw.plot_timeseries(DATA.Spd40mN, date_to='2017-10-01 00:00')
    bw.plot_timeseries(DATA.Spd40mN, date_from='2017-09-01', date_to='2017-10-01 00:00', x_tick_label_angle=25)
    bw.plot_timeseries(DATA.Spd40mN, date_from='2017-09-01', date_to='2017-10-01 00:00', y_limits=(0, None))
    bw.plot_timeseries(DATA.Spd40mN, date_from='2017-09-01', date_to='2017-10-01 00:00', y_limits=None)
    bw.plot_timeseries(DATA.Spd40mN, date_from='2017-09-01', date_to='2017-10-01 00:00', y_limits=(0, 25))
    bw.plot_timeseries(DATA.Spd40mN, date_from='2017-09-01', date_to='2017-10-01 00:00', y_limits=(None, 25))
    bw.plot_timeseries(DATA[['Spd40mN', 'Spd60mS', 'T2m']], line_colors=['#009991', '#171a28', '#726e83'],
                       figure_size=(20, 4))
    time_series_plot = bw.plot_timeseries(
        DATA[['Spd40mN', 'Spd60mS', 'T2m']], line_colors=['#009991', '#171a28', '#726e83'],figure_size=(20, 4), 
        external_legend=True, legend_fontsize=14, show_grid=True)
    axes = time_series_plot.get_axes()
    ax = axes[0]
    x_gridlines = ax.xaxis.get_gridlines()
    y_gridlines = ax.yaxis.get_gridlines()
    axes = time_series_plot.get_axes()[0]
    legend = axes.get_legend()
    actual_fontsize = legend.get_texts()[0].get_fontsize()

    assert len(x_gridlines) > 0, "No X-axis gridlines found"
    assert len(y_gridlines) > 0, "No Y-axis gridlines found"
    assert time_series_plot.axes[0].get_legend()._bbox_to_anchor is not None
    assert actual_fontsize == 14, f"Expected font size {14}, got {actual_fontsize}"

    time_series_subplot =  bw.analyse.plot._timeseries_subplot(DATA.index, DATA[['Spd40mN', 'Spd60mS', 'T2m']], 
                                                               line_colors=['#009991', '#171a28', '#726e83'], 
                                                               external_legend=True, legend_fontsize=6, show_grid=True)
    legend = time_series_subplot.get_legend()
    actual_fontsize = legend.get_texts()[0].get_fontsize()
    assert actual_fontsize == 6, f"Expected font size {6}, got {actual_fontsize}"

    columns_to_plot = [a for a in DATA.columns if "Spd" in a]
    time_series_subplot =  bw.analyse.plot._timeseries_subplot(DATA.index, DATA[columns_to_plot], 
                                                               external_legend=True, legend_fontsize=6, show_grid=True)    
    actual_colours = np.ones_like(columns_to_plot)
    for _, line in enumerate(time_series_subplot.get_lines()):
        for j, column in enumerate(columns_to_plot):
            if line.get_label() == column:
                actual_colours[j] = line.get_color()
    expected_colors = [
        '#9CC537', '#2E3743', '#9B2B2C', '#E57925', '#ffc008', '#AB8D60', 
        '#A4D29F', '#01958a', '#3D636F', '#A49E9D', '#DA9BA6', '#6e8c27',
        '#9CC537', '#2E3743', '#9B2B2C', '#E57925', '#ffc008', '#AB8D60', 
        '#A4D29F', '#01958a', '#3D636F', '#A49E9D', '#DA9BA6', '#6e8c27'
    ]
    assert all(a == e for a, e in zip(actual_colours, expected_colors))
    
    time_series_subplot =  bw.analyse.plot._timeseries_subplot(DATA.index, DATA[columns_to_plot], colourmap=True,
                                                               external_legend=True, legend_fontsize=6, show_grid=True)
    actual_colors = [line.get_color() for line in time_series_subplot.get_lines()]
    
    actual_colours = np.ones_like(columns_to_plot)
    for _, line in enumerate(time_series_subplot.get_lines()):
        for j, column in enumerate(columns_to_plot):
            if line.get_label() == column:
                actual_colours[j] = line.get_color()
    expected_colors = ['#9cc537', '#8abf41', '#78ba4b', '#65b454', '#53ae5e', '#41a968', '#2fa372', '#1c9d7b', '#0a9885', 
                       '#059288', '#0c8c85', '#138682', '#1a807f', '#217b7c', '#287579', '#2f6f75', '#366972', '#3d636f']
    assert all(a == e for a, e in zip(actual_colours, expected_colors))
    
    assert True


def test_plot_scatter():
    bw.plot_scatter(DATA.Spd80mN, DATA.Spd80mS)
    bw.plot_scatter(DATA.Spd80mN, DATA[['Spd80mS']])
    bw.plot_scatter(DATA.Dir78mS, DATA.Dir58mS, x_label='Dir78mS', y_label='Dir58mS',
                    x_limits=(50, 300), y_limits=(250, 300))
    
    bw.plot_scatter(DATA.Dir78mS, DATA.Dir58mS, x_label='Dir78mS', y_label='Dir58mS',
                    x_limits=(50, 300), y_limits=(250, 300), line_of_slope_1=True)
    
    plt_scatter = bw.plot_scatter(DATA.Spd80mN, DATA.Spd80mS, trendline_y=[0, 15],trendline_x=[0, 10],
                                  line_of_slope_1=True, x_label="Reference", y_label="Target")

    legend_data_pnt_text = 'Data points'
    legend_trendline_text = 'Regression line'
    legend_line_text = '1:1 line'
    assert plt_scatter.legend().get_texts()[0].get_text() == legend_data_pnt_text
    assert plt_scatter.legend().get_texts()[1].get_text() == legend_trendline_text      
    assert plt_scatter.legend().get_texts()[2].get_text() == legend_line_text

    plt_scatter_wdir = bw.plot_scatter_wdir(DATA.Dir78mS, DATA.Dir58mS, x_label='Reference', y_label='Target',
                         x_limits=(50, 300), y_limits=(250, 300))
    
    assert plt_scatter_wdir.legend().get_texts()[0].get_text() == legend_data_pnt_text
    assert plt_scatter_wdir.legend().get_texts()[1].get_text() == legend_line_text

    plt_scatter_wspd = bw.plot_scatter_wspd(DATA.Spd80mN, DATA.Spd80mS, x_label='Speed at 80m North',
                         y_label='Speed at 80m South', x_limits=(0, 25), y_limits=(0, 25))
    
    assert plt_scatter_wspd.legend().get_texts()[0].get_text() == legend_data_pnt_text
    assert plt_scatter_wspd.legend().get_texts()[1].get_text() == legend_line_text
    
    bw.plot_scatter_wspd(DATA.Spd80mN, DATA.Spd80mN, x_limits=(0, 25), y_limits=(0, 25))

    assert True


def test_plot_scatter_by_sector():
    bw.plot_scatter_by_sector(DATA.Spd80mN, DATA.Spd80mS, DATA.Dir78mS)
    bw.plot_scatter_by_sector(DATA.Spd80mN, DATA[['Spd80mS']], DATA.Dir78mS, sectors=6)
    bw.plot_scatter_by_sector(DATA.Dir78mS, DATA.Dir58mS, DATA.Dir38mS,
                              x_limits=(50, 300), y_limits=(250, 300), line_of_slope_1=True)
    bw.plot_scatter_by_sector(DATA.Spd80mN, DATA.Spd80mS, DATA.Dir78mS, trendline_y=DATA.Spd80mN*2,
                              x_limits=(0, 25), y_limits=(0, 25), axes_equal=False)

    assert True


def test_bar_subplot():

    # To plot data with pd.DatetimeIndex, multiple columns, with bars total width of 20 days and line_width=0.3
    average_data, coverage = bw.average_data_by_period(DATA[['Spd80mN', 'Spd80mS', 'Spd60mN']], period='1M',
                                                       return_coverage=True)
    bw.analyse.plot._bar_subplot(coverage, max_bar_axis_limit=1, total_width=20/31, line_width=0.3,
                                 bin_tick_label_format=DateFormatter("%Y-%m-%d"), vertical_bars=True)

    # To plot multiple subplots in a figure
    fig, axes = plt.subplots(1, 2)
    bw.analyse.plot._bar_subplot(coverage[['Spd80mN_Coverage', 'Spd80mS_Coverage']], max_bar_axis_limit=1,
                                 total_width=20/31, line_width=0.3, vertical_bars=True, ax=axes[0])
    bw.analyse.plot._bar_subplot(coverage['Spd60mN_Coverage'], max_bar_axis_limit=1, total_width=20/31,
                                 line_width=0.3, vertical_bars=True, ax=axes[1])

    # To plot data with integer data.index, multiple columns, horizontal bars, total_width=0.8 and
    # setting bin_tick_labels, subplot title and with legend
    test_data = pd.DataFrame.from_dict({'mast': [99.87, 99.87, 99.87], 'lidar': [97.11, 92.66, 88.82]})
    test_data.index = [50, 65, 80]
    fig = plt.figure(figsize=(15, 8))
    bw.analyse.plot._bar_subplot(test_data, x_label='Data Availability [%]', y_label='Measurement heights [m]',
                                 max_bar_axis_limit=100, bin_tick_labels=['a', 'b', 'c'],
                                 bar_tick_label_format=PercentFormatter(), subplot_title='coverage',
                                 legend=True, total_width=0.8, vertical_bars=False)

    # To plot data with integer data.index, multiple columns, horizontal bars and
    # setting minimum and maximum y axis limit
    bw.analyse.plot._bar_subplot(test_data, x_label='Data Availability [%]', y_label='Measurement heights [m]',
                                 max_bar_axis_limit=100, min_bin_axis_limit=0, max_bin_axis_limit=100,
                                 subplot_title='coverage', legend=True, vertical_bars=False)

    # To plot frequency distribution data with index as bin ranges (ie [-0.5, 0.5)), single column,
    # vertical bars and default total_width
    distribution = bw.analyse.analyse._derive_distribution(DATA['Spd80mN'].to_frame(),
                                                           var_to_bin_against=DATA['Spd80mN'].to_frame(),
                                                           aggregation_method='%frequency')
    fig = plt.figure(figsize=(15, 8))
    bw.analyse.plot._bar_subplot(distribution.replace([np.inf, -np.inf], np.NAN).dropna(), y_label='%frequency')

    assert True


def test_plot_freq_distribution():
    # Plot frequency distribution of only one variable, without x tick labels
    distribution = bw.analyse.analyse._derive_distribution(DATA['Spd40mN'],
                                                           var_to_bin_against=DATA['Spd40mN'], bins=None,
                                                           aggregation_method='%frequency').rename('Spd40mN')
    bw.analyse.plot.plot_freq_distribution(distribution.replace([np.inf, -np.inf], np.NAN).dropna(),
                                           max_y_value=None, x_tick_labels=[], x_label=None,
                                           y_label='%frequency')

    # Plot distribution of counts for multiple variables, having the bars to take the total_width
    distribution1 = bw.analyse.analyse._derive_distribution(DATA['Spd40mN'],
                                                            var_to_bin_against=DATA['Spd40mN'],
                                                            aggregation_method='count').rename('Spd40mN')
    distribution2 = bw.analyse.analyse._derive_distribution(DATA['Spd80mN'],
                                                            var_to_bin_against=DATA['Spd80mN'],
                                                            aggregation_method='count').rename('Spd80mN')

    # The below is a workaround to be able to work with pandas >= 0.24.0, < 0.25.0, this shouldbe replaced with
    # distributions = pd.concat([distribution1, distribution2], axis=1)
    # when these versions of pandas will not be supported anymore by brightwind library. This version of pandas
    # doesn't allow to concatenate two pandas.Series with a CategoricalIndex if having a different index length
    temp_dist = pd.concat([pd.DataFrame(distribution1).reset_index().rename(columns={'variable_bin': 'variable_bin1'}),
                           pd.DataFrame(distribution2).reset_index()], axis=1
                          ).set_index('variable_bin')
    distributions = temp_dist.drop(['variable_bin1'], axis=1)

    bw.analyse.plot.plot_freq_distribution(distributions,
                                           max_y_value=None, x_tick_labels=None, x_label=None,
                                           y_label='count', total_width=1, legend=True)
    assert True


def test_adjust_color_lightness():
    input_color = '#9CC537'
    r, g, b = tuple(255 * np.array(mpl.colors.to_rgb(input_color)))
    hue, lightness, saturation = rgb2hls(r / 255, g / 255, b / 255)
    r, g, b = tuple(255 * np.array(mpl.colors.to_rgb(bw_plt.COLOR_PALETTE._adjust_color_lightness(
        input_color, 0.1))))
    hue1, lightness1, saturation1 = rgb2hls(r / 255, g / 255, b / 255)
    assert (int(hue * 100) == int(hue1 * 100)) and (int(saturation * 100) == int(saturation1 * 100)) and \
           (lightness1 == 0.1)

    assert bw_plt.COLOR_PALETTE._adjust_color_lightness(
        input_color, 0.1) == "#20280b"  # darkest green, 10% of primary
    assert bw_plt.COLOR_PALETTE._adjust_color_lightness(
        input_color, 0.35) == "#6e8c27"  # dark green, 35% of primary
    assert bw_plt.COLOR_PALETTE._adjust_color_lightness(
        input_color, 0.8) == "#d8e9af"  # light green, 80% of primary
    assert bw_plt.COLOR_PALETTE._adjust_color_lightness(
        input_color, 0.9) == "#ecf4d7"  # light green, 90% of primary
    assert bw_plt.COLOR_PALETTE._adjust_color_lightness(
        input_color, 0.95) == "#f5f9eb"  # lightest green, 95% of primary


def test_ColorPalette():
    color_palette = bw.analyse.plot.COLOR_PALETTE
    assert color_palette.color_list == ['#9CC537', '#2E3743', '#9B2B2C', '#E57925', '#ffc008', '#AB8D60', '#A4D29F',
                                        '#01958a', '#3D636F', '#A49E9D', '#DA9BA6', '#6e8c27']

    color_palette.primary = '#3366CC'
    assert bw.analyse.plot.COLOR_PALETTE.primary == '#3366CC'
    assert color_palette.color_list[0] == bw.analyse.plot.COLOR_PALETTE.primary
    assert bw.analyse.plot.COLOR_PALETTE.color_map_colors[1] == color_palette.primary

    color_palette.secondary = '#726e83'
    assert bw.analyse.plot.COLOR_PALETTE.color_list[1] == '#726e83'
    assert bw.analyse.plot.COLOR_PALETTE.color_list[2] == '#9B2B2C'

    color_palette.primary_10 = '#0a1429'
    assert bw.analyse.plot.COLOR_PALETTE.primary_10 == '#0a1429'
    assert color_palette.color_map_colors[-1] == color_palette.primary_10

    color_palette.color_map_colors = ['#ccfffc', '#00b4aa', '#008079']
    assert bw.analyse.plot.COLOR_PALETTE.color_map_colors == ['#ccfffc', '#00b4aa', '#008079']

    color_palette.color_map_cyclical_colors = ['#ccfffc', '#00b4aa', '#008079', '#ccfffc']
    assert bw.analyse.plot.COLOR_PALETTE.color_map_cyclical_colors == ['#ccfffc', '#00b4aa', '#008079', '#ccfffc']


def test_plot_shear_by_sector():

    alpha = pd.Series({'345.0-15.0': 0.216, '15.0-45.0': 0.196, '45.0-75.0': 0.170, '75.0-105.0': 0.182,
                       '105.0-135.0': 0.148, '135.0-165.0': 0.129, '165.0-195.0': 0.156, '195.0-225.0': 0.159,
                       '225.0-255.0': 0.160, '255.0-285.0': 0.169, '285.0-315.0': 0.187, '315.0-345.0': 0.188})

    roughness = pd.Series({'345.0-15.0': 0.537, '15.0-45.0': 0.342, '45.0-75.0': 0.156, '75.0-105.0': 0.231,
                           '105.0-135.0': 0.223, '135.0-165.0': 0.124, '165.0-195.0': 0.135,
                           '195.0-225.0': 0.145, '225.0-255.0': 0.108, '255.0-285.0': 0.149,
                           '285.0-315.0': 0.263, '315.0-345.0': 0.275})

    wind_rose_plot, wind_rose_dist = bw.analyse.analyse.dist_by_dir_sector(DATA.Spd80mS, DATA.Dir78mS,
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

    alpha1 = alpha.copy()
    alpha1['135.0-165.0'] = np.nan
    roughness1 = roughness.copy()
    roughness1['135.0-165.0'] = np.inf

    # Plots shear by directional sectors with calculation method as 'power law'.
    bw.analyse.plot.plot_shear_by_sector(scale_variable=alpha1, wind_rose_data=wind_rose_dist,
                                         calc_method='power_law')

    # Plots shear by directional sectors with calculation method as 'log law'.
    bw.analyse.plot.plot_shear_by_sector(scale_variable=roughness1, wind_rose_data=wind_rose_dist,
                                         calc_method='log_law')

