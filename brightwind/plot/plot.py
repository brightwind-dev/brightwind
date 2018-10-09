import matplotlib.pyplot as plt
import calendar
import numpy as np
import pandas as pd
from ..analyse import analyse as freq_an
from ..analyse import shear as sh
from ..utils import utils
from ..transform import transform

plt.style.use(r'C:\Dropbox (brightwind)\RTD\repos-hadley\brightwind\brightwind\plot\bw.mplstyle')


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


def _scatter_plot(x, y, predicted_y=None, x_label="Reference", y_label="Target", title="", size=(10,10),
                  prediction_marker='k-'):
    fig2 = plt.figure(111)
    scat = fig2.add_subplot(111)
    scat.set_xlabel(x_label)
    scat.set_ylabel(y_label)
    scat.scatter(x, y, marker = '.', color='#9ACD32',alpha=0.5)
    fig2.set_figwidth(size[0])
    fig2.set_figheight(size[1])
    plt.title(title)
    if predicted_y is not None:
        plt.plot(x, predicted_y, prediction_marker)
        plt.legend(['Predicted','Original'])
    plt.show()


def plot_freq_distribution(data, max_speed=30, plot_colors=[bw_colors('light_green_for_gradient'),
                                        bw_colors('dark_green_for_gradient'),bw_colors('darkgreen')]):
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
    ax.set_xlim(-0.5, max_speed+0.5)
    ax.set_ylim(0, max(data)+5)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(b=True, axis='y', zorder=0)
    # ax.bar(result.index, result.values,facecolor='#9ACD32',edgecolor=['#6C9023' for i in range(len(result))],zorder=3)
    for frequency, bin in zip(data,x_data):
        ax.imshow(np.array([[plot_colors[0]], [plot_colors[1]]]),
                  interpolation='gaussian', extent=(bin-0.4, bin+0.4, 0, frequency), aspect='auto', zorder=3)
        ax.bar(bin, frequency, edgecolor=plot_colors[2], linewidth=0.3, fill=False, zorder=5)
    ax.set_title('Wind Speed Frequency Distribution')
    return ax.get_figure()


def plot_wind_rose(ext_data, freq_table=False):
    """
    Plot a wind rose from a direction data or a frequency table.
    """
    data = ext_data.copy()
    if freq_table:
        sectors = data.shape[1]
    else:
        sectors = data.shape[0]
    result = data.sum(axis=0)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.1, 0.1, 0.8,0.8], polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0,360, 360.0/sectors))
    ax.set_rgrids(np.arange(0,101,10), labels=[str(i)+'%' for i in np.arange(0,101,10)],angle=0)
    ax.bar(np.arange(0,2.0*np.pi, 2.0*np.pi/sectors), result, width=2.0*np.pi/sectors, bottom=0.0, color='#9ACD32',
           edgecolor=['#6C9023' for i in range(len(result))], alpha=0.8)
    ax.set_title('Wind Rose', loc='center')
    return ax.get_figure()


def plot_wind_rose_with_gradient(freq_table, gradient_colors=['#f5faea','#d6ebad','#b8dc6f','#9acd32','#7ba428', '#5c7b1e']):
    table = freq_table.copy()
    import matplotlib as mpl
    sectors=len(table.columns)
    table_binned=pd.DataFrame()
    if isinstance(table.index[0], pd.Interval):
        table.index = [i.mid for i in table.index]
    table_trans = table.T
    table_binned = pd.concat([table_binned, table_trans.loc[:, 0:3].sum(axis=1).rename(3)],axis=1)
    table_binned = pd.concat([table_binned, table_trans.loc[:, 4:6].sum(axis=1).rename(6)],axis=1)
    table_binned = pd.concat([table_binned, table_trans.loc[:, 7:9].sum(axis=1).rename(9)],axis=1)
    table_binned = pd.concat([table_binned, table_trans.loc[:, 10:12].sum(axis=1).rename(12)],axis=1)
    table_binned = pd.concat([table_binned, table_trans.loc[:, 13:15].sum(axis=1).rename(15)],axis=1)
    table_binned = pd.concat([table_binned, table_trans.loc[:, 16:].sum(axis=1).rename(18)], axis=1)
    table_binned = table_binned.T
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.1, 0.1, 0.8,0.8], polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0,360,360.0/sectors), zorder=2)
    ax.set_rgrids(np.linspace(0.1, max(table.sum(axis=0))+2.0,10),labels=[ '%.0f' % round(i)+'%' for i in
                                                    np.linspace(0.1, max(table.sum(axis=0))+2.0, 10)], angle=0, zorder=2)
    direction_bins = utils.get_direction_bin_array(sectors)[1:-2]
    direction_bins = np.insert(direction_bins,0,direction_bins[-2])
    ax.set_ylim(0, max(table.sum(axis=0))+3.0)
    angular_width = 2*np.pi/sectors - (np.pi/180)  # Leaving 1 degree gap

    def _choose_color(speed_bin):
        colors = gradient_colors
        bins = [0, 3.5, 6.5, 9.5, 12.5, 15.5, 18.5, 41]
        return colors[np.digitize([speed_bin], bins)[0]-1]

    for column in table_binned:
        radial_pos = 0.0
        angular_pos = (np.pi / 180.0) * float(column.split('-')[0])
        for speed_bin,frequency in zip(table_binned.index,table_binned[column]):
            color = _choose_color(speed_bin)
            patch = mpl.patches.Rectangle((angular_pos, radial_pos), angular_width, frequency, facecolor=color,
                                          edgecolor='#5c7b1e', linewidth=0.3, zorder=3)
            ax.add_patch(patch)
            radial_pos += frequency
    legend_patches = [mpl.patches.Patch(color=gradient_colors[0], label='0-3 m/s'),
                        mpl.patches.Patch(color=gradient_colors[1], label='4-6 m/s'),
                        mpl.patches.Patch(color=gradient_colors[2], label='7-9 m/s'),
                        mpl.patches.Patch(color=gradient_colors[3], label='10-12 m/s'),
                        mpl.patches.Patch(color=gradient_colors[4], label='13-15 m/s'),
                        mpl.patches.Patch(color=gradient_colors[5], label='15+ m/s')]
    ax.legend(handles=legend_patches)
    return ax.get_figure()


def plot_TI_by_speed(wdspd, wdspd_std, IEC_class=None, **kwargs):
    """
    Plot turbulence intensity graphs alongside with IEC standards
    :param TI_by_speed:
    :param IEC_Class: By default IEC class 2005 is used for custom class pass a dataframe. Note we have removed
            option to include IEC Class 1999 as no longer appropriate.
            This may need to be placed in a separate function when updated IEC standard is released
    :return: Plots turbulence intensity distribution by wind speed
    """

    # IEC Class 2005
    #
    if IEC_class is None:
        IEC_class = pd.DataFrame(np.zeros([26, 4]), columns=['Windspeed', 'IEC Class A', 'IEC Class B', 'IEC Class C'])
        for n in range(1, 26):
            IEC_class.iloc[n, 0] = n
            IEC_class.iloc[n, 1] = 0.16 * (0.75 + (5.6 / n))
            IEC_class.iloc[n, 2] = 0.14 * (0.75 + (5.6 / n))
            IEC_class.iloc[n, 3] = 0.12 * (0.75 + (5.6 / n))

    TI = freq_an.get_TI_by_speed(wdspd, wdspd_std, **kwargs)
    common_idxs = wdspd.index.intersection(wdspd_std.index)
    fig, ax = plt.subplots()
    ax.scatter(wdspd.loc[common_idxs], wdspd_std.loc[common_idxs]/wdspd.loc[common_idxs],
               color=bw_colors('green'), alpha=0.3, marker='.')
    ax.plot(TI.index.__array__(), TI.loc[:,'Mean_TI'].values, color=bw_colors('darkgreen'))[0].set_label('Mean_TI')
    ax.plot(TI.index.__array__(), TI.loc[:,'Rep_TI'].values, color=bw_colors('redline'))[0].set_label('Rep_TI')
    ax.plot(IEC_class.iloc[:, 0], IEC_class.iloc[:, 1], color=bw_colors('greyline'), linestyle='dashed')
    ax.plot(IEC_class.iloc[:, 0], IEC_class.iloc[:, 2], color=bw_colors('greyline'), linestyle='dashdot')
    ax.plot(IEC_class.iloc[:, 0], IEC_class.iloc[:, 3], color=bw_colors('greyline'), linestyle='dotted')
    ax.set_xlim(2, 25)
    ax.set_ylim(0, 0.6)
    ax.set_xticks(np.arange(2, 26, 1))
    ax.set_xlabel('Wind speed [m/s]')
    ax.set_ylabel('Turbulence Intensity')
    ax.grid(True)
    ax.legend()
    return ax.get_figure()


def plot_TI_by_sector(wdspd, wdspd_std, wddir, sectors=12, **kwargs):

    # First we need to calculate the Turbulence Intensity by sector by calling the sector function.
    TI = freq_an.get_TI_by_sector(wdspd, wdspd_std, wddir, sectors=sectors, **kwargs)
    # Next we convert the Median bin degree to radians for plotting
    TI['Polar degrees'] = np.radians(TI.index * (360 / sectors))
    # To complete the plot, we need to copy the first row and append a new last row.
    TI.loc[-1,:] = TI.loc[0, :]
    # Set Figure size, define it as polar, set north, set number of sectors to be displayed
    fig, ax = plt.subplots()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 360.0 / sectors))
    ax.tick_params(axis='y',labelsize=15)
    # ,grid_color='white',labelcolor='white
    # Convert name of Turbulence Intensity Avg Column so it will read well in legend.
    TI['Turbulence Intensity Average by sector'] = TI['Turbulence_Intensity_Avg']

    # Plot the Average turbulence Intensity and assign a title to the graph
    ax.plot(TI['Polar degrees'], TI['Mean_TI'], c=bw_colors('green'), linewidth=4)
    plt.title('Turbulence Intensity by Direction')

    # Set the max extent of the polar plot to be the max average sector turbulence + 0.1
    maxlevel = TI['Turbulence_Intensity_Avg'].max() + 0.1
    ax.set_ylim(0, maxlevel)

    # Add in comment at bottom of graph about what anemometer and wind vane are used.
    # ax.annotate('*Plot generated using Anemometer ' + speed_col_name + ' and Wind Vane ' + direction_col_name,
    #             xy=(120, 10), xycoords='figure pixels')

    # Finally produce a scatter plot of all of the Turbulence Intensity data points
    # data['Turbulence Intensity by datapoint'] = data[std_col_name] / data[speed_col_name]
    # data['Polar degrees'] = np.radians(data[direction_col_name])
    common_idxs = wdspd.index.intersection(wdspd_std.index).intersection(wddir.index)
    ax.scatter(np.radians(wddir.loc[common_idxs]), wdspd_std.loc[common_idxs]/wdspd.loc[common_idxs],
               c=bw_colors('asphault'), alpha=0.3, s=1)
    ax.legend(loc=8, framealpha=1)
    return ax.get_figure()


def plot_monthly_means(columns):
    if not isinstance(columns, list):
        columns = [columns]
    data = transform.average_data_by_period(pd.concat(columns, axis=1, join='outer'), period='1MS')
    fig, ax = plt.subplots()
    for i in range(0, len(data.columns)):
        ax.plot(data.index.__array__(), data.iloc[:, i].values)
    ax.set_ylabel('Wind speed [m/s]')
    ax.legend(data.columns)
    return ax.get_figure()



def plot_12x24_contours(data, title='Variable'):
    # Get Contour Plot of 12 month x 24 hour matrix of turbulence intensity
    # result = freq_an.get_12x24_TI_matrix(data,time_col_name,speed_col_name,std_col_name)
    fig, ax = plt.subplots()
    x = ax.contourf(data, cmap="Greens")
    cbar = plt.colorbar(x)
    cbar.ax.set_ylabel(title)
    ax.set_xlabel('Month of Year')
    ax.set_ylabel('Hour of Day')
    ax.set_xticks(np.arange(12), calendar.month_name[1:13])
    ax.set_yticks(np.arange(0, 24, 1))
    ax.set_title('Hourly Mean '+title+' Calendar Month')
    return ax.get_figure()


def plot_sector_ratio(data, speed_col_name_1, speed_col_name_2, direction_col_name, boom_dir_1=0, boom_dir_2=0,
                      booms=False):
    ####Refactoring needed as it relies on get_sector_ration. -Inder

    """Accepts a dataframe table, along with 2 anemometer names, and one wind vane name and plots the speed ratio
    by sector. Optionally can include anemometer boom directions also.
    :param data: dataframe of windspeed and direction data
    :param speed_col_name_1: Anemometer 1 column name in dataframe. This is divisor series.
    :param speed_col_name_2: Anemometer 2 column name in dataframe.
    :param direction_col_name: Wind Vane column name in dataframe.
    :param boom_dir_1: Boom direction in degrees of speed_col_name_1. Defaults to 0.
    :param boom_dir_2: Boom direction in degrees of speed_col_name_2. Defaults to 0.
    :param booms: Boolean function. True if you want booms displayed on chart, False if not. Default False.
    :returns A speed ratio plot showing average speed ratio by sector and scatter of individual datapoints.
    """

    # Get Speed Ratio table
    SectorRatio = freq_an.get_sector_ratio(data[speed_col_name_1], data[speed_col_name_2], data[direction_col_name])

    # Convert Speed Ratio table bins into polar coordinates
    SectorRatio['Polar degrees'] = np.radians(SectorRatio.index * (360 / 72))

    # Copy first line to last line so polar plot completes full circle.
    SectorRatio.loc[-1] = SectorRatio.loc[0, :]

    # Setup polar plot
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 360.0 / 72))
    ax.tick_params(axis='y', labelsize=15)

    # Rename column headings, as Sector Speed Ratio Average needs to show up in legend
    SectorRatio.columns = ['index', 'Sector Speed Ratio Average', 'Polar degrees']

    # Plot Sector Speed Ratio average and give chart a title
    ax.plot(SectorRatio['Polar degrees'], SectorRatio['Sector Speed Ratio Average'], c=bw_colors('green'), linewidth=4)
    plt.title('Speed Ratio by Direction')

    # Get max and min levels and set chart axes
    maxlevel = SectorRatio['Sector Speed Ratio Average'].max() + 0.05
    minlevel = SectorRatio['Sector Speed Ratio Average'].min() - 0.1
    ax.set_ylim(minlevel, maxlevel)

    # Add boom dimensions to chart, if required
    if booms == True:

        boom_dir_1 = np.radians(boom_dir_1)
        boom_dir_2 = np.radians(boom_dir_2)

        width = np.pi / 72
        radii = maxlevel

        ax.bar(boom_dir_1, radii, width=width, bottom=minlevel, color='orange')
        ax.bar(boom_dir_2, radii, width=width, bottom=minlevel, color='yellow')

        ax.annotate(
            '*Plot generated using ' + speed_col_name_2 + ' (yellow boom) divided by ' + speed_col_name_1 + ' (orange boom)',
            xy=(20, 10), xycoords='figure pixels')

    else:
        ax.annotate('*Plot generated using ' + speed_col_name_2 + ' divided by ' + speed_col_name_1,
                    xy=(20, 10), xycoords='figure pixels')

    # Finally produce a scatter plot of all of the Speed Ratio data points
    data['Speed Ratio by datapoint'] = data[speed_col_name_2] / data[speed_col_name_1]
    data['Polar degrees'] = np.radians(data[direction_col_name])
    ax.scatter(data['Polar degrees'], data['Speed Ratio by datapoint'], c=bw_colors('asphault'), alpha=0.3, s=1)

    plt.legend(loc=8, framealpha=1)

    plt.show()


def plot_shear(wind_speeds, heights):
    """
    Show derivation and output (alpha value) of shear calculation function for a given timestep.
    :param wind_speeds: List of wind speeds [m/s]
    :param heights: List of heights [m above ground]. The position of the height in the list must be the same position in the list as its
    corresponding wind speed value.
    :return:
        1) Log-log plot of speed and elevation data, including linear fit.
        2) Speed and elevation data plotted on regular scale, showing power law fit resulting from alpha value
    """

    alpha, wind_speedsfit = sh.calc_shear(wind_speeds, heights, plot=True)

    # PLOT INPUT AND MODELLED DATA ON LOG-LOG SCALE
    heightstrend = np.linspace(0.1, max(heights) + 2, 100)  # create variable to define (interpolated) power law trend
    plt.loglog(wind_speeds, heights, 'bo')  # plot input data on log log scale
    plt.loglog(wind_speedsfit(heightstrend), heightstrend, 'k--')  # Show interpolated power law trend
    plt.xlabel('Speed (m/s)')
    plt.ylabel('Elevation (m)')
    plt.legend(['Input data', 'Best fit line for power law (' r'$\alpha$ = %i)' % int(round(alpha))])
    plt.grid(True)
    plt.show()

    # PLOT INPUT AND MODELLED DATA ON REGULAR SCALE
    plt.plot(wind_speeds, heights, 'bo')  # plot input data
    plt.plot(wind_speedsfit(heightstrend), heightstrend, 'k--')  # Show interpolated power law trend
    plt.xlabel('Speed (m/s)')
    plt.ylabel('Elevation (m)')
    plt.legend(['Input data', 'Power law trend (' r'$\alpha$ = %i)' % int(round(alpha))])
    plt.ylim(0, max(heights) + 2)
    plt.xlim(0, max([max(wind_speeds), max(wind_speedsfit(heights))]) + 2)
    plt.grid(True)
    plt.show()
