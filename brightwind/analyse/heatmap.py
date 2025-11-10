import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Union
from brightwind.analyse.analyse import coverage
from brightwind.transform import transform as tf



__all__ = ['plot_heatmap',
           'monthly_heatmap']

def plot_heatmap(x, y, z, x_label, y_label, z_label, colormap,
                    colormap_lim_lower = None, colormap_lim_upper = None,
                    fig=None, ax=None, ax_label = None, include_colorbar=True):
    
    """
    Generate a heatmap from x, y, and z values with customizable labels and color mapping.

    Method:
    1.  Create a 2D grid from the x and y inputs and assign corresponding z values for each coordinate.
    2.  If no figure or axis is provided, create a new matplotlib figure and axis.
    3.  Plot the z values on the grid as a heatmap using the specified colormap.
    4.  Apply color limits to control the mapping of z values to colors.
    5.  Set x-axis, y-axis, and colorbar labels according to provided arguments.
    6.  Optionally apply an axis label to the subplot if ax_label is provided.
    7.  Return the figure and axis objects for further customisation or saving.

    :param x:                       Values for the x-axis grid.
    :type x:                        list, numpy.ndarray
    :param y:                       Values for the y-axis grid.
    :type y:                        list, numpy.ndarray
    :param z:                       Values corresponding to each (x, y) pair.
                                    1. If z is a list or numpy.ndarray, the z values will be written on an x by y grid as text
                                    and will also determine the colour mapping. Note z should have shape (shape(y), shape(x)).
                                    2. z can also be a tuple like (z_value, z_colour), where:
                                    - the first entry, z_value, is the value which will be written on the grid as text, and
                                    - the second entry, z_colour, is the value which will determine the colour mapping.
                                    This allows visualisation of multiple quantities on the same heatmap such as wind speed
                                    (z_value) per month (x) and year (y), coloured by the monthly coverage of the wind speed
                                    data (z_colour) - see example usage below.
    :type z:                        list, numpy.ndarray, tuple
    :param x_label:                 Label for the x-axis.
    :type x_label:                  str
    :param y_label:                 Label for the y-axis.
    :type y_label:                  str
    :param z_label:                 Label for the z values.
                                    1. If z is a list or numpy.ndarray, the z_label is for the colorbar and default plot title.
                                    2. If z is a tuple, z_label should also be a tuple with (z_value_label, z_colour_label):
                                    - z_value_label used in plot title.
                                    - z_colour_label is used to label the colorbar.
    :type z_label:                  str, tuple(str, str)
    :param colormap:                Matplotlib colormap name used for the heatmap
    :type colormap:                 str
    :param fig:                     Optional matplotlib figure object. If not provided, a new figure is created.
    :type fig:                      matplotlib.figure.Figure, optional
    :param ax:                      Optional matplotlib axis object. If not provided, a new axis is created.
    :type ax:                       matplotlib.axes.Axes, optional
    :param colormap_lim_lower:      Lower limit for the colormap normalization
    :type colormap_lim_lower:       float
    :param colormap_lim_upper:      Upper limit for the colormap normalization
    :type colormap_lim_upper:       float
    :param ax_label:                Optional label for the axis (e.g., subplot title or identifier).
    :type ax_label:                 str, optional
    :param include_colorbar:        If True, include a colorbar to the side of the plot showing which values
                                    the colours correspond to (default True).
    :type include_colorbar:         bool
    :returns:                       Figure and axis objects containing the heatmap plot.
    :rtype:                         tuple (matplotlib.figure.Figure, matplotlib.axes.Axes)

    **Example usage**
    ::

    import brightwind as bw
    import numpy as np
    import matplotlib.pyplot as plt

    years = [2021, 2022]
    months = [1, 2, 3]
    wspd = [[3, 2, 5],
            [8, 4, 10]]
    wspd_coverage = [[0, .1, .2],
                [0.2, 0, .3]]

    # heatmap of wind speed by month and year
    heatmap = bw.plot_heatmap(months, years, wspd, 'month', 'year', 'wind speed', colormap='viridis')

    # heatmap of wind speed by month and year with upper and lower limits for colormap and label (ax_label) above plot
    heatmap = bw.plot_heatmap(months, years, wspd, 'month', 'year', 'wind speed',
                              colormap='viridis', colormap_lim_lower=2, colormap_lim_upper=14, ax_label = 'Monthly wind speed')

    # heatmap with wind speed values annotated, coloured by coverage, colormap limited to between 0 and 1 (range of expected coverage values)
    heatmap = bw.plot_heatmap(months, years, (wspd, wspd_coverage), 'month', 'year', ('wind speed', 'coverage'), 'viridis',
                              ax_label='Wind speed coloured by coverage', colormap_lim_lower=0, colormap_lim_upper=1) 

    # heatmap of wind speed by directional bin, using matplotlib figure and axis
    direction = ['0-90', '90-180', '180-270', '270-360']
    wspd_cols = ['Spd1', 'Spd2']
    spd1 = [2, 3.5, 4, 5]
    spd2 = [4, 3, 5, 6]
    wspd = [spd1, spd2]

    figure, axis = plt.subplots(figsize = (8, 10))
    heatmap = bw.plot_heatmap(direction, wspd_cols, wspd, 'direction', 'measurement', 'speed', 'viridis',
                              ax_label='Wind Speed by Directional bin', fig=figure, ax=axis)          

    """
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    if isinstance(z, tuple):
        if len(z)!=2:
            raise ValueError('if z is a tuple, it must have length 2 like (z_value, z_colour)')
        z_value = z[0]
        z_colour = z[1]

        if np.shape(z_value)!=np.shape(z_colour):
            raise ValueError('if z is a tuple like (z_value, z_colour), then z_value and z_colour must have the same shape')

        if not isinstance(z_label, tuple) or len(z_label)!=2:
            raise ValueError('if z is a tuple like (z_value, z_colour), z_label must also be a tuple of length 2 like (z_value_label, z_colour_label)')
        z_value_label = z_label[0]
        z_colour_label = z_label[1]
    
    else:
                
        z_value = z
        z_colour = z
        z_value_label = z_label
        z_colour_label = z_label

    # check that the dimensions of x, y, z make sense
    shape_x =  np.shape(x)[0]
    shape_y = np.shape(y)[0]
    shape_z = np.shape(z_value)
    if shape_x == 1 or shape_y ==1:
        if shape_z[0] != shape_x*shape_y:
            raise ValueError(f'z must have dimensions of ({shape_x*shape_y},) for given x and y, not {shape_z}')
    else:
        if shape_z[0]!=shape_y or shape_z[1]!=shape_x:
            raise ValueError(f'z values (and z colours) must have shape of ({shape_y}, {shape_x}) for given x and y, not {shape_z}')

    # Create colormap and set color for NaNs
    cmap = plt.get_cmap(colormap).copy()
    cmap.set_bad(color='lightgrey')

    z_colour = np.where(z_colour==None, np.nan, z_colour).astype(float)

    im = ax.imshow(z_colour, cmap = cmap)

    im.set_clim(colormap_lim_lower, colormap_lim_upper)

    if include_colorbar:
        cbar = fig.colorbar(im, ax = ax, shrink = 0.6)
        cbar.set_label(z_colour_label)

    # Set tick labels
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x)
    ax.set_yticks(np.arange(len(y)))
    ax.set_yticklabels(y)

    # Label axes
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Create text annotations.
    for i in range(len(x)):
        for j in range(len(y)):
            value = '-' if np.isnan(z_value[j][i]) else z_value[j][i]
            text = ax.text(i, j, value,
                       ha="center", va="center", color="k")

    if ax_label:
        ax.set_title(ax_label)
    else:
        if z_value_label == z_colour_label:
            ax.set_title(f'{z_value_label} by {x_label} and {y_label}')
        else:
            ax.set_title(f'{z_value_label} by {x_label} and {y_label} coloured by {z_colour_label}')
    return fig


def _reindex_to_include_all_months_per_year(monthly_averaged_data):
    """
    Reindex a monthly dataset to include a timestamp for all calendar months for each year found in the data.

    For heatmap plots, it is necessary to have a timestamp for each month, so this function is used to add
    timestamps for missing months (Missing months are filled with nan values).

    :param data:    Data in the form of a Pandas DataFrame with datetime index and monthly timestamps.
    :type data:     pandas.Series
    :returns:       tuple:
                    - list of unique years in the timestamps of monthly_averaged_data
                    - copy of monthly_averaged_data with timestamps for all calendar months in the relevant years,
                    any months without data have np.nan values.
    :rtype:         tuple(list, pandas.Series)

    **Example Usage**
    ::

    import brightwind as bw
    data = bw.load_csv(bw.demo_datasets.demo_data)

    # monthly mean temperature values (data ends in November, so mean value for Dec 2017 missing)
    monthly_mean_temp = bw.average_data_by_period(data.T2m.loc['2017'], '1M')
    # function fills in the value for Dec 2017 with np.nan

    bw.analyse.heatmap._reindex_to_include_all_months_per_year(monthly_mean_temp)
    # 2017-01-01     2.246696
    # 2017-02-01     2.742041
    # 2017-03-01     4.671117
    # 2017-04-01     5.045067
    # 2017-05-01     9.862758
    # 2017-06-01    11.120782
    # 2017-07-01    11.753794
    # 2017-08-01    11.526499
    # 2017-09-01     9.832652
    # 2017-10-01     8.540742
    # 2017-11-01     4.217146
    # 2017-12-01          NaN
    """

    data = monthly_averaged_data.copy()
    years = data.index.year.unique()
    first_year = years.min()
    last_year = years.max()
    full_index = pd.date_range(start = str(first_year), end = str(last_year+1), freq = 'MS')
    data = data.reindex(full_index)
    data = data[data.index.year.isin(years)]

    return years, data


def monthly_heatmap(measurement_data,
                    label,
                    type,
                    fig = None,
                    ax = None,
                    include_colorbar = True,
                    include_overall_per_month = True):
    
    """"
    Produces heatmap of the given measurement_data timeseries according to the heatmap 'type' specified, either:
        - heatmap of monthly mean of the given timeseries (if type = 'mean') or monthly coverage (if type = 'coverage')
        - heatmap of monthly coverage of the given timeseries (if type = 'coverage') or
        - heatmap of monthly mean of the given timeseries, coloured by the monthly coverage (if type = 'mean_coloured_by_coverage').

    :param measurement_data:            Timeseries of measurement to produce monthly heatmap for.
                                        This function calculates the monthly mean or coverage internally so
                                        the input must have datetime index and the index is not required to
                                        have monthly timestamp frequency, for example measurement_data can be
                                        a 10-min timeseries of wind speed.
    :type measurement_data:             pandas.Series (with datetime index)
    :param label:                       Label for figure           
    :type label:                        str
    :param type:                        type of heatmap to produce, should be 'mean' or 'coverage' or 'mean_coloured_by_coverage'          
    :type type:                         str 
    :param fig:                         Optional matplotlib figure object. If not provided, a new figure is created.
    :type fig:                          matplotlib.figure.Figure, optional
    :param ax:                          Optional matplotlib axis object. If not provided, a new axis is created.
    :type ax:                           matplotlib.axes.Axes, optional
    :param include_colorbar:            If True, include a colorbar to the side of the plot showing which values
                                        the colours correspond to (default True).
    :type include_colorbar:             bool
    :param include_overall_per_month:   If True (default):
                                        - if type is 'mean' or 'mean_coloured_by_coverage', return the monthly
                                        mean of the measurement_data grouped by calendar month (e.g. group
                                        all available data points in measurement_data timeseries for month 1
                                        (Jan) and take the mean)
                                        - if type is 'coverage' include the total coverage per calendar month
                                        (the sum of the monthly coverage across all years of the dataset)
    :type include_overall_per_month:    bool
    :returns:                           heatmap plot of monthly coverage or mean of measurement_data
    :rtype:                             plot

    **Example usage**
    ::
    import brightwind as bw
    import matplotlib.pyplot as plt

    data = bw.load_csv(bw.demo_datasets.demo_data)

    # monthly coverage plot, no colorbar, overall coverage per month included by default
    temperature_coverage_heatmap = bw.monthly_heatmap(data.T2m, 'Temperature Coverage', type = 'coverage')

    # monthly coverage plot, with colorbar, overall coverage per month excluded
    coverage_heatmap = bw.monthly_heatmap(data.T2m, 'Temperature Coverage', type='coverage',
                                    include_colorbar=False, include_overall_per_month=False)

    # monthly mean plot, no colorbar, overall mean per month included by default
    mean_temp_heatmap = bw.monthly_heatmap(data.T2m, 'Mean Temperature', type = 'mean', include_colorbar=False)

    # use optional fig, ax arguments for control over matplotlib figure and axis objects
    # e.g. plot mean and coverage as subplots of a figure below
    import matplotlib.pyplot as plt

    figure, axs = plt.subplots(nrows = 2, ncols = 1, figsize = (12, 10))
    bw.monthly_heatmap(data.T2m, 'Mean Temperature', type='mean',
                    fig = figure, ax = axs[0])
    bw.monthly_heatmap(data.T2m, 'Temperature Coverage', type='coverage',
                    fig = figure, ax = axs[1])
    plt.show() 

    # monthly mean temperature plot, with colorbar, coloured by monthly coverage, overall mean per month included by default
    mean_temp_coloured_by_coverage = bw.monthly_heatmap(data.T2m, 'Mean Temperature', type = 'mean_coloured_by_coverage')
    """
    if type == 'coverage':
        # get coverage per month (and per year)
        monthly_values = coverage(measurement_data, period = '1M')
    
    elif type == 'mean':
        # get mean per month (and per year)
        monthly_values = tf.average_data_by_period(measurement_data, period = '1M')

    elif type == 'mean_coloured_by_coverage':
        # get coverage per month (and per year)
        monthly_cov_values = coverage(measurement_data, period = '1M')
        # get mean per month (and per year)
        monthly_mean_values = tf.average_data_by_period(measurement_data, period = '1M')
    
    else:
        raise ValueError(f'type should be one of "coverage" or "mean" or "mean_coloured_by_coverage"')
    
    if type == 'coverage' or type == 'mean':
        # reindex the monthly timestamps to include all months for the relevant years (fill missing months with np.nan)
        years, monthly_values = _reindex_to_include_all_months_per_year(monthly_values)

        # array to store z values for plot_heatmap
        monthly_values_array = []

        # get unique years for y axis of plot_heatmap
        for year in years:
            monthly_values_array.append(monthly_values[monthly_values.index.year ==year].values.round(2))

        if include_overall_per_month:
            if type == 'coverage':
                # get coverage per month (across all years)
                overall_per_month = np.sum(np.where(np.isnan(monthly_values_array), 0, monthly_values_array), axis = 0).round(2)

            elif type == 'mean':
                # get mean per month (across all years)
                overall_per_month = measurement_data.groupby(measurement_data.index.month).mean().round(2)
            
            # add a blank row for separation
            monthly_values_array.append(np.full_like(np.arange(1, 13), fill_value=np.nan, dtype=float))
            # add total
            if type == 'mean':
                overall_per_month = overall_per_month.values
            monthly_values_array.append(overall_per_month)
        
            # add labels for blank row and overall row for y axis labels
            years = list(years.values)
            years.append('')
            years.append('Overall')
        
    elif type == 'mean_coloured_by_coverage':
        # reindex the monthly timestamps to include all months for the relevant years (fill missing months with np.nan)
        years, monthly_mean_values = _reindex_to_include_all_months_per_year(monthly_mean_values)
        monthly_cov_values = _reindex_to_include_all_months_per_year(monthly_cov_values)[1]

        # array to store mean values for plot_heatmap text
        monthly_mean_values_array = []
        # array to store coverage values for plot_heatmap colours
        monthly_cov_values_array = []

        # get unique years for y axis of plot_heatmap
        for year in years:
            monthly_mean_values_array.append(monthly_mean_values[monthly_mean_values.index.year ==year].values.round(2))
            monthly_cov_values_array.append(monthly_cov_values[monthly_cov_values.index.year ==year].values.round(2))

        if include_overall_per_month:
            # get coverage per month (across all years)
            overall_cov_per_month = np.sum(np.where(np.isnan(monthly_cov_values_array), 0, monthly_cov_values_array), axis = 0).round(2)
            # get mean per month (across all years)
            overall_mean_per_month = measurement_data.groupby(measurement_data.index.month).mean().round(2)
            
            # add a blank row for separation
            monthly_mean_values_array.append(np.full_like(np.arange(1, 13), fill_value=np.nan, dtype=float))
            monthly_cov_values_array.append(np.full_like(np.arange(1, 13), fill_value=np.nan, dtype=float))

            # add total
            monthly_mean_values_array.append(overall_mean_per_month.values)
            monthly_cov_values_array.append(overall_cov_per_month)

            #  add labels for blank row and overall row for y axis labels
            years = list(years.values)
            years.append('')
            years.append('Overall')


    if type == 'coverage' or type == 'mean_coloured_by_coverage':
        base_cmap = plt.cm.RdYlGn

        # Define custom colormap for coverage
        # Values below 0.6 are red, 0.6-0.7 yellow and 0.7-1 green
        colors = [
            (0.0, base_cmap(0.0)),
            (0.6, base_cmap(0.0)),
            (0.7, base_cmap(0.5)),
            (1.0, base_cmap(1.0)),
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("coverage_cmap", colors)

        # define upper and lower limits for colormap (expected coverage values between 0 and 1)
        cmap_lim_lower = 0
        cmap_lim_upper = 1
            
    elif type == 'mean':
        # generic colormap and no set colormap limits for monthly mean plot
        cmap = 'Blues'
        cmap_lim_lower = None
        cmap_lim_upper = None

    if type == 'mean_coloured_by_coverage':
        # plot heatmap
        figure=plot_heatmap(x = np.arange(1, 13),
                            y = years,
                            z = (monthly_mean_values_array, monthly_cov_values_array),
                            x_label = 'Month',
                            y_label='Year',
                            z_label = ('mean', 'coverage'),
                            colormap = cmap,
                            colormap_lim_lower = cmap_lim_lower,
                            colormap_lim_upper = cmap_lim_upper,
                            fig=fig,
                            ax=ax,
                            ax_label=f'{label}',
                            include_colorbar=include_colorbar)

    else:
        # plot heatmap
        figure=plot_heatmap(x = np.arange(1, 13),
                            y = years,
                            z = monthly_values_array,
                            x_label = 'Month',
                            y_label='Year',
                            z_label = type,
                            colormap = cmap,
                            colormap_lim_lower = cmap_lim_lower,
                            colormap_lim_upper = cmap_lim_upper,
                            fig=fig,
                            ax=ax,
                            ax_label=f'{label}',
                            include_colorbar=include_colorbar)
    return figure