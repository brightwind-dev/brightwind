import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar

plt.style.use(r'C:\Dropbox (brightwind)\RTD\repos-hadley\wind-analyse-scripts\bw.mplstyle')


def get_distribution(var1_series, var2_series, var2_bin_array=np.arange(-0.5, 41, 1), var2_bin_labels=None,
                     aggregation_method='%frequency'):
    """Accepts 2 series of same/different variables and computes the distribution of first variable with respect to
    the bins of another variable.
    :param var1_series: Series of the variable whose distribution we need to find
    :param var2_series: Series of the variable which we want to bin
    :param var2_bin_array: Array of numbers where adjacent elements of array form a bin
    :param var2_bin_labels: Labels of bins to be used, uses (bin-start, bin-end] format by default
    :param aggregation_method: Statistical method used to find distribution it can be mean, max, min, std, count,
    describe, a custom function, etc,computes frequency in percentages by default
    :returns A dataframe/series with bins as row indexes and columns with statistics chosen by aggregation_method"""
    var1_series = var1_series.dropna()
    var2_series = var2_series.dropna()
    var2_binned_series = pd.cut(var2_series, var2_bin_array, right=False, labels=var2_bin_labels).rename('variable_bin')
    data = pd.concat([var1_series.rename('data'), var2_binned_series], join='inner', axis=1)
    if aggregation_method == '%frequency':
        return data.groupby(['variable_bin'])['data'].count().rename('%frequency')/len(data) * 100.0
    else:
        return data.groupby(['variable_bin'])['data'].agg(aggregation_method)


def _get_direction_bin_array(sectors):
    bin_start = 180.0/sectors
    direction_bins = np.arange(bin_start, 360, 360.0/sectors)
    direction_bins = np.insert(direction_bins, 0, 0)
    direction_bins = np.append(direction_bins, 360)
    return direction_bins


def _get_direction_bin_labels(sectors, direction_bins, zero_centred=True):
    map = dict()
    for i,lower_bound in enumerate(direction_bins[:sectors]):
        if i == 0 and zero_centred:
            map[i+1] = '{0}-{1}'.format(direction_bins[-2], direction_bins[1])
        else:
            map[i+1] = '{0}-{1}'.format(lower_bound, direction_bins[i+1])
    return map.values()


def get_binned_direction_series(direction_series, sectors, direction_bin_array=None):
    """ Accepts a series with wind directions and number of sectors  you want to divide.
    :param  direction_series: Series of directions to bin
    :param  sectors: number of direction sectors
    :param direction_bin_array: An optional parameter, if you want custom direction bins pass an array
                        of the bins. If nto specified direction_bins will be centered around 0
    :returns  A series with direction-bins, bins centered around 0 degree by default if direction_bin_array
    is not specified
    """
    if direction_bin_array is None:
        direction_bin_array = _get_direction_bin_array(sectors)

    def _get_direction_bin(wdir, bins):
        kwargs = {}
        if wdir == max(bins):
            kwargs['right'] = True
        else:
            kwargs['right'] = False
        bin_num = np.digitize([wdir], bins, **kwargs)[0]
        if bin_num == sectors+1:
            bin_num = 1
        return bin_num
    return direction_series.apply(_get_direction_bin,bins=direction_bin_array)


def get_distribution_by_wind_sector(var_series, direction_series, sectors=12, aggregation_method='%frequency',
                                    direction_bin_array=None, direction_bin_labels=None):
    """Accepts a series of a variable and  wind direction. Computes the distribution of first variable with respect to
    wind direction sectors
    :param var_series: Series of the variable whose distribution we need to find
    :param direction_series: Series of wind directions between [0-360]
    :param sectors: Number of sectors to bin direction to. The first sector is centered at 0 by default. To change that
            behaviour specify direction_bin_array
    :param aggregation_method: Statistical method used to find distribution it can be mean, max, min, std, count,
    describe, a custom function, etc. Computes frequency in percentages by default
    :param direction_bin_array: Optional, to change default behaviour of first sector centered at 0 assign an array of
    bins to this
    :param direction_bin_labels: Optional, you can specify an array of labels to be used for the bins. uses string
    labels of the format '30-90' by default
    :returns A dataframe/series with wind direction sector as row indexes and columns with statistics chosen by
    aggregation_method
    """
    var_series = var_series.dropna()
    direction_series = direction_series.dropna()
    if direction_bin_array is None:
        direction_bin_array = _get_direction_bin_array(sectors)
        zero_centered = True
    else:
        sectors = len(direction_bin_array)-1
        zero_centered = False
    if direction_bin_labels is None:
        direction_bin_labels = _get_direction_bin_labels(sectors, direction_bin_array, zero_centered)
    direction_binned_series = get_binned_direction_series(direction_series, sectors, direction_bin_array).rename('direction_bin')
    data = pd.concat([var_series.rename('data'), direction_binned_series], join='inner', axis=1)
    if aggregation_method == '%frequency':
        result = data.groupby(['direction_bin'])['data'].count().rename('%frequency')/len(data) *100.0
    else:
        result = data.groupby(['direction_bin'])['data'].agg(aggregation_method)
    result.index = direction_bin_labels
    return result


def get_freq_table(var_series, direction_series, var_bin_array=np.arange(-0.5,41,1), sectors=12, var_bin_labels=None,
                   direction_bin_array=None, direction_bin_labels=None,freq_as_percentage=True):
    """Accepts a variable series and direction series and computes a frequency table of percentages. Both variable and
    direction are binned
    :param var_series: Series of variable to be binned
    :param direction_series: Series of wind directions between [0-360]
    :param var_bin_array: Array of numbers where adjacent elements of array form a bin
    :param sectors: Number of sectors to bin direction to. The first sector is centered at 0 by default. To change that
            behaviour specify direction_bin_array
    :param var_bin_labels: Optional, an array of labels to use for variable bins
    :param direction_bin_array: Optional, to change default behaviour of first sector centered at 0 assign an array of
    bins to this
    :param direction_bin_labels: Optional, you can specify an array of labels to be used for the bins. uses string
    labels of the format '30-90' by default
    :param freq_as_percentage: Optional, True by default. Returns the frequency as percentages. To return just the count change
    it to False
    :returns A dataframe with row indexes as variable bins and columns as wind direction bins.
    """
    var_series = var_series.dropna()
    direction_series = direction_series.dropna()
    if direction_bin_array is None:
        direction_bin_array = _get_direction_bin_array(sectors)
        zero_centered = True
    else:
        sectors = len(direction_bin_array)-1
        zero_centered = False
    if direction_bin_labels is None:
        direction_bin_labels = _get_direction_bin_labels(sectors, direction_bin_array, zero_centered)
    var_binned_series = pd.cut(var_series, var_bin_array, right=False, labels=var_bin_labels).rename('variable_bin')
    direction_binned_series = get_binned_direction_series(direction_series, sectors, direction_bin_array).rename(
        'direction_bin')
    data = pd.concat([var_series.rename('var_data'), var_binned_series,direction_binned_series],axis=1).dropna()
    if freq_as_percentage:
        result = pd.crosstab(data.loc[:,'variable_bin'],data.loc[:,'direction_bin']) / len(data) *100.0
    else:
        result = pd.crosstab(data.loc[:, 'variable_bin'], data.loc[:, 'direction_bin'])
    result.columns = direction_bin_labels
    return result


def plot_wind_rose(data, freq_table=False,direction_col_name=0,sectors=12):
    """Plot a wind rose from a direction data or a frequency table.
    """
    if not freq_table:
        data = data.dropna(subset=[direction_col_name])
        data.loc[:, 'direction_bin'] = data[direction_col_name].apply(_map_direction_bin, bins=_get_direction_bin_array(sectors))
        result = data['direction_bin'].value_counts() / len(data['direction_bin']) * 100.0
        result.loc[1] += result.loc[sectors+1]
        result = result.drop(sectors+1, axis=0).sort_index()
    else:
        sectors= data.shape[1]
        result = data.sum(axis=0)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.1, 0.1, 0.8,0.8], polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0,360,360.0/sectors))
    ax.set_rgrids(np.arange(0,101,10),labels=[str(i)+'%' for i in np.arange(0,101,10)],angle=0)
    ax.bar(np.arange(0,2.0*np.pi,2.0*np.pi/sectors), result, width=2.0*np.pi/sectors, bottom=0.0,color='#9ACD32',edgecolor=['#6C9023' for i in range(len(result))],alpha=0.8)
    ax.set_title(str(direction_col_name)+' Wind Rose',loc='center')
    plt.show()


def plot_freq_distribution(data,max_speed=30):
    from matplotlib.ticker import PercentFormatter
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8,0.8])
    ax.set_xlabel('Speed [m/s]')
    ax.set_ylabel('Frequency [%]')
    ax.set_xticks(data.index)
    ax.set_xlim(-0.5,max_speed+0.5)
    ax.set_ylim(0,max(data)+5)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(b=True, axis='y', zorder=0)
    #ax.bar(result.index, result.values,facecolor='#9ACD32',edgecolor=['#6C9023' for i in range(len(result))],zorder=3)
    for frequency, bin in zip(data,data.index):
        ax.imshow(np.array([[[154, 205, 50]],[[215, 235, 173]]])/255.0,interpolation='gaussian',extent=(bin-0.4,bin+0.4,0,frequency),aspect='auto',zorder=3)
        ax.bar(bin, frequency, edgecolor='#6c9023', linewidth=0.3, fill=False, zorder=5)
    ax.set_title('Wind Speed Frequency Distribution')
    plt.show()


def plot_wind_rose_with_speed_3_bins(table):
    import matplotlib as mpl
    sectors=len(table.columns)
    table_binned=pd.DataFrame()
    table_trans = table.T
    table_binned = pd.concat([table_binned,table_trans.loc[:,0:3].sum(axis=1).rename(3)],axis=1)
    table_binned = pd.concat([table_binned,table_trans.loc[:,4:6].sum(axis=1).rename(6)],axis=1)
    table_binned = pd.concat([table_binned,table_trans.loc[:,7:9].sum(axis=1).rename(9)],axis=1)
    table_binned = pd.concat([table_binned,table_trans.loc[:,10:12].sum(axis=1).rename(12)],axis=1)
    table_binned = pd.concat([table_binned,table_trans.loc[:,13:15].sum(axis=1).rename(15)],axis=1)
    table_binned = pd.concat([table_binned, table_trans.loc[:, 16:].sum(axis=1).rename(18)], axis=1)
    table_binned = table_binned.T
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.1, 0.1, 0.8,0.8], polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0,360,360.0/sectors))
    ax.set_rgrids(np.linspace(0,max(table.sum(axis=0))+2.0,10),labels=[ '%.0f' % round(i)+'%' for i in np.linspace(0,max(table.sum(axis=0))+2.0,10)],angle=0)
    direction_bins = _get_direction_bin_array(sectors)[1:-2]
    direction_bins = np.insert(direction_bins,0,direction_bins[-2])
    ax.set_ylim(0,max(table.sum(axis=0))+3.0)
    angular_width = 2*np.pi/sectors - (np.pi/180) #Leaving 1 degree gap
    def _choose_color(speed_bin):
        colors = ['#f5faea','#d6ebad','#b8dc6f','#9acd32','#7ba428', '#5c7b1e']
        bins = [0,3.5,6.5,9.5,12.5,15.5,18.5,41]
        return(colors[np.digitize([speed_bin], bins)[0]-1])

    for column in table_binned:
        radial_pos = 0.0
        angular_pos = (np.pi / 180.0) * float(column.split('-')[0])
        for speed_bin,frequency in zip(table_binned.index,table_binned[column]):
            color = _choose_color(speed_bin)
            patch = mpl.patches.Rectangle((angular_pos, radial_pos), angular_width, frequency, facecolor=color,edgecolor='#5c7b1e',linewidth=0.3)
            ax.add_patch(patch)
            radial_pos += frequency
    legend_patches = [mpl.patches.Patch(color='#f5faea', label='0-3 m/s'),
                        mpl.patches.Patch(color='#d6ebad', label='4-6 m/s'),
                        mpl.patches.Patch(color='#b8dc6f', label='7-9 m/s'),
                        mpl.patches.Patch(color='#9acd32', label='10-12 m/s'),
                        mpl.patches.Patch(color='#7ba428', label='13-15 m/s'),
                      mpl.patches.Patch(color='#5c7b1e', label='15+ m/s')]
    ax.legend(handles=legend_patches)
    plt.show()


def plot_wind_rose_with_speed(table):
    import matplotlib as mpl
    sectors=len(table.columns)
    table_binned=pd.DataFrame()
    table_trans = table.T
    table_binned = pd.concat([table_binned,table_trans.loc[:,0:4].sum(axis=1).rename(4)],axis=1)
    table_binned = pd.concat([table_binned,table_trans.loc[:,5:8].sum(axis=1).rename(8)],axis=1)
    table_binned = pd.concat([table_binned,table_trans.loc[:,9:12].sum(axis=1).rename(12)],axis=1)
    table_binned = pd.concat([table_binned,table_trans.loc[:,13:16].sum(axis=1).rename(16)],axis=1)
    table_binned = pd.concat([table_binned,table_trans.loc[:,17:].sum(axis=1).rename(40)],axis=1)
    table_binned = table_binned.T

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.1, 0.1, 0.8,0.8], polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0,360,360.0/sectors))
    ax.set_rgrids(np.linspace(0,max(table.sum(axis=0))+2.0,10),labels=[ '%.0f' % round(i)+'%' for i in np.linspace(0,max(table.sum(axis=0))+2.0,10)],angle=0)
    direction_bins = _get_direction_bin_array(sectors)[1:-2]
    direction_bins = np.insert(direction_bins,0,direction_bins[-2])
    ax.set_ylim(0,max(table.sum(axis=0))+3.0)
    angular_width = 2*np.pi/sectors - (np.pi/180) #Leaving 1 degree gap

    def _choose_color(speed_bin):
        colors = ['#d6ebad','#b8dc6f','#9acd32','#7ba428', '#5c7b1e']
        bins = [0,4.5,8.5,12.5,16.5,41]
        return(colors[np.digitize([speed_bin], bins)[0]-1])

    for column in table_binned:
        radial_pos = 0.0
        angular_pos = (np.pi / 180.0) * float(column.split('-')[0])
        for speed_bin,frequency in zip(table_binned.index,table_binned[column]):
            color = _choose_color(speed_bin)
            patch = mpl.patches.Rectangle((angular_pos, radial_pos), angular_width, frequency, facecolor=color,edgecolor='#5c7b1e',linewidth=0.3)
            ax.add_patch(patch)
            radial_pos += frequency
    legend_patches = [mpl.patches.Patch(color='#d6ebad', label='0-4 m/s'),
                        mpl.patches.Patch(color='#b8dc6f', label='5-8 m/s'),
                        mpl.patches.Patch(color='#9acd32', label='9-12 m/s'),
                        mpl.patches.Patch(color='#7ba428', label='13-16 m/s'),
                        mpl.patches.Patch(color='#5c7b1e', label='16+ m/s')]
    ax.legend(handles=legend_patches)
    plt.show()

def get_time_continuity(data,time_col_name,time_interval):
    #Get time continuity is a function that sees if there are any gaps bigger than the time interval specified in the
    #data, and brings back a table of values.

    #Note this first part of the function should be obselete, as the data should be preppred before it is used in
    #mast analyse, with timestamp converted correctly to pandas dataframe. Month, and day can get mixed up creating
    #multiple errors if this is not done correctly. The same fuction is used, but with parameters yearfirst or dayfirst
    data[time_col_name] = pd.to_datetime(data[time_col_name])

    #Next sort values by time, then re-index dataframe
    data = data.sort_values(by=time_col_name)
    data = data.reset_index()
    data = data.drop(columns=['index'])
    #Create a new column, if interval ok, column is True, if interval is not, column is false
    data['Interval'] = (data[time_col_name].diff()[1:] == np.timedelta64(time_interval, 'm'))

    #Where the column is False create a new dataframe, data_problems. Reset the index in this, and apply minus 1, then apply
    #index again. This provides the index for the timestamp prior to the gap.
    data_problems = data.loc[data['Interval'] == False]
    data_problems = data_problems.reset_index()
    data_problems['index'] = data_problems['index'] - 1
    data_problems = data_problems.set_index('index')
    data_problems['Timestamp before'] = data[time_col_name]

    #Next calculate the days lost for each gap
    data_problems['Days Lost'] = (data_problems[time_col_name] - data_problems['Timestamp before']) / np.timedelta64(1,'m')
    data_problems['Days Lost'] = data_problems['Days Lost'] / (1440)

    Time_continuity = pd.DataFrame(
        {'Start Date': data_problems['Timestamp before'], 'End Date': data_problems[time_col_name],
         'Days Lost': data_problems['Days Lost']})
    return Time_continuity

def get_monthly_coverage(data, time_col_name,time_interval):

    #Convert the timestamp column to a datetime variable in pandas
    data[time_col_name] = pd.to_datetime(data[time_col_name])

    #Group data by month and count number of values for each column for each month
    data = data.set_index(time_col_name).groupby(pd.Grouper(freq='M')).count()

    #Divide column values by total records possible each month to return coverage
    data['Divsor'] = (data.index.day * (24*60/time_interval))
    data = data.loc[:,:].div(data['Divsor'],axis=0)
    data = data.drop(columns=['Divsor'])

    #Now format the datetime index as MMM-YYYY
    data = data.reset_index()
    data[time_col_name] = data[time_col_name].apply(lambda x: x.strftime('%b-%Y'))
    data = data.set_index(time_col_name)
    return data

def get_monthly_means(data,time_col_name):

    # Convert the timestamp column to a datetime variable in pandas
    data[time_col_name] = pd.to_datetime(data[time_col_name])

    # Group data by month and count number of values for each column for each month
    data = data.set_index(time_col_name).groupby(pd.Grouper(freq='M')).mean()

    # Now format the datetime index as MMM-YYYY
    data = data.reset_index()
    data[time_col_name] = data[time_col_name].apply(lambda x: x.strftime('%b-%Y'))
    data = data.set_index(time_col_name)
    return data

def get_basic_stats(data,time_col_name):
    #Get basic stats for dataframe, mean, max, min and count
    data1 = data.describe().loc[['mean', 'max', 'min', 'count']]
    data1 = data1.T
    data1 = data1.reset_index()
    data1 = data1.rename(columns={'index': 'Instrument'})

    #The describe function does not return the min, max, count and mean for the timestamp column.
    #So we have calculated these separately here, and then added them back to the data1 above.
    data[time_col_name] = pd.to_datetime(data[time_col_name])
    Timestamp_min = data[time_col_name].min()
    Timestamp_max = data[time_col_name].max()
    Timestamp_mean = pd.Timedelta(Timestamp_max - Timestamp_min) / 2
    Timestamp_mean = Timestamp_min + Timestamp_mean
    Timestamp_count = data[time_col_name].count()

    data = data1.append({'Instrument': time_col_name, 'mean': Timestamp_mean, 'max': Timestamp_max, 'min': Timestamp_min,
                  'count': Timestamp_count}, ignore_index=True)
    #After data is appended, then the Timestamp row has to be ordered to be the first row.
    target_row = data.index.max()
    idx = [target_row] + [i for i in range(len(data)) if i != target_row]
    data = data.iloc[idx]
    data = data.reset_index(drop=True)
    return data

def plot_monthly_means(data,time_col_name):
    #Get table of monthly means from data passed
    data = get_monthly_means(data, time_col_name)

    #Make Timestamp its own column and not an index
    data = data.reset_index()

    #Setup figure for plotting, then plot all columns in dataframe
    plt.figure(figsize=(15, 7.5))
    for i in range(1, len(data.columns)):
        plt.plot(data.iloc[:, 0], data.iloc[:, i])
    plt.ylabel('Wind speed [m/s]')
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

def get_TI_by_Speed(data,speed_col_name,std_col_name):

    #Takes a dataframe, pulls the speed and standard deviation column.
    #This is then binned by windspeed and the average TI and count of TI extracted.

    data = data.dropna(subset=[speed_col_name, std_col_name])
    data['Turbulence_Intensity'] = data[std_col_name] / data[speed_col_name]
    speed_bins = np.arange(-0.5, 41, 1)
    # data['bins'] = pd.cut(data[speed_col_name], speed_bins,right=False)
    data = pd.concat([data, data.loc[:, speed_col_name].apply(_map_speed_bin, bins=speed_bins).rename('bins')], axis=1)
    max_bin = data['bins'].max()
    grouped = data.groupby(['bins'])
    grouped1 = grouped['Turbulence_Intensity'].mean()
    grouped2 = grouped['Turbulence_Intensity'].count()
    grouped3 = grouped[std_col_name].std()
    grouped4 = grouped['Turbulence_Intensity'].quantile(.9)
    speed_bins = np.arange(0, max_bin + 1, 1)
    charTI = grouped1 + (grouped3 / speed_bins)
    TI = pd.DataFrame({str(speed_col_name) + '_TI_Avg': grouped1,
                       str(speed_col_name) + '_TI_Count': grouped2, str(std_col_name) + '_SigmaSigma': grouped3,
                       str(speed_col_name) + '_CharTI': charTI, str(speed_col_name) + '_RepTI': grouped4})
    TI.at[0, str(speed_col_name) + '_CharTI'] = 0
    TI = TI.reset_index()
    return TI

def plot_TI_by_Speed(data,speed_col_name,std_col_name):

    #IEC Class 2005
    #Note we have removed option to include IEC Class 1999 as no longer appropriate.
    #This may need to be placed in a separate function when updated IEC standard is released

    columns = ['Windspeed', 'IEC Class A', 'IEC Class B', 'IEC Class C']
    IEC_Class_2005 = pd.DataFrame(np.zeros([26, 4]), columns=columns)

    for n in range(1, 26):
        IEC_Class_2005.iloc[n, 0] = n
        IEC_Class_2005.iloc[n, 1] = 0.16 * (0.75 + (5.6 / n))
        IEC_Class_2005.iloc[n, 2] = 0.14 * (0.75 + (5.6 / n))
        IEC_Class_2005.iloc[n, 3] = 0.12 * (0.75 + (5.6 / n))

    #Get Average Turbulence Intensity and Representative Turbulence Intensity for the plot
    TI = get_TI_by_Speed(data, speed_col_name, std_col_name)
    data['Turbulence_Intensity'] = data[std_col_name] / data[speed_col_name]

    #Plot Figure
    plt.figure(figsize=(15, 7.5))
    plt.scatter([data[speed_col_name]], [data['Turbulence_Intensity']], color=BWcolors('green'), s=1, alpha=0.3)
    plt.plot(TI.iloc[:, 0], TI.iloc[:, 1], color=BWcolors('darkgreen'))
    plt.plot(TI.iloc[:, 0], TI.iloc[:, 5], color=BWcolors('redline'))
    plt.plot(IEC_Class_2005.iloc[:, 0], IEC_Class_2005.iloc[:, 1], color=BWcolors('greyline'))
    plt.plot(IEC_Class_2005.iloc[:, 0], IEC_Class_2005.iloc[:, 2], color=BWcolors('greyline'))
    plt.plot(IEC_Class_2005.iloc[:, 0], IEC_Class_2005.iloc[:, 3], color=BWcolors('greyline'))
    plt.axis([2, 25, 0, 0.6])
    plt.xticks(np.arange(2, 26, 1))
    plt.xlabel('Wind speed [m/s]')
    plt.ylabel('Turbulence Intensity')
    # plt.title('Turbulence Intensity by Windspeed for ' + str(speed_col))
    plt.grid(True)
    plt.legend()
    plt.show()


def get_TI_by_sector(data,speed_col_name,std_col_name,direction_col_name,sectors,min_speed):

    data = data.dropna(subset=[speed_col_name, std_col_name, direction_col_name])
    data['Turbulence_Intensity'] = data[std_col_name] / data[speed_col_name]

    #Reduce dataframe by excluding all values below Minimum Speed
    data = data[data[speed_col_name] >= min_speed]

    #Here we take away half of the sector value from all of the direction column and if below zero we add 360.
    #Essentially rotates values by the appropriate amount, then when you bin from zero up to 360, you are actually
    #binning from half way through each sector, and get over the issue of having to add bins on either side of 360 degrees
    data[direction_col_name] = data[direction_col_name] - (360 / (sectors * 2))
    data.loc[data[direction_col_name] < 0, direction_col_name] += 360

    #Get Direction Bins, and then group data by bins while calculating mean TI and count the values
    #Note that sectors in Mast Analysis sheet calculated differently, first sector is not divided on either side of 0.
    #We have implemented sector division here.

    direction_bins = np.arange(0, 360 + (360 / sectors), 360 / sectors)
    data = pd.concat([data, data.loc[:, direction_col_name].apply(_map_direction_bin,bins=direction_bins).rename('Direction Bin')], axis=1)

    grouped = data.groupby(['Direction Bin'])
    grouped1 = grouped['Turbulence_Intensity'].mean()
    grouped2 = grouped['Turbulence_Intensity'].count()

    result = pd.DataFrame({'Turbulence_Intensity_Avg': grouped1, 'Turbulence_Intensity_Count': grouped2})
    result = result.reset_index()

    #Convert direction bins to actual bins, i.e. to include 345-15 degrees, and drop 345-360 sector

    direction_row = _get_direction_bin_dict(_get_direction_bin_array(sectors), sectors)
    new_bins = pd.Series(direction_row, name='bins', dtype='category')
    new_bins = new_bins.reset_index()
    result['Direction bin'] = new_bins['bins']
    result = result[['Direction bin', 'Turbulence_Intensity_Avg', 'Turbulence_Intensity_Count']]
    #Note results differ from Mast Analysis sheet when same binning used. Error was found to be associated with use of mod
    #function in access. Mod is used to convert values that are above 360 degrees, back to nomrmal degrees.
    #In practice, high unlikely any direction values will exceed 360 degrees. When this was removed, results matched exactly.
    return result


def BWcolors(BWcolor):
    #Define color scheme to be used across graphs, and tables.
    if BWcolor == 'green':
        BWcolor = [156, 197, 55]
    elif BWcolor == 'asphault':
        BWcolor = [46, 55, 67]
    elif BWcolor == 'greyline':
        BWcolor = [108, 120, 134]
    elif BWcolor == 'darkgreen':
        BWcolor = [108, 144, 35]
    elif BWcolor == 'redline':
        BWcolor = [255, 0, 0]
    else:
        BWcolor = [156, 197, 55]

    BWcolor[:]=[x / 255 for x in BWcolor]
    return BWcolor


def plot_TI_by_sector(data,speed_col_name,std_col_name,direction_col_name,sectors,min_speed):

    #First we need to calculate the Turbulence Intensity by sector by calling the sector function.
    TI = get_TI_by_sector(data, speed_col_name, std_col_name, direction_col_name, sectors, min_speed)

    #Next we convert the Median bin degree to radians for plotting
    TI['Polar degrees'] = np.radians(TI.index * (360 / sectors))

    #To complete the plot, we need to copy the first row and append a new last row.
    TI.loc[-1] = TI.loc[0, :]

    #Set Figure size, define it as polar, set north, set number of sectors to be displayed
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 360.0 / sectors))
    ax.tick_params(axis='y',labelsize=15)
    #,grid_color='white',labelcolor='white
    #Convert name of Turbulence Intensity Avg Column so it will read well in legend.
    TI['Turbulence Intensity Average by sector'] = TI['Turbulence_Intensity_Avg']

    #Plot the Average turbulence Intensity and assign a title to the graph
    ax.plot(TI['Polar degrees'], TI['Turbulence Intensity Average by sector'], c=BWcolors('green'), linewidth=4)
    plt.title('Turbulence Intensity by Direction')

    #Set the max extent of the polar plot to be the max average sector turbulence + 0.1
    maxlevel = TI['Turbulence_Intensity_Avg'].max() + 0.1
    ax.set_ylim(0, maxlevel)

    #Add in comment at bottom of graph about what anemometer and wind vane are used.
    ax.annotate('*Plot generated using Anemometer ' + speed_col_name + ' and Wind Vane ' + direction_col_name,
                xy=(120, 10), xycoords='figure pixels')

    #Finally produce a scatter plot of all of the Turbulence Intensity data points
    data['Turbulence Intensity by datapoint'] = data[std_col_name] / data[speed_col_name]
    data['Polar degrees'] = np.radians(data[direction_col_name])
    ax.scatter(data['Polar degrees'], data['Turbulence Intensity by datapoint'], c=BWcolors('asphault'), alpha=0.3, s=1)

    plt.legend(loc=8, framealpha=1)

    plt.show()


def get_12x24_TI_matrix(data,time_col_name,speed_col_name,std_col_name):
    #Get the 12 month x 24 hour matrix of turbulence intensity
    data = data.dropna(subset=[time_col_name, speed_col_name, std_col_name])
    data['Turbulence_Intensity'] = data.loc[:,std_col_name] / data.loc[:,speed_col_name]
    data[time_col_name] = pd.to_datetime(data[time_col_name])

    data['Month'] = data.loc[:,time_col_name].dt.month
    data['Hour'] = data.loc[:,time_col_name].dt.hour

    result = data.pivot_table(index='Hour', columns='Month', values='Turbulence_Intensity')
    return result


def plot_12x24_TI_Contours(data,time_col_name,speed_col_name,std_col_name):
    #Get Contour Plot of 12 month x 24 hour matrix of turbulence intensity
    result = get_12x24_TI_matrix(data,time_col_name,speed_col_name,std_col_name)
    plt.figure(figsize=(15, 7.5))
    x = plt.contourf(result, cmap="Greens")
    cbar = plt.colorbar(x)
    cbar.ax.set_ylabel('Turbulence Intensity')
    plt.xlabel('Month of Year')
    plt.ylabel('Hour of Day')
    plt.xticks(np.arange(12), calendar.month_name[1:13])
    plt.yticks(np.arange(0, 24, 1))
    plt.title('Hourly Mean Turbulence Intensity by Calendar Month')
    plt.show()

def _map_speed_bin(wdspd, bins):
    # Copy of Inders function, can be removed once this is integrated into main library
    kwargs = {}
    if wdspd == max(bins):
        kwargs['right'] = True
    bin = bins[np.digitize([wdspd], bins, **kwargs)[0]]
    bin_lower = bins[np.digitize([wdspd], bins, **kwargs)[0]-1]
    return np.digitize([wdspd], bins, **kwargs)[0]-1.0

def _get_direction_bin_dict(direction_bins,sectors):
    # Copy of Inders function, can be removed once this is integrated into main library
    map = dict()
    for i,lower_bound in enumerate(direction_bins[:sectors]):
        if i==0:
            map[i+1] = '{0}-{1}'.format(direction_bins[-2],direction_bins[1])
        else:
            map[i+1] = '{0}-{1}'.format(lower_bound,direction_bins[i+1])
    return map