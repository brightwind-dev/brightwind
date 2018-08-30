import pandas as pd
import numpy as np


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
    :returns A DataFrame/Series with bins as row indexes and columns with statistics chosen by aggregation_method"""
    var1_series = var1_series.dropna()
    var2_series = var2_series.dropna()
    var2_binned_series = pd.cut(var2_series, var2_bin_array, right=False, labels=var2_bin_labels).rename('variable_bin')
    data = pd.concat([var1_series.rename('data'), var2_binned_series], join='inner', axis=1)
    if aggregation_method == '%frequency':
        return data.groupby(['variable_bin'])['data'].count().rename('%frequency')/len(data) * 100.0
    else:
        return data.groupby(['variable_bin'])['data'].agg(aggregation_method)


def get_direction_bin_array(sectors):
    bin_start = 180.0/sectors
    direction_bins = np.arange(bin_start, 360, 360.0/sectors)
    direction_bins = np.insert(direction_bins, 0, 0)
    direction_bins = np.append(direction_bins, 360)
    return direction_bins


def _get_direction_bin_labels(sectors, direction_bins, zero_centred=True):
    mapper = dict()
    for i, lower_bound in enumerate(direction_bins[:sectors]):
        if i == 0 and zero_centred:
            mapper[i+1] = '{0}-{1}'.format(direction_bins[-2], direction_bins[1])
        else:
            mapper[i+1] = '{0}-{1}'.format(lower_bound, direction_bins[i+1])
    return mapper.values()


def map_direction_bin(wdir, bins, sectors):
    kwargs = {}
    if wdir == max(bins):
        kwargs['right'] = True
    else:
        kwargs['right'] = False
    bin_num = np.digitize([wdir], bins, **kwargs)[0]
    if bin_num == sectors+1:
        bin_num = 1
    return bin_num


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
        direction_bin_array = get_direction_bin_array(sectors)
    return direction_series.apply(map_direction_bin, bins=direction_bin_array, sectors=sectors)


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
        direction_bin_array = get_direction_bin_array(sectors)
        zero_centered = True
    else:
        sectors = len(direction_bin_array)-1
        zero_centered = False
    if direction_bin_labels is None:
        direction_bin_labels = _get_direction_bin_labels(sectors, direction_bin_array, zero_centered)
    direction_binned_series = get_binned_direction_series(direction_series, sectors, direction_bin_array)\
        .rename('direction_bin')
    data = pd.concat([var_series.rename('data'), direction_binned_series], join='inner', axis=1)
    if aggregation_method == '%frequency':
        result = data.groupby(['direction_bin'])['data'].count().rename('%frequency')/len(data) * 100.0
    else:
        result = data.groupby(['direction_bin'])['data'].agg(aggregation_method)
    result.index = direction_bin_labels
    return result


def get_freq_table(var_series, direction_series, var_bin_array=np.arange(-0.5, 41, 1), sectors=12, var_bin_labels=None,
                   direction_bin_array=None, direction_bin_labels=None, freq_as_percentage=True):
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
    :param freq_as_percentage: Optional, True by default. Returns the frequency as percentages. To return just the count
    change it to False
    :returns A DataFrame with row indexes as variable bins and columns as wind direction bins.
    """
    var_series = var_series.dropna()
    direction_series = direction_series.dropna()
    if direction_bin_array is None:
        direction_bin_array = get_direction_bin_array(sectors)
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


def get_time_continuity(data, time_col_name, time_interval):
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

    time_continuity = pd.DataFrame(
        {'Start Date': data_problems['Timestamp before'], 'End Date': data_problems[time_col_name],
         'Days Lost': data_problems['Days Lost']})
    return time_continuity


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


def get_TI_by_Speed(data,speed_col_name,std_col_name):
    #Takes a dataframe, pulls the speed and standard deviation column.
    #This is then binned by windspeed and the average TI and count of TI extracted.

    data = data.dropna(subset=[speed_col_name, std_col_name])
    data['Turbulence_Intensity'] = data[std_col_name] / data[speed_col_name]
    speed_bins = np.arange(-0.5, 41, 1)
    # data['bins'] = pd.cut(data[speed_col_name], speed_bins,right=False)
    data = pd.concat([data, data.loc[:, speed_col_name].apply(map_speed_bin, bins=speed_bins).rename('bins')], axis=1)
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
    data = pd.concat([data, data.loc[:, direction_col_name].apply(map_direction_bin,bins=direction_bins).rename('Direction Bin')], axis=1)

    grouped = data.groupby(['Direction Bin'])
    grouped1 = grouped['Turbulence_Intensity'].mean()
    grouped2 = grouped['Turbulence_Intensity'].count()


    result = pd.DataFrame({'Turbulence_Intensity_Avg': grouped1, 'Turbulence_Intensity_Count': grouped2})
    result = result.reset_index()

    #Convert direction bins to actual bins, i.e. to include 345-15 degrees, and drop 345-360 sector

    direction_row = _get_direction_bin_dict(get_direction_bin_array(sectors), sectors)
    new_bins = pd.Series(direction_row, name='bins', dtype='category')
    new_bins = new_bins.reset_index()
    result['Direction bin'] = new_bins['bins']
    result = result[['Direction bin', 'Turbulence_Intensity_Avg', 'Turbulence_Intensity_Count']]
    #Note results differ from Mast Analysis sheet when same binning used. Error was found to be associated with use of mod
    #function in access. Mod is used to convert values that are above 360 degrees, back to nomrmal degrees.
    #In practice, high unlikely any direction values will exceed 360 degrees. When this was removed, results matched exactly.
    return result


def get_12x24_TI_matrix(data,time_col_name,speed_col_name,std_col_name):
    #Get the 12 month x 24 hour matrix of turbulence intensity
    data = data.dropna(subset=[time_col_name, speed_col_name, std_col_name])
    data['Turbulence_Intensity'] = data.loc[:,std_col_name] / data.loc[:,speed_col_name]
    data[time_col_name] = pd.to_datetime(data[time_col_name])

    data['Month'] = data.loc[:,time_col_name].dt.month
    data['Hour'] = data.loc[:,time_col_name].dt.hour

    result = data.pivot_table(index='Hour', columns='Month', values='Turbulence_Intensity')
    return result


def map_speed_bin(wdspd, bins):
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