import pandas as pd
from transform import transform as tf


def get_coverage(ref, target, averaging_prd, aggregation_method_ref='mean', aggregation_method_target='mean'):
    """
    Accepts ref and target data and returns the coverage of concurrent data.
    :param ref: Reference data
    :type ref: pandas.Series
    :param target: Target data
    :type target: pandas.Series
    :param averaging_prd: Groups data by the period specified by period.

            * 2T, 2 min for minutely average
            * Set period to 1D for a daily average, 3D for three hourly average, similarly 5D, 7D, 15D etc.
            * Set period to 1H for hourly average, 3H for three hourly average and so on for 5H, 6H etc.
            * Set period to 1MS for monthly average
            * Set period to 1AS fo annual average

    :type averaging_prd: str
    :return:
    """
    ref_overlap, target_overlap = tf._get_overlapping_data(ref.sort_index().dropna(), target.sort_index().dropna(), averaging_prd)
    from pandas.tseries.frequencies import to_offset
    ref_resolution = tf._get_data_resolution(ref_overlap.index)
    target_resolution = tf._get_data_resolution(target_overlap.index)
    if ref_resolution > target_resolution:
        target_overlap = tf.average_data_by_period(target_overlap, to_offset(ref_resolution), filter=True,
                                                   coverage_threshold=1,aggregation_method=aggregation_method_target)
    if ref_resolution < target_resolution:
        ref_overlap = tf.average_data_by_period(ref_overlap, to_offset(target_resolution), filter=True,
                                                coverage_threshold=1,aggregation_method=aggregation_method_ref)
    common_idxs, data_pts = tf._common_idxs(ref_overlap, target_overlap)
    ref_concurrent = ref_overlap.loc[common_idxs]
    target_concurrent = target_overlap.loc[common_idxs]
    return pd.concat([tf.average_data_by_period(ref_concurrent, averaging_prd, filter=False,
                                         coverage_threshold=0,
                                         aggregation_method=aggregation_method_ref)]+list(
           tf.average_data_by_period(target_concurrent, averaging_prd, filter=False,
                                     coverage_threshold=0,
                                     aggregation_method=aggregation_method_target, return_coverage=True)), axis=1)


def mean_of_monthly_means(df: pd.DataFrame) -> pd.DataFrame:
    """ Return series of mean of momthly means for each column in the dataframe with timestamp as the index.
        Calculate the monthly mean for each calendar month and then average the resulting 12 months.
    """
    monthly_df: pd.DataFrame = df.groupby(df.index.month).mean()
    momm_series: pd.Series = monthly_df.mean()
    momm_df: pd.DataFrame = pd.DataFrame([momm_series], columns=['MOMM'])
    return momm_df


def calc_target_value_by_linear_model(ref_value: float, slope: float, offset: float):
    """
    :rtype: np.float64
    """
    return (ref_value*slope) + offset


def calc_lt_ref_speed(data: pd.DataFrame, date_from: str='', date_to: str=''):
    """Calculates and returns long term reference speed. Accepts a dataframe
    with timestamps as index column and another column with wind-speed. You can also specify
    date_from and date_to to calculate the long term reference speed for only that period.
    :param: data: Pandas dataframe with timestamp as index and a column with wind-speed
    :param: date_from: Start date as string in format YYYY-MM-DD
    :param: date_to: End date as string in format YYYY-MM-DD
    :returns: Long term reference speed
    """
    import datetime
    if (isinstance(date_from, datetime.date) or isinstance(date_from, datetime.datetime))\
        and (isinstance(date_to,datetime.date) or isinstance(date_to, datetime.datetime)):
        data = data.loc[date_from:date_to, :]
    elif date_from and date_to:
        import datetime as dt
        date_from = dt.datetime.strptime(date_from[:10], "%Y-%m-%d")
        date_to = dt.datetime.strptime(date_to[:10], "%Y-%m-%d")
        data = data.loc[date_from:date_to, :]
    return mean_of_monthly_means(data).get_value(index=0, col='MOMM')
