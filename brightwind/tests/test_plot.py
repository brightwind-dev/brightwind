import pytest
import brightwind as bw


def test_plot_timeseries():
    data = bw.load_csv(bw.datasets.demo_data)
    bw.plot_timeseries(data[['Spd40mN', 'Spd60mS', 'T2m']])
    bw.plot_timeseries(data[['Spd40mN']], date_from='2017-09-01')
    bw.plot_timeseries(data.Spd40mN, date_from='2017-09-01')
    bw.plot_timeseries(data.Spd40mN, date_to='2017-10-01')
    bw.plot_timeseries(data.Spd40mN, date_from='2017-09-01', date_to='2017-10-01')
    bw.plot_timeseries(data.Spd40mN, date_from='2017-09-01', date_to='2017-10-01', y_limits=(0, None))
    bw.plot_timeseries(data.Spd40mN, date_from='2017-09-01', date_to='2017-10-01', y_limits=None)
    bw.plot_timeseries(data.Spd40mN, date_from='2017-09-01', date_to='2017-10-01', y_limits=(0, 25))
    bw.plot_timeseries(data.Spd40mN, date_from='2017-09-01', date_to='2017-10-01', y_limits=(None, 25))

    assert True


def test_plot_scatter():
    data = bw.load_csv(bw.datasets.demo_data)
    graph = bw.plot_scatter(data.Spd80mN, data.Spd80mS)
    graph = bw.plot_scatter(data.Spd80mN, data[['Spd80mS']])
    bw.plot_scatter(data.Dir78mS, data.Dir58mS, x_axis_title='Dir78mS', y_axis_title='Dir58mS',
                    x_limits=(50, 300), y_limits=(250, 300))
    bw.plot_scatter_wdir(data.Dir78mS, data.Dir58mS, x_axis_title='Reference', y_axis_title='Target',
                         x_limits=(50, 300), y_limits=(250, 300))
    bw.plot_scatter_wspd(data.Spd80mN, data.Spd80mS, x_axis_title='Speed at 80m North',
                         y_axis_title='Speed at 80m South', x_limits=(0, 25), y_limits=(0, 25))

    assert True
