import pytest
import brightwind as bw


def test_plot_timeseries():
    df = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)
    graph = bw.plot_timeseries(df[['Spd40mN', 'Spd60mS', 'T2m']])
    graph = bw.plot_timeseries(df.Spd40mN, date_from='04-21-2016')
    graph = bw.plot_timeseries(df.Spd40mN, date_from='01-21-2016', date_to='02-28-2016')

    assert True
