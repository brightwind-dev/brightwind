import pytest
import brightwind as bw


def test_export_tab_file():
    df = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)
    graph, tab = bw.freq_table(df.Spd40mN, df.Dir38mS, return_data=True)
    bw.export_tab_file(tab, 80, 10, 10, file_name='campbell_tab_file_test')


def test_export_to_csv():

    df = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_site_data)

    # to export a .csv file with a specified name to a specific folder
    bw.export_csv(df, file_name='export_to_csv_test')

    # to export a .tab file
    bw.export_csv(df, file_name='export_to_csv_tab_test.tab', sep='\t')
