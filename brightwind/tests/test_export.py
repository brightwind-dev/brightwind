import pytest
import brightwind as bw

DATA = bw.load_csv(bw.datasets.demo_data)
DATA = bw.apply_cleaning(DATA, bw.datasets.demo_cleaning_file)


def test_export_tab_file():
    graph, tab = bw.freq_table(DATA.Spd40mN, DATA.Dir38mS, return_data=True)
    bw.export_tab_file(tab, 80, 10, 10, file_name='export_tab_file.tab')
    graph, tab = bw.freq_table(DATA[['Spd40mN']], DATA[['Dir38mS']], return_data=True)
    bw.export_tab_file(tab, 80, 10, 10, file_name='export_tab_file_1.tab')
    assert True


def test_export_to_csv():
    bw.export_csv(DATA, file_name='export_to_csv')

    bw.export_csv(DATA, file_name='export_to_csv_tab.tab', sep='\t')
    assert True
