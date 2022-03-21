import pytest
import brightwind as bw
import os

DATA = bw.load_csv(bw.demo_datasets.demo_data)
DATA = bw.apply_cleaning(DATA, bw.demo_datasets.demo_cleaning_file)

cwd = os.getcwd()
os.makedirs(os.path.join(cwd, 'temp'), exist_ok=True)


def test_export_tab_file():
    # test with file path provided
    folder = r'temp'
    graph, tab = bw.freq_table(DATA.Spd40mN, DATA.Dir38mS, return_data=True)
    bw.export_tab_file(tab, 80, 10, 10, file_name='export_tab_file.tab', folder_path=folder)
    graph, tab = bw.freq_table(DATA[['Spd40mN']], DATA[['Dir38mS']], return_data=True)
    bw.export_tab_file(tab, 80, 10, 10, file_name='export_tab_file_1.tab', folder_path=folder)
    assert True


def test_export_to_csv():
    folder = r'temp'
    bw.export_csv(DATA, file_name='export_to_csv', folder_path=folder)

    bw.export_csv(DATA, file_name='export_to_csv_tab.tab', folder_path=folder, sep='\t')
    assert True
