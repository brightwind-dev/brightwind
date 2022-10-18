import pytest
import brightwind as bw
import os
import numpy as np

DATA = bw.load_csv(bw.demo_datasets.demo_data)
DATA = bw.apply_cleaning(DATA, bw.demo_datasets.demo_cleaning_file)

# Temp folder used to save test files
TEMP_FOLDER = 'temp'
cwd = os.getcwd()
os.makedirs(os.path.join(cwd, TEMP_FOLDER), exist_ok=True)


def test_export_tab_file():
    graph, tab = bw.freq_table(DATA.Spd40mN, DATA.Dir38mS, return_data=True)
    bw.export_tab_file(tab, 80, 10, 10, file_name='export_tab_file.tab', folder_path=TEMP_FOLDER)
    graph, tab = bw.freq_table(DATA[['Spd40mN']], DATA[['Dir38mS']], return_data=True)
    bw.export_tab_file(tab, 80, 10, 10, file_name='export_tab_file_1.tab', folder_path=TEMP_FOLDER)
    assert True


def test_export_to_csv():
    bw.export_csv(DATA, file_name='export_to_csv', folder_path=TEMP_FOLDER)

    bw.export_csv(DATA, file_name='export_to_csv_tab.tab', folder_path=TEMP_FOLDER, sep='\t')
    assert True

def test_calc_mean_speed_of_freq_tab():
    fig, freq_tab = bw.freq_table(DATA.Spd80mN, DATA.Dir38mS, return_data=True)
    assert round(bw.export.export._calc_mean_speed_of_freq_tab(freq_tab), 5) == 7.51925
    fig, freq_tab = bw.freq_table(DATA.Spd80mN, DATA.Dir38mS, var_bin_array=np.arange(0, 41, 0.5), return_data=True)
    assert round(bw.export.export._calc_mean_speed_of_freq_tab(freq_tab), 5) == 7.5206
