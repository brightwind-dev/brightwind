import pytest
from ..analyse.analyse import monthly_means, SectorRatio

from ..load.load import load_csv
import brightwind.datasets

def test_monthly_means():
    #Load data
    monthly_means(load_csv(brightwind.datasets.shell_flats_80m_csv))
    monthly_means(load_csv(brightwind.datasets.shell_flats_80m_csv)[['WS70mA100NW_Avg','WS70mA100SE_Avg',
                                                                          'WS50mA100NW_Avg','WS50mA100SE_Avg',
                                                                          'WS20mA100CB1_Avg','WS20mA100CB2_Avg']],
                        return_data=True)
    monthly_means(load_csv(brightwind.datasets.shell_flats_80m_csv).WS80mWS425NW_Avg)
    monthly_means(load_csv(brightwind.datasets.shell_flats_80m_csv).WS80mWS425NW_Avg, return_data=True)
    assert True

def test_sector_ratio_by_sector():
    data = load_csv(brightwind.datasets.shell_flats_80m_csv)
    SectorRatio.by_sector(data['WS70mA100NW_Avg'], data['WS70mA100SE_Avg'], data['WD50mW200PNW_VAvg'],
                          sectors = 72, boom_dir_1 = 315, boom_dir_2 = 135,return_data=True)[1]
    assert True