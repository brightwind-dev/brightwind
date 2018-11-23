import pytest
from ..load.load import load_timeseries
import brightwind.datasets

def test_creyap():
    load_timeseries(brightwind.datasets.creyap_80m_csv)
    load_timeseries(brightwind.datasets.creyap_50m_csv)
    load_timeseries(brightwind.datasets.merra2_west)

