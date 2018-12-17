import pytest
from ..load.load import load_csv
import brightwind.datasets


def test_creyap():
    load_csv(brightwind.datasets.shell_flats_80m_csv)
    load_csv(brightwind.datasets.shell_flats_50m_csv)
    load_csv(brightwind.datasets.shell_flats_merra)

