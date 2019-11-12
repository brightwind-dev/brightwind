import pytest
from brightwind.load.load import load_csv
import brightwind.datasets


def test_creyap():
    load_csv(brightwind.datasets.demo_data)

