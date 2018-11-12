import pytest
from ..analyse import monthly_means


def test_monthly_means():
    #Load data
    assert monthly_means(data) == 6.0