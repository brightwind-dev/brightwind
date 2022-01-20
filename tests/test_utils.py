import pytest
from brightwind.utils import utils


def test_rename_equal_elements_between_two_inputs():

    input1 = ['Spd80mNT', 'Spd80mN', 'Spd50mN', 'Spd60mN']
    input2 = ['Spd80mN', 'Spd50mN']

    assert utils._rename_equal_elements_between_two_inputs(input1, input2) == [
        'Spd80mNT', 'Spd80mN_1', 'Spd50mN_1', 'Spd60mN']
    assert utils._rename_equal_elements_between_two_inputs('Spd80mN', input2, input1_suffix='_ref'
                                                           ) == 'Spd80mN_ref'
    assert utils._rename_equal_elements_between_two_inputs(input1, 'Spd80mN', input1_suffix='_ref') == [
        'Spd80mNT', 'Spd80mN_ref', 'Spd50mN', 'Spd60mN']
