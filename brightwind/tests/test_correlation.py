import pytest
import brightwind as bw
import pandas as pd
import numpy as np


def test_synthesize():
    ref_spd = bw.load_csv('../datasets/demo/MERRA-2_NE_2000-01-01_2017-06-30.csv')
    target_spd = bw.load_csv('../datasets/demo/demo_data.csv')
    ord_lst_sq = bw.Correl.OrdinaryLeastSquares(ref_spd['WS50m_m/s'], target_spd['Spd80mN'],averaging_prd='3D')
    ord_lst_sq.run(show_params=False)
    synth_spd = ord_lst_sq.synthesize()

    target_spd_avg = bw.transform.transform.average_data_by_period(target_spd['Spd80mN'], '3D', return_coverage=False)
    test = pd.concat([target_spd_avg, synth_spd[target_spd_avg .index[0]:
                                                target_spd_avg .index[-1]]], axis=1)[~pd.isnull(target_spd_avg)]

    assert len(np.unique(np.diff(synth_spd.index))) == 1

    assert test['Spd80mN'].equals(test['Spd80mN_Synthesized'])