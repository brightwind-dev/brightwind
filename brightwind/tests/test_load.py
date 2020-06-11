import pytest
import brightwind as bw


def test_apply_cleaning_windographer():
    data = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_data)
    data_clnd = bw.apply_cleaning_windographer(data, bw.datasets.demo_windographer_flagging_log)
    data_clnd2 = bw.apply_cleaning_windographer(data, bw.datasets.demo_windographer_flagging_log2,
                                                time_format='%d/%m/%Y %H:%M')

    assert (data_clnd2.fillna(-999) == data_clnd.fillna(-999)).all().all()


def test_apply_cleaning():
    data = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_data)
    data_clnd2 = bw.apply_cleaning_windographer(data, bw.datasets.demo_windographer_flagging_log2,
                                                time_format='%d/%m/%Y %H:%M')
    data_clnd3 = bw.apply_cleaning(data, bw.datasets.demo_cleaning_file)
    data_clnd4 = bw.apply_cleaning(data, '../datasets/demo/demo_cleaning_file2.csv', time_format='%d/%m/%Y %H:%M')

    assert (data_clnd2.drop(['RECORD', 'Site', 'LoggerID'], axis=1).fillna(-999) ==
            data_clnd3.drop(['RECORD', 'Site', 'LoggerID'], axis=1).fillna(-999)).all().all()

    assert (data_clnd3.drop(['RECORD', 'Site', 'LoggerID'], axis=1).fillna(-999) ==
            data_clnd4.drop(['RECORD', 'Site', 'LoggerID'], axis=1).fillna(-999)).all().all()

