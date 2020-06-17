import pytest
import brightwind as bw


def test_apply_cleaning_windographer():
    data = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_data)
    data_clnd = bw.apply_cleaning_windographer(data, bw.datasets.demo_windographer_flagging_log)
    data_clnd2 = bw.apply_cleaning_windographer(data, bw.datasets.demo_windographer_flagging_log2, dayfirst=True)

    assert (data_clnd2.fillna(-999) == data_clnd.fillna(-999)).all().all()


def test_apply_cleaning():
    data = bw.load_campbell_scientific(bw.datasets.demo_campbell_scientific_data)
    data_clnd2 = bw.apply_cleaning_windographer(data, bw.datasets.demo_windographer_flagging_log2, dayfirst=True)
    data_clnd3 = bw.apply_cleaning(data, bw.datasets.demo_cleaning_file)
    data_clnd4 = bw.apply_cleaning(data, '../datasets/demo/demo_cleaning_file2.csv', dayfirst=True)

    assert (data_clnd2.drop(['RECORD', 'Site', 'LoggerID'], axis=1).fillna(-999) ==
            data_clnd3.drop(['RECORD', 'Site', 'LoggerID'], axis=1).fillna(-999)).all().all()

    assert (data_clnd3.drop(['RECORD', 'Site', 'LoggerID'], axis=1).fillna(-999) ==
            data_clnd4.drop(['RECORD', 'Site', 'LoggerID'], axis=1).fillna(-999)).all().all()


def test_load_csv():
    data = bw.load_csv('../datasets/demo/demo_data.csv')
    data2 = bw.load_csv('../datasets/demo/demo_data2.csv', dayfirst=True)
    data3 = bw.load_csv('../datasets/demo/demo_data3.csv', dayfirst=True)
    data4 = bw.load_csv('../datasets/demo/demo_data4.csv', dayfirst=True)

    assert (data.fillna(-999) == data2.fillna(-999)).all().all()
    assert (data.fillna(-999) == data3.fillna(-999)).all().all()
    assert (data.fillna(-999) == data4.fillna(-999)).all().all()


def test_load_windographer_txt():
    data = bw.load_windographer_txt('../datasets/demo/windographer_demo_data.txt')
    data1 = bw.load_windographer_txt('../datasets/demo/windographer_demo_data1.txt', dayfirst=True)
    data2 = bw.load_windographer_txt('../datasets/demo/windographer_demo_data2.txt', dayfirst=True)

    assert (data.fillna(-999) == data1.fillna(-999)).all().all()
    assert (data.fillna(-999) == data2.fillna(-999)).all().all()

