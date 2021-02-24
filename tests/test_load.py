import pytest
import brightwind as bw
import os

DEMO_DATA_FOLDER = os.path.join(os.path.dirname(__file__), '../brightwind/demo_datasets')


def test_apply_cleaning_windographer():
    data = bw.load_campbell_scientific(bw.demo_datasets.demo_campbell_scientific_data)
    data_clnd = bw.apply_cleaning_windographer(data, bw.demo_datasets.demo_windographer_flagging_log)
    data_clnd2 = bw.apply_cleaning_windographer(data, bw.demo_datasets.demo_windographer_flagging_log2, dayfirst=True)

    assert (data_clnd2.fillna(-999) == data_clnd.fillna(-999)).all().all()


def test_apply_cleaning():
    data = bw.load_campbell_scientific(bw.demo_datasets.demo_campbell_scientific_data)
    data_clnd2 = bw.apply_cleaning_windographer(data, bw.demo_datasets.demo_windographer_flagging_log2, dayfirst=True)
    data_clnd3 = bw.apply_cleaning(data, bw.demo_datasets.demo_cleaning_file)
    data_clnd4 = bw.apply_cleaning(data, os.path.join(DEMO_DATA_FOLDER, 'demo_cleaning_file2.csv'), dayfirst=True)
    data_clnd5 = bw.apply_cleaning(data, os.path.join(DEMO_DATA_FOLDER, 'demo_cleaning_file3.csv'), dayfirst=True)

    assert (data_clnd2.drop(['RECORD', 'Site', 'LoggerID'], axis=1).fillna(-999) ==
            data_clnd3.drop(['RECORD', 'Site', 'LoggerID'], axis=1).fillna(-999)).all().all()

    assert (data_clnd3.drop(['RECORD', 'Site', 'LoggerID'], axis=1).fillna(-999) ==
            data_clnd4.drop(['RECORD', 'Site', 'LoggerID'], axis=1).fillna(-999)).all().all()

    assert (data_clnd3.drop(['RECORD', 'Site', 'LoggerID'], axis=1).fillna(-999) ==
            data_clnd5.drop(['RECORD', 'Site', 'LoggerID'], axis=1).fillna(-999)).all().all()


def test_load_csv():
    data = bw.load_csv(os.path.join(DEMO_DATA_FOLDER, 'demo_data.csv'))
    data2 = bw.load_csv(os.path.join(DEMO_DATA_FOLDER, 'demo_data2.csv'), dayfirst=True)
    data3 = bw.load_csv(os.path.join(DEMO_DATA_FOLDER, 'demo_data3.csv'), dayfirst=True)
    data4 = bw.load_csv(os.path.join(DEMO_DATA_FOLDER, 'demo_data4.csv'), dayfirst=True)

    assert (data['2016-01-09 15:30:00':'2016-01-10 23:50:00'].fillna(-999) ==
            data2['2016-01-09 15:30:00':'2016-01-10 23:50:00'].fillna(-999)).all().all()
    assert (data['2016-01-09 15:30:00':'2016-01-10 23:50:00'].fillna(-999) ==
            data3['2016-01-09 15:30:00':'2016-01-10 23:50:00'].fillna(-999)).all().all()
    assert (data['2016-01-09 15:30:00':'2016-01-10 23:50:00'].fillna(-999) ==
            data4['2016-01-09 15:30:00':'2016-01-10 23:50:00'].fillna(-999)).all().all()


def test_load_windographer_txt():
    data = bw.load_windographer_txt(os.path.join(DEMO_DATA_FOLDER, 'windographer_demo_data.txt'))
    data1 = bw.load_windographer_txt(os.path.join(DEMO_DATA_FOLDER, 'windographer_demo_data1.txt'), dayfirst=True)
    data2 = bw.load_windographer_txt(os.path.join(DEMO_DATA_FOLDER, 'windographer_demo_data2.txt'), dayfirst=True)

    assert (data['2016-01-09 15:30:00':'2016-01-10 23:50:00'].fillna(-999) ==
            data1['2016-01-09 15:30:00':'2016-01-10 23:50:00'].fillna(-999)).all().all()
    assert (data['2016-01-09 15:30:00':'2016-01-10 23:50:00'].fillna(-999) ==
            data2['2016-01-09 15:30:00':'2016-01-10 23:50:00'].fillna(-999)).all().all()


def test_load_campbell_scientific():
    data = bw.load_campbell_scientific(os.path.join(DEMO_DATA_FOLDER, 'campbell_scientific_demo_data.csv'))
    data1 = bw.load_campbell_scientific(os.path.join(DEMO_DATA_FOLDER, 'campbell_scientific_demo_data1.csv'), dayfirst=True)

    assert (data['2016-01-09 15:30:00':'2016-01-10 23:50:00'].fillna(-999) ==
            data1['2016-01-09 15:30:00':'2016-01-10 23:50:00'].fillna(-999)).all().all()
