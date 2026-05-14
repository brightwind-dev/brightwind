import pytest
import brightwind as bw
import os
import warnings
import pandas as pd
import numpy as np
import json
from io import BytesIO
from unittest.mock import patch, MagicMock

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

    cleaning_dict = {0: {'Sensor': 'All', 'Start': '2016-01-09 15:30:00', 'Stop': '2016-01-09 17:10:00',
                         'Reason': 'Installation'},
                     1: {'Sensor': 'Spd', 'Start': '2016-03-09 06:20:00', 'Stop': '2016-03-11',
                         'Reason': 'Icing'},
                     2: {'Sensor': 'Dir', 'Start': '2016-03-09 06:20:00', 'Stop': '2016-03-11',
                         'Reason': 'Icing'},
                     3: {'Sensor': 'Spd', 'Start': '2016-03-29', 'Stop': '2016-03-30 07:10:00',
                         'Reason': 'Icing'},
                     4: {'Sensor': 'Dir', 'Start': '2016-03-29 ', 'Stop': '2016-03-30 07:10:00',
                         'Reason': 'Icing'}}
    data_clnd6 = bw.apply_cleaning(data, pd.DataFrame(cleaning_dict).T)

    assert (data_clnd2.drop(['RECORD', 'Site', 'LoggerID'], axis=1).fillna(-999) ==
            data_clnd3.drop(['RECORD', 'Site', 'LoggerID'], axis=1).fillna(-999)).all().all()

    assert (data_clnd3.drop(['RECORD', 'Site', 'LoggerID'], axis=1).fillna(-999) ==
            data_clnd4.drop(['RECORD', 'Site', 'LoggerID'], axis=1).fillna(-999)).all().all()

    assert (data_clnd3.drop(['RECORD', 'Site', 'LoggerID'], axis=1).fillna(-999) ==
            data_clnd5.drop(['RECORD', 'Site', 'LoggerID'], axis=1).fillna(-999)).all().all()

    assert np.isnan(data_clnd6.Spd40mN['2016-03-09 06:20:00'])
    assert not np.isnan(data_clnd6.Spd40mN['2016-03-09 06:10:00'])
    assert np.isnan(data_clnd6.Spd40mN['2016-03-10 23:50:00'])
    assert not np.isnan(data_clnd6.Spd40mN['2016-03-11 00:00:00'])
    assert not np.isnan(data_clnd6.Spd40mN['2016-03-28 23:50'])
    assert np.isnan(data_clnd6.Spd40mN['2016-03-29 00:00'])


def test_apply_cleaning_rules():
    data = bw.load_csv(bw.demo_datasets.demo_data)
    data_clnd = bw.apply_cleaning_rules(data, bw.demo_datasets.demo_cleaning_rules_file)

    date_from = "2016-02-01T00:00"
    date_to = "2017-08-31T23:59"
    data_cleaned_test = data[data["T2m"] <= 5][date_from:date_to]


    assert data_clnd["Spd80mN"].min() >= 10
    assert data_clnd["T2m"][date_from:date_to].max() <= 5
    assert np.allclose(data_clnd["Spd60mS"][date_from:date_to].min(), data_cleaned_test["Spd60mS"].min())
    assert np.allclose(data_clnd["Spd80mS"][date_from:date_to].max(), data_cleaned_test["Spd80mS"].max())
    assert np.allclose(data_clnd["Spd60mS"][date_from:date_to].max(), data_cleaned_test["Spd60mS"].max())
    assert np.allclose(data_clnd["Spd80mSMax"][date_from:date_to].max(), data_cleaned_test["Spd80mSMax"].max())
    assert np.allclose(data_clnd["Spd60mSStd"][date_from:date_to].max(), data_cleaned_test["Spd60mSStd"].max())
    assert data_clnd[data["T2m"] > 5][date_from:date_to]["T2m"].isna().all()
    assert data_clnd[data["T2m"] > 5][date_from:date_to]["Spd60mS"].isna().all()
    assert data_clnd[data["T2m"] > 5][date_from:date_to]["Spd80mS"].isna().all()
    assert data_clnd[data["T2m"] > 5][date_from:date_to]["Spd80mSStd"].isna().all()
    assert data_clnd[data["T2m"] > 5][date_from:date_to]["Spd80mSMax"].isna().all()

    data_clnd = bw.apply_cleaning_rules(data, bw.demo_datasets.demo_cleaning_rules_file, replacement_text="-")
    assert (data_clnd[data["T2m"] > 5][date_from:date_to]["T2m"] == "-").all()
    assert (data_clnd[data["T2m"] > 5][date_from:date_to]["Spd60mS"] == "-").all()
    assert (data_clnd[data["T2m"] > 5][date_from:date_to]["Spd80mS"] == "-").all()
    before_range = data_clnd[data["T2m"] > 5][:"2016-01-31T23:59"]
    after_range = data_clnd[data["T2m"] > 5]["2017-09-01T00:00":]
    assert (before_range["T2m"] != "-").all()
    assert (after_range["T2m"] != "-").all()
    assert (before_range["Spd60mS"] != "-").all()
    assert (after_range["Spd60mS"] != "-").all()
    assert (before_range["Spd80mS"] != "-").all()
    assert (after_range["Spd80mS"] != "-").all()


    with open(bw.demo_datasets.demo_cleaning_rules_file) as file:
        cleaning_json = json.load(file)

    cleaning_json[0]['rule']['date_to'] = "2016-02-01T00:00:00"

    data_clnd = bw.apply_cleaning_rules(data, cleaning_json)
    assert data_clnd["Spd80mN"][:date_from].min() >= 10
    assert data_clnd["Spd80mN"][date_from:].min() < 10
    assert data_clnd[data["T2m"] > 5]["Spd80mN"][date_from:date_to].isna().all()
    assert data_clnd[data["T2m"] > 5]["Spd60mN"][date_from:date_to] .isna().all()

    del cleaning_json[0]['rule']
    with pytest.raises(ValueError) as except_info:
        bw.apply_cleaning_rules(data, cleaning_json)
    assert "There is a problem with the validity of the supplied JSON please check the errors above" in str(except_info.value)

    data_original = data.copy(deep=True)
    bw.apply_cleaning_rules(data, bw.demo_datasets.demo_cleaning_rules_file, replacement_text="-", inplace=True)
    assert (data[data_original["T2m"] > 5][date_from:date_to]["T2m"] == "-").all()
    assert (data[data_original["T2m"] > 5][date_from:date_to]["Spd60mS"] == "-").all()
    assert (data[data_original["T2m"] > 5][date_from:date_to]["Spd80mS"] == "-").all()


def _synthetic_cleaning_df():
    """Deterministic synthetic DataFrame for nested-condition / time-range tests."""
    index = pd.date_range('2016-01-01', '2016-03-31 23:50', freq='10min')
    n = len(index)
    return pd.DataFrame(
        {
            'Spd': np.linspace(0, 20, n),
            'Dir': np.linspace(0, 359, n),
            'T2m': np.linspace(-10, 30, n),
        },
        index=index,
    )


def test_apply_cleaning_rules_nested_and_time_range():
    df = _synthetic_cleaning_df()

    # 1. Flat condition regression — equivalent to old single-condition behaviour.
    rules = [{'rule': {
        'clean_out': [{'assembled_column_name': 'Spd'}],
        'conditions': {'assembled_column_name': 'Spd', 'comparison_operator_id': 1, 'comparator_value': 10},
    }}]
    cleaned = bw.apply_cleaning_rules(df, rules)
    expected_mask = df['Spd'] < 10
    assert cleaned.loc[expected_mask, 'Spd'].isna().all()
    assert cleaned.loc[~expected_mask, 'Spd'].notna().all()

    # 2. AND of two single comparisons — both must be true.
    rules = [{'rule': {
        'clean_out': [{'assembled_column_name': 'Spd'}],
        'conditions': {'and': [
            {'assembled_column_name': 'Spd', 'comparison_operator_id': 1, 'comparator_value': 10},
            {'assembled_column_name': 'T2m', 'comparison_operator_id': 1, 'comparator_value': 5},
        ]},
    }}]
    cleaned = bw.apply_cleaning_rules(df, rules)
    expected_mask = (df['Spd'] < 10) & (df['T2m'] < 5)
    assert cleaned.loc[expected_mask, 'Spd'].isna().all()
    assert cleaned.loc[~expected_mask, 'Spd'].notna().all()

    # 3. OR of two single comparisons — either may be true.
    rules = [{'rule': {
        'clean_out': [{'assembled_column_name': 'Spd'}],
        'conditions': {'or': [
            {'assembled_column_name': 'Spd', 'comparison_operator_id': 1, 'comparator_value': 5},
            {'assembled_column_name': 'T2m', 'comparison_operator_id': 3, 'comparator_value': 20},
        ]},
    }}]
    cleaned = bw.apply_cleaning_rules(df, rules)
    expected_mask = (df['Spd'] < 5) | (df['T2m'] > 20)
    assert cleaned.loc[expected_mask, 'Spd'].isna().all()
    assert cleaned.loc[~expected_mask, 'Spd'].notna().all()

    # 4. NOT — negation of a single comparison.
    rules = [{'rule': {
        'clean_out': [{'assembled_column_name': 'Spd'}],
        'conditions': {'not': {'assembled_column_name': 'Spd', 'comparison_operator_id': 1, 'comparator_value': 10}},
    }}]
    cleaned = bw.apply_cleaning_rules(df, rules)
    expected_mask = ~(df['Spd'] < 10)
    assert cleaned.loc[expected_mask, 'Spd'].isna().all()
    assert cleaned.loc[~expected_mask, 'Spd'].notna().all()

    # 5. Nested AND containing OR.
    rules = [{'rule': {
        'clean_out': [{'assembled_column_name': 'Spd'}],
        'conditions': {'and': [
            {'assembled_column_name': 'Dir', 'comparison_operator_id': 4, 'comparator_value': 180},
            {'or': [
                {'assembled_column_name': 'Spd', 'comparison_operator_id': 1, 'comparator_value': 3},
                {'assembled_column_name': 'Spd', 'comparison_operator_id': 3, 'comparator_value': 15},
            ]},
        ]},
    }}]
    cleaned = bw.apply_cleaning_rules(df, rules)
    expected_mask = (df['Dir'] >= 180) & ((df['Spd'] < 3) | (df['Spd'] > 15))
    assert cleaned.loc[expected_mask, 'Spd'].isna().all()
    assert cleaned.loc[~expected_mask, 'Spd'].notna().all()

    # 6. time_range_conditions alone (combined with an always-true conditions block).
    rules = [{'rule': {
        'clean_out': [{'assembled_column_name': 'Spd'}],
        'conditions': {'assembled_column_name': 'Spd', 'comparison_operator_id': 4, 'comparator_value': -1},
        'time_range_conditions': {'and': [
            {'value': '2016-02-01T00:00:00', 'comparison_operator_id': 4},
            {'value': '2016-03-01T00:00:00', 'comparison_operator_id': 1},
        ]},
    }}]
    cleaned = bw.apply_cleaning_rules(df, rules)
    expected_mask = (df.index >= pd.Timestamp('2016-02-01')) & (df.index < pd.Timestamp('2016-03-01'))
    assert cleaned.loc[expected_mask, 'Spd'].isna().all()
    assert cleaned.loc[~expected_mask, 'Spd'].notna().all()

    # 7. time_range_conditions ANDed with a measurement condition.
    rules = [{'rule': {
        'clean_out': [{'assembled_column_name': 'Spd'}],
        'conditions': {'assembled_column_name': 'Spd', 'comparison_operator_id': 1, 'comparator_value': 10},
        'time_range_conditions': {'and': [
            {'value': '2016-02-01T00:00:00', 'comparison_operator_id': 4},
            {'value': '2016-03-01T00:00:00', 'comparison_operator_id': 1},
        ]},
    }}]
    cleaned = bw.apply_cleaning_rules(df, rules)
    in_window = (df.index >= pd.Timestamp('2016-02-01')) & (df.index < pd.Timestamp('2016-03-01'))
    expected_mask = (df['Spd'] < 10) & in_window
    assert cleaned.loc[expected_mask, 'Spd'].isna().all()
    assert cleaned.loc[~expected_mask, 'Spd'].notna().all()

    # 8. time_range_conditions combined with date_from/date_to — all three ANDed.
    rules = [{'rule': {
        'clean_out': [{'assembled_column_name': 'Spd'}],
        'conditions': {'assembled_column_name': 'Spd', 'comparison_operator_id': 4, 'comparator_value': -1},
        'date_from': '2016-01-15T00:00:00',
        'date_to': '2016-03-15T00:00:00',
        'time_range_conditions': {'and': [
            {'value': '2016-02-01T00:00:00', 'comparison_operator_id': 4},
            {'value': '2016-04-01T00:00:00', 'comparison_operator_id': 1},
        ]},
    }}]
    cleaned = bw.apply_cleaning_rules(df, rules)
    expected_mask = (
        (df.index >= pd.Timestamp('2016-01-15')) & (df.index < pd.Timestamp('2016-03-15')) &
        (df.index >= pd.Timestamp('2016-02-01')) & (df.index < pd.Timestamp('2016-04-01'))
    )
    assert cleaned.loc[expected_mask, 'Spd'].isna().all()
    assert cleaned.loc[~expected_mask, 'Spd'].notna().all()

    # 9. Validation error — single comparison missing comparator_value.
    rules = [{'rule': {
        'clean_out': [{'assembled_column_name': 'Spd'}],
        'conditions': {'assembled_column_name': 'Spd', 'comparison_operator_id': 1},
    }}]
    with pytest.raises(ValueError) as except_info:
        bw.apply_cleaning_rules(df, rules)
    assert "validity of the supplied JSON" in str(except_info.value)

    # 10. Stat-type expansion — clean_out on the avg column also cleans all stat variants present.
    index = pd.date_range('2016-01-01', '2016-01-02 23:50', freq='10min')
    n = len(index)
    df_stats = pd.DataFrame(
        {
            'Spd_40m': np.linspace(0, 20, n),
            'Spd_40m_sd': np.linspace(0, 2, n),
            'Spd_40m_max': np.linspace(0, 25, n),
            'Spd_40m_min': np.linspace(0, 15, n),
            'OtherCol': np.linspace(0, 1, n),
        },
        index=index,
    )
    rules = [{'rule': {
        'clean_out': [{'assembled_column_name': 'Spd_40m'}],
        'conditions': {'assembled_column_name': 'Spd_40m', 'comparison_operator_id': 1, 'comparator_value': 10},
    }}]
    cleaned = bw.apply_cleaning_rules(df_stats, rules)
    mask = df_stats['Spd_40m'] < 10
    for col in ('Spd_40m', 'Spd_40m_sd', 'Spd_40m_max', 'Spd_40m_min'):
        assert cleaned.loc[mask, col].isna().all()
        assert cleaned.loc[~mask, col].notna().all()
    assert cleaned['OtherCol'].notna().all()

    # New fields measurement_point_uuid and statistic_type_id are accepted but ignored.
    rules = [{'rule': {
        'clean_out': [{
            'assembled_column_name': 'Spd_40m',
            'measurement_point_uuid': '00000000-0000-0000-0000-000000000000',
            'statistic_type_id': 'avg',
        }],
        'conditions': {
            'assembled_column_name': 'Spd_40m',
            'comparison_operator_id': 1,
            'comparator_value': 10,
            'measurement_point_uuid': '00000000-0000-0000-0000-000000000000',
            'statistic_type_id': 'avg',
        },
    }}]
    cleaned = bw.apply_cleaning_rules(df_stats, rules)
    assert cleaned.loc[mask, 'Spd_40m'].isna().all()


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
    
    # test loading files from folder
    bw.export_csv(data[:'2016-01-09 17:00'], os.path.join(DEMO_DATA_FOLDER, 'temp_test_data_first_chunk.csv'))
    bw.export_csv(data['2016-01-09 17:10':'2016-01-09 18:00'], os.path.join(DEMO_DATA_FOLDER, 'temp_test_data_second_chunk.csv'))
    with patch('brightwind.load.load._list_files', return_value=[
        os.path.join(DEMO_DATA_FOLDER, 'temp_test_data_first_chunk.csv'),
        os.path.join(DEMO_DATA_FOLDER, 'temp_test_data_second_chunk.csv')]):
        assert isinstance(bw.load_csv(DEMO_DATA_FOLDER, '.csv'), pd.DataFrame)
    # Remove the temp files
    if os.path.exists(os.path.join(DEMO_DATA_FOLDER, 'temp_test_data_first_chunk.csv')):
        os.remove(os.path.join(DEMO_DATA_FOLDER, 'temp_test_data_first_chunk.csv'))
    if os.path.exists(os.path.join(DEMO_DATA_FOLDER, 'temp_test_data_second_chunk.csv')):
        os.remove(os.path.join(DEMO_DATA_FOLDER, 'temp_test_data_second_chunk.csv'))


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


@pytest.mark.skipif(
    not (
        (os.environ.get('BRIGHTHUB_CLIENT_ID') and os.environ.get('BRIGHTHUB_CLIENT_SECRET'))
        or (os.environ.get('BRIGHTHUB_EMAIL') and os.environ.get('BRIGHTHUB_PASSWORD'))
    ),
    reason="Either BRIGHTHUB_CLIENT_ID and BRIGHTHUB_CLIENT_SECRET, or BRIGHTHUB_EMAIL and BRIGHTHUB_PASSWORD "
           "must be set to run this integration test against the live BrightHub API."
)
def test_load_brighthub():

    plant_uuid = '7a58497e-bee1-42a2-8084-c47a5cf213b7'
    measurement_station_uuid = '9344e576-6d5a-45f0-9750-2a7528ebfa14'
    test_period_demo_data = {'start_date': '2016-01-09T15:30:00', 'end_date': '2017-11-23T10:50:00'}

    # To get a specific plant
    # assert bw.LoadBrightHub.get_plants(plant_uuid=plant_uuid)[
    #            'plant_type_id'].values[0] == 'onshore_wind'

    measurement_stations2 = bw.LoadBrightHub.get_measurement_stations()
    device_types = measurement_stations2['measurement_station_type'].values

    measurement_stations_lidar = bw.LoadBrightHub.get_measurement_stations(measurement_station_type='lidar')
    device_types_lidar = measurement_stations_lidar['measurement_station_type'].unique()

    assert 'lidar' in device_types
    assert 'mast' in device_types
    assert ['lidar'] == device_types_lidar

    measurement_stations_lidar_mast = bw.LoadBrightHub.get_measurement_stations(
        measurement_station_type=['lidar', 'mast'])
    device_types_lidar_mast = measurement_stations_lidar_mast['measurement_station_type'].unique()

    assert ['lidar', 'mast'] == sorted(device_types_lidar_mast)
    # To get a specific measurement station
    measurement_stations = bw.LoadBrightHub.get_measurement_stations(measurement_station_uuid=measurement_station_uuid)
    measurement_stations_json = bw.LoadBrightHub.get_measurement_stations(
        measurement_station_uuid=measurement_station_uuid, return_df=False
        )

    assert type(measurement_stations) == pd.DataFrame
    assert type(measurement_stations_json) == list
    assert type(measurement_stations_json[0]) == dict
    assert measurement_stations_json[0]["uuid"] == measurement_station_uuid

    # # Doesn't work anymore as more than 1 station is returned now.
    # measurement_stations2 = bw.LoadBrightHub.get_measurement_stations(plant_uuid=plant_uuid)
    # assert (measurement_stations2.dropna(axis=1) == measurement_stations.dropna(axis=1)).all().all() 
    # measurement_stations = bw.LoadBrightHub.get_measurement_stations(measurement_station_uuid=measurement_station_uuid)
    # assert (.dropna(    # Doesn't work anymore as more than 1 station is returned now.
    #     axis=1) == measurement_stations.dropna(axis=1)).all().all()

    # To get the data model for a specific measurement station
    assert bw.LoadBrightHub.get_data_model(measurement_station_uuid=measurement_station_uuid
                                           )['author'] == 'Brighthub'

    # To get start and end date of data for a specific measurement station
    assert bw.LoadBrightHub.get_start_end_dates(measurement_station_uuid=measurement_station_uuid
                                                ) == test_period_demo_data

    # To get data for a specific time period for a specific measurement station
    data_csv = bw.LoadBrightHub.get_data(measurement_station_uuid=measurement_station_uuid, date_from='2016-12-01',
                                         date_to='2017-01-01', file_extension='.csv')
    for col in ['Spd80mN', 'Spd80mS', 'Dir78mS']:
        assert col in data_csv.columns

    # Same window as parquet — confirms the API accepts file_extension='.parquet' and returns equivalent data.
    data_parquet = bw.LoadBrightHub.get_data(measurement_station_uuid=measurement_station_uuid, date_from='2016-12-01',
                                             date_to='2017-01-01', file_extension='.parquet')
    assert list(data_parquet.columns) == list(data_csv.columns)
    assert len(data_parquet) == len(data_csv)
    assert data_parquet.index.name == 'Timestamp'

    # To get cleaning log
    cleaning_log_df = bw.LoadBrightHub.get_cleaning_log(measurement_station_uuid=measurement_station_uuid)
    assert len(cleaning_log_df) != 0

    # To get cleaning rules
    cleaning_rules_json = bw.LoadBrightHub.get_cleaning_rules(measurement_station_uuid=measurement_station_uuid)
    assert cleaning_rules_json[0]['measurement_location_uuid'] == measurement_station_uuid


def test_load_brighthub_get_data_default_emits_deprecation_warning():
    api_response = MagicMock(status_code=200)
    api_response.json.return_value = {'url': 'https://example.com/presigned'}
    csv_response = MagicMock(text='Timestamp,Spd80mN\n2016-06-01 00:00:00,7.1\n')

    with patch('brightwind.load.load.LoadBrightHub._brighthub_request', return_value=api_response) as mock_request, \
            patch('brightwind.load.load.requests.get', return_value=csv_response):
        with pytest.warns(DeprecationWarning, match="file_extension"):
            df = bw.LoadBrightHub.get_data(measurement_station_uuid='uuid-1')

    assert mock_request.call_args.kwargs['params']['file_extension'] == '.csv'
    assert df.index.name == 'Timestamp'


def test_load_brighthub_get_data_csv_explicit_no_warning():
    api_response = MagicMock(status_code=200)
    api_response.json.return_value = {'url': 'https://example.com/presigned'}
    csv_response = MagicMock(text='Timestamp,Spd80mN\n2016-06-01 00:00:00,7.1\n')

    with patch('brightwind.load.load.LoadBrightHub._brighthub_request', return_value=api_response) as mock_request, \
            patch('brightwind.load.load.requests.get', return_value=csv_response):
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            df = bw.LoadBrightHub.get_data(measurement_station_uuid='uuid-1', file_extension='.csv')

    assert mock_request.call_args.kwargs['params']['file_extension'] == '.csv'
    assert isinstance(df.index, pd.DatetimeIndex)


def test_load_brighthub_get_data_parquet():
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        pytest.importorskip('fastparquet', reason="parquet engine (pyarrow or fastparquet) required to round-trip parquet bytes")

    index = pd.to_datetime(['2016-06-01 00:00:00', '2016-06-01 00:10:00'])
    index.name = 'Timestamp'
    sample = pd.DataFrame({'Spd80mN': [7.1, 7.3]}, index=index)
    buffer = BytesIO()
    sample.to_parquet(buffer, compression='snappy')

    api_response = MagicMock(status_code=200)
    api_response.json.return_value = {'url': 'https://example.com/presigned'}
    parquet_response = MagicMock(content=buffer.getvalue())

    with patch('brightwind.load.load.LoadBrightHub._brighthub_request', return_value=api_response) as mock_request, \
            patch('brightwind.load.load.requests.get', return_value=parquet_response):
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            df = bw.LoadBrightHub.get_data(measurement_station_uuid='uuid-1', file_extension='.parquet')

    assert mock_request.call_args.kwargs['params']['file_extension'] == '.parquet'
    assert df.index.name == 'Timestamp'
    assert isinstance(df.index, pd.DatetimeIndex)
    assert np.allclose(df['Spd80mN'].values, sample['Spd80mN'].values)


def test_load_brighthub_get_data_invalid_extension():
    with patch('brightwind.load.load.LoadBrightHub._brighthub_request') as mock_request, \
            patch('brightwind.load.load.requests.get') as mock_get:
        with pytest.raises(ValueError, match="Unsupported file_extension"):
            bw.LoadBrightHub.get_data(measurement_station_uuid='uuid-1', file_extension='.xlsx')

    mock_request.assert_not_called()
    mock_get.assert_not_called()


def test_load_brighthub_get_data_parquet_missing_engine_raises_importerror():
    api_response = MagicMock(status_code=200)
    api_response.json.return_value = {'url': 'https://example.com/presigned'}
    parquet_response = MagicMock(content=b'irrelevant')

    with patch('brightwind.load.load.LoadBrightHub._brighthub_request', return_value=api_response), \
            patch('brightwind.load.load.requests.get', return_value=parquet_response), \
            patch('brightwind.load.load.pd.read_parquet', side_effect=ImportError("no engine")):
        with pytest.raises(ImportError, match=r"brightwind\[parquet\].*brightwind\[parquet-fastparquet\]"):
            bw.LoadBrightHub.get_data(measurement_station_uuid='uuid-1', file_extension='.parquet')
