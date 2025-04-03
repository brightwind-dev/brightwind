import os

__all__ = ['demo_data',
           'demo_campbell_scientific_data',
           'demo_windographer_data',
           'demo_merra2_NW',
           'demo_merra2_NE',
           'demo_merra2_SE',
           'demo_merra2_SW',
           'demo_windographer_flagging_log',
           'demo_windographer_flagging_log2',
           'demo_cleaning_file',
           'demo_wra_data_model',
           'iea43_wra_data_model_schema_v1_0',
           'iea43_wra_data_model_v1_0',
           'iea43_wra_data_model_schema_v1_2',
           'floating_lidar_iea43_wra_data_model_v1_2',
           'solar_iea43_wra_data_model_v1_3',
           'sodar_iea43_wra_data_model_v1_3',
           'demo_floating_lidar_data',
           'floating_lidar_demo_iea43_wra_data_model_v1_3'
           ]


demo_data = os.path.join(os.path.dirname(__file__), 'demo_data.csv')
demo_campbell_scientific_data = os.path.join(os.path.dirname(__file__), 'campbell_scientific_demo_data.csv')
demo_windographer_data = os.path.join(os.path.dirname(__file__), 'windographer_demo_data.txt')

demo_merra2_NW = os.path.join(os.path.dirname(__file__), 'MERRA-2_NW_2000-01-01_2017-06-30.csv')
demo_merra2_NE = os.path.join(os.path.dirname(__file__), 'MERRA-2_NE_2000-01-01_2017-06-30.csv')
demo_merra2_SE = os.path.join(os.path.dirname(__file__), 'MERRA-2_SE_2000-01-01_2017-06-30.csv')
demo_merra2_SW = os.path.join(os.path.dirname(__file__), 'MERRA-2_SW_2000-01-01_2017-06-30.csv')

demo_windographer_flagging_log = os.path.join(os.path.dirname(__file__), 'windographer_flagging_log.txt')
demo_windographer_flagging_log2 = os.path.join(os.path.dirname(__file__), 'windographer_flagging_log2.txt')
demo_cleaning_file = os.path.join(os.path.dirname(__file__), 'demo_cleaning_file.csv')

demo_wra_data_model = os.path.join(os.path.dirname(__file__), 'demo_data_iea43_wra_data_model.json')

iea43_wra_data_model_schema_v1_0 = os.path.join(os.path.dirname(__file__), 'iea43_wra_data_model.schema_v1_0.json')
iea43_wra_data_model_v1_0 = os.path.join(os.path.dirname(__file__), 'iea43_wra_data_model_v1_0.json')

iea43_wra_data_model_schema_v1_2 = os.path.join(os.path.dirname(__file__), 'iea43_wra_data_model.schema_v1_2.json')
floating_lidar_iea43_wra_data_model_v1_2 = os.path.join(os.path.dirname(__file__),
                                                        'floating_lidar_demo_iea43_wra_data_model_v1_2.json')
solar_iea43_wra_data_model_v1_3 = os.path.join(os.path.dirname(__file__),
                                                        'solar_iea43_wra_data_model_v1_3.json')
sodar_iea43_wra_data_model_v1_3 = os.path.join(os.path.dirname(__file__),
                                                        'sodar_iea43_wra_data_model_v1_3.json')

demo_data_adjusted_for_testing = os.path.join(os.path.dirname(__file__), 'demo_data_adjusted_for_testing.csv')
demo_floating_lidar_data = os.path.join(os.path.dirname(__file__),
                                                        'demo_floating_lidar_data.csv')
floating_lidar_demo_iea43_wra_data_model_v1_3 = os.path.join(os.path.dirname(__file__),
                                                        'floating_lidar_demo_iea43_wra_data_model_v1_3.json')


def datasets_available():
    """
    Example datasets that can be used with the library.


    **Example usage**
    ::
        import brightwind as bw

        all_datasets_available = ['demo_data', 'demo_campbell_scientific_data', 'demo_merra2_NW',
           'demo_merra2_NE', 'demo_merra2_SE', 'demo_merra2_SW', 'demo_windographer_data']
        demo_data = bw.load_campbell_scientific(bw.demo_datasets.demo_campbell_scientific_data)
        demo_data = bw.load_csv(bw.demo_datasets.demo_data)
        demo_windog_data = bw.load_windographer_txt(bw.demo_datasets.demo_windographer_data)

    """

    return None
