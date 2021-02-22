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
           'iea43_wra_data_model_schema',
           'iea43_wra_data_model',
           'demo_wra_data_model']

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

iea43_wra_data_model_schema = os.path.join(os.path.dirname(__file__), 'iea43_wra_data_model.schema.json')
iea43_wra_data_model = os.path.join(os.path.dirname(__file__), 'iea43_wra_data_model.json')
demo_wra_data_model = os.path.join(os.path.dirname(__file__), 'demo_data_data_model.json')

demo_data_adjusted_for_testing = os.path.join(os.path.dirname(__file__), 'demo_data_adjusted.csv')


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
