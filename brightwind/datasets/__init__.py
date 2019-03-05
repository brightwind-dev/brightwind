
import os

__all__ = ['demo_site_data', 'demo_merra2_NW', 'demo_merra2_NE', 'demo_merra2_SE', 'demo_merra2_SW',
           'shell_flats_80m_csv', 'shell_flats_50m_csv', 'shell_flats_merra']

shell_flats_80m_csv = os.path.join(os.path.dirname(__file__), 'offshore-CREYAP-2-data-pack', 'Shell_Flats_1_80mHAT.csv')
shell_flats_50m_csv = os.path.join(os.path.dirname(__file__), 'offshore-CREYAP-2-data-pack', 'Shell_Flats_2_50mHAT.csv')
shell_flats_merra = os.path.join(os.path.dirname(__file__), 'offshore-CREYAP-2-data-pack', 'MERRA_W03.332_N54.000.csv')

demo_site_data = os.path.join(os.path.dirname(__file__), 'demo', 'campbell_scientific_demo_site_data_clean.csv')
demo_merra2_NW = os.path.join(os.path.dirname(__file__), 'demo', 'MERRA-2_NW_2000-01-01_2017-06-30.csv')
demo_merra2_NE = os.path.join(os.path.dirname(__file__), 'demo', 'MERRA-2_NE_2000-01-01_2017-06-30.csv')
demo_merra2_SE = os.path.join(os.path.dirname(__file__), 'demo', 'MERRA-2_SE_2000-01-01_2017-06-30.csv')
demo_merra2_SW = os.path.join(os.path.dirname(__file__), 'demo', 'MERRA-2_SW_2000-01-01_2017-06-30.csv')


def datasets_available():
    """
    Example datasets that can be used with the library.



    **Example usage**
    ::
        import brightwind as bw

        all_datasets_available = ['demo_site_data', 'demo_merra2_NW', 'demo_merra2_NE', 'demo_merra2_SE', 'demo_merra2_SW',
               'shell_flats_80m_csv', 'shell_flats_50m_csv', 'shell_flats_merra']
        shell_flats_80m_csv = bw.load_csv(bw.datasets.shell_flats_80m_csv)
        shell_flats_50m_csv = bw.load_csv(bw.datasets.shell_flats_50m_csv)
        shell_flats_merra = bw.load_csv(bw.datasets.shell_flats_merra)

    """

    return None
