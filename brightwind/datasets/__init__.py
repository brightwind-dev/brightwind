"""
Example datasets that can be used with the library.

Usage:
import brightwind as bw

creyap_80m = bw.load_csv(bw.datasets.creyap_80m_csv)
creyap_50m = bw.load_csv(bw.datasets.creyap_50m_csv)
merra_west = bw.load_csv(bw.datasets.merra_west)

"""
import os

__all__ = ['shell_flats_80m_csv', 'shell_flats_50m_csv', 'shell_flats_merra']

shell_flats_80m_csv = os.path.join(os.path.dirname(__file__), 'offshore-CREYAP-2-data-pack', 'Shell_Flats_1_80mHAT.csv')
shell_flats_50m_csv = os.path.join(os.path.dirname(__file__), 'offshore-CREYAP-2-data-pack', 'Shell_Flats_2_50mHAT.csv')
shell_flats_merra = os.path.join(os.path.dirname(__file__), 'offshore-CREYAP-2-data-pack', 'MERRA_W03.332_N54.000.csv')
