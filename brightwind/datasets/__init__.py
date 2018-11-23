"""
Example datasets that can be used with the library.

Usage:
import brightwind as bw

creyap_80m = bw.load_timeseries(bw.datasets.creyap_80m_csv)
creyap_50m = bw.load_timeseries(bw.datasets.creyap_50m_csv)
merra_west = bw.load_timeseries(bw.datasets.merra_west)

"""
__all__ = ['creyap_80m_csv', 'creyap_50m_csv', 'merra2_west']

import os

creyap_80m_csv = os.path.join(os.path.dirname(__file__), 'Shell_Flats_1_80mHAT.csv')
creyap_50m_csv = os.path.join(os.path.dirname(__file__), 'Shell_Flats_2_50mHAT.csv')
merra_west = os.path.join(os.path.dirname(__file__), 'MERRA_W03.332_N54.000.csv')
