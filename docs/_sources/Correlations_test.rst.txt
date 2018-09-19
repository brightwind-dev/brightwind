
Castlecraig
~~~~~~~~~~~

.. code:: ipython3

    import ipywidgets as widgets
    import pandas as pd
    import sys
    path = r'C:\BW\RTD\repos-hadley\brightwind'
    if path not in sys.path:
        sys.path.insert(1, path)
    from analyse import correlation as bw
    from load import reanalysis
    from plot import plot
    from transform import transform
    
    import importlib
    importlib.reload(bw)
    importlib.reload(reanalysis)
    importlib.reload(plot)
    importlib.reload(transform)




.. parsed-literal::

    <module 'transform.transform' from 'C:\\BW\\RTD\\repos-hadley\\brightwind\\transform\\transform.py'>



.. code:: ipython3

    merra2_dfs_1996 = reanalysis.get_merra2_nearest_nodes('55.5','-8.5', from_date='1996-01-01', to_date='2016-12-31')


.. parsed-literal::

    2018-09-07 09:26:44.785952 - Run start for get_merra_2_dataframe.
    2018-09-07 09:26:44.785952 - http://52.16.60.214:3306/merra/55.5/-8.75/1996-01-01/2016-12-31
    2018-09-07 09:26:55.976074 - Finished
    2018-09-07 09:26:55.976074 - Run start for get_merra_2_dataframe.
    2018-09-07 09:26:55.976074 - http://52.16.60.214:3306/merra/55.5/-8.125/1996-01-01/2016-12-31
    2018-09-07 09:27:06.378920 - Finished
    2018-09-07 09:27:06.382129 - Run start for get_merra_2_dataframe.
    2018-09-07 09:27:06.382129 - http://52.16.60.214:3306/merra/55.0/-8.125/1996-01-01/2016-12-31
    2018-09-07 09:27:16.631337 - Finished
    2018-09-07 09:27:16.646959 - Run start for get_merra_2_dataframe.
    2018-09-07 09:27:16.646959 - http://52.16.60.214:3306/merra/55.0/-8.75/1996-01-01/2016-12-31
    2018-09-07 09:27:27.024563 - Finished
    

.. code:: ipython3

    import pickle
    with open ('merra2_dfs_castlecraig_1996.pkl','wb') as file:
        pickle.dump(merra2_dfs_1996,file)
        

.. code:: ipython3

    merra2_dfs[0].data.to_csv('Castlecraig_MERRA2_correlations_test.csv')

.. code:: ipython3

    import pickle
    merra2_dfs = pickle.load(open('merra2_dfs_castlecraig_1996.pkl','rb'))

.. code:: ipython3

    site_file_path = r'C:\Dropbox (brightwind)\RTD\BrightData\MERA\2018-05-25_MERA-LT-reference-quality\site-data\M352_Calib_Cleaned_20jan2011onwards.csv'
    
    if site_file_path.endswith('.csv'):
        site_data = pd.read_csv(site_file_path)
    else:
        site_data = pd.read_excel(site_file_path)
    index_wdgt = widgets.Dropdown(
        options=site_data.columns,
        description='Index column:',style = {'description_width': 'initial'},
        disabled=False
    )
    print("Choose index column:")
    index_wdgt
    
    
    


.. parsed-literal::

    Choose index column:
    


.. parsed-literal::

    A Jupyter Widget


.. code:: ipython3

    
    site_data = site_data.set_index(pd.DatetimeIndex(site_data[index_wdgt.value]))
    

.. code:: ipython3

    target_wdgt = widgets.Dropdown(options=site_data.columns, description='Target:', style = {'description_width': 'initial'},disabled=False)
    ref_merra2_wdgt = widgets.Dropdown(options=merra2_dfs[0].data.columns, description='MERRA2:',style = {'description_width': 'initial'}, disabled=False)
    items = [target_wdgt, ref_merra2_wdgt]
    box_layout = widgets.Layout(display='flex', flex_flow='row', align_items='stretch')
    box = widgets.Box(children=items, layout=box_layout)
    box
    



.. parsed-literal::

    A Jupyter Widget


.. code:: ipython3

    merra2_dfs[1].data.resample('10D', axis=0, closed='left', label='left',base=0,
                                    convention='start', kind='timestamp').count().divide(transform._max_coverage_count(merra2_dfs[1].data.index, merra2_dfs[1].data.resample('10D', axis=0, closed='left', label='left',base=0,
                                    convention='start', kind='timestamp').mean().index), axis=0)




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>PS_hPa</th>
          <th>T2M_degC</th>
          <th>WD50m_deg</th>
          <th>WS50m_ms</th>
        </tr>
        <tr>
          <th>DateTime</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1996-01-01</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-01-11</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-01-21</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-01-31</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-02-10</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-02-20</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-03-01</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-03-11</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-03-21</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-03-31</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-04-10</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-04-20</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-04-30</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-05-10</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-05-20</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-05-30</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-06-09</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-06-19</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-06-29</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-07-09</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-07-19</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-07-29</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-08-08</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-08-18</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-08-28</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-09-07</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-09-17</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-09-27</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-10-07</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-10-17</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>2016-03-06</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-03-16</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-03-26</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-04-05</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-04-15</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-04-25</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-05-05</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-05-15</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-05-25</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-06-04</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-06-14</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-06-24</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-07-04</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-07-14</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-07-24</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-08-03</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-08-13</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-08-23</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-09-02</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-09-12</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-09-22</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-10-02</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-10-12</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-10-22</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-11-01</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-11-11</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-11-21</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-12-01</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-12-11</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-12-21</th>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
        </tr>
      </tbody>
    </table>
    <p>767 rows × 4 columns</p>
    </div>



.. code:: ipython3

    transform._max_coverage_count(merra2_dfs[1].data.index, merra2_dfs[1].data.resample('10D', axis=0, closed='left', label='left',base=0,
                                    convention='start', kind='timestamp').mean().index)




.. parsed-literal::

    DatetimeIndex(['1996-01-01', '1996-01-11', '1996-01-21', '1996-01-31',
                   '1996-02-10', '1996-02-20', '1996-03-01', '1996-03-11',
                   '1996-03-21', '1996-03-31',
                   ...
                   '2016-09-22', '2016-10-02', '2016-10-12', '2016-10-22',
                   '2016-11-01', '2016-11-11', '2016-11-21', '2016-12-01',
                   '2016-12-11', '2016-12-21'],
                  dtype='datetime64[ns]', name='DateTime', length=767, freq='10D')



.. code:: ipython3

    transform.get_coverage(merra2_dfs[1].data)




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>1996-01-01 00:00:00</th>
          <th>1996-02-01 00:00:00</th>
          <th>1996-03-01 00:00:00</th>
          <th>1996-04-01 00:00:00</th>
          <th>1996-05-01 00:00:00</th>
          <th>1996-06-01 00:00:00</th>
          <th>1996-07-01 00:00:00</th>
          <th>1996-08-01 00:00:00</th>
          <th>1996-09-01 00:00:00</th>
          <th>1996-10-01 00:00:00</th>
          <th>...</th>
          <th>2016-07-01 00:00:00</th>
          <th>2016-08-01 00:00:00</th>
          <th>2016-09-01 00:00:00</th>
          <th>2016-10-01 00:00:00</th>
          <th>2016-11-01 00:00:00</th>
          <th>2016-12-01 00:00:00</th>
          <th>PS_hPa</th>
          <th>T2M_degC</th>
          <th>WD50m_deg</th>
          <th>WS50m_ms</th>
        </tr>
        <tr>
          <th>DateTime</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1996-01-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1996-02-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1996-03-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1996-04-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1996-05-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1996-06-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1996-07-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1996-08-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1996-09-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1996-10-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1996-11-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1996-12-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1997-01-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1997-02-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1997-03-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1997-04-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1997-05-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1997-06-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1997-07-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1997-08-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1997-09-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1997-10-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1997-11-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1997-12-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1998-01-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1998-02-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1998-03-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1998-04-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1998-05-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1998-06-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>2014-07-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2014-08-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2014-09-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2014-10-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2014-11-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2014-12-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2015-01-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2015-02-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2015-03-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2015-04-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2015-05-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2015-06-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2015-07-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2015-08-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2015-09-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2015-10-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2015-11-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2015-12-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2016-01-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2016-02-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2016-03-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2016-04-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2016-05-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2016-06-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2016-07-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2016-08-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2016-09-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2016-10-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2016-11-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2016-12-01</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    <p>252 rows × 256 columns</p>
    </div>



.. code:: ipython3

    transform._get_data_resolution(merra2_dfs[1].data.index) < transform._get_data_resolution(site_data.index)




.. parsed-literal::

    False



.. code:: ipython3

    transform.average_data_by_period(merra2_dfs[1].data[ref_merra2_wdgt.value], period='10D', filter=True, return_coverage=True)[1]
    




.. parsed-literal::

    DateTime
    1996-01-01    1.0
    1996-01-11    1.0
    1996-01-21    1.0
    1996-01-31    1.0
    1996-02-10    1.0
    1996-02-20    1.0
    1996-03-01    1.0
    1996-03-11    1.0
    1996-03-21    1.0
    1996-03-31    1.0
    1996-04-10    1.0
    1996-04-20    1.0
    1996-04-30    1.0
    1996-05-10    1.0
    1996-05-20    1.0
    1996-05-30    1.0
    1996-06-09    1.0
    1996-06-19    1.0
    1996-06-29    1.0
    1996-07-09    1.0
    1996-07-19    1.0
    1996-07-29    1.0
    1996-08-08    1.0
    1996-08-18    1.0
    1996-08-28    1.0
    1996-09-07    1.0
    1996-09-17    1.0
    1996-09-27    1.0
    1996-10-07    1.0
    1996-10-17    1.0
                 ... 
    2016-03-06    1.0
    2016-03-16    1.0
    2016-03-26    1.0
    2016-04-05    1.0
    2016-04-15    1.0
    2016-04-25    1.0
    2016-05-05    1.0
    2016-05-15    1.0
    2016-05-25    1.0
    2016-06-04    1.0
    2016-06-14    1.0
    2016-06-24    1.0
    2016-07-04    1.0
    2016-07-14    1.0
    2016-07-24    1.0
    2016-08-03    1.0
    2016-08-13    1.0
    2016-08-23    1.0
    2016-09-02    1.0
    2016-09-12    1.0
    2016-09-22    1.0
    2016-10-02    1.0
    2016-10-12    1.0
    2016-10-22    1.0
    2016-11-01    1.0
    2016-11-11    1.0
    2016-11-21    1.0
    2016-12-01    1.0
    2016-12-11    1.0
    2016-12-21    1.0
    Freq: 240H, Length: 767, dtype: float64



.. code:: ipython3

    from sklearn.linear_model import LinearRegression
    combined = LinearRegression()
    # combined_data = pd.concat([valentia['wdsp'][overlap_idx],merra2['WS50m_m/s'][overlap_idx]], axis=1)
    # combined.fit(combined_data, hourly_site_data['WS5_ms_Avg'][overlap_idx].values.reshape(-1,1))
    
    x, y = bw._preprocess_data_for_correlations(pd.concat([merra2_dfs[i].data[ref_merra2_wdgt.value] for i in range(0,4)],axis=1,join='inner'), 
                     site_data[target_wdgt.value], averaging_prd='1H', coverage_threshold=0.8)
    combined.fit(mlr.data.iloc[:,:len(mlr.data.columns)-1], mlr.data.iloc[:,-1])
    combined.coef_, combined.intercept_




.. parsed-literal::

    (array([-0.0813603 , -0.17854418,  1.31509098, -0.23565439]),
     1.3117160988494376)



.. code:: ipython3

    mlr = bw.MultipleLinearRegression(ref=[merra2_dfs[i].data[ref_merra2_wdgt.value] for i in range(0,4)],target=site_data[target_wdgt.value], averaging_prd='1H', 
                                      coverage_threshold=0.8)

.. code:: ipython3

    mlr.run()

.. code:: ipython3

    mlr.params




.. parsed-literal::

    {'offset': 1.3117160988494521,
     'slope': array([-0.0813603 , -0.17854418,  1.31509098, -0.23565439])}



.. code:: ipython3

    mlr.data




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>ref_spd_1</th>
          <th>ref_spd_2</th>
          <th>ref_spd_3</th>
          <th>ref_spd_4</th>
          <th>target_spd</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2011-01-20 09:00:00</th>
          <td>5.883</td>
          <td>5.506</td>
          <td>4.386</td>
          <td>5.090</td>
          <td>5.368333</td>
        </tr>
        <tr>
          <th>2011-01-20 10:00:00</th>
          <td>5.956</td>
          <td>5.502</td>
          <td>4.276</td>
          <td>5.104</td>
          <td>5.158333</td>
        </tr>
        <tr>
          <th>2011-01-20 11:00:00</th>
          <td>6.246</td>
          <td>5.631</td>
          <td>4.297</td>
          <td>5.285</td>
          <td>5.115000</td>
        </tr>
        <tr>
          <th>2011-01-20 12:00:00</th>
          <td>6.612</td>
          <td>5.865</td>
          <td>4.424</td>
          <td>5.554</td>
          <td>5.998333</td>
        </tr>
        <tr>
          <th>2011-01-20 13:00:00</th>
          <td>6.872</td>
          <td>6.060</td>
          <td>4.550</td>
          <td>5.757</td>
          <td>5.250000</td>
        </tr>
        <tr>
          <th>2011-01-20 14:00:00</th>
          <td>6.927</td>
          <td>6.099</td>
          <td>4.537</td>
          <td>5.763</td>
          <td>3.505000</td>
        </tr>
        <tr>
          <th>2011-01-20 15:00:00</th>
          <td>6.873</td>
          <td>6.068</td>
          <td>4.416</td>
          <td>5.613</td>
          <td>3.178333</td>
        </tr>
        <tr>
          <th>2011-01-20 16:00:00</th>
          <td>6.765</td>
          <td>6.028</td>
          <td>4.358</td>
          <td>5.497</td>
          <td>4.206667</td>
        </tr>
        <tr>
          <th>2011-01-20 17:00:00</th>
          <td>6.672</td>
          <td>5.977</td>
          <td>4.439</td>
          <td>5.520</td>
          <td>4.980000</td>
        </tr>
        <tr>
          <th>2011-01-20 18:00:00</th>
          <td>6.550</td>
          <td>5.863</td>
          <td>4.492</td>
          <td>5.513</td>
          <td>4.128333</td>
        </tr>
        <tr>
          <th>2011-01-20 19:00:00</th>
          <td>6.398</td>
          <td>5.718</td>
          <td>4.468</td>
          <td>5.434</td>
          <td>3.755000</td>
        </tr>
        <tr>
          <th>2011-01-20 20:00:00</th>
          <td>6.363</td>
          <td>5.686</td>
          <td>4.475</td>
          <td>5.400</td>
          <td>3.078333</td>
        </tr>
        <tr>
          <th>2011-01-20 21:00:00</th>
          <td>6.378</td>
          <td>5.766</td>
          <td>4.579</td>
          <td>5.416</td>
          <td>3.020000</td>
        </tr>
        <tr>
          <th>2011-01-20 22:00:00</th>
          <td>6.330</td>
          <td>5.810</td>
          <td>4.653</td>
          <td>5.384</td>
          <td>3.570000</td>
        </tr>
        <tr>
          <th>2011-01-20 23:00:00</th>
          <td>6.200</td>
          <td>5.744</td>
          <td>4.649</td>
          <td>5.332</td>
          <td>4.566667</td>
        </tr>
        <tr>
          <th>2011-01-21 00:00:00</th>
          <td>6.011</td>
          <td>5.616</td>
          <td>4.584</td>
          <td>5.231</td>
          <td>4.958333</td>
        </tr>
        <tr>
          <th>2011-01-21 01:00:00</th>
          <td>5.796</td>
          <td>5.453</td>
          <td>4.535</td>
          <td>5.125</td>
          <td>5.296667</td>
        </tr>
        <tr>
          <th>2011-01-21 02:00:00</th>
          <td>5.600</td>
          <td>5.315</td>
          <td>4.436</td>
          <td>5.015</td>
          <td>5.156667</td>
        </tr>
        <tr>
          <th>2011-01-21 03:00:00</th>
          <td>5.481</td>
          <td>5.126</td>
          <td>4.385</td>
          <td>5.062</td>
          <td>5.585000</td>
        </tr>
        <tr>
          <th>2011-01-21 04:00:00</th>
          <td>5.671</td>
          <td>5.175</td>
          <td>4.530</td>
          <td>5.322</td>
          <td>5.595000</td>
        </tr>
        <tr>
          <th>2011-01-21 05:00:00</th>
          <td>5.951</td>
          <td>5.382</td>
          <td>4.742</td>
          <td>5.598</td>
          <td>4.701667</td>
        </tr>
        <tr>
          <th>2011-01-21 06:00:00</th>
          <td>6.178</td>
          <td>5.545</td>
          <td>4.933</td>
          <td>5.866</td>
          <td>4.765000</td>
        </tr>
        <tr>
          <th>2011-01-21 07:00:00</th>
          <td>6.255</td>
          <td>5.582</td>
          <td>4.952</td>
          <td>5.896</td>
          <td>2.610000</td>
        </tr>
        <tr>
          <th>2011-01-21 08:00:00</th>
          <td>6.053</td>
          <td>5.405</td>
          <td>4.758</td>
          <td>5.701</td>
          <td>2.940000</td>
        </tr>
        <tr>
          <th>2011-01-21 09:00:00</th>
          <td>5.553</td>
          <td>4.966</td>
          <td>4.355</td>
          <td>5.282</td>
          <td>2.220000</td>
        </tr>
        <tr>
          <th>2011-01-21 10:00:00</th>
          <td>4.967</td>
          <td>4.445</td>
          <td>3.825</td>
          <td>4.718</td>
          <td>1.343333</td>
        </tr>
        <tr>
          <th>2011-01-21 11:00:00</th>
          <td>4.471</td>
          <td>3.995</td>
          <td>3.316</td>
          <td>4.222</td>
          <td>1.376667</td>
        </tr>
        <tr>
          <th>2011-01-21 12:00:00</th>
          <td>4.128</td>
          <td>3.634</td>
          <td>2.912</td>
          <td>3.896</td>
          <td>0.690000</td>
        </tr>
        <tr>
          <th>2011-01-21 13:00:00</th>
          <td>3.887</td>
          <td>3.372</td>
          <td>2.644</td>
          <td>3.620</td>
          <td>0.733333</td>
        </tr>
        <tr>
          <th>2011-01-21 14:00:00</th>
          <td>3.522</td>
          <td>3.017</td>
          <td>2.413</td>
          <td>3.248</td>
          <td>0.790000</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>2015-06-17 22:00:00</th>
          <td>11.439</td>
          <td>11.406</td>
          <td>9.462</td>
          <td>10.201</td>
          <td>6.286667</td>
        </tr>
        <tr>
          <th>2015-06-17 23:00:00</th>
          <td>11.576</td>
          <td>11.520</td>
          <td>9.490</td>
          <td>10.370</td>
          <td>6.333333</td>
        </tr>
        <tr>
          <th>2015-06-18 00:00:00</th>
          <td>11.509</td>
          <td>11.566</td>
          <td>9.670</td>
          <td>10.516</td>
          <td>5.370000</td>
        </tr>
        <tr>
          <th>2015-06-18 01:00:00</th>
          <td>10.932</td>
          <td>11.194</td>
          <td>9.565</td>
          <td>10.173</td>
          <td>5.281667</td>
        </tr>
        <tr>
          <th>2015-06-18 02:00:00</th>
          <td>10.716</td>
          <td>10.780</td>
          <td>9.083</td>
          <td>9.715</td>
          <td>7.096667</td>
        </tr>
        <tr>
          <th>2015-06-18 03:00:00</th>
          <td>10.990</td>
          <td>10.887</td>
          <td>9.113</td>
          <td>10.037</td>
          <td>7.898333</td>
        </tr>
        <tr>
          <th>2015-06-18 04:00:00</th>
          <td>10.877</td>
          <td>10.839</td>
          <td>9.005</td>
          <td>9.887</td>
          <td>6.601667</td>
        </tr>
        <tr>
          <th>2015-06-18 05:00:00</th>
          <td>10.713</td>
          <td>10.708</td>
          <td>8.785</td>
          <td>9.757</td>
          <td>8.035000</td>
        </tr>
        <tr>
          <th>2015-06-18 06:00:00</th>
          <td>10.652</td>
          <td>10.718</td>
          <td>8.976</td>
          <td>9.795</td>
          <td>8.233333</td>
        </tr>
        <tr>
          <th>2015-06-18 07:00:00</th>
          <td>10.561</td>
          <td>10.755</td>
          <td>9.533</td>
          <td>9.812</td>
          <td>9.306667</td>
        </tr>
        <tr>
          <th>2015-06-18 08:00:00</th>
          <td>10.424</td>
          <td>10.570</td>
          <td>9.615</td>
          <td>9.720</td>
          <td>10.073333</td>
        </tr>
        <tr>
          <th>2015-06-18 09:00:00</th>
          <td>10.389</td>
          <td>10.458</td>
          <td>9.529</td>
          <td>9.596</td>
          <td>11.288333</td>
        </tr>
        <tr>
          <th>2015-06-18 10:00:00</th>
          <td>10.573</td>
          <td>10.577</td>
          <td>9.552</td>
          <td>9.630</td>
          <td>11.843333</td>
        </tr>
        <tr>
          <th>2015-06-18 11:00:00</th>
          <td>10.686</td>
          <td>10.790</td>
          <td>9.716</td>
          <td>9.726</td>
          <td>11.493333</td>
        </tr>
        <tr>
          <th>2015-06-18 12:00:00</th>
          <td>10.806</td>
          <td>10.972</td>
          <td>9.871</td>
          <td>9.819</td>
          <td>11.251667</td>
        </tr>
        <tr>
          <th>2015-06-18 13:00:00</th>
          <td>10.898</td>
          <td>11.219</td>
          <td>10.065</td>
          <td>9.944</td>
          <td>9.471667</td>
        </tr>
        <tr>
          <th>2015-06-18 14:00:00</th>
          <td>10.884</td>
          <td>11.251</td>
          <td>10.175</td>
          <td>10.029</td>
          <td>8.925000</td>
        </tr>
        <tr>
          <th>2015-06-18 15:00:00</th>
          <td>10.713</td>
          <td>11.110</td>
          <td>10.145</td>
          <td>9.962</td>
          <td>8.605000</td>
        </tr>
        <tr>
          <th>2015-06-18 16:00:00</th>
          <td>10.496</td>
          <td>10.908</td>
          <td>10.044</td>
          <td>9.831</td>
          <td>9.148333</td>
        </tr>
        <tr>
          <th>2015-06-18 17:00:00</th>
          <td>10.304</td>
          <td>10.629</td>
          <td>9.841</td>
          <td>9.646</td>
          <td>9.288333</td>
        </tr>
        <tr>
          <th>2015-06-18 18:00:00</th>
          <td>10.158</td>
          <td>10.359</td>
          <td>9.542</td>
          <td>9.466</td>
          <td>7.561667</td>
        </tr>
        <tr>
          <th>2015-06-18 19:00:00</th>
          <td>10.037</td>
          <td>10.171</td>
          <td>9.116</td>
          <td>9.253</td>
          <td>7.331667</td>
        </tr>
        <tr>
          <th>2015-06-18 20:00:00</th>
          <td>9.900</td>
          <td>9.873</td>
          <td>8.494</td>
          <td>8.991</td>
          <td>7.200000</td>
        </tr>
        <tr>
          <th>2015-06-18 21:00:00</th>
          <td>9.726</td>
          <td>9.728</td>
          <td>8.386</td>
          <td>8.815</td>
          <td>8.670000</td>
        </tr>
        <tr>
          <th>2015-06-18 22:00:00</th>
          <td>9.512</td>
          <td>9.551</td>
          <td>8.116</td>
          <td>8.530</td>
          <td>9.340000</td>
        </tr>
        <tr>
          <th>2015-06-18 23:00:00</th>
          <td>9.423</td>
          <td>9.376</td>
          <td>7.771</td>
          <td>8.339</td>
          <td>8.268333</td>
        </tr>
        <tr>
          <th>2015-06-19 00:00:00</th>
          <td>9.459</td>
          <td>9.509</td>
          <td>8.006</td>
          <td>8.350</td>
          <td>8.053333</td>
        </tr>
        <tr>
          <th>2015-06-19 01:00:00</th>
          <td>9.490</td>
          <td>9.669</td>
          <td>8.177</td>
          <td>8.331</td>
          <td>8.423333</td>
        </tr>
        <tr>
          <th>2015-06-19 02:00:00</th>
          <td>9.221</td>
          <td>9.518</td>
          <td>7.950</td>
          <td>7.980</td>
          <td>8.026667</td>
        </tr>
        <tr>
          <th>2015-06-19 03:00:00</th>
          <td>8.839</td>
          <td>9.278</td>
          <td>7.673</td>
          <td>7.547</td>
          <td>8.200000</td>
        </tr>
      </tbody>
    </table>
    <p>38653 rows × 5 columns</p>
    </div>



.. code:: ipython3

    mlr.synthesize()




.. parsed-literal::

    1996-01-01 00:00:00    10.786788
    1996-01-01 01:00:00    10.314553
    1996-01-01 02:00:00    10.173168
    1996-01-01 03:00:00     9.756164
    1996-01-01 04:00:00     9.593828
    1996-01-01 05:00:00     9.975427
    1996-01-01 06:00:00     9.610730
    1996-01-01 07:00:00     9.403509
    1996-01-01 08:00:00     9.378400
    1996-01-01 09:00:00     9.328809
    1996-01-01 10:00:00     9.219646
    1996-01-01 11:00:00     9.042950
    1996-01-01 12:00:00     8.636226
    1996-01-01 13:00:00     8.241108
    1996-01-01 14:00:00     8.074203
    1996-01-01 15:00:00     8.166895
    1996-01-01 16:00:00     8.060465
    1996-01-01 17:00:00     7.884909
    1996-01-01 18:00:00     7.775125
    1996-01-01 19:00:00     7.704580
    1996-01-01 20:00:00     7.662440
    1996-01-01 21:00:00     7.567080
    1996-01-01 22:00:00     7.539512
    1996-01-01 23:00:00     7.298382
    1996-01-02 00:00:00     7.150031
    1996-01-02 01:00:00     7.024435
    1996-01-02 02:00:00     7.064542
    1996-01-02 03:00:00     7.248798
    1996-01-02 04:00:00     7.398593
    1996-01-02 05:00:00     7.501256
                             ...    
    2015-06-17 22:00:00     6.286667
    2015-06-17 23:00:00     6.333333
    2015-06-18 00:00:00     5.370000
    2015-06-18 01:00:00     5.281667
    2015-06-18 02:00:00     7.096667
    2015-06-18 03:00:00     7.898333
    2015-06-18 04:00:00     6.601667
    2015-06-18 05:00:00     8.035000
    2015-06-18 06:00:00     8.233333
    2015-06-18 07:00:00     9.306667
    2015-06-18 08:00:00    10.073333
    2015-06-18 09:00:00    11.288333
    2015-06-18 10:00:00    11.843333
    2015-06-18 11:00:00    11.493333
    2015-06-18 12:00:00    11.251667
    2015-06-18 13:00:00     9.471667
    2015-06-18 14:00:00     8.925000
    2015-06-18 15:00:00     8.605000
    2015-06-18 16:00:00     9.148333
    2015-06-18 17:00:00     9.288333
    2015-06-18 18:00:00     7.561667
    2015-06-18 19:00:00     7.331667
    2015-06-18 20:00:00     7.200000
    2015-06-18 21:00:00     8.670000
    2015-06-18 22:00:00     9.340000
    2015-06-18 23:00:00     8.268333
    2015-06-19 00:00:00     8.053333
    2015-06-19 01:00:00     8.423333
    2015-06-19 02:00:00     8.026667
    2015-06-19 03:00:00     8.200000
    Length: 170615, dtype: float64



.. code:: ipython3

    ordinary = bw.OrdinaryLeastSquares(merra2_dfs[0].data[ref_merra2_wdgt.value], 
                                      site_data[target_wdgt.value], averaging_prd='1D', 
                                      coverage_threshold=0.8)
    
    

.. code:: ipython3

    sum((ordinary.data['target_spd'] - ordinary.data['ref_spd'])**2)




.. parsed-literal::

    3477.0883452264575



.. code:: ipython3

    ordinary.run()
    ordinary.show_params()
    ordinary.plot()
    ordinary.averaging_prd
    ordinary.synthesize()


.. parsed-literal::

    {'slope': 0.76424017513653641, 'offset': 1.2869551044078436, 'r2': 0.90475689691199157}
    


.. image:: output_22_1.png




.. parsed-literal::

    1996-01-01     8.948590
    1996-01-02     8.661554
    1996-01-03    13.098574
    1996-01-04     6.810183
    1996-01-05     8.973810
    1996-01-06    14.888711
    1996-01-07    14.096130
    1996-01-08    12.677445
    1996-01-09    11.931643
    1996-01-10     8.259150
    1996-01-11    12.218169
    1996-01-12     7.186189
    1996-01-13     8.862231
    1996-01-14     8.890158
    1996-01-15     7.577830
    1996-01-16     7.217236
    1996-01-17    11.852448
    1996-01-18     8.895539
    1996-01-19     7.182431
    1996-01-20     9.939046
    1996-01-21     8.520074
    1996-01-22     7.501693
    1996-01-23     9.165475
    1996-01-24     9.007500
    1996-01-25    10.689147
    1996-01-26     9.044057
    1996-01-27    10.614092
    1996-01-28     8.483614
    1996-01-29     7.459691
    1996-01-30     6.989906
                    ...    
    2015-05-20     7.633819
    2015-05-21     8.998958
    2015-05-22     8.083125
    2015-05-23     6.037708
    2015-05-24     8.020486
    2015-05-25     6.013056
    2015-05-26     7.278889
    2015-05-27     8.191528
    2015-05-28     9.129028
    2015-05-29     7.594931
    2015-05-30     6.686458
    2015-05-31    11.129028
    2015-06-01    13.511111
    2015-06-02     8.627500
    2015-06-03     5.934167
    2015-06-04     5.652431
    2015-06-05    12.585972
    2015-06-06    13.950486
    2015-06-07     6.966528
    2015-06-08     6.498472
    2015-06-09     4.410000
    2015-06-10     4.397847
    2015-06-11     3.618403
    2015-06-12     7.128542
    2015-06-13     8.171250
    2015-06-14     7.255278
    2015-06-15     3.797708
    2015-06-16     7.159375
    2015-06-17     9.427778
    2015-06-18     8.649306
    Length: 7109, dtype: float64



.. code:: ipython3

    ordinary




.. parsed-literal::

    <analyse.correlation.OridnaryLeastSquares at 0x1d26947aa20>



.. code:: ipython3

    regress = bw.OrthogonalLeastSquares(merra2_dfs[0].data[ref_merra2_wdgt.value], 
                                      site_data[target_wdgt.value], averaging_prd='1D', 
                                      coverage_threshold=0.8)

.. code:: ipython3

    regress.long_term_ref_speed()




.. parsed-literal::

    8.6902412103389981



.. code:: ipython3

    regress.num_data_pts




.. parsed-literal::

    1609



.. code:: ipython3

    regress.run()
    regress.show_params()
    regress.plot()
    regress.synthesize()
    


.. parsed-literal::

    Beta: [ 0.64407486  1.47482962]
    Beta Std Error: [ 0.00720134  0.07880838]
    Beta Covariance: [[  4.63980625e-05  -4.65937745e-04]
     [ -4.65937745e-04   5.55671888e-03]]
    Residual Variance: 1.1177028860575626
    Inverse Condition #: 0.03459057763923457
    Reason(s) for Halting:
      Sum of squares convergence
    Model output: None
    {'slope': 0.64407486013506543, 'offset': 1.4748296182552052, 'r2': 0.81711891398682301}
    


.. image:: output_27_1.png




.. parsed-literal::

    1996-01-01     9.333616
    1996-01-02     9.054866
    1996-01-03    12.284311
    1996-01-04     5.821154
    1996-01-05     8.992176
    1996-01-06    14.415367
    1996-01-07    14.974129
    1996-01-08    12.195966
    1996-01-09    12.450563
    1996-01-10     9.793486
    1996-01-11    12.189605
    1996-01-12     6.498265
    1996-01-13     8.167465
    1996-01-14     7.621665
    1996-01-15     7.465021
    1996-01-16     7.851493
    1996-01-17    12.337635
    1996-01-18     9.120776
    1996-01-19     7.048305
    1996-01-20     9.848581
    1996-01-21     8.727891
    1996-01-22     7.420499
    1996-01-23     8.651434
    1996-01-24     8.932063
    1996-01-25    10.606469
    1996-01-26     9.213953
    1996-01-27    11.128143
    1996-01-28     8.721477
    1996-01-29     7.159569
    1996-01-30     7.103749
                    ...    
    2015-05-21     8.998958
    2015-05-22     8.083125
    2015-05-23     6.037708
    2015-05-24     8.020486
    2015-05-25     6.013056
    2015-05-26     7.278889
    2015-05-27     8.191528
    2015-05-28     9.129028
    2015-05-29     7.594931
    2015-05-30     6.686458
    2015-05-31    11.129028
    2015-06-01    13.511111
    2015-06-02     8.627500
    2015-06-03     5.934167
    2015-06-04     5.652431
    2015-06-05    12.585972
    2015-06-06    13.950486
    2015-06-07     6.966528
    2015-06-08     6.498472
    2015-06-09     4.410000
    2015-06-10     4.397847
    2015-06-11     3.618403
    2015-06-12     7.128542
    2015-06-13     8.171250
    2015-06-14     7.255278
    2015-06-15     3.797708
    2015-06-16     7.159375
    2015-06-17     9.427778
    2015-06-18     8.649306
    2015-06-19     8.175833
    Length: 7111, dtype: float64



.. code:: ipython3

    merra2_dfs[0].longitude, merra2_dfs[0].latitude




.. parsed-literal::

    ('-8.125', '55.0')



.. code:: ipython3

    speed_sort = bw.SpeedSort(ref['WS50m_ms'], ref['WD50m_deg'], site_data['A_Avg1'],
                              site_data['WindDir_AVG'],'1H', 0)


::


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-72-d8b8843b8523> in <module>()
    ----> 1 speed_sort = bw.SpeedSort(ref['WS50m_ms'], ref['WD50m_deg'], site_data['A_Avg1'],
          2                           site_data['WindDir_AVG'],'1H', 0)
    

    NameError: name 'ref' is not defined


.. code:: ipython3

    speed_sort.run()
    speed_sort.synthesize()


.. parsed-literal::

    {'Ref_cutoff_for_speed': 4.0, 'Ref_veer_cutoff': 5.177485042292863, 'Target_veer_cutoff': 4.9858107348287, 'Overall_average_veer': 2.1928558734672117}
    Processing sector: 1
    {'slope': 0.82886143267667023, 'offset': 1.9490861758847409}
    Processing sector: 2
    {'slope': 0.84971093618543736, 'offset': 1.5164545578398485}
    Processing sector: 3
    {'slope': 0.97172606339029188, 'offset': -0.080334130128519199}
    Processing sector: 4
    {'slope': 0.99374704279384074, 'offset': -0.16013396734383889}
    Processing sector: 5
    {'slope': 0.93477756144975721, 'offset': -0.024926640010193957}
    Processing sector: 6
    {'slope': 1.0441416987003833, 'offset': -1.2503755025201464}
    Processing sector: 7
    {'slope': 1.0077451180480734, 'offset': -1.0670993138475238}
    Processing sector: 8
    {'slope': 0.90950167543588323, 'offset': -0.76100683873290631}
    Processing sector: 9
    {'slope': 0.86650650662934992, 'offset': -0.053186413884664496}
    Processing sector: 10
    {'slope': 0.82029081056829745, 'offset': 0.74897802811954595}
    Processing sector: 11
    {'slope': 0.7350981406748458, 'offset': 1.5922674066220006}
    Processing sector: 12
    {'slope': 0.83006967590343606, 'offset': 1.5463510418953881}
    



.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>0</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2011-01-20 09:00:00</th>
          <td>5.368333</td>
        </tr>
        <tr>
          <th>2011-01-20 10:00:00</th>
          <td>5.158333</td>
        </tr>
        <tr>
          <th>2011-01-20 11:00:00</th>
          <td>5.115000</td>
        </tr>
        <tr>
          <th>2011-01-20 12:00:00</th>
          <td>5.998333</td>
        </tr>
        <tr>
          <th>2011-01-20 13:00:00</th>
          <td>5.250000</td>
        </tr>
        <tr>
          <th>2011-01-20 14:00:00</th>
          <td>3.505000</td>
        </tr>
        <tr>
          <th>2011-01-20 15:00:00</th>
          <td>3.178333</td>
        </tr>
        <tr>
          <th>2011-01-20 16:00:00</th>
          <td>4.206667</td>
        </tr>
        <tr>
          <th>2011-01-20 17:00:00</th>
          <td>4.980000</td>
        </tr>
        <tr>
          <th>2011-01-20 18:00:00</th>
          <td>4.128333</td>
        </tr>
        <tr>
          <th>2011-01-20 19:00:00</th>
          <td>3.755000</td>
        </tr>
        <tr>
          <th>2011-01-20 20:00:00</th>
          <td>3.078333</td>
        </tr>
        <tr>
          <th>2011-01-20 21:00:00</th>
          <td>3.020000</td>
        </tr>
        <tr>
          <th>2011-01-20 22:00:00</th>
          <td>3.570000</td>
        </tr>
        <tr>
          <th>2011-01-20 23:00:00</th>
          <td>4.566667</td>
        </tr>
        <tr>
          <th>2011-01-21 00:00:00</th>
          <td>4.958333</td>
        </tr>
        <tr>
          <th>2011-01-21 01:00:00</th>
          <td>5.296667</td>
        </tr>
        <tr>
          <th>2011-01-21 02:00:00</th>
          <td>5.156667</td>
        </tr>
        <tr>
          <th>2011-01-21 03:00:00</th>
          <td>5.585000</td>
        </tr>
        <tr>
          <th>2011-01-21 04:00:00</th>
          <td>5.595000</td>
        </tr>
        <tr>
          <th>2011-01-21 05:00:00</th>
          <td>4.701667</td>
        </tr>
        <tr>
          <th>2011-01-21 06:00:00</th>
          <td>4.765000</td>
        </tr>
        <tr>
          <th>2011-01-21 07:00:00</th>
          <td>2.610000</td>
        </tr>
        <tr>
          <th>2011-01-21 08:00:00</th>
          <td>2.940000</td>
        </tr>
        <tr>
          <th>2011-01-21 09:00:00</th>
          <td>2.220000</td>
        </tr>
        <tr>
          <th>2011-01-21 10:00:00</th>
          <td>1.343333</td>
        </tr>
        <tr>
          <th>2011-01-21 11:00:00</th>
          <td>1.376667</td>
        </tr>
        <tr>
          <th>2011-01-21 12:00:00</th>
          <td>0.690000</td>
        </tr>
        <tr>
          <th>2011-01-21 13:00:00</th>
          <td>0.733333</td>
        </tr>
        <tr>
          <th>2011-01-21 14:00:00</th>
          <td>0.790000</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
        </tr>
        <tr>
          <th>2015-06-17 22:00:00</th>
          <td>6.286667</td>
        </tr>
        <tr>
          <th>2015-06-17 23:00:00</th>
          <td>6.333333</td>
        </tr>
        <tr>
          <th>2015-06-18 00:00:00</th>
          <td>5.370000</td>
        </tr>
        <tr>
          <th>2015-06-18 01:00:00</th>
          <td>5.281667</td>
        </tr>
        <tr>
          <th>2015-06-18 02:00:00</th>
          <td>7.096667</td>
        </tr>
        <tr>
          <th>2015-06-18 03:00:00</th>
          <td>7.898333</td>
        </tr>
        <tr>
          <th>2015-06-18 04:00:00</th>
          <td>6.601667</td>
        </tr>
        <tr>
          <th>2015-06-18 05:00:00</th>
          <td>8.035000</td>
        </tr>
        <tr>
          <th>2015-06-18 06:00:00</th>
          <td>8.233333</td>
        </tr>
        <tr>
          <th>2015-06-18 07:00:00</th>
          <td>9.306667</td>
        </tr>
        <tr>
          <th>2015-06-18 08:00:00</th>
          <td>10.073333</td>
        </tr>
        <tr>
          <th>2015-06-18 09:00:00</th>
          <td>11.288333</td>
        </tr>
        <tr>
          <th>2015-06-18 10:00:00</th>
          <td>11.843333</td>
        </tr>
        <tr>
          <th>2015-06-18 11:00:00</th>
          <td>11.493333</td>
        </tr>
        <tr>
          <th>2015-06-18 12:00:00</th>
          <td>11.251667</td>
        </tr>
        <tr>
          <th>2015-06-18 13:00:00</th>
          <td>9.471667</td>
        </tr>
        <tr>
          <th>2015-06-18 14:00:00</th>
          <td>8.925000</td>
        </tr>
        <tr>
          <th>2015-06-18 15:00:00</th>
          <td>8.605000</td>
        </tr>
        <tr>
          <th>2015-06-18 16:00:00</th>
          <td>9.148333</td>
        </tr>
        <tr>
          <th>2015-06-18 17:00:00</th>
          <td>9.288333</td>
        </tr>
        <tr>
          <th>2015-06-18 18:00:00</th>
          <td>7.561667</td>
        </tr>
        <tr>
          <th>2015-06-18 19:00:00</th>
          <td>7.331667</td>
        </tr>
        <tr>
          <th>2015-06-18 20:00:00</th>
          <td>7.200000</td>
        </tr>
        <tr>
          <th>2015-06-18 21:00:00</th>
          <td>8.670000</td>
        </tr>
        <tr>
          <th>2015-06-18 22:00:00</th>
          <td>9.340000</td>
        </tr>
        <tr>
          <th>2015-06-18 23:00:00</th>
          <td>8.268333</td>
        </tr>
        <tr>
          <th>2015-06-19 00:00:00</th>
          <td>8.053333</td>
        </tr>
        <tr>
          <th>2015-06-19 01:00:00</th>
          <td>8.423333</td>
        </tr>
        <tr>
          <th>2015-06-19 02:00:00</th>
          <td>8.026667</td>
        </tr>
        <tr>
          <th>2015-06-19 03:00:00</th>
          <td>8.200000</td>
        </tr>
      </tbody>
    </table>
    <p>38653 rows × 1 columns</p>
    </div>



.. code:: ipython3

    speed_sort.show_params()


.. parsed-literal::

    {'Ref_cutoff_for_speed': 4.0, 'Ref_veer_cutoff': 5.177485042292863, 'Target_veer_cutoff': 4.9858107348287, 'Overall_average_veer': 2.1928558734672117, 1: {'slope': 0.82886143267667023, 'offset': 1.9490861758847409, 'target_cutoff': 4.9733333333333327, 'num_pts_for_speed_fit': 1583, 'num_total_pts': 2008, 'average_veer': -6.685867895545315, 'num_pts_for_veer': 1302}, 2: {'slope': 0.84959435735488953, 'offset': 1.517603639843772, 'target_cutoff': 4.3700000000000001, 'num_pts_for_speed_fit': 1260, 'num_total_pts': 1669, 'average_veer': -6.295823665893272, 'num_pts_for_veer': 862}, 3: {'slope': 0.9716146996240278, 'offset': -0.079344491439719, 'target_cutoff': 3.5833333333333335, 'num_pts_for_speed_fit': 923, 'num_total_pts': 1259, 'average_veer': -2.872072072072072, 'num_pts_for_veer': 555}, 4: {'slope': 0.99374704279384074, 'offset': -0.16013396734383889, 'target_cutoff': 3.8783333333333334, 'num_pts_for_speed_fit': 1198, 'num_total_pts': 1499, 'average_veer': 3.123110151187905, 'num_pts_for_veer': 926}, 5: {'slope': 0.93482016475530239, 'offset': -0.02540836042643857, 'target_cutoff': 3.9133333333333327, 'num_pts_for_speed_fit': 2305, 'num_total_pts': 2598, 'average_veer': 1.6128526645768024, 'num_pts_for_veer': 1914}, 6: {'slope': 1.0440258219393279, 'offset': -1.2490355178442938, 'target_cutoff': 3.473333333333334, 'num_pts_for_speed_fit': 2939, 'num_total_pts': 3305, 'average_veer': 10.046491228070176, 'num_pts_for_veer': 2280}, 7: {'slope': 1.0075876674179476, 'offset': -1.0651156711141425, 'target_cutoff': 3.4283333333333332, 'num_pts_for_speed_fit': 3571, 'num_total_pts': 3949, 'average_veer': 12.906561319134319, 'num_pts_for_veer': 2911}, 8: {'slope': 0.90950949318627872, 'offset': -0.76111330927796761, 'target_cutoff': 3.1666666666666665, 'num_pts_for_speed_fit': 4542, 'num_total_pts': 4885, 'average_veer': 5.886650550631211, 'num_pts_for_veer': 3723}, 9: {'slope': 0.86667609393065237, 'offset': -0.055493537117882852, 'target_cutoff': 3.5683333333333334, 'num_pts_for_speed_fit': 5453, 'num_total_pts': 5846, 'average_veer': 1.7051502145922748, 'num_pts_for_veer': 4660}, 10: {'slope': 0.82029081056829745, 'offset': 0.74897802811954595, 'target_cutoff': 4.083333333333333, 'num_pts_for_speed_fit': 5170, 'num_total_pts': 5603, 'average_veer': -1.4156316916488223, 'num_pts_for_veer': 4670}, 11: {'slope': 0.7350981406748458, 'offset': 1.5922674066220006, 'target_cutoff': 4.3850000000000007, 'num_pts_for_speed_fit': 3136, 'num_total_pts': 3558, 'average_veer': -2.770755422587883, 'num_pts_for_veer': 2674}, 12: {'slope': 0.83006967590343606, 'offset': 1.5463510418953881, 'target_cutoff': 4.8583333333333334, 'num_pts_for_speed_fit': 1996, 'num_total_pts': 2474, 'average_veer': -2.9408926417370327, 'num_pts_for_veer': 1658}}
    

.. code:: ipython3

    speed_sort.plot()



.. image:: output_32_0.png



.. image:: output_32_1.png



.. image:: output_32_2.png



.. image:: output_32_3.png



.. image:: output_32_4.png



.. image:: output_32_5.png



.. image:: output_32_6.png



.. image:: output_32_7.png



.. image:: output_32_8.png



.. image:: output_32_9.png



.. image:: output_32_10.png



.. image:: output_32_11.png


.. code:: ipython3

    speed_sort.get_result_table()
    




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>average_veer</th>
          <th>num_pts_for_speed_fit</th>
          <th>num_pts_for_veer</th>
          <th>num_total_pts</th>
          <th>offset</th>
          <th>slope</th>
          <th>target_cutoff</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1</th>
          <td>-6.685868</td>
          <td>1583</td>
          <td>1302</td>
          <td>2008</td>
          <td>1.949086</td>
          <td>0.828861</td>
          <td>4.973333</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-6.295824</td>
          <td>1260</td>
          <td>862</td>
          <td>1669</td>
          <td>1.517604</td>
          <td>0.849594</td>
          <td>4.370000</td>
        </tr>
        <tr>
          <th>3</th>
          <td>-2.872072</td>
          <td>923</td>
          <td>555</td>
          <td>1259</td>
          <td>-0.079344</td>
          <td>0.971615</td>
          <td>3.583333</td>
        </tr>
        <tr>
          <th>4</th>
          <td>3.123110</td>
          <td>1198</td>
          <td>926</td>
          <td>1499</td>
          <td>-0.160134</td>
          <td>0.993747</td>
          <td>3.878333</td>
        </tr>
        <tr>
          <th>5</th>
          <td>1.612853</td>
          <td>2305</td>
          <td>1914</td>
          <td>2598</td>
          <td>-0.025408</td>
          <td>0.934820</td>
          <td>3.913333</td>
        </tr>
        <tr>
          <th>6</th>
          <td>10.046491</td>
          <td>2939</td>
          <td>2280</td>
          <td>3305</td>
          <td>-1.249036</td>
          <td>1.044026</td>
          <td>3.473333</td>
        </tr>
        <tr>
          <th>7</th>
          <td>12.906561</td>
          <td>3571</td>
          <td>2911</td>
          <td>3949</td>
          <td>-1.065116</td>
          <td>1.007588</td>
          <td>3.428333</td>
        </tr>
        <tr>
          <th>8</th>
          <td>5.886651</td>
          <td>4542</td>
          <td>3723</td>
          <td>4885</td>
          <td>-0.761113</td>
          <td>0.909509</td>
          <td>3.166667</td>
        </tr>
        <tr>
          <th>9</th>
          <td>1.705150</td>
          <td>5453</td>
          <td>4660</td>
          <td>5846</td>
          <td>-0.055494</td>
          <td>0.866676</td>
          <td>3.568333</td>
        </tr>
        <tr>
          <th>10</th>
          <td>-1.415632</td>
          <td>5170</td>
          <td>4670</td>
          <td>5603</td>
          <td>0.748978</td>
          <td>0.820291</td>
          <td>4.083333</td>
        </tr>
        <tr>
          <th>11</th>
          <td>-2.770755</td>
          <td>3136</td>
          <td>2674</td>
          <td>3558</td>
          <td>1.592267</td>
          <td>0.735098</td>
          <td>4.385000</td>
        </tr>
        <tr>
          <th>12</th>
          <td>-2.940893</td>
          <td>1996</td>
          <td>1658</td>
          <td>2474</td>
          <td>1.546351</td>
          <td>0.830070</td>
          <td>4.858333</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    speed_sort.overall_veer




.. parsed-literal::

    2.1928558734672117



.. code:: ipython3

    result['average_veer'].mean()




.. parsed-literal::

    1.0249810615590242



.. code:: ipython3

    sum(result['num_total_pts'])




.. parsed-literal::

    38653



.. code:: ipython3

    sum(result['num_pts_for_speed_fit'])




.. parsed-literal::

    34076



.. code:: ipython3

    sum(result['num_pts_for_veer'])




.. parsed-literal::

    28135



.. code:: ipython3

    bulkratio = bw.BulkSpeedRatio(merra2_dfs[0].data[ref_merra2_wdgt.value], 
                                      site_data[target_wdgt.value],'1H', 0.9)

.. code:: ipython3

    bulkratio.run()

.. code:: ipython3

    bulkratio.params




.. parsed-literal::

    {'slope': 0.7908900932150227}



.. code:: ipython3

    bulkratio.synthesize()




.. parsed-literal::

    1996-01-01 00:00:00    11.887078
    1996-01-01 01:00:00    11.634784
    1996-01-01 02:00:00    11.282047
    1996-01-01 03:00:00    10.952246
    1996-01-01 04:00:00    10.901629
    1996-01-01 05:00:00    10.764805
    1996-01-01 06:00:00    10.518838
    1996-01-01 07:00:00    10.189828
    1996-01-01 08:00:00     9.928834
    1996-01-01 09:00:00     9.595870
    1996-01-01 10:00:00     9.324594
    1996-01-01 11:00:00     9.251832
    1996-01-01 12:00:00     9.178280
    1996-01-01 13:00:00     9.175907
    1996-01-01 14:00:00     9.224942
    1996-01-01 15:00:00     9.217033
    1996-01-01 16:00:00     9.203588
    1996-01-01 17:00:00     9.081791
    1996-01-01 18:00:00     8.979766
    1996-01-01 19:00:00     8.883278
    1996-01-01 20:00:00     8.649174
    1996-01-01 21:00:00     8.247402
    1996-01-01 22:00:00     7.893083
    1996-01-01 23:00:00     7.637626
    1996-01-02 00:00:00     7.356069
    1996-01-02 01:00:00     7.056321
    1996-01-02 02:00:00     6.920288
    1996-01-02 03:00:00     7.188400
    1996-01-02 04:00:00     7.517410
    1996-01-02 05:00:00     7.802922
                             ...    
    2015-06-17 22:00:00     6.286667
    2015-06-17 23:00:00     6.333333
    2015-06-18 00:00:00     5.370000
    2015-06-18 01:00:00     5.281667
    2015-06-18 02:00:00     7.096667
    2015-06-18 03:00:00     7.898333
    2015-06-18 04:00:00     6.601667
    2015-06-18 05:00:00     8.035000
    2015-06-18 06:00:00     8.233333
    2015-06-18 07:00:00     9.306667
    2015-06-18 08:00:00    10.073333
    2015-06-18 09:00:00    11.288333
    2015-06-18 10:00:00    11.843333
    2015-06-18 11:00:00    11.493333
    2015-06-18 12:00:00    11.251667
    2015-06-18 13:00:00     9.471667
    2015-06-18 14:00:00     8.925000
    2015-06-18 15:00:00     8.605000
    2015-06-18 16:00:00     9.148333
    2015-06-18 17:00:00     9.288333
    2015-06-18 18:00:00     7.561667
    2015-06-18 19:00:00     7.331667
    2015-06-18 20:00:00     7.200000
    2015-06-18 21:00:00     8.670000
    2015-06-18 22:00:00     9.340000
    2015-06-18 23:00:00     8.268333
    2015-06-19 00:00:00     8.053333
    2015-06-19 01:00:00     8.423333
    2015-06-19 02:00:00     8.026667
    2015-06-19 03:00:00     8.200000
    Length: 170621, dtype: float64



Testing Direction Binning by Ciaran's tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from analyse import frequency_analysis as freq
    dir_test = pd.read_csv('direction_test.csv')
    dir_test['bw_tool'] = freq.get_binned_direction_series(dir_test['Direction'], sectors=12) - 1.0
    dir_test ['difference'] = abs(dir_test['bw_tool']  - dir_test["Ciaran's"])
    dir_test.to_csv('direction_test.csv')

Testing averaging by Ciaran's tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    speed_sort.data.to_csv('averaging_check.csv')

.. code:: ipython3

    speed_sort.data.head(5)




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>ref_spd</th>
          <th>target_spd</th>
          <th>ref_dir</th>
          <th>target_dir</th>
          <th>ref_dir_bin</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2011-01-20 09:00:00</th>
          <td>4.386</td>
          <td>5.368333</td>
          <td>205.0</td>
          <td>221.520378</td>
          <td>8</td>
        </tr>
        <tr>
          <th>2011-01-20 10:00:00</th>
          <td>4.276</td>
          <td>5.158333</td>
          <td>207.0</td>
          <td>212.812301</td>
          <td>8</td>
        </tr>
        <tr>
          <th>2011-01-20 11:00:00</th>
          <td>4.297</td>
          <td>5.115000</td>
          <td>206.0</td>
          <td>218.325316</td>
          <td>8</td>
        </tr>
        <tr>
          <th>2011-01-20 12:00:00</th>
          <td>4.424</td>
          <td>5.998333</td>
          <td>205.0</td>
          <td>220.561748</td>
          <td>8</td>
        </tr>
        <tr>
          <th>2011-01-20 13:00:00</th>
          <td>4.550</td>
          <td>5.250000</td>
          <td>206.0</td>
          <td>232.589117</td>
          <td>8</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    avg_test = pd.read_csv('averaging_check.csv')
    avg_test = avg_test.set_index(pd.DatetimeIndex(avg_test['Timestamp']))
    avg_test = pd.concat([avg_test,speed_sort.data], axis=1, join='inner')
    avg_test['ref_spd_diff'] = abs(avg_test['ref_spd'] - avg_test['ref_spd_C'])
    avg_test['target_spd_diff'] = abs(avg_test['target_spd'] - avg_test['target_spd_C'])
    avg_test['ref_dir_diff'] = abs(avg_test['ref_dir'] - avg_test['ref_dir_C'])
    avg_test['target_dir_diff'] = abs(avg_test['target_dir'] - avg_test['target_dir_C'])
    avg_test.head(10)
    




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Timestamp</th>
          <th>target_spd_C</th>
          <th>target_dir_C</th>
          <th>ref_spd_C</th>
          <th>ref_dir_C</th>
          <th>ref_spd</th>
          <th>target_spd</th>
          <th>ref_dir</th>
          <th>target_dir</th>
          <th>ref_dir_bin</th>
          <th>ref_spd_diff</th>
          <th>target_spd_diff</th>
          <th>ref_dir_diff</th>
          <th>target_dir_diff</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2011-01-20 09:00:00</th>
          <td>1/20/2011 09:00</td>
          <td>5.368333</td>
          <td>221.520371</td>
          <td>4.386</td>
          <td>205</td>
          <td>4.386</td>
          <td>5.368333</td>
          <td>205.0</td>
          <td>221.520378</td>
          <td>8</td>
          <td>1.560000e-07</td>
          <td>6.666668e-09</td>
          <td>2.842171e-14</td>
          <td>7.602772e-06</td>
        </tr>
        <tr>
          <th>2011-01-20 10:00:00</th>
          <td>1/20/2011 10:00</td>
          <td>5.158333</td>
          <td>212.812302</td>
          <td>4.276</td>
          <td>207</td>
          <td>4.276</td>
          <td>5.158333</td>
          <td>207.0</td>
          <td>212.812301</td>
          <td>8</td>
          <td>2.300000e-08</td>
          <td>3.133333e-08</td>
          <td>5.684342e-14</td>
          <td>5.447800e-07</td>
        </tr>
        <tr>
          <th>2011-01-20 11:00:00</th>
          <td>1/20/2011 11:00</td>
          <td>5.115000</td>
          <td>218.325317</td>
          <td>4.297</td>
          <td>206</td>
          <td>4.297</td>
          <td>5.115000</td>
          <td>206.0</td>
          <td>218.325316</td>
          <td>8</td>
          <td>6.900000e-08</td>
          <td>2.290000e-07</td>
          <td>2.842171e-14</td>
          <td>1.786469e-06</td>
        </tr>
        <tr>
          <th>2011-01-20 12:00:00</th>
          <td>1/20/2011 12:00</td>
          <td>5.998333</td>
          <td>220.561752</td>
          <td>4.424</td>
          <td>205</td>
          <td>4.424</td>
          <td>5.998333</td>
          <td>205.0</td>
          <td>220.561748</td>
          <td>8</td>
          <td>2.140000e-07</td>
          <td>1.206667e-07</td>
          <td>0.000000e+00</td>
          <td>4.438059e-06</td>
        </tr>
        <tr>
          <th>2011-01-20 13:00:00</th>
          <td>1/20/2011 13:00</td>
          <td>5.250000</td>
          <td>232.589111</td>
          <td>4.550</td>
          <td>206</td>
          <td>4.550</td>
          <td>5.250000</td>
          <td>206.0</td>
          <td>232.589117</td>
          <td>8</td>
          <td>1.910000e-07</td>
          <td>0.000000e+00</td>
          <td>2.842171e-14</td>
          <td>5.797391e-06</td>
        </tr>
        <tr>
          <th>2011-01-20 14:00:00</th>
          <td>1/20/2011 14:00</td>
          <td>3.505000</td>
          <td>233.952560</td>
          <td>4.537</td>
          <td>209</td>
          <td>4.537</td>
          <td>3.505000</td>
          <td>209.0</td>
          <td>233.952562</td>
          <td>8</td>
          <td>1.790000e-07</td>
          <td>1.140000e-07</td>
          <td>0.000000e+00</td>
          <td>1.417876e-06</td>
        </tr>
        <tr>
          <th>2011-01-20 15:00:00</th>
          <td>1/20/2011 15:00</td>
          <td>3.178333</td>
          <td>232.907303</td>
          <td>4.416</td>
          <td>213</td>
          <td>4.416</td>
          <td>3.178333</td>
          <td>213.0</td>
          <td>232.907300</td>
          <td>8</td>
          <td>1.110000e-07</td>
          <td>5.133333e-08</td>
          <td>0.000000e+00</td>
          <td>3.231382e-06</td>
        </tr>
        <tr>
          <th>2011-01-20 16:00:00</th>
          <td>1/20/2011 16:00</td>
          <td>4.206666</td>
          <td>244.597748</td>
          <td>4.358</td>
          <td>216</td>
          <td>4.358</td>
          <td>4.206667</td>
          <td>216.0</td>
          <td>244.597741</td>
          <td>8</td>
          <td>1.980000e-07</td>
          <td>1.966667e-07</td>
          <td>0.000000e+00</td>
          <td>7.023625e-06</td>
        </tr>
        <tr>
          <th>2011-01-20 17:00:00</th>
          <td>1/20/2011 17:00</td>
          <td>4.980000</td>
          <td>238.875351</td>
          <td>4.439</td>
          <td>217</td>
          <td>4.439</td>
          <td>4.980000</td>
          <td>217.0</td>
          <td>238.875353</td>
          <td>8</td>
          <td>1.300000e-07</td>
          <td>1.900000e-08</td>
          <td>0.000000e+00</td>
          <td>1.883546e-06</td>
        </tr>
        <tr>
          <th>2011-01-20 18:00:00</th>
          <td>1/20/2011 18:00</td>
          <td>4.128334</td>
          <td>253.291061</td>
          <td>4.492</td>
          <td>217</td>
          <td>4.492</td>
          <td>4.128333</td>
          <td>217.0</td>
          <td>253.291055</td>
          <td>8</td>
          <td>1.030000e-07</td>
          <td>2.356667e-07</td>
          <td>0.000000e+00</td>
          <td>6.372372e-06</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    avg_test.to_csv('averaging_test_results.csv')

.. code:: ipython3

    avg_test[avg_test['ref_dir_bin']>0]




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Timestamp</th>
          <th>target_spd_C</th>
          <th>target_dir_C</th>
          <th>ref_spd_C</th>
          <th>ref_dir_C</th>
          <th>ref_spd</th>
          <th>target_spd</th>
          <th>ref_dir</th>
          <th>target_dir</th>
          <th>ref_dir_bin</th>
          <th>ref_spd_diff</th>
          <th>target_spd_diff</th>
          <th>ref_dir_diff</th>
          <th>target_dir_diff</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2011-01-21 19:00:00</th>
          <td>1/21/2011 19:00</td>
          <td>5.591667</td>
          <td>359.371643</td>
          <td>1.240</td>
          <td>298</td>
          <td>1.240</td>
          <td>5.591667</td>
          <td>359.371646</td>
          <td>359.371646</td>
          <td>1</td>
          <td>1.000000e-08</td>
          <td>3.133333e-08</td>
          <td>61.371646</td>
          <td>2.754636e-06</td>
        </tr>
        <tr>
          <th>2011-01-22 00:00:00</th>
          <td>1/22/2011</td>
          <td>6.281667</td>
          <td>19.265867</td>
          <td>1.238</td>
          <td>303</td>
          <td>1.238</td>
          <td>6.281667</td>
          <td>19.265868</td>
          <td>19.265868</td>
          <td>2</td>
          <td>3.500000e-08</td>
          <td>8.933333e-08</td>
          <td>283.734132</td>
          <td>8.440202e-07</td>
        </tr>
        <tr>
          <th>2011-01-22 01:00:00</th>
          <td>1/22/2011 01:00</td>
          <td>7.693333</td>
          <td>16.968378</td>
          <td>1.156</td>
          <td>323</td>
          <td>1.156</td>
          <td>7.693333</td>
          <td>16.968379</td>
          <td>16.968379</td>
          <td>2</td>
          <td>1.800000e-08</td>
          <td>1.843333e-07</td>
          <td>306.031621</td>
          <td>4.579017e-07</td>
        </tr>
        <tr>
          <th>2011-01-22 02:00:00</th>
          <td>1/22/2011 02:00</td>
          <td>7.733333</td>
          <td>14.157602</td>
          <td>1.181</td>
          <td>347</td>
          <td>1.181</td>
          <td>7.733333</td>
          <td>14.157603</td>
          <td>14.157603</td>
          <td>1</td>
          <td>6.000000e-09</td>
          <td>2.223333e-07</td>
          <td>332.842397</td>
          <td>4.216543e-07</td>
        </tr>
        <tr>
          <th>2011-01-22 03:00:00</th>
          <td>1/22/2011 03:00</td>
          <td>7.726666</td>
          <td>12.857865</td>
          <td>1.320</td>
          <td>7</td>
          <td>1.320</td>
          <td>7.726667</td>
          <td>12.857865</td>
          <td>12.857865</td>
          <td>1</td>
          <td>5.200000e-08</td>
          <td>2.156667e-07</td>
          <td>5.857865</td>
          <td>1.289103e-07</td>
        </tr>
        <tr>
          <th>2011-01-22 04:00:00</th>
          <td>1/22/2011 04:00</td>
          <td>7.598333</td>
          <td>17.429571</td>
          <td>1.446</td>
          <td>17</td>
          <td>1.446</td>
          <td>7.598333</td>
          <td>17.429571</td>
          <td>17.429571</td>
          <td>2</td>
          <td>2.000000e-08</td>
          <td>2.566667e-08</td>
          <td>0.429571</td>
          <td>4.950137e-07</td>
        </tr>
        <tr>
          <th>2011-01-22 05:00:00</th>
          <td>1/22/2011 05:00</td>
          <td>6.801667</td>
          <td>25.819942</td>
          <td>1.464</td>
          <td>18</td>
          <td>1.464</td>
          <td>6.801667</td>
          <td>25.819943</td>
          <td>25.819943</td>
          <td>2</td>
          <td>1.300000e-08</td>
          <td>7.033333e-08</td>
          <td>7.819943</td>
          <td>7.846084e-07</td>
        </tr>
        <tr>
          <th>2011-01-22 06:00:00</th>
          <td>1/22/2011 06:00</td>
          <td>7.570000</td>
          <td>20.417307</td>
          <td>1.402</td>
          <td>14</td>
          <td>1.402</td>
          <td>7.570000</td>
          <td>20.417308</td>
          <td>20.417308</td>
          <td>2</td>
          <td>5.000000e-08</td>
          <td>1.720000e-07</td>
          <td>6.417308</td>
          <td>7.469265e-07</td>
        </tr>
        <tr>
          <th>2011-01-22 07:00:00</th>
          <td>1/22/2011 07:00</td>
          <td>7.678333</td>
          <td>17.365040</td>
          <td>1.336</td>
          <td>9</td>
          <td>1.336</td>
          <td>7.678333</td>
          <td>17.365040</td>
          <td>17.365040</td>
          <td>2</td>
          <td>3.400000e-08</td>
          <td>5.133333e-08</td>
          <td>8.365040</td>
          <td>2.234193e-07</td>
        </tr>
        <tr>
          <th>2011-01-22 08:00:00</th>
          <td>1/22/2011 08:00</td>
          <td>7.535000</td>
          <td>19.518656</td>
          <td>1.274</td>
          <td>7</td>
          <td>1.274</td>
          <td>7.535000</td>
          <td>19.518656</td>
          <td>19.518656</td>
          <td>2</td>
          <td>4.900000e-08</td>
          <td>1.530000e-07</td>
          <td>12.518656</td>
          <td>1.789803e-07</td>
        </tr>
        <tr>
          <th>2011-01-22 09:00:00</th>
          <td>1/22/2011 09:00</td>
          <td>6.678333</td>
          <td>19.565651</td>
          <td>1.352</td>
          <td>4</td>
          <td>1.352</td>
          <td>6.678333</td>
          <td>19.565651</td>
          <td>19.565651</td>
          <td>2</td>
          <td>2.000000e-09</td>
          <td>5.133333e-08</td>
          <td>15.565651</td>
          <td>1.160042e-07</td>
        </tr>
        <tr>
          <th>2011-01-22 10:00:00</th>
          <td>1/22/2011 10:00</td>
          <td>7.853333</td>
          <td>17.493196</td>
          <td>1.642</td>
          <td>3</td>
          <td>1.642</td>
          <td>7.853333</td>
          <td>17.493196</td>
          <td>17.493196</td>
          <td>2</td>
          <td>4.000000e-08</td>
          <td>1.396667e-07</td>
          <td>14.493196</td>
          <td>5.026487e-07</td>
        </tr>
        <tr>
          <th>2011-02-27 18:00:00</th>
          <td>2/27/2011 18:00</td>
          <td>7.801667</td>
          <td>351.760681</td>
          <td>6.788</td>
          <td>360</td>
          <td>6.788</td>
          <td>7.801667</td>
          <td>0.000000</td>
          <td>351.760677</td>
          <td>1</td>
          <td>1.070000e-07</td>
          <td>7.033333e-08</td>
          <td>360.000000</td>
          <td>3.796442e-06</td>
        </tr>
        <tr>
          <th>2011-02-27 21:00:00</th>
          <td>2/27/2011 21:00</td>
          <td>5.066667</td>
          <td>3.769779</td>
          <td>4.456</td>
          <td>360</td>
          <td>4.456</td>
          <td>5.066667</td>
          <td>0.000000</td>
          <td>3.769779</td>
          <td>1</td>
          <td>1.490000e-07</td>
          <td>6.366667e-08</td>
          <td>360.000000</td>
          <td>4.950075e-09</td>
        </tr>
        <tr>
          <th>2011-03-13 01:00:00</th>
          <td>3/13/2011 01:00</td>
          <td>7.468333</td>
          <td>316.468231</td>
          <td>12.797</td>
          <td>360</td>
          <td>12.797</td>
          <td>7.468333</td>
          <td>0.000000</td>
          <td>316.468227</td>
          <td>1</td>
          <td>7.000000e-08</td>
          <td>8.933333e-08</td>
          <td>360.000000</td>
          <td>4.346548e-06</td>
        </tr>
        <tr>
          <th>2011-04-20 23:00:00</th>
          <td>4/20/2011 23:00</td>
          <td>6.230000</td>
          <td>351.476593</td>
          <td>1.454</td>
          <td>3</td>
          <td>1.454</td>
          <td>6.230000</td>
          <td>351.476583</td>
          <td>351.476583</td>
          <td>1</td>
          <td>4.000000e-09</td>
          <td>1.900000e-08</td>
          <td>348.476583</td>
          <td>1.016684e-05</td>
        </tr>
        <tr>
          <th>2011-06-07 11:00:00</th>
          <td>6/7/2011 11:00</td>
          <td>5.288333</td>
          <td>251.171112</td>
          <td>0.395</td>
          <td>144</td>
          <td>0.395</td>
          <td>5.288333</td>
          <td>251.171108</td>
          <td>251.171108</td>
          <td>8</td>
          <td>1.100000e-08</td>
          <td>8.266667e-08</td>
          <td>107.171108</td>
          <td>3.920994e-06</td>
        </tr>
        <tr>
          <th>2011-06-07 15:00:00</th>
          <td>6/7/2011 15:00</td>
          <td>1.800000</td>
          <td>113.500580</td>
          <td>3.366</td>
          <td>360</td>
          <td>3.366</td>
          <td>1.800000</td>
          <td>0.000000</td>
          <td>113.500577</td>
          <td>1</td>
          <td>6.300000e-08</td>
          <td>4.800000e-08</td>
          <td>360.000000</td>
          <td>2.513807e-06</td>
        </tr>
        <tr>
          <th>2011-06-11 18:00:00</th>
          <td>6/11/2011 18:00</td>
          <td>5.353333</td>
          <td>8.936969</td>
          <td>1.271</td>
          <td>343</td>
          <td>1.271</td>
          <td>5.353333</td>
          <td>8.936969</td>
          <td>8.936969</td>
          <td>1</td>
          <td>2.800000e-08</td>
          <td>1.396667e-07</td>
          <td>334.063031</td>
          <td>4.108324e-07</td>
        </tr>
        <tr>
          <th>2011-07-20 09:00:00</th>
          <td>7/20/2011 09:00</td>
          <td>5.906667</td>
          <td>3.887907</td>
          <td>6.890</td>
          <td>360</td>
          <td>6.890</td>
          <td>5.906667</td>
          <td>0.000000</td>
          <td>3.887907</td>
          <td>1</td>
          <td>1.340000e-07</td>
          <td>8.933333e-08</td>
          <td>360.000000</td>
          <td>5.022433e-08</td>
        </tr>
        <tr>
          <th>2011-07-20 17:00:00</th>
          <td>7/20/2011 17:00</td>
          <td>8.635000</td>
          <td>354.235046</td>
          <td>7.558</td>
          <td>360</td>
          <td>7.558</td>
          <td>8.635000</td>
          <td>0.000000</td>
          <td>354.235045</td>
          <td>1</td>
          <td>8.800000e-08</td>
          <td>2.290000e-07</td>
          <td>360.000000</td>
          <td>1.790734e-06</td>
        </tr>
        <tr>
          <th>2012-03-01 20:00:00</th>
          <td>3/1/2012 20:00</td>
          <td>5.718333</td>
          <td>262.674438</td>
          <td>1.564</td>
          <td>312</td>
          <td>1.564</td>
          <td>5.718333</td>
          <td>262.674446</td>
          <td>262.674446</td>
          <td>10</td>
          <td>1.000000e-08</td>
          <td>8.933333e-08</td>
          <td>49.325554</td>
          <td>7.338326e-06</td>
        </tr>
        <tr>
          <th>2012-03-01 21:00:00</th>
          <td>3/1/2012 21:00</td>
          <td>5.481667</td>
          <td>264.543304</td>
          <td>1.221</td>
          <td>259</td>
          <td>1.221</td>
          <td>5.481667</td>
          <td>264.543292</td>
          <td>264.543292</td>
          <td>10</td>
          <td>4.400000e-08</td>
          <td>1.016667e-07</td>
          <td>5.543292</td>
          <td>1.283565e-05</td>
        </tr>
        <tr>
          <th>2012-04-23 23:00:00</th>
          <td>4/23/2012 23:00</td>
          <td>6.285000</td>
          <td>346.167053</td>
          <td>4.610</td>
          <td>360</td>
          <td>4.610</td>
          <td>6.285000</td>
          <td>0.000000</td>
          <td>346.167047</td>
          <td>1</td>
          <td>1.340000e-07</td>
          <td>1.530000e-07</td>
          <td>360.000000</td>
          <td>5.794007e-06</td>
        </tr>
        <tr>
          <th>2012-06-10 17:00:00</th>
          <td>6/10/2012 17:00</td>
          <td>2.995000</td>
          <td>13.691919</td>
          <td>2.587</td>
          <td>360</td>
          <td>2.587</td>
          <td>2.995000</td>
          <td>0.000000</td>
          <td>13.691919</td>
          <td>1</td>
          <td>1.070000e-07</td>
          <td>1.140000e-07</td>
          <td>360.000000</td>
          <td>3.352309e-07</td>
        </tr>
        <tr>
          <th>2012-06-17 19:00:00</th>
          <td>6/17/2012 19:00</td>
          <td>5.776667</td>
          <td>31.216002</td>
          <td>1.075</td>
          <td>335</td>
          <td>1.075</td>
          <td>5.776667</td>
          <td>31.216001</td>
          <td>31.216001</td>
          <td>2</td>
          <td>4.800000e-08</td>
          <td>2.566667e-08</td>
          <td>303.783999</td>
          <td>4.127759e-07</td>
        </tr>
        <tr>
          <th>2012-07-13 22:00:00</th>
          <td>7/13/2012 22:00</td>
          <td>9.631667</td>
          <td>352.922852</td>
          <td>7.072</td>
          <td>360</td>
          <td>7.072</td>
          <td>9.631667</td>
          <td>0.000000</td>
          <td>352.922861</td>
          <td>1</td>
          <td>2.700000e-08</td>
          <td>4.703333e-07</td>
          <td>360.000000</td>
          <td>9.182117e-06</td>
        </tr>
        <tr>
          <th>2012-08-05 00:00:00</th>
          <td>8/5/2012</td>
          <td>10.463333</td>
          <td>358.264404</td>
          <td>7.098</td>
          <td>360</td>
          <td>7.098</td>
          <td>10.463333</td>
          <td>0.000000</td>
          <td>358.264412</td>
          <td>1</td>
          <td>5.000000e-08</td>
          <td>2.033333e-07</td>
          <td>360.000000</td>
          <td>7.853286e-06</td>
        </tr>
        <tr>
          <th>2012-08-07 16:00:00</th>
          <td>8/7/2012 16:00</td>
          <td>3.288333</td>
          <td>16.753290</td>
          <td>3.427</td>
          <td>360</td>
          <td>3.427</td>
          <td>3.288333</td>
          <td>0.000000</td>
          <td>16.753290</td>
          <td>1</td>
          <td>4.600000e-08</td>
          <td>8.266667e-08</td>
          <td>360.000000</td>
          <td>3.026859e-09</td>
        </tr>
        <tr>
          <th>2012-11-26 08:00:00</th>
          <td>11/26/2012 08:00</td>
          <td>11.510000</td>
          <td>356.870331</td>
          <td>12.995</td>
          <td>360</td>
          <td>12.995</td>
          <td>11.510000</td>
          <td>0.000000</td>
          <td>356.870327</td>
          <td>1</td>
          <td>1.100000e-07</td>
          <td>2.300000e-07</td>
          <td>360.000000</td>
          <td>4.168447e-06</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>2013-11-20 19:00:00</th>
          <td>11/20/2013 19:00</td>
          <td>10.505000</td>
          <td>352.348969</td>
          <td>15.169</td>
          <td>360</td>
          <td>15.169</td>
          <td>10.505000</td>
          <td>0.000000</td>
          <td>352.348983</td>
          <td>1</td>
          <td>3.300000e-07</td>
          <td>1.100000e-07</td>
          <td>360.000000</td>
          <td>1.435761e-05</td>
        </tr>
        <tr>
          <th>2013-11-24 10:00:00</th>
          <td>11/24/2013 10:00</td>
          <td>2.351667</td>
          <td>40.884998</td>
          <td>1.878</td>
          <td>360</td>
          <td>1.878</td>
          <td>2.351667</td>
          <td>0.000000</td>
          <td>40.884997</td>
          <td>1</td>
          <td>2.100000e-08</td>
          <td>2.233333e-08</td>
          <td>360.000000</td>
          <td>8.727059e-07</td>
        </tr>
        <tr>
          <th>2014-01-18 13:00:00</th>
          <td>1/18/2014 13:00</td>
          <td>4.430000</td>
          <td>357.259644</td>
          <td>0.178</td>
          <td>248</td>
          <td>0.178</td>
          <td>4.430000</td>
          <td>357.259639</td>
          <td>357.259639</td>
          <td>6</td>
          <td>3.000000e-09</td>
          <td>1.720000e-07</td>
          <td>109.259639</td>
          <td>4.993587e-06</td>
        </tr>
        <tr>
          <th>2014-01-18 14:00:00</th>
          <td>1/18/2014 14:00</td>
          <td>5.671667</td>
          <td>21.540751</td>
          <td>0.801</td>
          <td>351</td>
          <td>0.801</td>
          <td>5.671667</td>
          <td>21.540751</td>
          <td>21.540751</td>
          <td>4</td>
          <td>1.000000e-09</td>
          <td>4.466667e-08</td>
          <td>329.459249</td>
          <td>4.004333e-07</td>
        </tr>
        <tr>
          <th>2014-01-28 18:00:00</th>
          <td>1/28/2014 18:00</td>
          <td>10.631667</td>
          <td>354.281738</td>
          <td>10.588</td>
          <td>360</td>
          <td>10.588</td>
          <td>10.631667</td>
          <td>0.000000</td>
          <td>354.281727</td>
          <td>1</td>
          <td>3.000000e-07</td>
          <td>4.733333e-07</td>
          <td>360.000000</td>
          <td>1.080524e-05</td>
        </tr>
        <tr>
          <th>2014-03-06 17:00:00</th>
          <td>3/6/2014 17:00</td>
          <td>7.960000</td>
          <td>212.689621</td>
          <td>1.184</td>
          <td>211</td>
          <td>1.184</td>
          <td>7.960000</td>
          <td>212.689618</td>
          <td>212.689618</td>
          <td>8</td>
          <td>1.500000e-08</td>
          <td>3.800000e-08</td>
          <td>1.689618</td>
          <td>2.991798e-06</td>
        </tr>
        <tr>
          <th>2014-04-18 21:00:00</th>
          <td>4/18/2014 21:00</td>
          <td>5.306667</td>
          <td>199.452270</td>
          <td>0.193</td>
          <td>255</td>
          <td>0.193</td>
          <td>5.306667</td>
          <td>199.452277</td>
          <td>199.452277</td>
          <td>6</td>
          <td>4.000000e-09</td>
          <td>1.843333e-07</td>
          <td>55.547723</td>
          <td>6.070649e-06</td>
        </tr>
        <tr>
          <th>2014-04-18 22:00:00</th>
          <td>4/18/2014 22:00</td>
          <td>5.178333</td>
          <td>207.008041</td>
          <td>0.588</td>
          <td>124</td>
          <td>0.588</td>
          <td>5.178333</td>
          <td>207.008049</td>
          <td>207.008049</td>
          <td>3</td>
          <td>0.000000e+00</td>
          <td>5.133333e-08</td>
          <td>83.008049</td>
          <td>7.144502e-06</td>
        </tr>
        <tr>
          <th>2014-04-29 15:00:00</th>
          <td>4/29/2014 15:00</td>
          <td>6.530000</td>
          <td>179.562332</td>
          <td>1.815</td>
          <td>203</td>
          <td>1.815</td>
          <td>6.530000</td>
          <td>179.562329</td>
          <td>179.562329</td>
          <td>7</td>
          <td>5.700000e-08</td>
          <td>2.100000e-07</td>
          <td>23.437671</td>
          <td>2.842059e-06</td>
        </tr>
        <tr>
          <th>2014-04-29 16:00:00</th>
          <td>4/29/2014 16:00</td>
          <td>6.983333</td>
          <td>193.170487</td>
          <td>1.562</td>
          <td>206</td>
          <td>1.562</td>
          <td>6.983333</td>
          <td>193.170489</td>
          <td>193.170489</td>
          <td>7</td>
          <td>3.600000e-08</td>
          <td>2.223333e-07</td>
          <td>12.829511</td>
          <td>2.393348e-06</td>
        </tr>
        <tr>
          <th>2014-04-29 17:00:00</th>
          <td>4/29/2014 17:00</td>
          <td>6.641667</td>
          <td>182.656982</td>
          <td>1.226</td>
          <td>200</td>
          <td>1.226</td>
          <td>6.641667</td>
          <td>182.656985</td>
          <td>182.656985</td>
          <td>7</td>
          <td>4.900000e-08</td>
          <td>2.223333e-07</td>
          <td>17.343015</td>
          <td>2.808822e-06</td>
        </tr>
        <tr>
          <th>2014-04-29 18:00:00</th>
          <td>4/29/2014 18:00</td>
          <td>7.518333</td>
          <td>169.829681</td>
          <td>1.169</td>
          <td>168</td>
          <td>1.169</td>
          <td>7.518333</td>
          <td>169.829684</td>
          <td>169.829684</td>
          <td>7</td>
          <td>3.000000e-08</td>
          <td>1.016667e-07</td>
          <td>1.829684</td>
          <td>2.412369e-06</td>
        </tr>
        <tr>
          <th>2014-06-19 07:00:00</th>
          <td>6/19/2014 07:00</td>
          <td>8.073334</td>
          <td>10.565766</td>
          <td>7.423</td>
          <td>360</td>
          <td>7.423</td>
          <td>8.073333</td>
          <td>0.000000</td>
          <td>10.565766</td>
          <td>1</td>
          <td>1.410000e-07</td>
          <td>4.066667e-07</td>
          <td>360.000000</td>
          <td>3.488248e-07</td>
        </tr>
        <tr>
          <th>2014-06-19 09:00:00</th>
          <td>6/19/2014 09:00</td>
          <td>8.953333</td>
          <td>9.143786</td>
          <td>7.463</td>
          <td>360</td>
          <td>7.463</td>
          <td>8.953333</td>
          <td>0.000000</td>
          <td>9.143787</td>
          <td>1</td>
          <td>1.790000e-07</td>
          <td>4.323333e-07</td>
          <td>360.000000</td>
          <td>1.791341e-07</td>
        </tr>
        <tr>
          <th>2014-07-10 08:00:00</th>
          <td>7/10/2014 08:00</td>
          <td>3.555000</td>
          <td>317.186585</td>
          <td>2.763</td>
          <td>360</td>
          <td>2.763</td>
          <td>3.555000</td>
          <td>0.000000</td>
          <td>317.186570</td>
          <td>1</td>
          <td>1.100000e-08</td>
          <td>6.700000e-08</td>
          <td>360.000000</td>
          <td>1.460114e-05</td>
        </tr>
        <tr>
          <th>2014-08-06 06:00:00</th>
          <td>8/6/2014 06:00</td>
          <td>4.660000</td>
          <td>263.196625</td>
          <td>0.627</td>
          <td>115</td>
          <td>0.627</td>
          <td>4.660000</td>
          <td>263.196628</td>
          <td>263.196628</td>
          <td>8</td>
          <td>2.600000e-08</td>
          <td>1.530000e-07</td>
          <td>148.196628</td>
          <td>3.174322e-06</td>
        </tr>
        <tr>
          <th>2014-08-23 22:00:00</th>
          <td>8/23/2014 22:00</td>
          <td>6.260000</td>
          <td>358.868866</td>
          <td>1.358</td>
          <td>327</td>
          <td>1.358</td>
          <td>6.260000</td>
          <td>358.868851</td>
          <td>358.868851</td>
          <td>1</td>
          <td>4.000000e-08</td>
          <td>2.290000e-07</td>
          <td>31.868851</td>
          <td>1.494344e-05</td>
        </tr>
        <tr>
          <th>2014-09-07 22:00:00</th>
          <td>9/7/2014 22:00</td>
          <td>8.095000</td>
          <td>350.695801</td>
          <td>5.284</td>
          <td>360</td>
          <td>5.284</td>
          <td>8.095000</td>
          <td>0.000000</td>
          <td>350.695816</td>
          <td>1</td>
          <td>8.000000e-08</td>
          <td>2.670000e-07</td>
          <td>360.000000</td>
          <td>1.479973e-05</td>
        </tr>
        <tr>
          <th>2014-09-08 08:00:00</th>
          <td>9/8/2014 08:00</td>
          <td>3.891667</td>
          <td>335.527313</td>
          <td>3.918</td>
          <td>360</td>
          <td>3.918</td>
          <td>3.891667</td>
          <td>0.000000</td>
          <td>335.527321</td>
          <td>1</td>
          <td>1.700000e-08</td>
          <td>1.566667e-08</td>
          <td>360.000000</td>
          <td>7.570486e-06</td>
        </tr>
        <tr>
          <th>2014-09-08 09:00:00</th>
          <td>9/8/2014 09:00</td>
          <td>2.426667</td>
          <td>341.957581</td>
          <td>4.044</td>
          <td>360</td>
          <td>4.044</td>
          <td>2.426667</td>
          <td>0.000000</td>
          <td>341.957581</td>
          <td>1</td>
          <td>1.490000e-07</td>
          <td>7.033333e-08</td>
          <td>360.000000</td>
          <td>4.087426e-07</td>
        </tr>
        <tr>
          <th>2014-09-20 02:00:00</th>
          <td>9/20/2014 02:00</td>
          <td>8.878333</td>
          <td>357.488922</td>
          <td>7.890</td>
          <td>360</td>
          <td>7.890</td>
          <td>8.878333</td>
          <td>0.000000</td>
          <td>357.488908</td>
          <td>1</td>
          <td>1.340000e-07</td>
          <td>2.413333e-07</td>
          <td>360.000000</td>
          <td>1.369100e-05</td>
        </tr>
        <tr>
          <th>2014-11-04 07:00:00</th>
          <td>11/4/2014 07:00</td>
          <td>10.248333</td>
          <td>340.986938</td>
          <td>10.629</td>
          <td>360</td>
          <td>10.629</td>
          <td>10.248333</td>
          <td>0.000000</td>
          <td>340.986932</td>
          <td>1</td>
          <td>2.900000e-07</td>
          <td>3.533333e-07</td>
          <td>360.000000</td>
          <td>6.852077e-06</td>
        </tr>
        <tr>
          <th>2015-01-18 22:00:00</th>
          <td>1/18/2015 22:00</td>
          <td>7.791667</td>
          <td>338.369171</td>
          <td>6.688</td>
          <td>360</td>
          <td>6.688</td>
          <td>7.791667</td>
          <td>0.000000</td>
          <td>338.369170</td>
          <td>1</td>
          <td>2.020000e-07</td>
          <td>1.586667e-07</td>
          <td>360.000000</td>
          <td>8.616506e-07</td>
        </tr>
        <tr>
          <th>2015-02-03 02:00:00</th>
          <td>2/3/2015 02:00</td>
          <td>5.178333</td>
          <td>244.490112</td>
          <td>11.484</td>
          <td>360</td>
          <td>11.484</td>
          <td>5.178333</td>
          <td>0.000000</td>
          <td>244.490112</td>
          <td>1</td>
          <td>2.100000e-07</td>
          <td>5.133333e-08</td>
          <td>360.000000</td>
          <td>7.886141e-07</td>
        </tr>
        <tr>
          <th>2015-04-07 20:00:00</th>
          <td>4/7/2015 20:00</td>
          <td>4.546667</td>
          <td>49.758533</td>
          <td>0.309</td>
          <td>168</td>
          <td>0.309</td>
          <td>4.546667</td>
          <td>49.758534</td>
          <td>49.758534</td>
          <td>8</td>
          <td>1.500000e-08</td>
          <td>4.466667e-08</td>
          <td>118.241466</td>
          <td>4.689388e-07</td>
        </tr>
        <tr>
          <th>2015-04-08 23:00:00</th>
          <td>4/8/2015 23:00</td>
          <td>6.326667</td>
          <td>198.659347</td>
          <td>1.865</td>
          <td>259</td>
          <td>1.865</td>
          <td>6.326667</td>
          <td>198.659341</td>
          <td>198.659341</td>
          <td>8</td>
          <td>1.000000e-08</td>
          <td>1.653333e-07</td>
          <td>60.340659</td>
          <td>6.262045e-06</td>
        </tr>
        <tr>
          <th>2015-04-09 00:00:00</th>
          <td>4/9/2015</td>
          <td>5.836667</td>
          <td>207.099670</td>
          <td>1.509</td>
          <td>229</td>
          <td>1.509</td>
          <td>5.836667</td>
          <td>207.099667</td>
          <td>207.099667</td>
          <td>8</td>
          <td>5.600000e-08</td>
          <td>8.266667e-08</td>
          <td>21.900333</td>
          <td>3.703586e-06</td>
        </tr>
        <tr>
          <th>2015-04-24 23:00:00</th>
          <td>4/24/2015 23:00</td>
          <td>6.471667</td>
          <td>352.625244</td>
          <td>6.149</td>
          <td>360</td>
          <td>6.149</td>
          <td>6.471667</td>
          <td>0.000000</td>
          <td>352.625258</td>
          <td>1</td>
          <td>1.680000e-07</td>
          <td>1.463333e-07</td>
          <td>360.000000</td>
          <td>1.385736e-05</td>
        </tr>
        <tr>
          <th>2015-05-05 13:00:00</th>
          <td>5/5/2015 13:00</td>
          <td>11.453333</td>
          <td>235.495499</td>
          <td>1.153</td>
          <td>104</td>
          <td>1.153</td>
          <td>11.453333</td>
          <td>235.495498</td>
          <td>235.495498</td>
          <td>9</td>
          <td>3.000000e-09</td>
          <td>4.333333e-07</td>
          <td>131.495498</td>
          <td>9.436083e-07</td>
        </tr>
        <tr>
          <th>2015-05-05 14:00:00</th>
          <td>5/5/2015 14:00</td>
          <td>11.910000</td>
          <td>237.188995</td>
          <td>0.883</td>
          <td>245</td>
          <td>0.883</td>
          <td>11.910000</td>
          <td>237.188999</td>
          <td>237.188999</td>
          <td>11</td>
          <td>1.600000e-08</td>
          <td>1.500000e-07</td>
          <td>7.811001</td>
          <td>3.152304e-06</td>
        </tr>
      </tbody>
    </table>
    <p>76 rows × 14 columns</p>
    </div>



.. code:: ipython3

    class test_base:
        def check(a):
            print(a)
        def test():
            print("I am in base class")
    class test_child(test_base):
        def print_t(self):
            print('i am printing')
        def test(self):
            print("I am in child class")
            self.print_t();
    obj = test_child()
    obj.test()


.. parsed-literal::

    I am in child class
    i am printing
    

.. code:: ipython3

    for i in speed_sort.data.groupby(['ref_dir_bin']):
        data = pd.read_csv('sector'+str(i[0])+'_test.csv',index_col=False)
        result = pd.concat([i[1].reset_index(),data],axis=1)
        result.to_csv('sector'+str(i[0])+'_test_compare.csv')

.. code:: ipython3

    transform.get_coverage(site_data['A_Avg1'], '1M')




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>A_Avg1</th>
          <th>Count</th>
          <th>Coverage</th>
        </tr>
        <tr>
          <th>TIMESTAMP</th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2011-01-01</th>
          <td>6.271714</td>
          <td>1674</td>
          <td>0.375000</td>
        </tr>
        <tr>
          <th>2011-02-01</th>
          <td>8.261815</td>
          <td>4032</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2011-03-01</th>
          <td>6.237083</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2011-04-01</th>
          <td>7.001463</td>
          <td>4320</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2011-05-01</th>
          <td>10.643631</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2011-06-01</th>
          <td>6.381873</td>
          <td>4320</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2011-07-01</th>
          <td>5.910354</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2011-08-01</th>
          <td>5.820753</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2011-09-01</th>
          <td>9.641574</td>
          <td>4320</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2011-10-01</th>
          <td>8.988519</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2011-11-01</th>
          <td>9.722176</td>
          <td>4320</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2011-12-01</th>
          <td>10.660375</td>
          <td>4421</td>
          <td>0.990367</td>
        </tr>
        <tr>
          <th>2012-01-01</th>
          <td>9.965945</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2012-02-01</th>
          <td>8.544796</td>
          <td>4176</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2012-03-01</th>
          <td>7.635374</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2012-04-01</th>
          <td>7.577502</td>
          <td>4320</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2012-05-01</th>
          <td>6.659218</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2012-06-01</th>
          <td>6.629965</td>
          <td>4320</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2012-07-01</th>
          <td>6.681994</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2012-08-01</th>
          <td>7.233291</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2012-09-01</th>
          <td>8.611248</td>
          <td>4320</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2012-10-01</th>
          <td>6.956897</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2012-11-01</th>
          <td>8.257398</td>
          <td>4320</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2012-12-01</th>
          <td>8.699671</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2013-01-01</th>
          <td>8.746629</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2013-02-01</th>
          <td>7.986039</td>
          <td>4032</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2013-03-01</th>
          <td>8.704713</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2013-04-01</th>
          <td>8.790891</td>
          <td>4320</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2013-05-01</th>
          <td>8.306667</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2013-06-01</th>
          <td>6.471755</td>
          <td>4320</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2013-07-01</th>
          <td>5.398351</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2013-08-01</th>
          <td>6.909590</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2013-09-01</th>
          <td>7.475595</td>
          <td>4320</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2013-10-01</th>
          <td>8.129203</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2013-11-01</th>
          <td>7.671782</td>
          <td>4320</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2013-12-01</th>
          <td>10.977363</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2014-01-01</th>
          <td>8.978392</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2014-02-01</th>
          <td>10.497240</td>
          <td>4032</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2014-03-01</th>
          <td>8.817621</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2014-04-01</th>
          <td>7.168616</td>
          <td>4320</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2014-05-01</th>
          <td>6.652160</td>
          <td>4463</td>
          <td>0.999776</td>
        </tr>
        <tr>
          <th>2014-06-01</th>
          <td>5.944350</td>
          <td>4320</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2014-07-01</th>
          <td>6.227993</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2014-08-01</th>
          <td>7.773627</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2014-09-01</th>
          <td>5.501780</td>
          <td>4320</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2014-10-01</th>
          <td>8.626987</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2014-11-01</th>
          <td>7.019639</td>
          <td>4320</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2014-12-01</th>
          <td>9.287343</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2015-01-01</th>
          <td>10.514998</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2015-02-01</th>
          <td>8.207768</td>
          <td>4032</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2015-03-01</th>
          <td>9.474996</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2015-04-01</th>
          <td>6.517377</td>
          <td>4320</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2015-05-01</th>
          <td>8.430623</td>
          <td>4464</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2015-06-01</th>
          <td>7.656105</td>
          <td>2611</td>
          <td>0.604398</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    test1, test2=bw._preprocess_dir_data_for_correlations(ref['WS50m_ms'], ref['WD50m_deg'], site_data['A_Avg1'],
                              site_data['WindDir_AVG'],'5H', 0)
    test1


.. parsed-literal::

                             E  Count
    DateTime                         
    2011-01-20 00:00:00  197.0      5
    2011-01-20 05:00:00  204.0      5
    2011-01-20 10:00:00  207.0      5
    2011-01-20 15:00:00  216.0      5
    2011-01-20 20:00:00  221.0      5
    2011-01-21 01:00:00  224.0      5
    2011-01-21 06:00:00  223.0      5
    2011-01-21 11:00:00  234.0      5
    2011-01-21 16:00:00  281.0      5
    2011-01-21 21:00:00  302.0      5
    2011-01-22 02:00:00   10.0      5
    2011-01-22 07:00:00    5.0      5
    2011-01-22 12:00:00    9.0      5
    2011-01-22 17:00:00   23.0      5
    2011-01-22 22:00:00   18.0      5
    2011-01-23 03:00:00   21.0      5
    2011-01-23 08:00:00   22.0      5
    2011-01-23 13:00:00  346.0      5
    2011-01-23 18:00:00  338.0      5
    2011-01-23 23:00:00  303.0      5
    2011-01-24 04:00:00  297.0      5
    2011-01-24 09:00:00  294.0      5
    2011-01-24 14:00:00  300.0      5
    2011-01-24 19:00:00  304.0      5
    2011-01-25 00:00:00  300.0      5
    2011-01-25 05:00:00  311.0      5
    2011-01-25 10:00:00  329.0      5
    2011-01-25 15:00:00  336.0      5
    2011-01-25 20:00:00  356.0      5
    2011-01-26 01:00:00   20.0      5
    ...                    ...    ...
    2018-06-24 22:00:00  252.0      5
    2018-06-25 03:00:00  242.0      5
    2018-06-25 08:00:00  259.0      5
    2018-06-25 13:00:00  232.0      5
    2018-06-25 18:00:00   16.0      5
    2018-06-25 23:00:00   96.0      5
    2018-06-26 04:00:00  142.0      5
    2018-06-26 09:00:00  141.0      5
    2018-06-26 14:00:00  157.0      5
    2018-06-26 19:00:00  155.0      5
    2018-06-27 00:00:00  173.0      5
    2018-06-27 05:00:00  182.0      5
    2018-06-27 10:00:00  174.0      5
    2018-06-27 15:00:00   34.0      5
    2018-06-27 20:00:00   60.0      5
    2018-06-28 01:00:00  130.0      5
    2018-06-28 06:00:00   11.0      5
    2018-06-28 11:00:00   20.0      5
    2018-06-28 16:00:00   21.0      5
    2018-06-28 21:00:00   27.0      5
    2018-06-29 02:00:00   35.0      5
    2018-06-29 07:00:00   54.0      5
    2018-06-29 12:00:00   80.0      5
    2018-06-29 17:00:00  109.0      5
    2018-06-29 22:00:00  148.0      5
    2018-06-30 03:00:00  147.0      5
    2018-06-30 08:00:00  152.0      5
    2018-06-30 13:00:00  152.0      5
    2018-06-30 18:00:00  217.0      5
    2018-06-30 23:00:00  286.0      1
    
    [13052 rows x 2 columns]
                             E  Count
    TIMESTAMP                        
    2011-01-20 05:00:00  222.0      6
    2011-01-20 10:00:00  223.0     30
    2011-01-20 15:00:00  246.0     30
    2011-01-20 20:00:00  264.0     30
    2011-01-21 01:00:00  271.0     30
    2011-01-21 06:00:00  349.0     30
    2011-01-21 11:00:00  328.0     30
    2011-01-21 16:00:00    1.0     30
    2011-01-21 21:00:00   24.0     30
    2011-01-22 02:00:00   18.0     30
    2011-01-22 07:00:00   18.0     30
    2011-01-22 12:00:00    8.0     30
    2011-01-22 17:00:00   12.0     30
    2011-01-22 22:00:00   20.0     30
    2011-01-23 03:00:00   27.0     30
    2011-01-23 08:00:00   31.0     30
    2011-01-23 13:00:00    2.0     30
    2011-01-23 18:00:00  337.0     30
    2011-01-23 23:00:00  318.0     30
    2011-01-24 04:00:00  291.0     30
    2011-01-24 09:00:00  287.0     30
    2011-01-24 14:00:00  301.0     30
    2011-01-24 19:00:00  295.0     30
    2011-01-25 00:00:00  308.0     30
    2011-01-25 05:00:00  311.0     30
    2011-01-25 10:00:00  327.0     30
    2011-01-25 15:00:00  336.0     30
    2011-01-25 20:00:00  357.0     30
    2011-01-26 01:00:00   10.0     30
    2011-01-26 06:00:00   21.0     30
    ...                    ...    ...
    2015-06-13 00:00:00    1.0     30
    2015-06-13 05:00:00    5.0     30
    2015-06-13 10:00:00  353.0     30
    2015-06-13 15:00:00  351.0     30
    2015-06-13 20:00:00  346.0     30
    2015-06-14 01:00:00  358.0     30
    2015-06-14 06:00:00    2.0     30
    2015-06-14 11:00:00   11.0     30
    2015-06-14 16:00:00   16.0     30
    2015-06-14 21:00:00   31.0     30
    2015-06-15 02:00:00  146.0     30
    2015-06-15 07:00:00  205.0     30
    2015-06-15 12:00:00  205.0     30
    2015-06-15 17:00:00  256.0     30
    2015-06-15 22:00:00  261.0     30
    2015-06-16 03:00:00  215.0     30
    2015-06-16 08:00:00  205.0     30
    2015-06-16 13:00:00  214.0     30
    2015-06-16 18:00:00  225.0     30
    2015-06-16 23:00:00  251.0     30
    2015-06-17 04:00:00  251.0     30
    2015-06-17 09:00:00  276.0     30
    2015-06-17 14:00:00  287.0     30
    2015-06-17 19:00:00  289.0     30
    2015-06-18 00:00:00  284.0     30
    2015-06-18 05:00:00  287.0     30
    2015-06-18 10:00:00  279.0     30
    2015-06-18 15:00:00  277.0     30
    2015-06-18 20:00:00  287.0     30
    2015-06-19 01:00:00  286.0     13
    
    [7733 rows x 2 columns]
    



.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>E</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2011-01-20 05:00:00</th>
          <td>204.0</td>
        </tr>
        <tr>
          <th>2011-01-20 10:00:00</th>
          <td>207.0</td>
        </tr>
        <tr>
          <th>2011-01-20 15:00:00</th>
          <td>216.0</td>
        </tr>
        <tr>
          <th>2011-01-20 20:00:00</th>
          <td>221.0</td>
        </tr>
        <tr>
          <th>2011-01-21 01:00:00</th>
          <td>224.0</td>
        </tr>
        <tr>
          <th>2011-01-21 06:00:00</th>
          <td>223.0</td>
        </tr>
        <tr>
          <th>2011-01-21 11:00:00</th>
          <td>234.0</td>
        </tr>
        <tr>
          <th>2011-01-21 16:00:00</th>
          <td>281.0</td>
        </tr>
        <tr>
          <th>2011-01-21 21:00:00</th>
          <td>302.0</td>
        </tr>
        <tr>
          <th>2011-01-22 02:00:00</th>
          <td>10.0</td>
        </tr>
        <tr>
          <th>2011-01-22 07:00:00</th>
          <td>5.0</td>
        </tr>
        <tr>
          <th>2011-01-22 12:00:00</th>
          <td>9.0</td>
        </tr>
        <tr>
          <th>2011-01-22 17:00:00</th>
          <td>23.0</td>
        </tr>
        <tr>
          <th>2011-01-22 22:00:00</th>
          <td>18.0</td>
        </tr>
        <tr>
          <th>2011-01-23 03:00:00</th>
          <td>21.0</td>
        </tr>
        <tr>
          <th>2011-01-23 08:00:00</th>
          <td>22.0</td>
        </tr>
        <tr>
          <th>2011-01-23 13:00:00</th>
          <td>346.0</td>
        </tr>
        <tr>
          <th>2011-01-23 18:00:00</th>
          <td>338.0</td>
        </tr>
        <tr>
          <th>2011-01-23 23:00:00</th>
          <td>303.0</td>
        </tr>
        <tr>
          <th>2011-01-24 04:00:00</th>
          <td>297.0</td>
        </tr>
        <tr>
          <th>2011-01-24 09:00:00</th>
          <td>294.0</td>
        </tr>
        <tr>
          <th>2011-01-24 14:00:00</th>
          <td>300.0</td>
        </tr>
        <tr>
          <th>2011-01-24 19:00:00</th>
          <td>304.0</td>
        </tr>
        <tr>
          <th>2011-01-25 00:00:00</th>
          <td>300.0</td>
        </tr>
        <tr>
          <th>2011-01-25 05:00:00</th>
          <td>311.0</td>
        </tr>
        <tr>
          <th>2011-01-25 10:00:00</th>
          <td>329.0</td>
        </tr>
        <tr>
          <th>2011-01-25 15:00:00</th>
          <td>336.0</td>
        </tr>
        <tr>
          <th>2011-01-25 20:00:00</th>
          <td>356.0</td>
        </tr>
        <tr>
          <th>2011-01-26 01:00:00</th>
          <td>20.0</td>
        </tr>
        <tr>
          <th>2011-01-26 06:00:00</th>
          <td>30.0</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
        </tr>
        <tr>
          <th>2015-06-13 00:00:00</th>
          <td>4.0</td>
        </tr>
        <tr>
          <th>2015-06-13 05:00:00</th>
          <td>5.0</td>
        </tr>
        <tr>
          <th>2015-06-13 10:00:00</th>
          <td>356.0</td>
        </tr>
        <tr>
          <th>2015-06-13 15:00:00</th>
          <td>347.0</td>
        </tr>
        <tr>
          <th>2015-06-13 20:00:00</th>
          <td>353.0</td>
        </tr>
        <tr>
          <th>2015-06-14 01:00:00</th>
          <td>6.0</td>
        </tr>
        <tr>
          <th>2015-06-14 06:00:00</th>
          <td>13.0</td>
        </tr>
        <tr>
          <th>2015-06-14 11:00:00</th>
          <td>19.0</td>
        </tr>
        <tr>
          <th>2015-06-14 16:00:00</th>
          <td>34.0</td>
        </tr>
        <tr>
          <th>2015-06-14 21:00:00</th>
          <td>68.0</td>
        </tr>
        <tr>
          <th>2015-06-15 02:00:00</th>
          <td>124.0</td>
        </tr>
        <tr>
          <th>2015-06-15 07:00:00</th>
          <td>192.0</td>
        </tr>
        <tr>
          <th>2015-06-15 12:00:00</th>
          <td>237.0</td>
        </tr>
        <tr>
          <th>2015-06-15 17:00:00</th>
          <td>240.0</td>
        </tr>
        <tr>
          <th>2015-06-15 22:00:00</th>
          <td>200.0</td>
        </tr>
        <tr>
          <th>2015-06-16 03:00:00</th>
          <td>198.0</td>
        </tr>
        <tr>
          <th>2015-06-16 08:00:00</th>
          <td>199.0</td>
        </tr>
        <tr>
          <th>2015-06-16 13:00:00</th>
          <td>212.0</td>
        </tr>
        <tr>
          <th>2015-06-16 18:00:00</th>
          <td>234.0</td>
        </tr>
        <tr>
          <th>2015-06-16 23:00:00</th>
          <td>241.0</td>
        </tr>
        <tr>
          <th>2015-06-17 04:00:00</th>
          <td>252.0</td>
        </tr>
        <tr>
          <th>2015-06-17 09:00:00</th>
          <td>274.0</td>
        </tr>
        <tr>
          <th>2015-06-17 14:00:00</th>
          <td>284.0</td>
        </tr>
        <tr>
          <th>2015-06-17 19:00:00</th>
          <td>283.0</td>
        </tr>
        <tr>
          <th>2015-06-18 00:00:00</th>
          <td>285.0</td>
        </tr>
        <tr>
          <th>2015-06-18 05:00:00</th>
          <td>282.0</td>
        </tr>
        <tr>
          <th>2015-06-18 10:00:00</th>
          <td>279.0</td>
        </tr>
        <tr>
          <th>2015-06-18 15:00:00</th>
          <td>281.0</td>
        </tr>
        <tr>
          <th>2015-06-18 20:00:00</th>
          <td>283.0</td>
        </tr>
        <tr>
          <th>2015-06-19 01:00:00</th>
          <td>283.0</td>
        </tr>
      </tbody>
    </table>
    <p>7733 rows × 1 columns</p>
    </div>



.. code:: ipython3

    bw._preprocess_data_for_correlations(ref.loc['2006':,'WS50m_ms'], ref.loc['2006':,'WS50m_ms'],'6M',0)




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>WS50m_ms</th>
          <th>Count</th>
          <th>Coverage</th>
        </tr>
        <tr>
          <th>DateTime</th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2006-01-01</th>
          <td>8.521690</td>
          <td>4344</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2006-07-01</th>
          <td>9.020437</td>
          <td>4416</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2007-01-01</th>
          <td>8.980777</td>
          <td>4344</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2007-07-01</th>
          <td>8.521915</td>
          <td>4416</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2008-01-01</th>
          <td>9.322370</td>
          <td>4368</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2008-07-01</th>
          <td>8.654504</td>
          <td>4416</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2009-01-01</th>
          <td>8.843457</td>
          <td>4344</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2009-07-01</th>
          <td>8.675714</td>
          <td>4416</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2010-01-01</th>
          <td>7.109253</td>
          <td>4344</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2010-07-01</th>
          <td>8.206018</td>
          <td>4416</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2011-01-01</th>
          <td>8.497811</td>
          <td>4344</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2011-07-01</th>
          <td>9.415128</td>
          <td>4416</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2012-01-01</th>
          <td>8.474168</td>
          <td>4368</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2012-07-01</th>
          <td>8.311534</td>
          <td>4416</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2013-01-01</th>
          <td>8.777592</td>
          <td>4344</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2013-07-01</th>
          <td>8.755722</td>
          <td>4416</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2014-01-01</th>
          <td>8.549137</td>
          <td>4344</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2014-07-01</th>
          <td>8.166914</td>
          <td>4416</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2015-01-01</th>
          <td>9.447409</td>
          <td>4344</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2015-07-01</th>
          <td>8.829938</td>
          <td>4416</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-01-01</th>
          <td>8.364278</td>
          <td>4368</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2016-07-01</th>
          <td>8.455631</td>
          <td>4416</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2017-01-01</th>
          <td>8.593398</td>
          <td>4344</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2017-07-01</th>
          <td>8.692274</td>
          <td>4416</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2018-01-01</th>
          <td>8.514523</td>
          <td>4344</td>
          <td>1.0</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    pd.concat([ref['WD50m_deg'],transform.offset_wind_direction(ref['WD50m_deg'],-122)],axis=1)




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>WD50m_deg</th>
          <th>WD50m_deg</th>
        </tr>
        <tr>
          <th>DateTime</th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1996-01-01 00:00:00</th>
          <td>121.0</td>
          <td>359.0</td>
        </tr>
        <tr>
          <th>1996-01-01 01:00:00</th>
          <td>122.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>1996-01-01 02:00:00</th>
          <td>123.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1996-01-01 03:00:00</th>
          <td>124.0</td>
          <td>2.0</td>
        </tr>
        <tr>
          <th>1996-01-01 04:00:00</th>
          <td>125.0</td>
          <td>3.0</td>
        </tr>
        <tr>
          <th>1996-01-01 05:00:00</th>
          <td>134.0</td>
          <td>12.0</td>
        </tr>
        <tr>
          <th>1996-01-01 06:00:00</th>
          <td>140.0</td>
          <td>18.0</td>
        </tr>
        <tr>
          <th>1996-01-01 07:00:00</th>
          <td>146.0</td>
          <td>24.0</td>
        </tr>
        <tr>
          <th>1996-01-01 08:00:00</th>
          <td>152.0</td>
          <td>30.0</td>
        </tr>
        <tr>
          <th>1996-01-01 09:00:00</th>
          <td>155.0</td>
          <td>33.0</td>
        </tr>
        <tr>
          <th>1996-01-01 10:00:00</th>
          <td>157.0</td>
          <td>35.0</td>
        </tr>
        <tr>
          <th>1996-01-01 11:00:00</th>
          <td>161.0</td>
          <td>39.0</td>
        </tr>
        <tr>
          <th>1996-01-01 12:00:00</th>
          <td>164.0</td>
          <td>42.0</td>
        </tr>
        <tr>
          <th>1996-01-01 13:00:00</th>
          <td>164.0</td>
          <td>42.0</td>
        </tr>
        <tr>
          <th>1996-01-01 14:00:00</th>
          <td>166.0</td>
          <td>44.0</td>
        </tr>
        <tr>
          <th>1996-01-01 15:00:00</th>
          <td>168.0</td>
          <td>46.0</td>
        </tr>
        <tr>
          <th>1996-01-01 16:00:00</th>
          <td>168.0</td>
          <td>46.0</td>
        </tr>
        <tr>
          <th>1996-01-01 17:00:00</th>
          <td>169.0</td>
          <td>47.0</td>
        </tr>
        <tr>
          <th>1996-01-01 18:00:00</th>
          <td>170.0</td>
          <td>48.0</td>
        </tr>
        <tr>
          <th>1996-01-01 19:00:00</th>
          <td>171.0</td>
          <td>49.0</td>
        </tr>
        <tr>
          <th>1996-01-01 20:00:00</th>
          <td>170.0</td>
          <td>48.0</td>
        </tr>
        <tr>
          <th>1996-01-01 21:00:00</th>
          <td>169.0</td>
          <td>47.0</td>
        </tr>
        <tr>
          <th>1996-01-01 22:00:00</th>
          <td>169.0</td>
          <td>47.0</td>
        </tr>
        <tr>
          <th>1996-01-01 23:00:00</th>
          <td>169.0</td>
          <td>47.0</td>
        </tr>
        <tr>
          <th>1996-01-02 00:00:00</th>
          <td>169.0</td>
          <td>47.0</td>
        </tr>
        <tr>
          <th>1996-01-02 01:00:00</th>
          <td>168.0</td>
          <td>46.0</td>
        </tr>
        <tr>
          <th>1996-01-02 02:00:00</th>
          <td>166.0</td>
          <td>44.0</td>
        </tr>
        <tr>
          <th>1996-01-02 03:00:00</th>
          <td>167.0</td>
          <td>45.0</td>
        </tr>
        <tr>
          <th>1996-01-02 04:00:00</th>
          <td>171.0</td>
          <td>49.0</td>
        </tr>
        <tr>
          <th>1996-01-02 05:00:00</th>
          <td>173.0</td>
          <td>51.0</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>2018-06-29 18:00:00</th>
          <td>97.0</td>
          <td>335.0</td>
        </tr>
        <tr>
          <th>2018-06-29 19:00:00</th>
          <td>105.0</td>
          <td>343.0</td>
        </tr>
        <tr>
          <th>2018-06-29 20:00:00</th>
          <td>117.0</td>
          <td>355.0</td>
        </tr>
        <tr>
          <th>2018-06-29 21:00:00</th>
          <td>129.0</td>
          <td>7.0</td>
        </tr>
        <tr>
          <th>2018-06-29 22:00:00</th>
          <td>142.0</td>
          <td>20.0</td>
        </tr>
        <tr>
          <th>2018-06-29 23:00:00</th>
          <td>149.0</td>
          <td>27.0</td>
        </tr>
        <tr>
          <th>2018-06-30 00:00:00</th>
          <td>152.0</td>
          <td>30.0</td>
        </tr>
        <tr>
          <th>2018-06-30 01:00:00</th>
          <td>151.0</td>
          <td>29.0</td>
        </tr>
        <tr>
          <th>2018-06-30 02:00:00</th>
          <td>147.0</td>
          <td>25.0</td>
        </tr>
        <tr>
          <th>2018-06-30 03:00:00</th>
          <td>145.0</td>
          <td>23.0</td>
        </tr>
        <tr>
          <th>2018-06-30 04:00:00</th>
          <td>146.0</td>
          <td>24.0</td>
        </tr>
        <tr>
          <th>2018-06-30 05:00:00</th>
          <td>148.0</td>
          <td>26.0</td>
        </tr>
        <tr>
          <th>2018-06-30 06:00:00</th>
          <td>149.0</td>
          <td>27.0</td>
        </tr>
        <tr>
          <th>2018-06-30 07:00:00</th>
          <td>150.0</td>
          <td>28.0</td>
        </tr>
        <tr>
          <th>2018-06-30 08:00:00</th>
          <td>153.0</td>
          <td>31.0</td>
        </tr>
        <tr>
          <th>2018-06-30 09:00:00</th>
          <td>154.0</td>
          <td>32.0</td>
        </tr>
        <tr>
          <th>2018-06-30 10:00:00</th>
          <td>153.0</td>
          <td>31.0</td>
        </tr>
        <tr>
          <th>2018-06-30 11:00:00</th>
          <td>151.0</td>
          <td>29.0</td>
        </tr>
        <tr>
          <th>2018-06-30 12:00:00</th>
          <td>148.0</td>
          <td>26.0</td>
        </tr>
        <tr>
          <th>2018-06-30 13:00:00</th>
          <td>146.0</td>
          <td>24.0</td>
        </tr>
        <tr>
          <th>2018-06-30 14:00:00</th>
          <td>146.0</td>
          <td>24.0</td>
        </tr>
        <tr>
          <th>2018-06-30 15:00:00</th>
          <td>149.0</td>
          <td>27.0</td>
        </tr>
        <tr>
          <th>2018-06-30 16:00:00</th>
          <td>156.0</td>
          <td>34.0</td>
        </tr>
        <tr>
          <th>2018-06-30 17:00:00</th>
          <td>164.0</td>
          <td>42.0</td>
        </tr>
        <tr>
          <th>2018-06-30 18:00:00</th>
          <td>174.0</td>
          <td>52.0</td>
        </tr>
        <tr>
          <th>2018-06-30 19:00:00</th>
          <td>189.0</td>
          <td>67.0</td>
        </tr>
        <tr>
          <th>2018-06-30 20:00:00</th>
          <td>214.0</td>
          <td>92.0</td>
        </tr>
        <tr>
          <th>2018-06-30 21:00:00</th>
          <td>242.0</td>
          <td>120.0</td>
        </tr>
        <tr>
          <th>2018-06-30 22:00:00</th>
          <td>264.0</td>
          <td>142.0</td>
        </tr>
        <tr>
          <th>2018-06-30 23:00:00</th>
          <td>286.0</td>
          <td>164.0</td>
        </tr>
      </tbody>
    </table>
    <p>197208 rows × 2 columns</p>
    </div>


