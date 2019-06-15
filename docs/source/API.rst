.. _api_ref:



API reference
=============

Datasets
---------

.. currentmodule:: brightwind.datasets

.. autosummary::
    :toctree:

    datasets_available

Load
-----

.. currentmodule:: brightwind.load.load

.. autosummary::
    :toctree:

    load_csv
    load_campbell_scientific
    load_excel
    load_brightdata

Analyse
---------

.. currentmodule:: brightwind.analyse.analyse

.. autosummary::
    :toctree:

    concurrent_coverage
    monthly_means
    momm
    distribution
    distribution_by_wind_speed
    distribution_by_dir_sector
    freq_table
    time_continuity_gaps
    coverage
    basic_stats
    twelve_by_24
    TI
    SectorRatio
    Shear

.. currentmodule:: brightwind.analyse.plot

.. autosummary::
    :toctree:

    plot_timeseries


Correlation
------------

.. currentmodule:: brightwind.analyse.correlation

.. autosummary::
    :toctree:

    OrdinaryLeastSquares
    OrthogonalLeastSquares
    MultipleLinearRegression
    SimpleSpeedRatio
    SpeedSort
    SVR

Transform
-----------

.. currentmodule:: brightwind.transform.transform

.. autosummary::
    :toctree:

    average_data_by_period
    adjust_slope_offset
    scale_wind_speed
    offset_wind_direction
    _get_data_resolution
    offset_timestamps


Export
-----------

.. currentmodule:: brightwind.export.export

.. autosummary::
    :toctree:

    export_tab_file