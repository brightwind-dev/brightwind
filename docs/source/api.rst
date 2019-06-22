.. _api_ref:



API reference
=============

Datasets
---------

.. currentmodule:: brightwind.datasets

.. autosummary::
    :toctree: generated

    datasets_available

Load
-----

.. currentmodule:: brightwind.load.load

.. autosummary::
    :toctree: generated

    load_csv
    load_campbell_scientific
    load_excel
    load_brightdata

Analyse
---------

.. currentmodule:: brightwind.analyse.analyse

.. autosummary::
    :toctree: generated

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
    wspd_ratio_by_dir_sector
    Shear

.. currentmodule:: brightwind.analyse.plot

.. autosummary::
    :toctree: generated

    plot_timeseries


Correlation
------------

.. currentmodule:: brightwind.analyse.correlation

.. autosummary::
    :toctree: generated

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
    :toctree: generated

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
    :toctree: generated

    export_csv
    export_tab_file