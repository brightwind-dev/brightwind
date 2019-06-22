.. _api_ref:



API reference
=============

Load
-----

.. currentmodule:: brightwind.load.load

.. autosummary::
    :toctree: generated

    load_csv
    load_campbell_scientific
    load_excel
    load_windographer_txt
    load_brightdata
    load_cleaning_file
    apply_cleaning
    apply_cleaning_windographer

Analyse
---------

.. currentmodule:: brightwind.analyse.analyse

.. autosummary::
    :toctree: generated

    time_continuity_gaps
    basic_stats
    coverage
    concurrent_coverage
    monthly_means
    momm
    sector_ratio
    dist
    dist_of_wind_speed
    distribution_by_dir_sector
    dist_12x24
    freq_distribution
    freq_table
    TI
    Shear
    calc_air_density

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
    offset_timestamps


Export
-----------

.. currentmodule:: brightwind.export.export

.. autosummary::
    :toctree: generated

    export_csv
    export_tab_file


Datasets
---------

.. currentmodule:: brightwind.datasets

.. autosummary::
    :toctree: generated

    datasets_available