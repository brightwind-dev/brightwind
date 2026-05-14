# Changelog – Brightwind Python Library
All notable changes to this project will be documented in this file. If you make a notable change to the project, please add a line describing the change to the "unreleased" section. The maintainers will make an effort to keep the [Github Releases](https://github.com/brightwind-dev/brightwind/releases) page up to date with this changelog. The format is based on [Semantic Versioning](https://semver.org/) e.g. '1.1.0'

Given a version number MAJOR.MINOR.PATCH, increment the:

    1. MAJOR version when you make incompatible API changes,
    2. MINOR version when you add functionality in a backwards compatible manner, and
    3. PATCH version when you make backwards compatible bug fixes.

Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.

---
## [2.7.0]
XX-May-2026

### New Features and Enhancements
1. Updated `apply_cleaning_rules()` to support nested boolean conditions (`and`/`or`/`not`) inside the `conditions` block and a new optional `time_range_conditions` block on cleaning rules, in line with the updated BrightHub `/measurement-locations/{uuid}/cleaning-rules` API response. Existing flat-condition cleaning rule files continue to work unchanged. The new `measurement_point_uuid` and `statistic_type_id` fields on `conditions` and `clean_out` items are accepted but currently ignored — column resolution still relies on `assembled_column_name`. The `cleaning_rule.schema.json` was extended with a `time_condition` definition.
1. Rewrote `README.md` and `contributing.md` to align with the [docs site](https://brightwind-dev.github.io/brightwind-docs/), with clearer install options (venv / conda), a quick-start example, and an expanded contributing guide covering the fork workflow, editable dev install and running the test suite. ([#602](https://github.com/brightwind-dev/brightwind/issues/602))
2. `test_load_brighthub` is now skipped automatically when no BrightHub credentials are available — either `BRIGHTHUB_CLIENT_ID` and `BRIGHTHUB_CLIENT_SECRET`, or the deprecated `BRIGHTHUB_EMAIL` and `BRIGHTHUB_PASSWORD` — so the test suite runs cleanly without BrightHub credentials. ([#602](https://github.com/brightwind-dev/brightwind/issues/602))
1. Added support for reading Parquet files from BrightHub in `LoadBrightHub.get_data()` via a new `file_extension` argument (one of `'.csv'` or `'.parquet'`). Default remains `'.csv'` for backwards compatibility; Parquet is expected to become the default in the next major release. Reading Parquet requires a parquet engine — install with `pip install brightwind[parquet]` (pyarrow) or `pip install brightwind[parquet-fastparquet]` (fastparquet). ([#601](https://github.com/brightwind-dev/brightwind/issues/601))

### Bug Fixes
1. 

### Deprecated
1. The implicit default of `file_extension='.csv'` in `LoadBrightHub.get_data()` is deprecated and will change to `'.parquet'` in the next major release. Calls that do not pass `file_extension` explicitly will emit a `DeprecationWarning`. Pass `file_extension='.csv'` to keep the current behaviour, or `file_extension='.parquet'` to opt in early. ([#601](https://github.com/brightwind-dev/brightwind/issues/601))

---
## [2.6.0]
02-Apr-2026

### New Features and Enhancements
1. Updated `apply_wspd_slope_offset_adj()` in order to also apply slope and offset adjustments to related columns (for 'avg', 'min', 'max', 'gust', 'median', 'mode', 'range' and 'sd' statistic types) in addition to the main variable itself. It should be noted given lack of required information available no correction is applied to 'ti' 'ti30sec' or 'sum'. ([#575](https://github.com/brightwind-dev/brightwind/issues/575))
1. Updated `MeasurementStation` function `_mast_section_geometry()` to extract mast section geometry properties. This is called by a user with `MeasurementStation.mast_section_geometry`. ([#362](https://github.com/brightwind-dev/brightwind/issues/362))
1. Added `export_tws_file()` to allow export of a WindSim .tws formated timeseries climatology file. ([#305](https://github.com/brightwind-dev/brightwind/issues/305))
1. Updated `plot_scatter()`, `plot_scatter_wdir()`, `plot_scatter_wspd()`, `CorrelBase.plot()` and `SpeedSort.plot_wind_directions()`, to accept an `ax` parameter to support use in subplots. ([#596](https://github.com/brightwind-dev/brightwind/issues/596))
1. Updated `scale_air_pressure_to_height()` to accept air temperature and air pressure time series at different heights, the user can now specify a separate height for the reference temperature time series. This is then used to scale the time series to the target height before scaled air pressure is calculated. ([#562](https://github.com/brightwind-dev/brightwind/issues/562))

### Bug Fixes
1. Fixed two bugs on `SpeedSort.synthesize()` accounting for NaNs and when direction is 360. ([#406](https://github.com/brightwind-dev/brightwind/issues/406))

---
## [2.5.0]
12-Feb-2026

### New Features and Enhancements
1. Updated `calc_air_density()` to incorporate additional methods. The methods differ in their approach to deriving water vapour pressure using `_calc_water_vapour_pressure_Pa()` and include the pre-existing 'IEC' method (default). The two new methods are based on the Herman Wobus approximation (`_calc_water_saturation_vapour_pressure_Pa()`) and require either relative humidity or dew point temperature as inputs. ([#539](https://github.com/brightwind-dev/brightwind/issues/539)).
2. Added `calc_rel_humidity_from_dew_point()` to calculate relative humidity from air temperature and dew point temperature. ([#581](https://github.com/brightwind-dev/brightwind/issues/581)).
3. Updated `MeasurementStation` function `__get_properties()` in order to raise child properties when flattening the dictionary to include sub-lists and sub-dictionaries when raising the following parts of the data model 'mast_properties', 'logger_measurement_config', 'sensor' and 'mounting_arrangement'. This is called by a user with `MeasurementStation.properties` or `MeasurementStation.measurements.properties` ([#576](https://github.com/brightwind-dev/brightwind/issues/576))
4. Updated `apply_wind_vane_deadband_offset()` to use same core function than `apply_device_orientation_offset()` for the adjustment. ([#498](https://github.com/brightwind-dev/brightwind/issues/498))
5. Updated `apply_device_orientation_offset()` and `apply_wind_vane_deadband_offset()` in order to also apply directional adjustment to related columns (for 'avg', 'min', 'max' and 'gust' statistic types) in addition to the main variable itself. ([#574](https://github.com/brightwind-dev/brightwind/issues/574)).

### Bug Fixes
1. Fixed two bugs on `apply_device_orientation_offset()` reported table and prints. ([#571](https://github.com/brightwind-dev/brightwind/issues/571))
2. Fixed bug on `apply_wind_vane_deadband_offset()` reported table and prints. ([#569](https://github.com/brightwind-dev/brightwind/issues/569))

---
## [2.4.0]

### New Features and Enhancements
1. Added `scale_air_pressure_to_height()` to output an air pressure value for any height based on reference air temperature and air pressure values at a different measurement height. ([#531](https://github.com/brightwind-dev/brightwind/issues/531))
2. Added `scale_air_density_to_height()` to output an air density value for any height by applying a constant lapse rate to a known reference air density value at a reference measurement height. ([#534](https://github.com/brightwind-dev/brightwind/issues/534)) 
3. Added `scale_air_temperature_to_height()` to output an air temperature value for any height by applying a constant lapse rate to a known reference air temperature value at a reference measurement height. ([#530](https://github.com/brightwind-dev/brightwind/issues/530))
4. Updated `calc_air_density()` to include relative humidity as suggested in IEC 61400-12-1. ([#535](https://github.com/brightwind-dev/brightwind/issues/535))
   1. Added deprecation warning the `specific_gas_constant` argument of `calc_air_density()` will be removed in v3.0.
   1. Added deprecation warning the scaling of air density to height within `calc_air_density()` will be removed in v3.0. Users should use `scale_air_density_to_height()` separately instead.
5. Added `apply_scale_factor()` to scale data by the scale_factor. ([#541](https://github.com/brightwind-dev/brightwind/issues/541))
6. Added optional output to `apply_wind_vane_deadband_offset()` which provides a results table showing the applied offset. ([#520](https://github.com/brightwind-dev/brightwind/issues/520))
7. Added optional output to `apply_device_orientation_offset()` which provides a results table showing the applied offset. ([#521](https://github.com/brightwind-dev/brightwind/issues/521)).
8. Updated `LoadBrightHub()` to use BrightHub API Keys. ([#550](https://github.com/brightwind-dev/brightwind/issues/550))
   1. Added deprecation warning the username and password method of authenticating in BrightHub will be removed in a future version.
1. Updated `time_continuity_gaps()` in order to take an argument `minimum_gap_length` which allows the user to filter the time gaps returned. (Issue [#545](https://github.com/brightwind-dev/brightwind/issues/545))
1. Updated `offset_timestamps()`to include last timestamp when date_to is unspecified, so that offset is applied to the entire record if date_to not specified. ([#504](https://github.com/brightwind-dev/brightwind/issues/504))

### Deprecated
1. `LoadBrightdata()` is deprecated and will be removed in version 3.0. Please use `LoadBrightHub()` instead to continue accessing reanalysis data.
2. `LoadBrightHub()` authentication using email and password is deprecated and will be removed in v3.0. Please migrate to API key authentication. Create and manage API keys at: https://brighthub.io/account-settings/settings.
3. `specific_gas_constant` argument of `calc_air_density()` is deprecated and will be removed in v3.0 because the updated method depends on the gas constants for dry air and water vapour, making this argument redundant.
4. The scaling of air density to height within `calc_air_density()` is deprecated and will be removed in v3.0. Users should use `scale_air_density_to_height()` separately instead.
5. Support for **Python ≤ 3.10** is deprecated and will be dropped in v3.0.0. Users should upgrade to Python 3.11 or newer. 
6. Support for **Pandas ≤ 2.2** is deprecated and will be dropped in v3.0.0. Users should upgrade to Pandas 2.3 or newer.

### Bug Fixes
1. Fixed pandas<3.0.0 and numpy<2.3.1 dependencies, ([#458](https://github.com/brightwind-dev/brightwind/issues/458))
1. Fixed pandas deprecating warnings that were linked to frequency strings, .groupby() and .map(). ([#407](https://github.com/brightwind-dev/brightwind/issues/407), [#415](https://github.com/brightwind-dev/brightwind/issues/415) and [#445](https://github.com/brightwind-dev/brightwind/issues/445))


---
## [2.4.0]
17-Dec-2025

### New Features and Enhancements
1. Added `scale_air_pressure_to_height()` to output an air pressure value for any height based on reference air temperature and air pressure values at a different measurement height. ([#531](https://github.com/brightwind-dev/brightwind/issues/531))
2. Added `scale_air_density_to_height()` to output an air density value for any height by applying a constant lapse rate to a known reference air density value at a reference measurement height. ([#534](https://github.com/brightwind-dev/brightwind/issues/534)) 
3. Added `scale_air_temperature_to_height()` to output an air temperature value for any height by applying a constant lapse rate to a known reference air temperature value at a reference measurement height. ([#530](https://github.com/brightwind-dev/brightwind/issues/530))
4. Updated `calc_air_density()` to include relative humidity as suggested in IEC 61400-12-1. ([#535](https://github.com/brightwind-dev/brightwind/issues/535))
   1. Added deprecation warning the `specific_gas_constant` argument of `calc_air_density()` will be removed in v3.0.
   1. Added deprecation warning the scaling of air density to height within `calc_air_density()` will be removed in v3.0. Users should use `scale_air_density_to_height()` separately instead.
5. Added `apply_scale_factor()` to scale data by the scale_factor. ([#541](https://github.com/brightwind-dev/brightwind/issues/541))
6. Added optional output to `apply_wind_vane_deadband_offset()` which provides a results table showing the applied offset. ([#520](https://github.com/brightwind-dev/brightwind/issues/520))
7. Added optional output to `apply_device_orientation_offset()` which provides a results table showing the applied offset. ([#521](https://github.com/brightwind-dev/brightwind/issues/521)).
8. Updated `LoadBrightHub()` to use BrightHub API Keys. ([#550](https://github.com/brightwind-dev/brightwind/issues/550))
   1. Added deprecation warning the username and password method of authenticating in BrightHub will be removed in a future version.
9. Updated `time_continuity_gaps()` in order to take an argument `minimum_gap_length` which allows the user to filter the time gaps returned. (Issue [#545](https://github.com/brightwind-dev/brightwind/issues/545))
10. Updated `offset_timestamps()`to include last timestamp when date_to is unspecified, so that offset is applied to the entire record if date_to not specified. ([#504](https://github.com/brightwind-dev/brightwind/issues/504))

### Deprecated
1. `LoadBrightdata()` is deprecated and will be removed in version 3.0. Please use `LoadBrightHub()` instead to continue accessing reanalysis data.
2. `LoadBrightHub()` authentication using email and password is deprecated and will be removed in v3.0. Please migrate to API key authentication. Create and manage API keys at: https://brighthub.io/account-settings/settings.
3. `specific_gas_constant` argument of `calc_air_density()` is deprecated and will be removed in v3.0 because the updated method depends on the gas constants for dry air and water vapour, making this argument redundant.
4. The scaling of air density to height within `calc_air_density()` is deprecated and will be removed in v3.0. Users should use `scale_air_density_to_height()` separately instead.
5. Support for **Python ≤ 3.10** is deprecated and will be dropped in v3.0.0. Users should upgrade to Python 3.11 or newer. 
6. Support for **Pandas ≤ 2.2** is deprecated and will be dropped in v3.0.0. Users should upgrade to Pandas 2.3 or newer.

### Bug Fixes
1. Fixed pandas<3.0.0 and numpy<2.3.1 dependencies, ([#458](https://github.com/brightwind-dev/brightwind/issues/458))
2. Fixed pandas deprecating warnings that were linked to frequency strings, .groupby() and .map(). ([#407](https://github.com/brightwind-dev/brightwind/issues/407), [#415](https://github.com/brightwind-dev/brightwind/issues/415) and [#445](https://github.com/brightwind-dev/brightwind/issues/445))

---
## [2.3.0]
14-Apr-2025

This update brings a comprehensive set of **bug fixes** and **enhancements** across the Brightwind library. 
Key improvements include more reliable wind and solar data handling, expanded plotting capabilities (including 
colormap support and better legends), and the introduction of new functions for downloading and applying cleaning rules 
and device orientation offsets. 
Enhanced error messages and schema validation strengthen data integrity and user feedback.

### Bug Fixes
1. Fixed legend display in `plot_scatter_wspd` and `plot_scatter_wdir`, ensuring correct labeling in plots. ([#443](https://github.com/brightwind-dev/brightwind/issues/443))  
2. Corrected `is_file` usage, resolving an error when checking file existence. ([#447](https://github.com/brightwind-dev/brightwind/issues/447))  
3. Fixed bug in `_Measurements_get_table`, which affected how measurement tables were retrieved. ([#421](https://github.com/brightwind-dev/brightwind/issues/421))  
4. Made `_Measurements_get_names` public and fixed an issue with how names were retrieved. ([#450](https://github.com/brightwind-dev/brightwind/issues/450))  
5. Improved sensor calibration logic in `_Measurements__get_properties` for multi-type sensors by correctly selecting the appropriate calibration. ([#449](https://github.com/brightwind-dev/brightwind/issues/449))  
6. Fixed error for solar sites in `_Measurements__get_properties`, improving compatibility. ([#453](https://github.com/brightwind-dev/brightwind/issues/453))  
8. Improved support for solar and sodar sites in `_LoggerMainConfigs__get_properties`. ([#454](https://github.com/brightwind-dev/brightwind/issues/454))  
7. Resolved calculation issues in `dist()` function when constant values are sent as well as improved error messages. ([#459](https://github.com/brightwind-dev/brightwind/issues/459))  
9. Fixed issues in `monthly_means()`, correcting plot output and handling of missing months. ([#452](https://github.com/brightwind-dev/brightwind/issues/452), [#413](https://github.com/brightwind-dev/brightwind/issues/413))  
10. Corrected `Shear.TimeOfDay()` logic when months are missing along with improving error message and fixing plot labels. ([#441](https://github.com/brightwind-dev/brightwind/issues/441))

### New Features and Enhancements

1. Updated `monthly_means()` and `plot_monthly_means()` to improve output and visual clarity. ([#452](https://github.com/brightwind-dev/brightwind/issues/452), [#413](https://github.com/brightwind-dev/brightwind/issues/413))  
2. Enhanced `plot_timeseries()` to support more than 12 lines and introduced colormap-based coloring. ([#457](https://github.com/brightwind-dev/brightwind/issues/457), [#492](https://github.com/brightwind-dev/brightwind/issues/492))  
3. Added error feedback in `Shear`: More informative messages returned when no valid data is available. ([#205](https://github.com/brightwind-dev/brightwind/issues/205))  
4. `Shear.TimeSeries` and `TimeOfDay` now return `np.nan` for timestamps with no valid data. ([#205](https://github.com/brightwind-dev/brightwind/issues/205))  
5. Enhanced `LoadBrightHub.get_measurement_stations()` to support device type filtering and optional dictionary return format. ([#287](https://github.com/brightwind-dev/brightwind/issues/287), [#378](https://github.com/brightwind-dev/brightwind/issues/378))  
6. Added `LoadBrightHub.get_cleaning_rules()`, allowing programmatic retrieval of cleaning rules for a station. ([#461](https://github.com/brightwind-dev/brightwind/issues/461))  
7. Added `load.apply_cleaning_rules()`, enabling automated column cleaning based on BrightHub rules. ([#462](https://github.com/brightwind-dev/brightwind/issues/462))  
8. New function: `apply_device_orientation_offset()` adjusts wind direction data based on remote sensing device orientation as stated in it's data model. ([#451](https://github.com/brightwind-dev/brightwind/issues/451))  
9. Schema validation added to `MeasurementStation()`, ensuring input data matches the expected data model version. ([#489](https://github.com/brightwind-dev/brightwind/issues/489))


## [2.2.1]
1. Bug fix some users encounter with `plot.plot_shear_time_of_day()` (Issue [#429](https://github.com/brightwind-dev/brightwind/issues/429)).


## [2.2.0]
1. Modify `Correl.OrdinaryLeastSquares()` to force the intercept to pass through the origin (Issue [#412](https://github.com/brightwind-dev/brightwind/issues/412)).
1. Update `LoadBrightHub.get_data()` to use a new API (Issue [#419](https://github.com/brightwind-dev/brightwind/issues/419)).
1. Added new function `LoadBrightHub.get_cleaning_log()` to pull the cleaning log for a particular measurement station on BrightHub (Issue [#405](https://github.com/brightwind-dev/brightwind/issues/405)).
1. Added new function `LoadBrightHub.get_reanalysis()` to pull reanalysis datasets from BrightHub (Issue [#431](https://github.com/brightwind-dev/brightwind/issues/431)).
1. Modify `load.apply_cleaning()` and `apply_cleaning_windographer()` to clean columns specified in cleaning file by 
matching the sensor name from the beginning of the string. (Issue [#249](https://github.com/brightwind-dev/brightwind/issues/249)).


## [2.1.0]
1. Update behaviour of `time_continuity_gaps` to find any gap that
is not equal to the derived temporal resolution.
2. Added `data_resolution` argument to `average_data_by_period`, `monthly_means`, `coverage` and 
  `merge_datasets_by_period` functions (Issue [#297](https://github.com/brightwind-dev/brightwind/issues/297))
3. Update to work with Pandas 1.3.2. This mostly includes depreciating pd.Timedelta and using pd.DateOffset instead. (Pull request [#312](https://github.com/brightwind-dev/brightwind/pull/312)).
4. Update to work with Pandas 2.0.1, due to `date_format` input update for `pandas.to_datetime`. (Pull request [#387](https://github.com/brightwind-dev/brightwind/issues/387)).
5. Update to work with matplotlib 3.5.2 and bug fix for plot_freq_distribution and dist functions (Issue [#315](https://github.com/brightwind-dev/brightwind/issues/315)). 
6. Update to work with numpy>=1.20.0 when pandas=0.25.3. (Issue [#344](https://github.com/brightwind-dev/brightwind/issues/344)). 
7. Addressed all Future and Deprecation warnings for matplotlib<=3.6.3, numpy<=1.24.1, pandas<=1.5.3. (Issue [#356](https://github.com/brightwind-dev/brightwind/issues/356)).
8. In`Correl` fix issue when duplicate column names are sent to `SpeedSort` (Issue [#304](https://github.com/brightwind-dev/brightwind/issues/304))
9. Added subplotting functionality to `sector_ratio` and improved user control of plotting (Issue [#309](https://github.com/brightwind-dev/brightwind/issues/309))
10. Allow `dist()` function to take a pd.DataFrame so user can plot multiple distributions on the same plot. (Issue [#264](https://github.com/brightwind-dev/brightwind/issues/264))
    1. As part of this added subplotting functionality for bar plots
11. Allow `freq_table()` function to derive a seasonal adjusted frequency distribution if user sets 'seasonal_adjustment' 
to true. (Issue [#334](https://github.com/brightwind-dev/brightwind/issues/334))
    1. As part of this, added 'monthly_coverage_threshold' option for the user to ensure good coverage months. 
12. In `freq_table` added option to give as input target wind speed we want the mean frequency distribution to have 
(Issue [#269](https://github.com/brightwind-dev/brightwind/issues/269)).
13. Allow `freq_table` function to apply a `coverage_threshold` for both seasonal adjusted and base methods. (Issue [#386](https://github.com/brightwind-dev/brightwind/issues/386))
14. Updated `plot_timeseries` to use a subplot function (`_timeseries_subplot`) and added arguments _x_label_, _y_label_, _x_tick_label_angle_, 
_line_colors_, _legend_ and _figure_size_. (Issue [#349](https://github.com/brightwind-dev/brightwind/issues/349)).
15. In `average_data_by_period()` fixed issue when wind direction average is derived for a period equal to the data resolution period 
(Issue [#319](https://github.com/brightwind-dev/brightwind/issues/319)).
16. In `average_data_by_period()` fixed issue when wind direction average is derived for a period equal to the data resolution period (Issue [#319](https://github.com/brightwind-dev/brightwind/issues/319)).
17. Fixed bugs for `TI.by_speed` and `TI.by_sector` and added tests. Solved versions issue that were raised from Pandas 1.3.3. (Issue [#317](https://github.com/brightwind-dev/brightwind/issues/317)).
18. Address errors and warnings generated for `Shear.TimeOfDay` and `Shear` when pandas >=1.0.0 (Issue [#347](https://github.com/brightwind-dev/brightwind/issues/347)).
19. In `_calc_mean_speed_of_freq_tab` for `export_tab_file` fix issue around using wind speed bins less than 1 m/s (Issue [#359](https://github.com/brightwind-dev/brightwind/issues/359)).
20. Update to work with versions 1.0 to 1.2 of IEA WIND Task 43 WRA Data Model (Issue [#306](https://github.com/brightwind-dev/brightwind/issues/306)).
21. Updated `LoadBrightHub` URL and generalised functions used for connecting to BrightHub platform without using `boto3` (Issue [#355](https://github.com/brightwind-dev/brightwind/issues/355)).
22. Removed hardcoded colours for `Shear.TimeOfDay` plots when `plot_type` is 'step' or 'line' and added a colour map. (Issue [#376](https://github.com/brightwind-dev/brightwind/issues/376)).
23. Fixed bug for `SpeedSort` where the `sector_predict` function was not interpolating data using two fit lines. (Issue [#377](https://github.com/brightwind-dev/brightwind/issues/377)).
24. Updated `_ColorPalette` to automatically update color_list, color_map, color_map_cyclical and adjusted lightness color variables when main colors (primary, secondary etc.) are changed. (Issue [#381](https://github.com/brightwind-dev/brightwind/issues/381)).
25. Allow `momm` function to derive a seasonal adjusted mean of monthly mean, if user sets `seasonal_adjustment` to true, and allow to apply a `coverage_threshold` (Issue [#298](https://github.com/brightwind-dev/brightwind/issues/298))
26. Updated `slice_data`, `offset_timestamps`, `_LoadBWPlatform.get_data` functions to use 'less than' data_to if provided as input. (Issue [#385](https://github.com/brightwind-dev/brightwind/issues/385))


## [2.0.0]
- Major changes, notably
  - Incorporating the IEA WIND Task 43 WRA Data Model
  - Adding APIs to pull data from the BrightHub platform www.brightwindhub.com
  - Change license to MIT
  - Correl - add linear regression by direction sector
  - Correl - add different aggregation methods to both ref and target
  - better function to average wind directions
  - Bug fixes


## [1.0.0]
- Initial release
