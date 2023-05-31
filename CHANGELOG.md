# Changelog
All notable changes to this project will be documented in this file. If you make a notable change to the project, please add a line describing the change to the "unreleased" section. The maintainers will make an effort to keep the [Github Releases](https://github.com/brightwind-dev/brightwind/releases) page up to date with this changelog. The format is based on [Semantic Versioning](https://semver.org/) e.g. '1.1.0'

Given a version number MAJOR.MINOR.PATCH, increment the:

    1. MAJOR version when you make incompatible API changes,
    2. MINOR version when you add functionality in a backwards compatible manner, and
    3. PATCH version when you make backwards compatible bug fixes.

Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.

## [2.X.0]
1. Update behaviour of `time_continuity_gaps` to find any gap that
is not equal to the derived temporal resolution.
2. Added `data_resolution` argument to `average_data_by_period`, `monthly_means`, `coverage` and 
  `merge_datasets_by_period` functions (Issue #[297](https://github.com/brightwind-dev/brightwind/issues/297))
3. Update to work with Pandas 1.3.2. This mostly includes depreciating pd.Timedelta and using pd.DateOffset instead. (Pull request #[312](https://github.com/brightwind-dev/brightwind/pull/312)).
4. In`Correl` fix issue when duplicate column names are sent to `SpeedSort` (Issue #[304](https://github.com/brightwind-dev/brightwind/issues/304))
5. Added subplotting functionality to `sector_ratio` and improved user control of plotting (Issue #[309](https://github.com/brightwind-dev/brightwind/issues/309))
6. Allow `dist()` function to take a pd.DataFrame so user can plot multiple distributions on the same plot. (Issue #[264](https://github.com/brightwind-dev/brightwind/issues/264))
   1. As part of this added subplotting functionality for bar plots
7. Allow 'freq_table()' function to derive a seasonal adjusted frequency distribution if user sets 'seasonal_adjustment' 
to true. (Issue #[334](https://github.com/brightwind-dev/brightwind/issues/334))
   1. As part of this, added 'monthly_coverage_threshold' option for the user to ensure good coverage months.
8. Update to work with matplotlib 3.5.2 and bug fix for plot_freq_distribution and dist functions (Issue #[315](https://github.com/brightwind-dev/brightwind/issues/315)). 
9. Update to work with numpy>=1.20.0 when pandas=0.25.3. (Issue #[344](https://github.com/brightwind-dev/brightwind/issues/344)). 
10. Updated `plot_timeseries` to use a subplot function (`_timeseries_subplot`) and added arguments _x_label_, _y_label_, _x_tick_label_angle_, 
_line_colors_, _legend_ and _figure_size_. (Issue #[349](https://github.com/brightwind-dev/brightwind/issues/349)).
11. In `average_data_by_period()` fixed issue when wind direction average is derived for a period equal to the data resolution period 
(Issue #[319](https://github.com/brightwind-dev/brightwind/issues/319)).
13. In `freq_table` added option to give as input target wind speed we want the mean frequency distribution to have 
(Issue #[269](https://github.com/brightwind-dev/brightwind/issues/269)).
12. Fixed bugs for `TI.by_speed` and `TI.by_sector` and added tests. Solved versions issue that were raised from Pandas 1.3.3. (Issue #[317](https://github.com/brightwind-dev/brightwind/issues/317)).
11. Address errors and warnings generated for `Shear.TimeOfDay` and `Shear` when pandas >=1.0.0 (Issue #[347](https://github.com/brightwind-dev/brightwind/issues/347)).
12. In `average_data_by_period()` fixed issue when wind direction average is derived for a period equal to the data resolution period (Issue #[319](https://github.com/brightwind-dev/brightwind/issues/319)).
13. In `_calc_mean_speed_of_freq_tab` for `export_tab_file` fix issue around using wind speed bins less than 1 m/s (Issue #[359](https://github.com/brightwind-dev/brightwind/issues/359)).
14. Addressed all Future and Deprecation warnings for matplotlib<=3.6.3, numpy<=1.24.1, pandas<=1.5.3. (Issue #[356](https://github.com/brightwind-dev/brightwind/issues/356)).
15. Update to work with versions 1.0 to 1.2 of IEA WIND Task 43 WRA Data Model (Issue #[306](https://github.com/brightwind-dev/brightwind/issues/306)).
16. Updated `LoadBrightHub` URL and generalised functions used for connecting to BrightHub platform without using `boto3` (Issue #[355](https://github.com/brightwind-dev/brightwind/issues/355)).
17. Removed hardcoded colours for `Shear.TimeOfDay` plots when `plot_type` is 'step' or 'line' and added a colour map. (Issue #[376](https://github.com/brightwind-dev/brightwind/issues/376)).
18. Fixed bug for `SpeedSort` where the `sector_predict` function was not interpolating data using two fit lines. (Issue #[377](https://github.com/brightwind-dev/brightwind/issues/377)).
19. Updated `_ColorPalette` to automatically update color_list, color_map, color_map_cyclical and adjusted lightness color variables when main colors (primary, secondary etc.) are changed. (Issue #[381](https://github.com/brightwind-dev/brightwind/issues/381)).
19. Allow `momm` function to derive a seasonal adjusted mean of monthly mean, if user sets `seasonal_adjustment` to true, and allow to apply a `coverage_threshold` (Issue #[298](https://github.com/brightwind-dev/brightwind/issues/298))
20. Updated `slice_data`, `offset_timestamps`, `_LoadBWPlatform.get_data` functions to use 'less than' data_to if provided as input. (Issue #[385](https://github.com/brightwind-dev/brightwind/issues/385))
21. Update to work with Pandas 2.0.1, due to `date_format` input update for `pandas.to_datetime`. (Pull request #[387](https://github.com/brightwind-dev/brightwind/issues/387)).
20. Allow `freq_table` function to apply a `coverage_threshold` for both seasonal adjusted and base methods. (Issue #[386](https://github.com/brightwind-dev/brightwind/issues/386))



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
