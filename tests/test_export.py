import pytest
import brightwind as bw
import os
import numpy as np
import pandas as pd

DATA = bw.load_csv(bw.demo_datasets.demo_data)
DATA = bw.apply_cleaning(DATA, bw.demo_datasets.demo_cleaning_file)

# Temp folder used to save test files
TEMP_FOLDER = 'temp'
cwd = os.getcwd()
os.makedirs(os.path.join(cwd, TEMP_FOLDER), exist_ok=True)


def test_export_tab_file():
    graph, tab = bw.freq_table(DATA.Spd40mN, DATA.Dir38mS, return_data=True)
    bw.export_tab_file(tab, 80, 10, 10, file_name='export_tab_file.tab', folder_path=TEMP_FOLDER)
    graph, tab = bw.freq_table(DATA[['Spd40mN']], DATA[['Dir38mS']], return_data=True)
    bw.export_tab_file(tab, 80, 10, 10, file_name='export_tab_file_1.tab', folder_path=TEMP_FOLDER)
    assert True


def test_export_tws_file():
    # Generate the files
    bw.export_tws_file(
        DATA.Spd40mN.iloc[:20], DATA.Dir38mS.iloc[:20], 10, 10, 80, DATA.Spd40mNStd.iloc[:20], 
        file_name='export_tws_file.tws', folder_path=TEMP_FOLDER
        )
    
    # Load the export file and test that it matches the test file
    with open(bw.demo_datasets.demo_tws_file, "r") as f:
        test_file = f.read()

    with open(os.path.join(TEMP_FOLDER, 'export_tws_file.tws'), "r") as f:
        new_file = f.read()
    # Skip the first line, which holds the brightwind version and a creation timestamp that will never match
    assert new_file.split('\n', 1)[1] == test_file.split('\n', 1)[1]


def _read_wti_file(file_name):
    """Read a .wti file into its number of sectors, sector headers and the three tables as pandas.DataFrames."""
    with open(os.path.join(TEMP_FOLDER, file_name), 'r') as f:
        lines = f.read().split('\n')
    sections = {'Mean_TI': [], 'Sigma_Sigma': [], 'Bin_Counts': []}
    headers = []
    section = 'Mean_TI'
    for line in lines[3:]:
        if line == '#Sigma Sigma':
            section = 'Sigma_Sigma'
        elif line == '#Bin Counts':
            section = 'Bin_Counts'
        elif line.startswith('#  \t'):
            headers.append(line)
        elif line:
            values = line.lstrip('#').split('\t')
            sections[section].append([int(values[0])] + [float(v) for v in values[1:]])
    tables = {name: pd.DataFrame(rows).set_index(0) for name, rows in sections.items()}
    return lines, headers, tables


def test_export_wti_file():
    # Test default inputs: header, sector centres, layout and contiguous wind speed rows from 1 m/s
    bw.export_wti_file(DATA.Spd80mN, DATA.Spd80mNStd, DATA.Dir78mS, file_name='export_wti_file.wti',
                       folder_path=TEMP_FOLDER)
    lines, headers, tables = _read_wti_file('export_wti_file.wti')
    assert lines[0].startswith('# export_wti_file created using brightwind version')
    assert lines[1] == '12'
    assert lines[2] == ''
    expected_header = '#  \t' + '\t'.join(f'{centre:7.1f}' for centre in np.arange(0, 360, 30))
    assert headers == [expected_header] * 3
    assert lines[3] == expected_header
    assert lines[4] == ' 1\t45.2233\t44.0331\t45.9428\t42.5892\t44.6871\t51.7704\t49.1049\t47.8597\t46.2701\t53.2412' \
                       '\t52.0801\t47.3334'
    # Rows are always 1 to 24 m/s, as WindFarmer fails to load rows above 24 m/s
    rows = list(range(1, 25))
    for table in tables.values():
        assert list(table.index) == rows
        assert table.shape[1] == 12

    # Test values against TI.by_speed_and_sector: cells with at least 5 TI values keep their own mean TI (in %) and
    # sigma-sigma, other cells take the values for the wind speed bin across all sectors, and counts are not filled.
    # The demo data has at least 5 TI values across all sectors in every bin up to 24 m/s, so no rows are held.
    ti_kwargs = dict(speed_bin_array=np.arange(0.5, 25, 1), speed_bin_labels=rows, min_speed=0.5, return_data=True)
    _, ti_dist = bw.TI.by_speed_and_sector(DATA.Spd80mN, DATA.Spd80mNStd, DATA.Dir78mS, **ti_kwargs)
    _, ti_all = bw.TI.by_speed_and_sector(DATA.Spd80mN, DATA.Spd80mNStd, DATA.Dir78mS, direction_bin_array=[0, 360],
                                          **ti_kwargs)
    counts = ti_dist['TI_Count'].reindex(rows).fillna(0).values
    np.testing.assert_array_equal(tables['Bin_Counts'].values, counts)
    all_sector_counts = ti_all['TI_Count'].iloc[:, 0].reindex(rows).fillna(0)
    assert (all_sector_counts >= 5).all()
    for stat, scale in [('Mean_TI', 100), ('Sigma_Sigma', 1)]:
        own = ti_dist[stat].reindex(rows).values * scale
        all_sectors = ti_all[stat].reindex(rows).values * scale
        keep = (counts >= 5) & ~np.isnan(own)
        np.testing.assert_allclose(tables[stat].values[keep], own[keep], atol=5e-5)
        replace = counts < 5
        np.testing.assert_allclose(tables[stat].values[replace], np.broadcast_to(all_sectors, own.shape)[replace],
                                   atol=5e-5)
    assert not tables['Mean_TI'].isna().any().any()
    assert not tables['Sigma_Sigma'].isna().any().any()

    # Test the held rows follow count_threshold: with 20, the highest wind speed bin with enough TI values across
    # all sectors is 23 m/s, so the 24 m/s row holds the 23 m/s values
    bw.export_wti_file(DATA.Spd80mN, DATA.Spd80mNStd, DATA.Dir78mS, count_threshold=20,
                       file_name='export_wti_file_threshold_20.wti', folder_path=TEMP_FOLDER)
    _, _, tables = _read_wti_file('export_wti_file_threshold_20.wti')
    assert all_sector_counts[all_sector_counts >= 20].index.max() == 23
    for stat in ['Mean_TI', 'Sigma_Sigma']:
        np.testing.assert_array_equal(tables[stat].loc[24].values, tables[stat].loc[23].values)
        assert not np.array_equal(tables[stat].loc[23].values, tables[stat].loc[22].values)

    # Test wind speeds above 24.5 m/s are not used, and rows above the highest wind speed bin with data hold its
    # values up to 24 m/s
    bw.export_wti_file(DATA.Spd80mN.clip(upper=15), DATA.Spd80mNStd, DATA.Dir78mS,
                       file_name='export_wti_file_max_15.wti', folder_path=TEMP_FOLDER)
    _, _, tables = _read_wti_file('export_wti_file_max_15.wti')
    assert list(tables['Mean_TI'].index) == rows
    for stat in ['Mean_TI', 'Sigma_Sigma']:
        for row in range(16, 25):
            np.testing.assert_array_equal(tables[stat].loc[row].values, tables[stat].loc[15].values)
    np.testing.assert_array_equal(tables['Bin_Counts'].loc[16:].values, 0)
    above_24 = DATA.Spd80mN.where(DATA.Spd80mN < 24.5, DATA.Spd80mN + 10)
    bw.export_wti_file(above_24, DATA.Spd80mNStd, DATA.Dir78mS, file_name='export_wti_file_above_24.wti',
                       folder_path=TEMP_FOLDER)
    _, _, tables_above_24 = _read_wti_file('export_wti_file_above_24.wti')
    _, _, tables = _read_wti_file('export_wti_file.wti')
    for name in tables:
        pd.testing.assert_frame_equal(tables_above_24[name], tables[name])

    # Test sectors, DataFrame inputs and a file_name without the .wti extension
    bw.export_wti_file(DATA[['Spd80mN']], DATA[['Spd80mNStd']], DATA[['Dir78mS']], sectors=16,
                       file_name='export_wti_file_16_sectors', folder_path=TEMP_FOLDER)
    lines, headers, tables = _read_wti_file('export_wti_file_16_sectors.wti')
    assert lines[1] == '16'
    assert headers[0] == '#  \t' + '\t'.join(f'{centre:7.1f}' for centre in np.arange(0, 360, 22.5))
    assert tables['Mean_TI'].shape == (24, 16)

    # Test count_threshold=0 only replaces cells with no value
    bw.export_wti_file(DATA.Spd80mN, DATA.Spd80mNStd, DATA.Dir78mS, count_threshold=0,
                       file_name='export_wti_file_threshold_0.wti', folder_path=TEMP_FOLDER)
    _, _, tables = _read_wti_file('export_wti_file_threshold_0.wti')
    own = ti_dist['Mean_TI'].reindex(rows).values * 100
    has_value = ~np.isnan(own)
    np.testing.assert_allclose(tables['Mean_TI'].values[has_value], own[has_value], atol=5e-5)

    # Test a wind speed bin with no data in any sector (2 m/s) is interpolated from its neighbours, and no bin has
    # count_threshold TI values, so the highest wind speed bin with data (3 m/s) is held up to 24 m/s
    idx = pd.date_range('2025-01-01', periods=4, freq='10min')
    gap = pd.DataFrame({'spd': [1.0, 1.0, 3.0, 3.0], 'std': [0.2, 0.4, 0.6, 1.0], 'dir': [0.0, 0.0, 90.0, 90.0]},
                       index=idx)
    bw.export_wti_file(gap.spd, gap['std'], gap.dir, sectors=4, file_name='export_wti_file_gap.wti',
                       folder_path=TEMP_FOLDER)
    _, _, tables = _read_wti_file('export_wti_file_gap.wti')
    assert list(tables['Mean_TI'].index) == rows
    np.testing.assert_allclose(tables['Mean_TI'].loc[1].values, 30.0, atol=5e-5)
    np.testing.assert_allclose(tables['Mean_TI'].loc[3].values, 80.0 / 3, atol=5e-5)
    np.testing.assert_allclose(tables['Mean_TI'].loc[2].values, (30.0 + 80.0 / 3) / 2, atol=5e-5)
    np.testing.assert_allclose(tables['Mean_TI'].loc[4:].values, 80.0 / 3, atol=5e-5)
    np.testing.assert_array_equal(tables['Bin_Counts'].loc[2].values, 0)
    np.testing.assert_array_equal(tables['Bin_Counts'].loc[4:].values, 0)

    # Test invalid inputs
    with pytest.raises(ValueError, match='count_threshold'):
        bw.export_wti_file(DATA.Spd80mN, DATA.Spd80mNStd, DATA.Dir78mS, count_threshold=-1, folder_path=TEMP_FOLDER)
    with pytest.raises(ValueError, match='sectors'):
        bw.export_wti_file(DATA.Spd80mN, DATA.Spd80mNStd, DATA.Dir78mS, sectors=0, folder_path=TEMP_FOLDER)
    with pytest.raises(ValueError, match='wspd and wdir must have the same index'):
        bw.export_wti_file(DATA.Spd80mN, DATA.Spd80mNStd, DATA.Dir78mS.shift(1, freq='10min'),
                           folder_path=TEMP_FOLDER)
    with pytest.raises(ValueError, match='wspd and wspd_std must have the same dimensions'):
        bw.export_wti_file(DATA.Spd80mN, DATA.Spd80mNStd.iloc[:100], DATA.Dir78mS, folder_path=TEMP_FOLDER)

    # Test a clear error when there is no data to export between 0.5 and 24.5 m/s
    for spd, wdir in [(gap.spd * np.nan, gap.dir), (gap.spd * 0.1, gap.dir), (gap.spd + 30, gap.dir),
                      (gap.spd, gap.dir * np.nan)]:
        with pytest.raises(ValueError, match='No TI values to bin.*0.5 to 24.5'):
            bw.export_wti_file(spd, gap['std'], wdir, folder_path=TEMP_FOLDER)

    # Test a warning is given for mean TI of 100 % or more
    high_ti = pd.DataFrame({'spd': [1.0] * 5, 'std': [1.2] * 5, 'dir': [0.0] * 5},
                           index=pd.date_range('2025-01-01', periods=5, freq='10min'))
    with pytest.warns(UserWarning, match='100 %'):
        bw.export_wti_file(high_ti.spd, high_ti['std'], high_ti.dir, file_name='export_wti_file_high_ti.wti',
                           folder_path=TEMP_FOLDER)


def test_export_to_csv():
    bw.export_csv(DATA, file_name='export_to_csv', folder_path=TEMP_FOLDER)

    bw.export_csv(DATA, file_name='export_to_csv_tab.tab', folder_path=TEMP_FOLDER, sep='\t')
    assert True


def test_calc_mean_speed_of_freq_tab():
    fig, freq_tab = bw.freq_table(DATA.Spd80mN, DATA.Dir38mS, return_data=True)
    assert round(bw.export.export._calc_mean_speed_of_freq_tab(freq_tab), 5) == 7.51925
    fig, freq_tab = bw.freq_table(DATA.Spd80mN, DATA.Dir38mS, var_bin_array=list(np.arange(0, 41, 0.5)),
                                  return_data=True)
    assert round(bw.export.export._calc_mean_speed_of_freq_tab(freq_tab), 5) == 7.5206
