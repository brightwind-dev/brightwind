import sys
import types

import pytest

import brightwind as bw

MM = bw.MeasurementStation(bw.demo_datasets.demo_wra_data_model)


def _fake_gmaps_module():
    """Create a minimal stand-in for the optional 'gmaps' library so no Google Maps API key or widgets are needed."""
    fake = types.ModuleType('gmaps')
    fake.configure = lambda api_key: None
    fake.figure = lambda **kwargs: types.SimpleNamespace(add_layer=lambda layer: None)
    fake.marker_layer = lambda *args, **kwargs: object()
    return fake


def test_plot_meas_station_on_gmap_raises_import_error_without_gmaps(monkeypatch):
    # A None entry in sys.modules makes 'import gmaps' raise ImportError even if the library is installed.
    monkeypatch.setitem(sys.modules, 'gmaps', None)
    with pytest.warns(FutureWarning):
        with pytest.raises(ImportError, match=r'brightwind\[maps\]'):
            bw.plot_meas_station_on_gmap(MM)


def test_plot_meas_station_on_gmap_deprecation_warning(monkeypatch):
    monkeypatch.setitem(sys.modules, 'gmaps', _fake_gmaps_module())
    monkeypatch.setenv('GMAPS_API_KEY', 'dummy_key')
    with pytest.warns(FutureWarning, match=r'removed or replaced in v3\.0\.0') as record:
        bw.plot_meas_station_on_gmap(MM)
    # stacklevel=2 attributes the warning to the calling code, not to brightwind.
    assert record[0].filename == __file__


def test_plot_meas_station_on_gmap_input_type(monkeypatch):
    monkeypatch.setitem(sys.modules, 'gmaps', _fake_gmaps_module())
    monkeypatch.setenv('GMAPS_API_KEY', 'dummy_key')
    with pytest.warns(FutureWarning):
        with pytest.raises(TypeError, match='MeasurementStation'):
            bw.plot_meas_station_on_gmap('not a measurement station')


def test_import_brightwind_does_not_import_gmaps():
    import subprocess
    code = "import sys; sys.modules['gmaps'] = None; import brightwind"
    result = subprocess.run([sys.executable, '-W', 'ignore', '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
