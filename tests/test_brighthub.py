import pytest
import requests
import brightwind as bw
from unittest.mock import patch, MagicMock
from brightwind.utils import brighthub


@pytest.fixture(autouse=True)
def _reset_auth():
    original = brighthub._BrighthubAuth.ID_TOKEN
    brighthub._BrighthubAuth.ID_TOKEN = 'token-1'
    yield
    brighthub._BrighthubAuth.ID_TOKEN = original


@pytest.mark.parametrize('method', ['GET', 'POST', 'PUT', 'DELETE', 'post'])
def test_brighthub_request_methods(method):
    with patch('brightwind.utils.brighthub.requests.request', return_value=MagicMock(status_code=200)) as mock_req:
        response = bw.brighthub_request(method, '/plants', params={'a': 1}, json={'b': 2})

    assert response.status_code == 200
    kwargs = mock_req.call_args.kwargs
    assert kwargs['method'] == method.upper()
    assert kwargs['url'].endswith('/plants')
    assert kwargs['headers'] == {'authorization': 'token-1'}
    assert kwargs['params'] == {'a': 1}
    assert kwargs['json'] == {'b': 2}
    assert kwargs['timeout'] == 10


def test_brighthub_request_unsupported_method_returns_411():
    with patch('brightwind.utils.brighthub.requests.request') as mock_req:
        response = bw.brighthub_request('PATCH', '/plants')

    assert response.status_code == 411
    mock_req.assert_not_called()


def test_brighthub_request_retries_on_connection_error():
    ok = MagicMock(status_code=200)
    side_effect = [requests.exceptions.ConnectionError(), requests.exceptions.Timeout(), ok]
    with patch('brightwind.utils.brighthub.requests.request', side_effect=side_effect) as mock_req,             patch('brightwind.utils.brighthub.time.sleep') as mock_sleep:
        response = bw.brighthub_request('GET', '/plants')

    assert response is ok
    assert mock_req.call_count == 3
    assert mock_sleep.call_count == 2


def test_brighthub_request_raises_after_max_attempts():
    with patch('brightwind.utils.brighthub.requests.request', side_effect=requests.exceptions.ConnectionError())             as mock_req, patch('brightwind.utils.brighthub.time.sleep'):
        with pytest.raises(requests.exceptions.ConnectionError):
            bw.brighthub_request('GET', '/plants')

    assert mock_req.call_count == 3


def test_brighthub_request_reauthenticates_once_on_expired_token():
    expired = MagicMock(status_code=401)
    expired.json.return_value = {'message': 'The incoming token has expired'}
    ok = MagicMock(status_code=200)

    def new_token():
        brighthub._BrighthubAuth.ID_TOKEN = 'token-2'

    with patch('brightwind.utils.brighthub.requests.request', side_effect=[expired, ok]) as mock_req,             patch.object(brighthub._BrighthubAuth, '_get_id_token', side_effect=new_token) as mock_auth:
        response = bw.brighthub_request('POST', '/plants')

    assert response is ok
    mock_auth.assert_called_once()
    assert mock_req.call_args.kwargs['headers'] == {'authorization': 'token-2'}


def test_get_brighthub_auth_id_token_returns_cached_token():
    with patch.object(brighthub._BrighthubAuth, '_get_id_token') as mock_auth:
        assert bw.get_brighthub_auth_id_token() == 'token-1'

    mock_auth.assert_not_called()


def test_get_brighthub_auth_id_token_authenticates_when_empty():
    brighthub._BrighthubAuth.ID_TOKEN = ''

    def new_token():
        brighthub._BrighthubAuth.ID_TOKEN = 'token-new'

    with patch.object(brighthub._BrighthubAuth, '_get_id_token', side_effect=new_token) as mock_auth:
        assert bw.get_brighthub_auth_id_token() == 'token-new'

    mock_auth.assert_called_once()


def test_legacy_load_brighthub_request_still_works_and_warns():
    response = MagicMock(status_code=200)
    with patch('brightwind.utils.brighthub.requests.request', return_value=response) as mock_req:
        with pytest.warns(FutureWarning, match='v3.0.0'):
            result = bw.LoadBrightHub._brighthub_request('/plants', params={'a': 1})

    assert result is response
    assert mock_req.call_args.kwargs['method'] == 'GET'
    assert mock_req.call_args.kwargs['params'] == {'a': 1}


def test_legacy_brighthub_auth_import_still_works_and_warns():
    import brightwind.load.load as load_module
    with pytest.warns(FutureWarning, match='v3.0.0'):
        legacy_class = load_module._BrighthubAuth

    assert legacy_class is brighthub._BrighthubAuth
