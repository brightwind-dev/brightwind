import json
import os
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
    side_effect = [requests.exceptions.ConnectionError(), requests.exceptions.ReadTimeout(), ok]
    with patch('brightwind.utils.brighthub.requests.request', side_effect=side_effect) as mock_req, \
            patch('brightwind.utils.brighthub.time.sleep') as mock_sleep:
        response = bw.brighthub_request('GET', '/plants')

    assert response is ok
    assert mock_req.call_count == 3
    assert mock_sleep.call_count == 2


def test_brighthub_request_raises_after_max_attempts():
    with patch('brightwind.utils.brighthub.requests.request', side_effect=requests.exceptions.ConnectionError()) \
            as mock_req, patch('brightwind.utils.brighthub.time.sleep'):
        with pytest.raises(requests.exceptions.ConnectionError):
            bw.brighthub_request('GET', '/plants')

    assert mock_req.call_count == 3


def test_brighthub_request_reauthenticates_once_on_401():
    expired = MagicMock(status_code=401)
    expired.json.return_value = {'message': 'Unauthorized'}
    ok = MagicMock(status_code=200)

    def new_token():
        brighthub._BrighthubAuth.ID_TOKEN = 'token-2'

    with patch('brightwind.utils.brighthub.requests.request', side_effect=[expired, ok]) as mock_req, \
            patch.object(brighthub._BrighthubAuth, '_get_id_token', side_effect=new_token) as mock_auth:
        response = bw.brighthub_request('POST', '/plants')

    assert response is ok
    mock_auth.assert_called_once()
    assert mock_req.call_args.kwargs['headers'] == {'authorization': 'token-2'}


@pytest.mark.parametrize('error', [requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError])
@pytest.mark.parametrize('method', ['POST', 'PUT', 'DELETE'])
def test_brighthub_request_does_not_retry_writes_after_ambiguous_errors(method, error):
    with patch('brightwind.utils.brighthub.requests.request', side_effect=error()) as mock_req, \
            patch('brightwind.utils.brighthub.time.sleep'):
        with pytest.raises(error):
            bw.brighthub_request(method, '/plants')

    assert mock_req.call_count == 1


@pytest.mark.parametrize('method', ['POST', 'PUT', 'DELETE'])
def test_brighthub_request_retries_writes_on_connect_timeout(method):
    ok = MagicMock(status_code=200)
    with patch('brightwind.utils.brighthub.requests.request',
               side_effect=[requests.exceptions.ConnectTimeout(), ok]) as mock_req, \
                       patch('brightwind.utils.brighthub.time.sleep'):
        assert bw.brighthub_request(method, '/plants') is ok

    assert mock_req.call_count == 2


@pytest.mark.parametrize('method, expected_calls', [('GET', 3), ('POST', 1)])
def test_brighthub_request_retries_503_for_get_only(method, expected_calls):
    with patch('brightwind.utils.brighthub.requests.request', return_value=MagicMock(status_code=503)) as mock_req, \
            patch('brightwind.utils.brighthub.time.sleep'):
        response = bw.brighthub_request(method, '/plants')

    assert response.status_code == 503
    assert mock_req.call_count == expected_calls


def test_brighthub_login_error_map_has_password_verifier():
    assert 'password_verifier' in brighthub._BrighthubAuth._BrighthubAuth__BRIGHTHUB_LOGIN_ERROR_MAP


def test_brighthub_module_does_not_import_brightwind():
    import ast
    with open(brighthub.__file__, encoding='utf-8') as f:
        tree = ast.parse(f.read())
    imported = [n.module if isinstance(n, ast.ImportFrom) else a.name
                for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in (n.names if isinstance(n, ast.Import) else [n])]
    assert not any(name and name.startswith('brightwind') for name in imported)


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


def test_brighthub_request_raises_auth_error_when_401_persists():
    unauthorised = MagicMock(status_code=401, text='{"message": "Unauthorized"}')
    with patch('brightwind.utils.brighthub.requests.request', return_value=unauthorised), \
            patch.object(brighthub._BrighthubAuth, '_get_id_token'):
        with pytest.raises(bw.BrighthubAuthError, match='unauthorised'):
            bw.brighthub_request('GET', '/measurement-locations/uuid/cleaning-log')


@pytest.mark.parametrize('status_code', [400, 401, 403])
def test_client_credentials_rejected_gives_clear_auth_error(status_code):
    brighthub._BrighthubAuth.CLIENT_ID = 'id'
    brighthub._BrighthubAuth.CLIENT_SECRET = 'secret'
    response = requests.Response()
    response.status_code = status_code
    try:
        with patch('brightwind.utils.brighthub.requests.post', return_value=response):
            with pytest.raises(bw.BrighthubAuthError, match='API key is invalid'):
                brighthub._BrighthubAuth._authenticate_with_client_credentials()
    finally:
        brighthub._BrighthubAuth.CLIENT_ID = ''
        brighthub._BrighthubAuth.CLIENT_SECRET = ''


def test_client_credentials_connection_error_is_reported_as_network_error():
    brighthub._BrighthubAuth.CLIENT_ID = 'id'
    brighthub._BrighthubAuth.CLIENT_SECRET = 'secret'
    try:
        with patch('brightwind.utils.brighthub.requests.post', side_effect=requests.exceptions.ConnectionError()):
            with pytest.raises(bw.BrighthubAuthError, match='Network or connection error'):
                brighthub._BrighthubAuth._authenticate_with_client_credentials()
    finally:
        brighthub._BrighthubAuth.CLIENT_ID = ''
        brighthub._BrighthubAuth.CLIENT_SECRET = ''


def test_load_brighthub_get_cleaning_log_forbidden_without_details_raises_value_error_not_key_error():
    response = MagicMock(status_code=403, text='{"message": "Forbidden"}')
    response.json.return_value = {'message': 'Forbidden'}
    with patch('brightwind.load.load.brighthub.brighthub_request', return_value=response):
        with pytest.raises(ValueError, match='Status Code 403'):
            bw.LoadBrightHub.get_cleaning_log(measurement_station_uuid='uuid-1')


def test_error_messages_use_the_base_uri():
    brighthub._BrighthubAuth.CLIENT_ID = 'id'
    brighthub._BrighthubAuth.CLIENT_SECRET = 'secret'
    response = requests.Response()
    response.status_code = 401
    try:
        with patch('brightwind.utils.brighthub._BASE_URI', 'https://api.example.test'), \
                patch('brightwind.utils.brighthub.requests.post', return_value=response):
            with pytest.raises(bw.BrighthubAuthError) as auth_error:
                brighthub._BrighthubAuth._authenticate_with_client_credentials()
    finally:
        brighthub._BrighthubAuth.CLIENT_ID = ''
        brighthub._BrighthubAuth.CLIENT_SECRET = ''
    assert 'https://api.example.test/auth/token' in str(auth_error.value)
    assert 'https://example.test/account-settings/settings' in str(auth_error.value)

    with patch('brightwind.utils.brighthub._BASE_URI', 'https://api.example.test'), \
            patch.object(brighthub._BrighthubAuth, '_get_id_token'), \
            patch('brightwind.utils.brighthub.requests.request', return_value=MagicMock(status_code=401, text='x')):
        with pytest.raises(bw.BrighthubAuthError, match='GET https://api.example.test/plants'):
            bw.brighthub_request('GET', '/plants')


def test_no_credentials_error_asks_for_client_id_and_secret():
    import os
    creds = ('BRIGHTHUB_CLIENT_ID', 'BRIGHTHUB_CLIENT_SECRET', 'BRIGHTHUB_EMAIL', 'BRIGHTHUB_PASSWORD')
    env = {k: v for k, v in os.environ.items() if k not in creds}
    saved = (brighthub._BrighthubAuth.ID_TOKEN, brighthub._BrighthubAuth.CLIENT_ID,
             brighthub._BrighthubAuth.CLIENT_SECRET, brighthub._BrighthubAuth.USERNAME,
             brighthub._BrighthubAuth.PASSWORD)
    brighthub._BrighthubAuth.ID_TOKEN = brighthub._BrighthubAuth.CLIENT_ID = ''
    brighthub._BrighthubAuth.CLIENT_SECRET = brighthub._BrighthubAuth.USERNAME = ''
    brighthub._BrighthubAuth.PASSWORD = ''
    try:
        with patch.dict('os.environ', env, clear=True):
            with pytest.raises(bw.BrighthubAuthError) as error:
                bw.get_brighthub_auth_id_token()
    finally:
        (brighthub._BrighthubAuth.ID_TOKEN, brighthub._BrighthubAuth.CLIENT_ID,
         brighthub._BrighthubAuth.CLIENT_SECRET, brighthub._BrighthubAuth.USERNAME,
         brighthub._BrighthubAuth.PASSWORD) = saved
    message = str(error.value)
    assert 'BRIGHTHUB_CLIENT_ID' in message and 'BRIGHTHUB_CLIENT_SECRET' in message
    assert 'BRIGHTHUB_EMAIL' not in message


@pytest.mark.parametrize('method', ['GET', 'POST', 'PUT', 'DELETE'])
@pytest.mark.parametrize('message', [
    'Missing Authentication Token',
    "Authorization header requires 'Credential' parameter. Authorization=abc",
])
def test_brighthub_request_gateway_rejection_raises_request_error_naming_method_and_url(method, message):
    rejected = requests.Response()
    rejected.status_code = 403
    rejected._content = ('{"message": "%s"}' % message).encode()
    with patch('brightwind.utils.brighthub._BASE_URI', 'https://api.example.test'), \
            patch('brightwind.utils.brighthub.requests.request', return_value=rejected):
        with pytest.raises(bw.BrighthubRequestError) as error:
            bw.brighthub_request(method.lower(), '/measurement-locations/uuid/issues')

    assert f'{method} https://api.example.test/measurement-locations/uuid/issues' in str(error.value)
    assert f'for the {method} method' in str(error.value)
    assert error.value.status_code == 403 and error.value.method == method
    assert isinstance(error.value, ValueError)


@pytest.mark.parametrize('body', [
    {'error': 'Forbidden', 'details': 'Download limit reached'},
    {'message': 'Some other forbidden reason'},
    {'message': 'Missing Authentication Token', 'details': 'x'},
])
def test_brighthub_request_other_403_responses_are_returned_unchanged(body):
    forbidden = requests.Response()
    forbidden.status_code = 403
    forbidden._content = json.dumps(body).encode()
    with patch('brightwind.utils.brighthub.requests.request', return_value=forbidden):
        assert bw.brighthub_request('GET', '/plants') is forbidden


def test_get_brighthub_base_uri_returns_base_uri_used_by_requests():
    assert bw.get_brighthub_base_uri() == brighthub._BASE_URI
    assert not bw.get_brighthub_base_uri().endswith('/')
    with patch('brightwind.utils.brighthub._BASE_URI', 'https://api.example.test'), \
            patch('brightwind.utils.brighthub.requests.request', return_value=MagicMock(status_code=200)) as mock_req:
        assert bw.get_brighthub_base_uri() == 'https://api.example.test'
        bw.brighthub_request('GET', '/plants')

    assert mock_req.call_args.kwargs['url'] == 'https://api.example.test/plants'


def test_clear_brighthub_auth_cache_resets_all_cached_values():
    auth = brighthub._BrighthubAuth
    saved = {n: getattr(auth, n) for n in
             ('ID_TOKEN', 'REFRESH_TOKEN', 'USERNAME', 'PASSWORD', 'CLIENT_ID', 'CLIENT_SECRET')}
    try:
        for name in saved:
            setattr(auth, name, 'value')
        bw.clear_brighthub_auth_cache()
        assert all(getattr(auth, name) == '' for name in saved)
    finally:
        for name, value in saved.items():
            setattr(auth, name, value)
