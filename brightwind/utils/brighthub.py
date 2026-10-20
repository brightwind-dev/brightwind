# This module is self-contained and only depends on the 'requests' package, so it can be copied into other
# code bases to interact with the BrightHub APIs.
import os
import time
import warnings
import requests

__all__ = ['brighthub_request',
           'get_brighthub_auth_id_token',
           'BrighthubAuthError']

_BASE_URI = os.getenv('BRIGHTHUB_BASE_URI', 'https://api.brighthub.io')
_SUPPORTED_METHODS = ('GET', 'POST', 'PUT', 'DELETE')
_RETRY_ON_STATUS_METHODS = ('GET',)
_RETRY_STATUS_CODES = (429, 502, 503, 504)
_MAX_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 3
_TIMEOUT_SECONDS = 10


class BrighthubAuthError(Exception):
    """Raised when authentication with BrightHub fails, e.g. invalid API key or an unauthorised request."""
    pass


def _get_environment_variable(name):
    if name not in os.environ:
        raise _BrighthubAuth.BrighthubAuthError('{} environmental variable is not set.'.format(name))
    return os.environ[name]


class _BrighthubAuth:
    """
    This class is used to define general functions that are then called by LoadBrightHub. Functions in this
    class are outside of LoadBrightHub and will be called only once during the analysis and this will avoid making
    multiple login to the Brighthub user pool.
    """

    # List possible errors encountered on Login
    __BRIGHTHUB_LOGIN_ERROR_MAP = {
        "not_authorized": "The Brighthub Email or Password is incorrect",
        "user_not_confirmed": "The User is not confirmed. Please confirm your email and try again.",
        "unexpected_error": "An unexpected error occurred.",
        "new_password_required": "Your password has expired or needs to be reset. "
                                 "Kindly reset your Brighthub password and try again.",
        "password_not_verified": "Could not verify your password. "
                                 "Please ensure you have confirmed your email and the password is correct.",
        "password_verifier": "Could not verify your password. "
                             "Please ensure you have confirmed your email and the password is correct.",
        "ms_sso_user": "Microsoft SSO users cannot authenticate using 'BRIGHTHUB_EMAIL' and 'BRIGHTHUB_PASSWORD'. "
                       "Please migrate to API key authentication."
    }
    ID_TOKEN = ''
    REFRESH_TOKEN = ''
    USERNAME = ''
    PASSWORD = ''
    CLIENT_ID = ''
    CLIENT_SECRET = ''

    # Kept as an attribute of this class for backwards compatibility
    BrighthubAuthError = BrighthubAuthError

    @staticmethod
    def _authenticate_with_client_credentials():
        """
        Authenticate a Brighthub user using the OAuth2 client credentials flow.

        This method retrieves the Brighthub client ID and client secret from environmental
        variables if not already assigned in the _BrighthubAuth class.

        :rtype: str
        :return: The JWT ID token for authenticated requests.
        """

        if not _BrighthubAuth.CLIENT_ID:
            _BrighthubAuth.CLIENT_ID = _get_environment_variable('BRIGHTHUB_CLIENT_ID')

        if not _BrighthubAuth.CLIENT_SECRET:
            _BrighthubAuth.CLIENT_SECRET = _get_environment_variable('BRIGHTHUB_CLIENT_SECRET')

        base_uri = os.getenv('BRIGHTHUB_BASE_URI', 'https://api.brighthub.io')
        auth_url = f"{base_uri}/auth/token"
        try:
            response = requests.post(
                url=auth_url,
                auth=requests.auth.HTTPBasicAuth(_BrighthubAuth.CLIENT_ID, _BrighthubAuth.CLIENT_SECRET),
                data={"grant_type": "client_credentials"},
                timeout=10
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            if status_code in (400, 401, 403):
                raise _BrighthubAuth.BrighthubAuthError(
                    f"Authentication failed (status code {status_code}). The BrightHub API key is invalid, expired or "
                    "revoked. Please check the BRIGHTHUB_CLIENT_ID and BRIGHTHUB_CLIENT_SECRET environmental "
                    "variables, and create a new API key at https://brighthub.io/account-settings/settings if needed."
                ) from e
            raise _BrighthubAuth.BrighthubAuthError(
                f"Authentication failed, BrightHub returned an unexpected response: {e}") from e
        except requests.exceptions.RequestException as e:
            raise _BrighthubAuth.BrighthubAuthError(
                f"Network or connection error during authentication: {e}") from e

        try:
            token_data = response.json()
        except ValueError:
            raise _BrighthubAuth.BrighthubAuthError(f"Invalid JSON response from Brighthub: {response.text}")

        if 'id_token' not in token_data:
            raise _BrighthubAuth.BrighthubAuthError(f"Authentication failed, no id_token returned: {token_data}")

        return token_data['id_token']

    @staticmethod
    def _get_cognito_request():
        """
        Function to form a request for Cognito Auth APIs
        """
        url = "https://cognito-idp.eu-west-1.amazonaws.com/"
        headers = {
            'X-Amz-Target': 'AWSCognitoIdentityProviderService.InitiateAuth',
            'Content-Type': 'application/x-amz-json-1.1'
        }
        client_id = os.getenv("BRIGHTHUB_USER_POOL_CLIENT_ID", "3qkkpikve578cbok46p136au3g")

        return url, headers, client_id
    
    @staticmethod
    def _authenticate_with_basic_auth():
        """Authenticate a Brighthub user using username and password (USER_PASSWORD_AUTH flow) 
        via Cognito and return the ID token and refresh token.

        :rtype: tuple
        :return: A tuple containing the JWT ID token and refresh token for authenticated requests.
        """
        warnings.warn(
            "Authentication using 'BRIGHTHUB_EMAIL' and 'BRIGHTHUB_PASSWORD' is deprecated in v2.4.0 and "
            "will be removed in v3.0.0. \nPlease migrate to API key authentication. "
            "Create and manage API keys at: https://brighthub.io/account-settings/settings. "
            "After generating a key, set the environment variables 'BRIGHTHUB_CLIENT_ID' and "
            "'BRIGHTHUB_CLIENT_SECRET'.",
            FutureWarning
        )
        if not _BrighthubAuth.USERNAME:
            _BrighthubAuth.USERNAME = _get_environment_variable('BRIGHTHUB_EMAIL')

        if not _BrighthubAuth.PASSWORD:
            _BrighthubAuth.PASSWORD = _get_environment_variable('BRIGHTHUB_PASSWORD')

        url, headers, client_id = _BrighthubAuth._get_cognito_request()

        body = {
            "AuthParameters": {
                "USERNAME": _BrighthubAuth.USERNAME,
                "PASSWORD": _BrighthubAuth.PASSWORD
            },
            "AuthFlow": "USER_PASSWORD_AUTH",
            "ClientId": client_id
        }
        response = requests.post(url, headers=headers, json=body)
        login_response = response.json()
        login_response_type = login_response.get("__type")

        # a login error occurred
        if login_response_type:
            if login_response_type == "NotAuthorizedException":
                raise _BrighthubAuth.BrighthubAuthError(
                    _BrighthubAuth.__BRIGHTHUB_LOGIN_ERROR_MAP["not_authorized"])
            elif login_response_type == "UserNotConfirmedException":
                raise _BrighthubAuth.BrighthubAuthError(
                    _BrighthubAuth.__BRIGHTHUB_LOGIN_ERROR_MAP["user_not_confirmed"])
            elif 'This account is linked to Microsoft SSO' in login_response.get("message", ""):
                raise _BrighthubAuth.BrighthubAuthError(
                    _BrighthubAuth.__BRIGHTHUB_LOGIN_ERROR_MAP["ms_sso_user"])
            else:
                raise _BrighthubAuth.BrighthubAuthError(
                    _BrighthubAuth.__BRIGHTHUB_LOGIN_ERROR_MAP["unexpected_error"])

        # challenge returned
        if login_response.get("ChallengeName"):
            if login_response["ChallengeName"] == "NEW_PASSWORD_REQUIRED":
                raise _BrighthubAuth.BrighthubAuthError(
                    _BrighthubAuth.__BRIGHTHUB_LOGIN_ERROR_MAP["new_password_required"])
            elif login_response["ChallengeName"] == "PASSWORD_VERIFIER":
                raise _BrighthubAuth.BrighthubAuthError(
                    _BrighthubAuth.__BRIGHTHUB_LOGIN_ERROR_MAP["password_verifier"])
            else:
                raise _BrighthubAuth.BrighthubAuthError(
                    _BrighthubAuth.__BRIGHTHUB_LOGIN_ERROR_MAP["unexpected_error"])

        # login successful
        id_token = login_response['AuthenticationResult']['IdToken']
        refresh_token = login_response['AuthenticationResult']['RefreshToken']
        return id_token, refresh_token

    @staticmethod
    def _get_id_token():
        """
        Retrieve an ID token for authenticating BrightHub API requests.

        This method attempts to authenticate using API client credentials if
        `BRIGHTHUB_CLIENT_ID` and `BRIGHTHUB_CLIENT_SECRET` are available. If they
        are not provided, it falls back to username/password authentication using
        `BRIGHTHUB_EMAIL` and `BRIGHTHUB_PASSWORD`.

        Username/password authentication is deprecated in v2.4.0 and will be removed in
        v3.0.0. Users should migrate to API key–based authentication.

        Upon success, this method stores the resulting ID token (and refresh token,
        when applicable) in the class-level attributes `ID_TOKEN` and
        `REFRESH_TOKEN`. In case of authentication failure, an appropriate
        exception is raised by the underlying authentication method.

        :rtype: dict
        :return: An empty dictionary upon successful authentication.
        """
        if _BrighthubAuth.REFRESH_TOKEN:
            try:
                _BrighthubAuth._brighthub_refresh_token()
                return {}
            except Exception:
                # If refresh fails, proceed to re-authenticate
                pass

        # Preferred authentication flow: client credentials (API key)
        try:
            _BrighthubAuth.ID_TOKEN = _BrighthubAuth._authenticate_with_client_credentials()
            return {}
        
        except Exception as e:
            if str(e) in [
                f'BRIGHTHUB_CLIENT_ID environmental variable is not set.',
                f'BRIGHTHUB_CLIENT_SECRET environmental variable is not set.'
            ]:
                # Fallback to basic auth if client credentials are not provided
                # This method is depreciated in v2.4.0 and will be removed in v3.0.0
                _BrighthubAuth.ID_TOKEN, _BrighthubAuth.REFRESH_TOKEN = _BrighthubAuth._authenticate_with_basic_auth()
            else:
                raise e

        return {}

    @staticmethod
    def _brighthub_refresh_token():
        """
        Function to generate a new token if the current id_token has expired. The new tokens are assigned to the global
        variables ID_TOKEN, REFRESH_TOKEN.
        In case of an error, an error message will be returned.
        """
        warnings.warn(
            "Refresh token authentication is depreciated in v2.4.0 and will be removed in v3.0.0. "
            "Please migrate to API key authentication.",
            DeprecationWarning,
            stacklevel=3
        )
        url, headers, cognito_client_id = _BrighthubAuth._get_cognito_request()

        body = {
            "AuthParameters": {
                "REFRESH_TOKEN": _BrighthubAuth.REFRESH_TOKEN
            },
            "AuthFlow": "REFRESH_TOKEN_AUTH",
            "ClientId": cognito_client_id
        }
        response = requests.post(url, headers=headers, json=body)
        login_response = response.json()

        # a login error occurred
        if login_response.get("__type"):
            if login_response["__type"] == "NotAuthorizedException":
                raise ImportError(_BrighthubAuth.__BRIGHTHUB_LOGIN_ERROR_MAP["not_authorized"])
            elif login_response["__type"] == "UserNotConfirmedException":
                raise ImportError(_BrighthubAuth.__BRIGHTHUB_LOGIN_ERROR_MAP["user_not_confirmed"])
            else:
                raise ImportError(_BrighthubAuth.__BRIGHTHUB_LOGIN_ERROR_MAP["unexpected_error"])

        id_token = login_response['AuthenticationResult']['IdToken']
        _BrighthubAuth.ID_TOKEN = id_token
        new_refresh_token = ""

        if login_response['AuthenticationResult'].get('RefreshToken'):
            new_refresh_token = login_response['AuthenticationResult']['RefreshToken']

        # add the refresh token only if it has been generated again
        # this functionality hasn't been tested as we could not replicate an expired refresh token
        if new_refresh_token:
            _BrighthubAuth.REFRESH_TOKEN = new_refresh_token

        return {}


def get_brighthub_auth_id_token():
    """
    Get the ID token used to authenticate requests to the BrightHub APIs.

    If no token is cached yet, this authenticates using the API key set in the BRIGHTHUB_CLIENT_ID and
    BRIGHTHUB_CLIENT_SECRET environmental variables. The token is cached and re-used by other BrightHub functions.

    The token is a credential, do not log it, share it or commit it to source control.

    :return: The JWT ID token to be sent in the 'authorization' header of BrightHub API requests.
    :rtype:  str

    **Example usage**
    ::
        import brightwind as bw

        token = bw.get_brighthub_auth_id_token()
        headers = {"authorization": token}

    """
    if not _BrighthubAuth.ID_TOKEN:
        _BrighthubAuth._get_id_token()
    return _BrighthubAuth.ID_TOKEN


def brighthub_request(method, url_extension, params=None, json=None, data=None):
    """
    Make a request to the BrightHub APIs. Authentication is handled automatically, see get_brighthub_auth_id_token().

    Up to 3 attempts are made, with a 3 second delay between attempts and a 10 second timeout. Connection timeouts are
    retried for all methods. Other connection errors, read timeouts and 429, 502, 503 and 504 responses are retried
    for GET only, as a POST, PUT or DELETE may already have been applied by the server and retrying could duplicate it.
    If a 401 is returned, a new token is requested and the request is made once more.

    :param method:        The HTTP method, one of 'GET', 'POST', 'PUT' or 'DELETE'. Not case sensitive.
    :type method:         str
    :param url_extension: The end of the url to be concatenated with the BrightHub base URI e.g. '/plants'.
    :type url_extension:  str
    :param params:        Optional. A dictionary, list of tuples or bytes to send as a query string. Default None.
    :type params:         dict, list(tuples), bytes
    :param json:          Optional. A JSON serialisable object to send as the body of the request. Default None.
    :type json:           dict, list
    :param data:          Optional. A dictionary, list of tuples or bytes to send as the body of the request.
                          Default None.
    :type data:           dict, list(tuples), bytes
    :return:              The response returned by the request. For an unsupported method a response with a
                          status code of 411 is returned and no request is made.
    :rtype:               requests.Response

    **Example usage**
    ::
        import brightwind as bw

        plants = bw.brighthub_request('GET', '/plants')
        response = bw.brighthub_request('PUT', '/plants/<plant_uuid>', json={'name': 'New name'})

    """
    method = str(method).upper()
    if method not in _SUPPORTED_METHODS:
        response = requests.Response()
        response.status_code = 411
        response.reason = "Unsupported method '{}'. Use one of {}.".format(method, ', '.join(_SUPPORTED_METHODS))
        return response

    if not _BrighthubAuth.ID_TOKEN:
        _BrighthubAuth._get_id_token()

    url = "{}{}".format(_BASE_URI, url_extension)

    def _send():
        for attempt in range(_MAX_ATTEMPTS):
            last_attempt = attempt == _MAX_ATTEMPTS - 1
            try:
                response = requests.request(method=method, url=url,
                                            headers={"authorization": _BrighthubAuth.ID_TOKEN},
                                            params=params, json=json, data=data, timeout=_TIMEOUT_SECONDS)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as error:
                # A connect timeout means the request never reached the server so it is safe to retry for every
                # method. For any other connection error or a read timeout the server may already have applied the
                # request, so only retry methods that do not write.
                safe_to_retry = (isinstance(error, requests.exceptions.ConnectTimeout)
                                 or method in _RETRY_ON_STATUS_METHODS)
                if last_attempt or not safe_to_retry:
                    raise
                time.sleep(_RETRY_DELAY_SECONDS)
                continue

            if response.status_code in _RETRY_STATUS_CODES and method in _RETRY_ON_STATUS_METHODS                     and not last_attempt:
                time.sleep(_RETRY_DELAY_SECONDS)
                continue
            return response

    response = _send()

    # If there is an auth error, the token has most likely expired so authenticate again and retry once
    if response.status_code == 401:
        _BrighthubAuth._get_id_token()
        response = _send()
        if response.status_code == 401:
            raise BrighthubAuthError(
                "BrightHub rejected the request as unauthorised (status code 401) even after re-authenticating. "
                "Please check your API key and that it has permission to access {} {}. Response: {}".format(
                    method, url_extension, response.text))

    return response
