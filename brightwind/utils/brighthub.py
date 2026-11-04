# This module is self-contained and only depends on the 'requests' package, so it can be copied into other
# code bases to interact with the BrightHub APIs.
import os
import time
import warnings
import requests

__all__ = ['request',
           'get_id_token',
           'BrighthubAuthError',
           'BrighthubRequestError']

_BASE_URI = os.getenv('BRIGHTHUB_BASE_URI', 'https://api.brighthub.io')
_SUPPORTED_METHODS = ('GET', 'POST', 'PUT', 'DELETE')
_RETRY_ON_STATUS_METHODS = ('GET',)
_RETRY_STATUS_CODES = (429, 502, 503, 504)
_DEFAULT_MAX_RETRIES = 2
_RETRY_DELAY_SECONDS = 3
# (connect, read) timeouts in seconds. The read timeout is how long to wait for the server to respond.
_DEFAULT_TIMEOUT = (10, 35)


class BrighthubAuthError(Exception):
    """Raised when authentication with BrightHub fails, e.g. invalid API key or an unauthorised request."""
    pass


class BrighthubRequestError(ValueError):
    """
    Raised when a request is rejected by the BrightHub API gateway before it reaches BrightHub, e.g. the endpoint
    does not exist for the HTTP method used. The status_code, method, url and response are available as attributes.
    """
    def __init__(self, message, status_code=None, method=None, url=None, response=None):
        super().__init__(message)
        self.status_code = status_code
        self.method = method
        self.url = url
        self.response = response


# Messages returned by the API gateway (not BrightHub) for an unknown route or a route that does not use API key auth
_GATEWAY_REJECTION_MESSAGES = ('Missing Authentication Token', 'Authorization header requires')


def _get_gateway_rejection_message(response):
    """Return the message if the response is an API gateway rejection, otherwise None."""
    if response.status_code != 403:
        return None
    try:
        body = response.json()
    except ValueError:
        return None
    # BrightHub's own errors include 'error' and 'details', a gateway rejection only has 'message'
    if (isinstance(body, dict) and list(body) == ['message'] and isinstance(body['message'], str)
            and any(known in body['message'] for known in _GATEWAY_REJECTION_MESSAGES)):
        return body['message']
    return None


def _get_web_uri(base_uri):
    # The BrightHub website is the API base URI without the 'api.' sub-domain e.g. https://brighthub.io
    return base_uri.replace('://api.', '://', 1).rstrip('/')


def _get_environment_variable(name):
    if name not in os.environ:
        raise _BrighthubAuth.BrighthubAuthError('{} environmental variable is not set.'.format(name))
    return os.environ[name]


class _BrighthubAuth:
    """
    This class holds the BrightHub authentication state and functions. It is used by request() and get_id_token()
    in this module, and so by LoadBrightHub. Credentials and tokens are cached at class level so that authentication
    happens only once during the analysis, avoiding multiple logins to the Brighthub user pool.
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
            print(response.json)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            if status_code in (400, 401, 403):
                raise _BrighthubAuth.BrighthubAuthError(
                    f"Authentication failed at {auth_url} (status code {status_code}). The BrightHub API key is "
                    "invalid, expired or revoked. Please check the BRIGHTHUB_CLIENT_ID and BRIGHTHUB_CLIENT_SECRET "
                    f"environmental variables, and create a new API key at {_get_web_uri(base_uri)}"
                    "/account-settings/settings if needed."
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
            f"Create and manage API keys at: {_get_web_uri(_BASE_URI)}/account-settings/settings. "
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
                # Fallback to basic auth if client credentials are not provided but legacy credentials are.
                # This method is depreciated in v2.4.0 and will be removed in v3.0.0
                has_legacy_credentials = ((_BrighthubAuth.USERNAME or os.getenv('BRIGHTHUB_EMAIL'))
                                          and (_BrighthubAuth.PASSWORD or os.getenv('BRIGHTHUB_PASSWORD')))
                if not has_legacy_credentials:
                    raise _BrighthubAuth.BrighthubAuthError(
                        "BrightHub API key not found. Please create an API key at "
                        f"{_get_web_uri(_BASE_URI)}/account-settings/settings and set the BRIGHTHUB_CLIENT_ID and "
                        "BRIGHTHUB_CLIENT_SECRET environmental variables. If you have just set them, restart your "
                        "Python environment for them to take effect.") from None
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


def get_id_token():
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

        token = bw.brighthub.get_id_token()
        headers = {"authorization": token}

    """
    if not _BrighthubAuth.ID_TOKEN:
        _BrighthubAuth._get_id_token()
    return _BrighthubAuth.ID_TOKEN


def request(url_extension, method='GET', params=None, json=None, data=None, timeout=_DEFAULT_TIMEOUT,
            max_retries=_DEFAULT_MAX_RETRIES):
    """
    Make a request to the BrightHub APIs. Authentication is handled automatically, see get_id_token().

    A failed request is retried up to max_retries times, with a 3 second delay between attempts. Connection timeouts
    are retried for all methods. Other connection errors, read timeouts and 429, 502, 503 and 504 responses are retried
    for GET only, as a POST, PUT or DELETE may already have been applied by the server and retrying could duplicate it.
    If a 401 is returned, a new token is requested and the request is made once more. A BrighthubAuthError is raised
    if it persists. A BrighthubRequestError is raised if the API gateway rejects the request with a 403, which means the
    endpoint does not exist for the method used or does not accept API key authentication.

    :param url_extension: The end of the url to be concatenated with the BrightHub base URI e.g. '/plants'.
    :type url_extension:  str
    :param method:        Optional. The HTTP method, one of 'GET', 'POST', 'PUT' or 'DELETE'. Not case sensitive.
                          Default 'GET'.
    :type method:         str
    :param params:        Optional. A dictionary, list of tuples or bytes to send as a query string. Default None.
    :type params:         dict, list(tuples), bytes
    :param json:          Optional. A JSON serialisable object to send as the body of the request. Default None.
    :type json:           dict, list
    :param data:          Optional. A dictionary, list of tuples or bytes to send as the body of the request.
                          Default None.
    :type data:           dict, list(tuples), bytes
    :param timeout:       Optional. The time in seconds to wait for the server, as a (connect, read) tuple or a single
                          value used for both. Default (10, 35). The read timeout is how long to wait for the server to
                          respond, so raise it for requests that are expected to take longer.
    :type timeout:        float, tuple(float, float)
    :param max_retries:   Optional. The maximum number of times a failed request is retried, so the request is made
                          at most max_retries + 1 times. Use 0 to turn retrying off. Writes (POST, PUT and DELETE)
                          are still only retried after a connect timeout. Default 2.
    :type max_retries:    int
    :return:              The response returned by the request.
    :rtype:               requests.Response
    :raises ValueError:   If the method is not one of 'GET', 'POST', 'PUT' or 'DELETE', or if max_retries is not a
                          non-negative integer. No request is made.

    **Example usage**
    ::
        import brightwind as bw

        plants = bw.brighthub.request('/plants')
        response = bw.brighthub.request('/plants/<plant_uuid>', method='PUT', json={'name': 'New name'})

        # If the name is needed on its own, it can be aliased when imported
        from brightwind.utils.brighthub import request as brighthub_request

    """
    method = str(method).upper()
    if method not in _SUPPORTED_METHODS:
        raise ValueError("Unsupported method '{}'. Use one of {}.".format(method, ', '.join(_SUPPORTED_METHODS)))
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer, got {!r}.".format(max_retries))

    if not _BrighthubAuth.ID_TOKEN:
        _BrighthubAuth._get_id_token()

    url = "{}{}".format(_BASE_URI, url_extension)

    def _send():
        for attempt in range(max_retries + 1):
            last_attempt = attempt == max_retries
            try:
                response = requests.request(method=method, url=url,
                                            headers={"authorization": _BrighthubAuth.ID_TOKEN},
                                            params=params, json=json, data=data, timeout=timeout)
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

            if (response.status_code in _RETRY_STATUS_CODES and method in _RETRY_ON_STATUS_METHODS
                    and not last_attempt):
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
                    method, url, response.text))

    gateway_message = _get_gateway_rejection_message(response)
    if gateway_message:
        raise BrighthubRequestError(
            "BrightHub rejected {method} {url} before it reached the API (status code 403: {message}). "
            "The endpoint may not exist for the {method} method, or may not accept API key authentication. "
            "Please check the path and the HTTP method.".format(method=method, url=url, message=gateway_message),
            status_code=response.status_code, method=method, url=url, response=response)

    return response
