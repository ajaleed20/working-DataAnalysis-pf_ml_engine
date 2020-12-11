import time

from authlib.integrations.requests_client import OAuth2Session

from services import logging_service

_logger = logging_service.get_logger(__name__)

token = None
client = None
configs = None


def config_oauth(configurations):
    global token
    global client
    global configs
    configs = configurations
    try:
        client = OAuth2Session(configs.IS_CLIENT_ID, configs.IS_CLIENT_SECRET, scope='idmgr permissions')
        token = client.fetch_token(configs.IS_ACCESS_TOKEN_URL, grant_type='client_credentials')
    except Exception as e:
        _logger.error(e)
        raise


def get_access_token():
    global token
    if token != None and client != None:
        if token['expires_at'] > time.time():
            return token['access_token']
        else:
            refresh_token()
            return token['access_token']


def refresh_token():
    global configs
    try:
        token = client.fetch_token(configs.IS,
                                   grant_type='client_credentials')
    except Exception as e:
        _logger(e)
        raise
    return token
