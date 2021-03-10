import requests
import json
import config
from utils.oauth import get_access_token


def get(service, controller, action, params):
    current_configs = config.get_current_config()
    base_url = current_configs.REMOTE_SERVICES[service]
    base_url += controller
    base_url += '/' + action
    header = {'Authorization': 'Bearer ' + get_access_token()}

    response = requests.get(base_url, params=params, headers=header)
    return json.loads(response.text)

def post(service, controller, action, params):
    current_configs = config.get_current_config()

    base_url = current_configs.REMOTE_SERVICES[service]
    base_url += controller
    base_url += '/' + action
    header = {'Authorization': 'Bearer ' + get_access_token()}

    response = requests.post(base_url, data=params, headers=header)
    return json.loads(response.text)
