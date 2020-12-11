class DemoConfig:

    IS_CLIENT_ID = 'POCKETFACTORY_PROD.CLIENT.CC'
    IS_CLIENT_SECRET = '3FAB5367-27AE-476D-9218-0565A7ACAFFA'
    IS_ACCESS_TOKEN_URL = 'https://pocketfactory-demo.ssidecisions.com:55335/core/connect/token'
    IS_Scopes = 'idmgr permissions'

    REMOTE_SERVICES = {
        'INFO_SERVICE': 'https://pocketfactory-demo.ssidecisions.com:44338/api/',
        'DATA_SERVICE': 'https://pocketfactory-demo.ssidecisions.com:44337/api/',
        'ANALYTICAL_SERVICE': 'https://pocketfactory-demo.ssidecisions.com:44340/api/'
    }
class ProdQaConfig:

    IS_CLIENT_ID = 'POCKETFACTORY_PROD.CLIENT.CC'
    IS_CLIENT_SECRET = '3FAB5367-27AE-476D-9218-0565A7ACAFFA'
    IS_ACCESS_TOKEN_URL = 'http://prodqa.northeurope.cloudapp.azure.com:55335/core/connect/token'
    IS_Scopes = 'idmgr permissions'

    REMOTE_SERVICES = {
        'INFO_SERVICE': 'http://prodqa.northeurope.cloudapp.azure.com:44338/api/',
        'DATA_SERVICE': 'http://prodqa.northeurope.cloudapp.azure.com:44337/api/',
        'ANALYTICAL_SERVICE': 'http://prodqa.northeurope.cloudapp.azure.com:44340/api/'
    }

class HbcConfig:
    IS_CLIENT_ID = 'POCKETFACTORY_PROD.CLIENT.CC'
    IS_CLIENT_SECRET = '3FAB5367-27AE-476D-9218-0565A7ACAFFA'
    IS_ACCESS_TOKEN_URL = 'https://pocketfactory-hbc.ssidecisions.com:55335/core/connect/token'
    IS_Scopes = 'idmgr permissions'

    REMOTE_SERVICES = {
        'INFO_SERVICE': 'https://pocketfactory-hbc.ssidecisions.com:44338/api/',
        'DATA_SERVICE': 'https://pocketfactory-hbc.ssidecisions.com:44337/api/',
        'ANALYTICAL_SERVICE': 'https://pocketfactory-hbc.ssidecisions.com:44340/api/'
    }

    # 'hbc for HBC instance
    # 'demo' for Demo instance
def get_current_config(instance='hbc'):
    if instance == 'demo':
        return DemoConfig
    elif instance == 'hbc':
        return HbcConfig
    elif instance == 'prod_qa':
        return ProdQaConfig