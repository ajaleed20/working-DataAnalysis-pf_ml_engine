from utils.enumerations import Services, RemoteControllers
from services.http_request_service import get

def get_by_id(id):
    return get(Services.info_service.value, RemoteControllers.lookup.value, 'get', {'id': id})


def get_by_ids(ids):
    return get(Services.info_service.value, RemoteControllers.lookup.value, 'GetByIds', {'ids': ids})

def get_by_category_and_value(category_code, value):
    return get(Services.info_service.value, RemoteControllers.lookup.value,
               'GetByCategoryAndValue', {'lookupCategoryCode': category_code, 'lookupValue':value})