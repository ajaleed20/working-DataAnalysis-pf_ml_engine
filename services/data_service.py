import pandas as pd
from dateutil import parser
from services.http_request_service import post
from utils.enumerations import GanualityLevel, Services, RemoteControllers
from utils.remote_dtos import HistoricTrendDTO



def get_data_by_ids_period_and_level(start_period, end_period, mp_ids, level=GanualityLevel.one_hour.value, include_missing_mp = False):
    dfs = []
    start_period = parser.parse(start_period)
    end_period = parser.parse(end_period)
    dto = HistoricTrendDTO(start_period, end_period, None, level, mp_ids)
    data = post(Services.data_service.value, RemoteControllers.data_trend.value, 'GetAggregatedDataByIDS',
                dto.__dict__)
    missing_mp_ids = []

    if ('Message' in data):
        raise Exception('unable to get data from data service')

    for id in mp_ids:
        if str(id) in data and len(data[str(id)]) > 0:
            mp_dict = dict(enumerate(data[str(id)]))
            df = pd.DataFrame.from_dict(mp_dict, "index")
            df.drop(columns='Description', axis=1, inplace=True)
            df.columns = ['ts', str(id)]
            df['ts'] = pd.to_datetime(df['ts'])
            df['ts'] = pd.to_datetime(df['ts']).dt.tz_localize(None)
            dfs.append(df)
        elif include_missing_mp:
            missing_mp_ids.append(id)
            dfs.append(pd.DataFrame(columns=['ts', str(id)]))
        else:
            raise Exception('No data found against MP id: ' + str(id))

    return dfs, missing_mp_ids
