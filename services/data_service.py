import pandas as pd
import numpy as np
from dateutil import parser
from services.http_request_service import post
from services.lookup_management_service import get_by_category_and_value
from utils.enumerations import GanualityLevel, Services, RemoteControllers
from utils.remote_dtos import HistoricTrendDTO
import pytz



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
            df['ts'] = df['ts'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')
            dfs.append(df)
        elif include_missing_mp:
            missing_mp_ids.append(id)
            dfs.append(pd.DataFrame(columns=['ts', str(id)]))
        else:
            raise Exception('No data found against MP id: ' + str(id))

    return dfs, missing_mp_ids


def get_data_by_ids_period_and_level_for_filling_time_relevant_kpis(start_period, end_period, mp_ids, level=GanualityLevel.one_hour.value, include_missing_mp = False):
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

            if id == 660:
                df.columns = ['ts', 'Article_'+str(id)]
               # fill_lookups(df)
            elif id == 13749:
                df.columns = ['ts', 'StatusFillingValveFiller-'+str(id)]
            elif id == 1637:
                df.columns = ['ts', 'FillingTime_' + str(id)]
            elif id == 1958:
                df.columns = ['ts', 'FillingStateValve_' + str(id)]
            elif id == 1797:
                df.columns = ['ts', 'FillingValueValve_' + str(id)]
            elif id == 557:
                df.columns = ['ts', 'FillingOrganUnderfilled_' + str(id)]
            elif id == 567:
                df.columns = ['ts', 'FillingOrganOverfilled_' + str(id)]
            else:
                df.columns = ['ts', str(id)]

            df['ts'] = pd.to_datetime(df['ts'])
            df['ts'] = df['ts'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')
            dfs.append(df)

        elif include_missing_mp:
            missing_mp_ids.append(id)
            dfs.append(pd.DataFrame(columns=['ts', str(id)]))
        else:
            raise Exception('No data found against MP id: ' + str(id))

    return dfs, missing_mp_ids

def fill_lookups(df):
    for index, row in df.iterrows():
        if row['Article_660'] is  not np.nan:
            lookup = get_by_category_and_value('Article', int(row['Article_660']));
            if isinstance(lookup,dict):
                if(lookup['Name'] is not None):
                    row['Article_660'].str = lookup['Name']
                    print(row['Article_660'])