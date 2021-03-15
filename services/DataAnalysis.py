from utils.enumerations import GanualityLevel,data_analysis
from utils.oauth import config_oauth
from services.helper_service import get_mp_data, get_freq_by_level
from services import data_service
import pandas as pd
import config
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime
from datetime import date, timedelta
import time


def analysis_to_file(res_df,mp_ids,sdate,edate):
    sp = sdate.replace('-', '')
    ep = edate.replace('-', '')
    start_date = sp.replace(':', '')
    end_date = ep.replace(':', '')

    string_ints = [str(int) for int in mp_ids]
    str_of_mpids = "_".join(string_ints)
    res_df.to_csv('DataAnalysis/DataAnalysis/'+'mpid_'+str_of_mpids+'_from_'+start_date+'_till_'+end_date+'_createdon_'+datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv', index = False, header=True)

def get_mp_data_data_analysis(start_period, end_period, mp_ids, level,df_fetched_dates, include_missing_mp= False):
    for i in range (data_analysis.start_range.value, data_analysis.end_range.value):
        for index, row in df_fetched_dates.iterrows():
            dfs, missing_mp_ids = data_service.get_data_by_ids_period_and_level(row['start_date'], row['end_date'], [i], level, include_missing_mp)
            res_df = pd.DataFrame({'ts': pd.date_range(start=row['end_date'], end=row['end_date'], freq=get_freq_by_level(level))})
            res_df['ts'] = res_df['ts'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')
            for df in dfs:
                res_df = pd.merge(res_df, df, on='ts', how='outer')
            res_df.sort_values(by=['ts'], inplace=True)
            res_df = res_df.drop_duplicates().reset_index(drop=True)
            res_df = res_df.fillna(method='ffill')
            res_df.dropna(inplace=True)
            analysis_to_file(res_df, [i], row['start_date'],row['end_date'] )
            #time.sleep(10)
            print([i],row['start_date'], row['end_date'])

def get_dates(sdate,edate):
    thisdict = {}
    x = sdate.split('-')
    y = edate.split('-')
    start_date = date(int(x[2]), int(x[0]), int(x[1]))
    end_date = date(int(y[2]), int(y[0]), int(y[1]))
    delta = timedelta(days=10)
    while start_date <= end_date:
        sd = start_date.strftime("%m-%d-%YT00:00:00")
        ed_tmp = start_date + delta
        ed = ed_tmp.strftime("%m-%d-%YT00:00:00")
        temp = start_date
        if ed_tmp >= end_date:
            sd = temp.strftime("%m-%d-%YT00:00:00")
            ed = end_date.strftime("%m-%d-%YT00:00:00")
            thisdict[sd] = ed
            print("last startdate:", sd)
            print("last enddate:", ed)
            break;
        thisdict[sd] = ed
        start_date += delta

    dataframe = pd.DataFrame(list(thisdict.items()), columns=['start_date', 'end_date'])

    return dataframe

def execute_data_analysis(mpid_var, start_period, end_period,granularity_level,df_fetched_dates):
   get_mp_data_data_analysis(start_period, end_period, mpid_var, granularity_level,df_fetched_dates)

#instance = data_analysis.instance.value
config_oauth(config.get_current_config())

try:
    df_fetched_dates = get_dates(data_analysis.start_period.value, data_analysis.end_period.value)
    execute_data_analysis(data_analysis.measuringpoint_var.value, data_analysis.start_period.value, data_analysis.end_period.value,data_analysis.granularity.value,df_fetched_dates)
    print("\nend of program successful with files saved to DataAnalysis folder, inside project folder:services")
    exit()
except Exception as e:
    print(e)
