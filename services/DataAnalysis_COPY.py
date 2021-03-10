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


def analysis_to_file(res_df,mp_ids,df_fetched_dates):
    string_ints = [str(int) for int in mp_ids]
    str_of_mpids = "_".join(string_ints)
    res_df.to_csv('DataAnalysis/DataAnalysis/'+ 'mpid_' + str_of_mpids + data_analysis.Filename_Analysis.value +  '_createdon_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv', index = False, header=True)

def threshold_diff(df,mpid):
    for threshold in data_analysis.thresholds.value:
        for i in range(len(mpid)+1, len(df.columns)):
            df_filter = df.loc[abs(df[df.columns[i]]) > threshold]
            if(df_filter.shape[0] >  0):
                df_filter.to_csv('DataAnalysis/DataAnalysis/' + data_analysis.Filename_Threshold.value + '_threshold_'+ str(threshold)
                                 +'_for_'+df.columns[i]+ '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv', index=False, header=True)

def plot_data_analysis_graphs(df):

    degrees = 70
    for i in range(1, len(df.columns)):
        res_df_ts = df['ts'].values
        res_df_data = df[df.columns[i]]
        plt.figure(figsize=(22, 22))
        plt.plot(res_df_ts, res_df_data, 'r')
        plt.xlabel('ts',fontsize=24,fontweight="bold")
        plt.ylabel(str(df.columns[i]),fontsize=24,fontweight="bold")
        plt.title('Historic Data for '+ str(df.columns[i]), fontsize=24,fontweight="bold")
        plt.xticks(rotation=degrees,weight = 'bold',fontsize=20)
        plt.yticks(weight = 'bold',fontsize=20)
        plt.savefig('Graphs/Graphs_DataAnalysis/' + 'Analysis_Historic Data_' + 'mp_id_' + str(df.columns[i]) + '_' + datetime.now().strftime("%Y%m%d-%H%M%S"))
        plt.show()

    for i in range(2, len(df.columns)):
        x = df[df.columns[1]]
        y = df[df.columns[i]]
        print(np.corrcoef(x, y))
        plt.figure(figsize=(22, 22))
        plt.scatter(x, y)
        plt.title('A plot to show the correlation between_' + str(df.columns[1]) + '_and_' + str(df.columns[i]),fontsize=24,fontweight="bold")
        plt.xlabel(str(df.columns[1]),fontsize=24,fontweight="bold")
        plt.ylabel(str(df.columns[i]),fontsize=24,fontweight="bold")
        plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='red')
        plt.xticks(rotation=degrees, weight = 'bold',fontsize=20)
        plt.yticks(weight = 'bold',fontsize=20)
        #plt.figure(figsize=(10,10))
        plt.savefig('Graphs/Graphs_DataAnalysis/' + 'Analysis_Correlated_Data_' + 'mp_id_' + str(df.columns[1]) + '_' + str(df.columns[i]) + '_' + datetime.now().strftime("%Y%m%d-%H%M%S"))
        plt.show()


def get_mp_data_data_analysis(start_period, end_period, mp_ids, level,df_fetched_dates, include_missing_mp= False):
    for index, row in df_fetched_dates.iterrows():
        dfs, missing_mp_ids = data_service.get_data_by_ids_period_and_level(row['start_date'], row['end_date'], mp_ids, level, include_missing_mp)
        res_df = pd.DataFrame({'ts': pd.date_range(start=row['end_date'], end=row['end_date'], freq=get_freq_by_level(level))})
        res_df['ts'] = res_df['ts'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')
        for df in dfs:
            res_df = pd.merge(res_df, df, on='ts', how='outer')
        res_df.sort_values(by=['ts'], inplace=True)
        res_df = res_df.drop_duplicates().reset_index(drop=True)
        res_df = res_df.fillna(method='ffill')
        res_df.dropna(inplace=True)
        analysis_to_file(res_df, mp_ids, df_fetched_dates)
        print(row['start_date'], row['end_date'])








    #
    # dfs, missing_mp_ids = data_service.get_data_by_ids_period_and_level(start_period, end_period, mp_ids,
    #                                                                                   level, include_missing_mp)
    # res_df = pd.DataFrame({'ts': pd.date_range(start=start_period, end=end_period, freq=get_freq_by_level(level))})
    #
    # res_df['ts'] = res_df['ts'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')
    #
    # for df in dfs:
    #     res_df = pd.merge(res_df, df, on='ts', how='outer')
    #
    # res_df.sort_values(by=['ts'], inplace=True)
    #
    # res_df = res_df.drop_duplicates().reset_index(drop=True)
    # res_df = res_df.fillna(method='ffill')
    # res_df.dropna(inplace=True)
    #
    # #plot_data_analysis_graphs(res_df)
    #
    # #for i in range(1,len(res_df.columns)-1):
    # #    res_df['diff_FlowValve008_1492 - ' + str(res_df.columns[i + 1])] = res_df[res_df.columns[1]] - res_df[res_df.columns[i + 1]]
    #
    #
    # analysis_to_file(res_df,mp_ids,df_fetched_dates)
    #
    # #threshold_diff(res_df,mp_ids)

def get_dates(sdate,edate):
    thisdict = {}
    x = sdate.split('-')
    y = edate.split('-')
    start_date = date(int(x[2]), int(x[0]), int(x[1]))
    #start_date = date(2020, 5, 31)
    end_date = date(int(y[2]), int(y[0]), int(y[1]))
    delta = timedelta(days=15)
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
    print("\nend of program successful with file saved to DataAnalysis and Graphs folders, inside project folder:services")
    exit()
except Exception as e:
    print(e)
