from utils.enumerations import GanualityLevel,data_analysis
from utils.oauth import config_oauth
from services.helper_service import get_mp_data, get_freq_by_level
from services import data_service
import pandas as pd
import config
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import stumpy
from matplotlib.patches import Rectangle
from matplotlib.pyplot import cm
from datetime import datetime

def get_stumpy_patterns(df):
    res_df = df
    res_df.dropna(inplace=True)
    res_df[res_df.columns[1]] = res_df[res_df.columns[1]].replace([np.inf, -np.inf], np.nan)
    res_df[res_df.columns[1]] = res_df[res_df.columns[1]].replace(0, np.nan)
    res_df.dropna(inplace=True)

    days_dict = {
        "Three-min": 6,
        "Five-min": 10,
        "Ten-min": 20,
        "Fifteen-min": 30,
        "Twenty-min": 40,
        "Half-Hour": 60,
    }
    m = days_dict['Half-Hour']
    mp = stumpy.stump(res_df[res_df.columns[1]], m)
    for index, value in np.ndenumerate(mp[:, 0]):
        mp[:, 0][index] = np.around(value, 2)

    a = mp[:, 0]
    min_value = mp[:, 0].min()
    max_value = mp[:, 0].max()
    min_index_row = np.where(a == min_value)
    max_index_row = np.where(a == max_value)
    print("Position/Index For Global Minima:", min_index_row)
    print("Value For Global Minima:", min_value)
    print("Position/Index For Global Maxima:", max_index_row)
    print("Value For Global Maxima:", max_value)

    fig, axs = plt.subplots(len(min_index_row)+1, sharex=True, gridspec_kw={'hspace': 0})
    plt.suptitle('Motif (Pattern) Discovery', fontsize='10')
    res_df_ts = res_df['ts'].values
    res_df_data = res_df[res_df.columns[1]]
    plt.figure(figsize=(22, 22))
    plt.ylabel(str(df.columns[1]), fontsize=24, fontweight="bold")
    axs[0].plot(res_df_data)
    axs[0].set_ylabel('Flow Valve 008', fontsize='05')
    axs[1].set_xlabel('Time', fontsize='05')
    axs[1].set_ylabel('Matrix Profile', fontsize='05')
    axs[1].plot(mp[:, 0])
    plt.autoscale()
    plt.show()

    xyz = list(min_index_row[0])

    fig_motif, axs_motif = plt.subplots(len(xyz), sharex=True, gridspec_kw={'hspace': 0})
    plt.suptitle('Matching Patterns', fontsize='10')
    color = iter(cm.rainbow(np.linspace(0, 10)))
    orig_df_ts = res_df['ts'].values
    orig_df_data = res_df[res_df.columns[1]]
    degrees = 70
    if (len(xyz) > 0):
        for i in range(len(xyz)):     #for i in range(1, len(res_df.columns)):
            orig_df_t = orig_df_ts[xyz[i]:xyz[i] + m]
            orig_df_d = orig_df_data[xyz[i]:xyz[i] + m]
            #res_df_data = res_df[df.columns[i]]
            plt.figure(figsize=(22, 22))
            plt.plot(orig_df_t, orig_df_d, 'r')
            plt.xlabel('ts', fontsize=24, fontweight="bold")
            plt.ylabel(str(xyz[i]), fontsize=24, fontweight="bold")
            plt.title('Historic Data for ' + str(xyz[i]), fontsize=24, fontweight="bold")
            plt.xticks(rotation=degrees, weight='bold', fontsize=20)
            plt.yticks(weight='bold', fontsize=20)
            plt.savefig('Graphs_Motifs/' + 'Motif_Historic Data_' + str(xyz[i]) + '_' + datetime.now().strftime("%Y%m%d-%H%M%S")+ '.png')
            plt.show()

                            #--------------------stump with varying window size--------------

    DAY_MULTIPLIER = 1
    x_axis_labels = df[(df.ts.dt.hour == 0)]['ts'].dt.strftime('%b %d').values[::DAY_MULTIPLIER]
    x_axis_labels = np.unique(x_axis_labels)
    x_axis_labels[1::2] = " "
    x_axis_labels, DAY_MULTIPLIER
    days_df = pd.DataFrame.from_dict(days_dict, orient='index', columns=['m'])
    days_df.head()

    fig, axs = plt.subplots(len(days_df), sharex=True, gridspec_kw={'hspace': 0})
    fig.text(0.5, -0.1, 'Subsequence Start Date', ha='center', fontsize='10')
    fig.text(0.08, 0.5, 'Matrix Profile', va='center', rotation='vertical', fontsize='10')
    for i, varying_m in enumerate(days_df['m'].values):
        mp = stumpy.stump(res_df[res_df.columns[1]], varying_m)
        axs[i].plot(mp[:, 0])
        title = f"m = {varying_m}"
        axs[i].set_title(title, fontsize=10, y=.5)

    plt.autoscale()
    plt.xticks(rotation=75)
    plt.suptitle('Matrix Profile Graph with Varying Window Sizes', fontsize='10')
    plt.savefig('Graphs_Motifs/' + 'Motif_VaryingWindowSizes_' + datetime.now().strftime("%Y%m%d-%H%M%S")+ '.png')
    plt.show()

def get_mp_data_data_analysis(start_period, end_period, mp_ids, level, include_missing_mp= False):
    dfs, missing_mp_ids = data_service.get_data_by_ids_period_and_level(start_period, end_period, mp_ids,
                                                                                      level, include_missing_mp)
    res_df = pd.DataFrame({'ts': pd.date_range(start=start_period, end=end_period, freq=get_freq_by_level(level))})

    res_df['ts'] = res_df['ts'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')

    for df in dfs:
        res_df = pd.merge(res_df, df, on='ts', how='outer')

    res_df.sort_values(by=['ts'], inplace=True)

    res_df = res_df.drop_duplicates().reset_index(drop=True)
    res_df = res_df.fillna(method='ffill')
    res_df.dropna(inplace=True)
    get_stumpy_patterns(res_df)

def execute_data_analysis(mpid_var, start_period, end_period,granularity_level):
   get_mp_data_data_analysis(start_period, end_period, mpid_var, granularity_level)


instance = data_analysis.instance.value
config_oauth(config.get_current_config())

try:
    execute_data_analysis(data_analysis.stumpy_measuringpoint_var.value, data_analysis.start_period.value, data_analysis.end_period.value,data_analysis.granularity.value)
    print("end of program successful with file saved to DataAnalysis folder, inside project folder.")
    exit()
except Exception as e:
    print(e)