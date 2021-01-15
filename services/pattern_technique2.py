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
import sys


def get_stumpy_query_pattern(df,Q_df):
    res_df = df
    res_df.dropna(inplace=True)
    res_df[res_df.columns[1]] = res_df[res_df.columns[1]].replace([np.inf, -np.inf], np.nan)
    res_df[res_df.columns[1]] = res_df[res_df.columns[1]].replace(0, np.nan)
    res_df.dropna(inplace=True)

    Q_res_df = Q_df
    Q_res_df.dropna(inplace=True)
    Q_res_df[Q_res_df.columns[1]] = Q_res_df[Q_res_df.columns[1]].replace([np.inf, -np.inf], np.nan)
    Q_res_df[Q_res_df.columns[1]] = Q_res_df[Q_res_df.columns[1]].replace(0, np.nan)
    Q_res_df.dropna(inplace=True)

    plt.suptitle('Pattern or Query Subsequence, Q_df', fontsize='10')
    plt.xlabel('Time', fontsize='10')
    plt.ylabel('Acceleration', fontsize='10')
    plt.plot(Q_res_df[Q_res_df.columns[0]],Q_res_df[Q_res_df.columns[1]], lw=2, color="C1")
    plt.xticks(rotation=70, weight='bold', fontsize=10)
    plt.show()

    days_dict = {
        "Three-min": 6,
        "Five-min": 10,
        "Ten-min": 20,
        "Fifteen-min": 30,
        "Twenty-min": 40,
        "Half-Hour": 60,
    }
    distance_profile = stumpy.core.mass(Q_res_df[Q_res_df.columns[1]], res_df[res_df.columns[1]])
    idx = np.argmin(distance_profile)
    print(f"The nearest neighbor to `Query Pattern` is located at index {idx} in `Original Data`")
    # Since MASS computes z-normalized Euclidean distances, we should z-normalize our subsequences before plotting
    temp_res_df = res_df[res_df.columns[1]]
    Q_z_norm = stumpy.core.z_norm(Q_res_df[Q_res_df.columns[1]].values)
    T_z_norm = stumpy.core.z_norm(temp_res_df.values[idx:idx + len(Q_res_df)])
    plt.suptitle('Comparing The Query (Orange) And Its Nearest Neighbor (Blue)', fontsize='15')
    plt.xlabel('Time', fontsize='10')
    plt.ylabel('Acceleration', fontsize='10')
    plt.plot(Q_z_norm, lw=2, color="C1")
    plt.plot(T_z_norm, lw=2, color="C2")
    plt.show()





    m = days_dict['Half-Hour']
    mp = stumpy.stump(res_df[res_df.columns[1]], m,ignore_trivial=True)
    for index, value in np.ndenumerate(mp[:, 0]):
        mp[:, 0][index] = np.around(value, 2)

    a = mp[:, 0]
    min_value = mp[:, 0].min()
    max_value = mp[:, 0].max()
    min_index_row = np.where(a == min_value)
    max_index_row = np.where(a == max_value)
    #no_nearest_nghbr = np.argwhere(mp[:, 0] == mp[:, 0].max()).flatten()[0]
    print("Position/Index For Global Minima:", min_index_row)
    print("Value For Global Minima:", min_value)
    print("Position/Index For Global Maxima:", max_index_row)
    print("Value For Global Maxima:", max_value)
    xyz = list(min_index_row[0])


    # for global minima and maxima--------
    appended_data = pd.DataFrame()
    fig = plt.figure()
    fig.set_size_inches(75, 75)
    #plt.figure(figsize=(100, 100))
    plt.rcParams['axes.linewidth'] = 2.5
    plt.plot(mp[:, 0])
    #plt.xlabel('ts', fontsize=24, fontweight="bold")
    plt.ylabel('Maxima = RedLine, Minima (Motifs) = GreenLine', fontsize=55, fontweight="bold")
    plt.title('Matrix Profile with Global Minima and Maxima', fontsize=55, fontweight="bold")
    plt.xticks(rotation=70, weight='bold', fontsize=40)
    plt.yticks(weight='bold', fontsize=40)
    plt.axvline(x=max_index_row, linestyle="dashed", lw = 8.0, color='red')
    for i in range(len(xyz)):
        plt.axvline(x=xyz[i], linestyle="dashed", lw = 8.0, color='green')
    plt.savefig('Graphs/Graphs_Motifs/' + 'GlobalMinimaAndMaxima_MatrixProfile' + '_' + datetime.now().strftime(
        "%Y%m%d-%H%M%S") + '.png')
    if (len(xyz) > 0):
        for i in range(len(xyz)):     #for i in range(1, len(res_df.columns)):
            data = res_df[xyz[i]:xyz[i] + m]
            data.columns = ['ts_'+str(xyz[i]), str(xyz[i])]
            data.reset_index(inplace=True)
            appended_data = pd.concat([appended_data,data], axis =1)
        appended_data.to_csv('DataAnalysis/MotifDataAnalysis/' + 'MotifData_for_' + str(xyz)
                             + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv',
                             index=False, header=True)
    plt.show()

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
            plt.savefig('Graphs/Graphs_Motifs/' + 'Motif_Historic Data_' + str(xyz[i]) + '_' + datetime.now().strftime("%Y%m%d-%H%M%S")+ '.png')
            plt.show()

#--------------------stump with varying window size--------------

    DAY_MULTIPLIER = 1
    x_axis_labels = df[(df.ts.dt.hour == 0)]['ts'].dt.strftime('%b %d').values[::DAY_MULTIPLIER]
    x_axis_labels = np.unique(x_axis_labels)
    x_axis_labels[1::2] = " "
    x_axis_labels, DAY_MULTIPLIER
    days_df = pd.DataFrame.from_dict(days_dict, orient='index', columns=['m'])
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
    plt.savefig('Graphs/Graphs_Motifs/' + 'Motif_VaryingWindowSizes_m' + datetime.now().strftime("%Y%m%d-%H%M%S")+ '.png')
    plt.show()

    m = 60
    orig1 = orig_df_data[xyz[0]:xyz[0] + m]
    orig2 = orig_df_data[xyz[2]:xyz[2] + m]
    fig, axs = plt.subplots(len(xyz))
    plt.suptitle('MPID DataSet', fontsize='20')
    axs[0].set_ylabel("MPID DataSet", fontsize='5')
    axs[0].plot(res_df[res_df.columns[1]], alpha=0.5, linewidth=0.5)
    plt.autoscale()
    plt.show()
    for i in range(len(xyz)):
        plt.axvline(x=xyz[i], linestyle="dashed", lw = 2.0, color='green')
    # axs[0].plot(orig1)
    # axs[0].plot(orig2)
    # rect = Rectangle((643, 0), m, 40, facecolor='lightgrey')
    # axs[0].add_patch(rect)
    # rect = Rectangle((8724, 0), m, 40, facecolor='lightgrey')
    # axs[0].add_patch(rect)
    for i in range(len(xyz)):
        axs[i].set_xlabel("Time", fontsize='10')
        axs[i].set_ylabel("Steam Flow", fontsize='10')
        axs[i].plot(orig_df_data[xyz[i]:xyz[i] + m], color='C1')
        plt.autoscale()
        plt.show()

    # axs[1].set_xlabel("Time", fontsize='20')
    # axs[1].set_ylabel("Steam Flow", fontsize='20')
    # axs[1].plot(orig1, color='C1')
    # axs[2].plot(orig2, color='C1')
    plt.autoscale()
    plt.show()

def get_mp_data_data_analysis(start_period, end_period, Q_start_period, Q_end_period, mp_ids, level, include_missing_mp= False):
    dfs, missing_mp_ids = data_service.get_data_by_ids_period_and_level(start_period, end_period, mp_ids,
                                                                                      level, include_missing_mp)
    Q_dfs, Q_missing_mp_ids = data_service.get_data_by_ids_period_and_level(Q_start_period, Q_end_period, mp_ids,
                                                                        level, include_missing_mp)

    res_df = pd.DataFrame({'ts': pd.date_range(start=start_period, end=end_period, freq=get_freq_by_level(level))})

    res_df['ts'] = res_df['ts'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')

    Q_res_df = pd.DataFrame({'ts': pd.date_range(start=Q_start_period, end=Q_end_period, freq=get_freq_by_level(level))})

    Q_res_df['ts'] = Q_res_df['ts'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')

    for df in dfs:
        res_df = pd.merge(res_df, df, on='ts', how='outer')
    res_df.sort_values(by=['ts'], inplace=True)
    res_df = res_df.drop_duplicates().reset_index(drop=True)
    res_df = res_df.fillna(method='ffill')
    res_df.dropna(inplace=True)

    for Q_df in Q_dfs:
        Q_res_df = pd.merge(Q_res_df, Q_df, on='ts', how='outer')

    Q_res_df.sort_values(by=['ts'], inplace=True)
    Q_res_df = Q_res_df.drop_duplicates().reset_index(drop=True)
    Q_res_df = Q_res_df.fillna(method='ffill')
    Q_res_df.dropna(inplace=True)

    get_stumpy_query_pattern(res_df, Q_res_df)

def execute_data_analysis(mpid_var, start_period, end_period,granularity_level,Q_start_period, Q_end_period):
   get_mp_data_data_analysis(start_period, end_period, Q_start_period, Q_end_period, mpid_var, granularity_level)


instance = data_analysis.instance.value
config_oauth(config.get_current_config())

try:
    execute_data_analysis(data_analysis.stumpy_measuringpoint_var.value, data_analysis.start_period.value, data_analysis.end_period.value,data_analysis.granularity.value,data_analysis.Q_start_period.value, data_analysis.Q_end_period.value)
    print("end of program successful with file saved to DataAnalysis folder, inside project folder.")
    exit()
except Exception as e:
    print(e)