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
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import sys


def get_stumpy_query_pattern(df,Q_df):
    #for original time series sequence
    res_df = df
    res_df.dropna(inplace=True)
    res_df[res_df.columns[1]] = res_df[res_df.columns[1]].replace([np.inf, -np.inf], np.nan)
    res_df[res_df.columns[1]] = res_df[res_df.columns[1]].replace(0, np.nan)
    res_df.dropna(inplace=True)

    # for query/input time series (sub-sequence)
    Q_res_df = Q_df
    Q_res_df.dropna(inplace=True)
    Q_res_df[Q_res_df.columns[1]] = Q_res_df[Q_res_df.columns[1]].replace([np.inf, -np.inf], np.nan)
    Q_res_df[Q_res_df.columns[1]] = Q_res_df[Q_res_df.columns[1]].replace(0, np.nan)
    Q_res_df.dropna(inplace=True)

    # plotting query/input time series (sub-sequence) along datetime
    plt.suptitle('Query Subsequence, Q_df', fontsize='10')
    plt.xlabel('Time', fontsize='5', y=0)
    plt.ylabel('Values', fontsize='7')
    plt.plot(Q_res_df[Q_res_df.columns[0]],Q_res_df[Q_res_df.columns[1]], lw=1, color="green")
    plt.xticks(rotation=70, weight='bold', fontsize=5)
    plt.xticks(weight='bold', fontsize=5)
    plt.savefig('Graphs/Graphs_Motifs/PatternTechnique2/' + 'Motif_Query_Pattern_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')

    days_dict = {
        "Three-min": 6,
        "Five-min": 10,
        "Ten-min": 20,
        "Fifteen-min": 30,
        "Twenty-min": 40,
        "Half-Hour": 60,
    }

    # Since MASS computes z-normalized Euclidean distances, we should z-normalize our subsequences before plotting
    distance_profile = stumpy.core.mass(Q_res_df[Q_res_df.columns[1]], res_df[res_df.columns[1]])
    idx = np.argmin(distance_profile)

    if (distance_profile[idx] > 30):
        print(f"The global minimum is above considerable range of motif detection")
    else:
        print(f"The nearest neighbor to `Query Pattern` is located at index {idx} in `Original Data`")
        temp_res_df = res_df[res_df.columns[1]]
        # plotting z normalization for query subsequence and provided time-series in temp_res_df
        Q_z_norm = stumpy.core.z_norm(Q_res_df[Q_res_df.columns[1]].values)
        T_z_norm = stumpy.core.z_norm(temp_res_df.values[idx:idx + len(Q_res_df)])
        fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
        plt.suptitle('Comparing z normalized Query subsequence and its possible nearest neighbor', fontsize='7')
        axs[0].set_title('Query Pattern', fontsize=5, y=0)
        axs[1].set_title('Closest Pattern', fontsize=5, y=0)
        axs[1].set_xlabel('Time')
        axs[0].set_ylabel('Query Pattern Values',fontsize=5)
        axs[1].set_ylabel('Closest Pattern Values',fontsize=5)
        # ylim_lower = -25
        # ylim_upper = 25
        # axs[0].set_ylim(ylim_lower, ylim_upper)
        # axs[1].set_ylim(ylim_lower, ylim_upper)
        axs[0].plot(Q_z_norm, c='green')
        axs[1].plot(T_z_norm, c='orange')
        plt.autoscale()
        plt.savefig('Graphs/Graphs_Motifs/PatternTechnique2/' + 'Motif_z_normalized_graphs_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')


        # plotting original data for query subsequence and provided time-series in temp_res_df
        fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
        plt.suptitle('Closest Pattern in Original Data (non normalized)', fontsize='7')
        axs[0].set_title('Query Pattern', fontsize=5, y=0)
        axs[1].set_title('Closest Pattern', fontsize=5, y=0)
        axs[1].set_xlabel('Time')
        axs[0].set_ylabel('Query Pattern Values',fontsize=5)
        axs[1].set_ylabel('Closest Pattern Values',fontsize=5)
        # ylim_lower = -25
        # ylim_upper = 25
        # axs[0].set_ylim(ylim_lower, ylim_upper)
        # axs[1].set_ylim(ylim_lower, ylim_upper)
        axs[0].plot(Q_res_df[Q_res_df.columns[1]].values, c = 'green')
        axs[1].plot(temp_res_df.values[idx:idx + len(Q_res_df)], c='orange')
        plt.autoscale()
        plt.savefig('Graphs/Graphs_Motifs/PatternTechnique2/' + 'OriginalData_graphs_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')

        plt.clf()
        plt.suptitle('Overlaying The Best Matching Motif', fontsize='15', fontweight='bold')
        plt.plot(Q_res_df[Q_res_df.columns[1]].values, label='Query Pattern')
        plt.plot(temp_res_df[idx:idx + len(Q_res_df)].values, label='Matching Pattern')
        plt.legend()
        plt.ylabel('Pattern Values')
        #plt.autoscale()
        plt.savefig(
            'Graphs/Graphs_Motifs/PatternTechnique2/' + 'Overlaying_sub-sequences_for_motif' + datetime.now().strftime(
                "%Y%m%d-%H%M%S") + '.png')


        appended_data = pd.DataFrame()
        data = res_df[idx:idx + len(Q_res_df)]
        data.columns = ['ts_'+str(idx), 'Data_'+ str(idx)]
        data.reset_index(inplace=True)
        appended_data = data
        data = Q_res_df
        data.columns = ['ts_Query_Motif' , 'Query_Motif_Data']
        data.reset_index(inplace=True)
        appended_data = pd.concat([appended_data, data], axis=1)
        appended_data.to_csv('DataAnalysis/MotifDataAnalysis/PatternTechnique2/' +  'MotifData_for_' + str(idx)
                                 + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv',
                                 index=False, header=True)
        plt.clf()

        #plotting matching query pattern on original data time-series
        fig = plt.figure()
        fig.set_size_inches(75, 75)
        plt.rcParams['axes.linewidth'] = 5.5
        plt.title('Original Dataset with Matching Query pattern', fontsize='80', fontweight="bold")
        plt.xlabel('Time', fontsize='50', fontweight="bold")
        plt.ylabel('Values', fontsize='50', fontweight="bold")
        plt.xticks(rotation=70, weight='bold', fontsize=60)
        plt.yticks(weight='bold', fontsize=60)
        plt.autoscale()
        plt.plot(res_df[res_df.columns[1]])
        # plt.text(2000, 4.5, 'Cement', color="black", fontsize=20)
        # plt.text(10000, 4.5, 'Cement', color="black", fontsize=20)
        # ax = plt.gca()
        # rect = Rectangle((5000, -4), 3000, 10, facecolor='lightgrey')
        # ax.add_patch(rect)
        # plt.text(6000, 4.5, 'Carpet', color="black", fontsize=20)
        plt.plot(range(idx, idx + len(Q_res_df)), temp_res_df.values[idx:idx + len(Q_res_df)], lw=2)
        plt.savefig('Graphs/Graphs_Motifs/PatternTechnique2/' + 'Matching_Pattern_From_OriginalData' + datetime.now().strftime(
            "%Y%m%d-%H%M%S") + '.png')


        # This simply returns the (sorted) positional indices of the top 16 smallest distances found in the distance_profile
        k = 5
        idxs = np.argpartition(distance_profile, k)[:k]
        idxs = idxs[np.argsort(distance_profile[idxs])]
        fig = plt.figure()
        fig.set_size_inches(75, 75)
        plt.rcParams['axes.linewidth'] = 5.5
        plt.title('Original Dataset with Matching Query pattern', fontsize='80', fontweight="bold")
        # plt.xlabel('Time', fontsize='20')
        # plt.ylabel('Acceleration', fontsize='20')
        plt.plot(res_df[res_df.columns[1]])
        plt.autoscale()
        # plt.text(2000, 4.5, 'Cement', color="black", fontsize=20)
        # plt.text(10000, 4.5, 'Cement', color="black", fontsize=20)
        # ax = plt.gca()
        # rect = Rectangle((5000, -4), 3000, 10, facecolor='lightgrey')
        # ax.add_patch(rect)
        # plt.text(6000, 4.5, 'Carpet', color="black", fontsize=20)
        for idx in idxs:
            plt.plot(range(idx, idx + len(Q_res_df)), temp_res_df.values[idx:idx + len(Q_res_df)], lw=2)
        plt.savefig('Graphs/Graphs_Motifs/PatternTechnique2/' + 'k_Closest_Matching_Patterns' + datetime.now().strftime(
                "%Y%m%d-%H%M%S") + '.png')


def get_mp_data_data_analysis(start_period, end_period, Q_start_period, Q_end_period, mp_ids, Q_mp_ids,level, include_missing_mp= False):
    dfs, missing_mp_ids = data_service.get_data_by_ids_period_and_level(start_period, end_period, mp_ids,
                                                                                      level, include_missing_mp)
    Q_dfs, Q_missing_mp_ids = data_service.get_data_by_ids_period_and_level(Q_start_period, Q_end_period, Q_mp_ids,
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

def execute_data_analysis(mpid_var, Q_mpid_var,start_period, end_period,granularity_level,Q_start_period, Q_end_period):
   get_mp_data_data_analysis(start_period, end_period, Q_start_period, Q_end_period, mpid_var,Q_mpid_var, granularity_level)


instance = data_analysis.instance.value
config_oauth(config.get_current_config())

try:
    execute_data_analysis(data_analysis.stumpy_measuringpoint_var.value, data_analysis.Q_stumpy_measuringpoint_var.value, data_analysis.start_period.value,data_analysis.end_period.value,data_analysis.granularity.value,data_analysis.Q_start_period.value, data_analysis.Q_end_period.value)
    print("\nEnd of program successful run with files  saved to DataAnalysis and Graph folders, inside project folder:services.")
    exit()
except Exception as e:
    print(e)