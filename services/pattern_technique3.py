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


def get_stumpy_query_pattern(A_df,B_df):

    #for original time series sequence
    A_res_df = A_df
    A_res_df.dropna(inplace=True)
    A_res_df[A_res_df.columns[1]] = A_res_df[A_res_df.columns[1]].replace([np.inf, -np.inf], np.nan)
    A_res_df[A_res_df.columns[1]] = A_res_df[A_res_df.columns[1]].replace(0, np.nan)
    A_res_df.dropna(inplace=True)

    # for query/input time series (sub-sequence)
    B_res_df = B_df
    B_res_df.dropna(inplace=True)
    B_res_df[B_res_df.columns[1]] = B_res_df[B_res_df.columns[1]].replace([np.inf, -np.inf], np.nan)
    B_res_df[B_res_df.columns[1]] = B_res_df[B_res_df.columns[1]].replace(0, np.nan)
    B_res_df.dropna(inplace=True)

    # plotting query/input time series (sub-sequence) along datetime
    plt.suptitle('Query Subsequence, Q_df', fontsize='10')
    plt.xlabel('Time', fontsize='5', y=0)
    plt.ylabel('Values', fontsize='7')
    plt.plot(B_res_df[B_res_df.columns[0]],B_res_df[B_res_df.columns[1]], lw=1, color="green")
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
    distance_profile = stumpy.core.mass(B_res_df[B_res_df.columns[1]], res_df[res_df.columns[1]])
    idx = np.argmin(distance_profile)
    print(f"The nearest neighbor to `Query Pattern` is located at index {idx} in `Original Data`")
    temp_res_df = A_res_df[A_res_df.columns[1]]
    # plotting z normalization for Buery subsequence and provided time-series in temp_res_df
    B_z_norm = stumpy.core.z_norm(B_res_df[B_res_df.columns[1]].values)
    T_z_norm = stumpy.core.z_norm(temp_res_df.values[idx:idx + len(B_res_df)])
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
    axs[0].plot(B_z_norm, c='green')
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
    axs[0].plot(B_res_df[B_res_df.columns[1]].values, c = 'green')
    axs[1].plot(temp_res_df.values[idx:idx + len(B_res_df)], c='orange')
    plt.autoscale()
    plt.savefig('Graphs/Graphs_Motifs/PatternTechnique2/' + 'OriginalData_graphs_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')

    appended_data = pd.DataFrame()
    data = A_res_df[idx:idx + len(B_res_df)]
    data.columns = ['ts_'+str(idx), 'index_'+ str(idx)]
    data.reset_index(inplace=True)
    appended_data = data
    data = B_res_df
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
    plt.plot(range(idx, idx + len(B_res_df)), temp_res_df.values[idx:idx + len(B_res_df)], lw=2)
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
        plt.plot(range(idx, idx + len(B_res_df)), temp_res_df.values[idx:idx + len(B_res_df)], lw=2)
    plt.savefig('Graphs/Graphs_Motifs/PatternTechnique2/' + 'k_Closest_Matching_Patterns' + datetime.now().strftime(
            "%Y%m%d-%H%M%S") + '.png')


def get_mp_data_data_analysis(A_start_period, A_end_period, B_start_period, B_end_period, A_mp_ids, B_mp_ids,level, include_missing_mp= False):
    A_dfs, A_missing_mp_ids = data_service.get_data_by_ids_period_and_level(A_start_period, A_end_period, A_mp_ids,
                                                                                      level, include_missing_mp)
    B_dfs, B_missing_mp_ids = data_service.get_data_by_ids_period_and_level(B_start_period, B_end_period, B_mp_ids,
                                                                        level, include_missing_mp)

    A_res_df = pd.DataFrame({'ts': pd.date_range(start=A_start_period, end=A_end_period, freq=get_freq_by_level(level))})

    A_res_df['ts'] = A_res_df['ts'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')

    B_res_df = pd.DataFrame({'ts': pd.date_range(start=B_start_period, end=B_end_period, freq=get_freq_by_level(level))})

    B_res_df['ts'] = B_res_df['ts'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')

    for A_df in A_dfs:
        A_res_df = pd.merge(A_res_df, A_df, on='ts', how='outer')
    A_res_df.sort_values(by=['ts'], inplace=True)
    A_res_df = A_res_df.drop_duplicates().reset_index(drop=True)
    A_res_df = A_res_df.fillna(method='ffill')
    A_res_df.dropna(inplace=True)

    for B_df in B_dfs:
        B_res_df = pd.merge(B_res_df, B_df, on='ts', how='outer')

    B_res_df.sort_values(by=['ts'], inplace=True)
    B_res_df = B_res_df.drop_duplicates().reset_index(drop=True)
    B_res_df = B_res_df.fillna(method='ffill')
    B_res_df.dropna(inplace=True)

    get_stumpy_query_pattern(A_res_df, B_res_df)

def execute_data_analysis(A_mpid_var,B_mpid_var, A_start_period, A_end_period, B_start_period, B_end_period,granularity_level):
   get_mp_data_data_analysis(A_start_period, A_end_period, B_start_period, B_end_period,A_mpid_var,B_mpid_var, granularity_level)


instance = data_analysis.instance.value
config_oauth(config.get_current_config())

try:
    execute_data_analysis(data_analysis.A_stumpy_measuringpoint_var, data_analysis.B_stumpy_measuringpoint_var,data_analysis.A_start_period.value, data_analysis.A_end_period.value,data_analysis.B_start_period.value, data_analysis.B_end_period.value,data_analysis.granularity.value)
    print("\nEnd of program successful run with files  saved to DataAnalysis and Graph folders, inside project folder:services.")
    exit()
except Exception as e:
    print(e)