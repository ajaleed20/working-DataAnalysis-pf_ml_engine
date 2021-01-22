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

    # plotting AB-JOIN time series along provided date range: , sharex=False, gridspec_kw={'hspace': 0}
    fig, axs = plt.subplots(2,figsize=(10,10))
    plt.suptitle('AB-JOIN Pattern Recognition Sequences For Original Data', fontsize='15', fontweight='bold')
    axs[0].set_title('Original Data For Sequence A', fontsize=8, y=0, fontweight='bold')
    axs[1].set_title('Original Data For Sequence B', fontsize=8, y=0, fontweight='bold')
    #axs[1].set_xlabel('Time')
    #axs[0].set_xlabel('Sequence A Index', fontsize=4, fontweight='bold')
    #axs[1].set_xlabel('Sequence B Index', fontsize=4, fontweight='bold')
    axs[0].set_ylabel('Sequence A Values', fontsize=8, fontweight='bold')
    axs[1].set_ylabel('Sequence B Values', fontsize=8, fontweight='bold')
    # ylim_lower = -25
    # ylim_upper = 25
    # axs[0].set_ylim(ylim_lower, ylim_upper)
    # axs[1].set_ylim(ylim_lower, ylim_upper)
    axs[0].plot(A_res_df[A_res_df.columns[1]], c='green')
    axs[1].plot(B_res_df[B_res_df.columns[1]], c='orange')
    plt.autoscale()
    fig.tight_layout(pad=3.0)
    plt.savefig('Graphs/Graphs_Motifs/PatternTechnique3/' + 'OriginalData_Graphs_for_A_and_B_' + datetime.now().strftime(
        "%Y%m%d-%H%M%S") + '.png')

    days_dict = {
        "Three-min": 6,
        "Five-min": 10,
        "Ten-min": 20,
        "Fifteen-min": 30,
        "Twenty-min": 40,
        "Half-Hour": 60,
    }
    m = days_dict['Half-Hour']

   ###############################################################################################################
   ## As a brief about the matrix profile data structure, each row of "TA_ClosestMatch_in_TB"             ##
   ## corresponds to each subsequence within T_A, the first column in "TA_ClosestMatch_in_TB" records           ##
   ## the matrix profile value for each subsequence in T_A(i.e.,the distance to its nearest neighbor            ##
   ## in T_B), and the second column in "TA_ClosestMatch_in_TB" keeps track of the index location of            ##
   ## the nearest neighbor subsequence in T_B.                                                                  ##
   ## One additional side note is that AB - joins are not symmetrical in general.That is, unlike a self - join, ##
   ## the order of the input time series matter.So, an AB - join will produce a different matrix profile than   ##
   ## a BA - join(i.e., for every subsequence in T_B, we find its closest subsequence in T_A).                  ##
   ###############################################################################################################

    TA_ClosestMatch_in_TB = stumpy.stump(T_A=A_res_df[A_res_df.columns[1]], m=m, T_B=B_res_df[B_res_df.columns[1]],ignore_trivial=False)
    TB_ClosestMatch_in_TA = stumpy.stump(T_A=B_res_df[B_res_df.columns[1]], m=m, T_B=A_res_df[A_res_df.columns[1]],ignore_trivial=False)

    TA_ClosestMatch_in_TB_index = TA_ClosestMatch_in_TB[:, 0].argmin()
    plt.clf()
    plt.suptitle('Matrix Profile with Global Minima', fontsize='15', fontweight='bold')
    plt.xlabel('Index')
    plt.ylabel('Euclidean Distance')
    plt.scatter(TA_ClosestMatch_in_TB_index, TA_ClosestMatch_in_TB[TA_ClosestMatch_in_TB_index, 0], c='red', s=100)
    plt.plot(TA_ClosestMatch_in_TB[:, 0])
    plt.savefig('Graphs/Graphs_Motifs/PatternTechnique3/' + 'MatrixProfile_for_AB-Join' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    print(f'GlobalMinima: For each subsequence in T_A, the motif is located at index {TA_ClosestMatch_in_TB_index} of T_A time-series')

    ClosestMatch_in_TB_index = TA_ClosestMatch_in_TB[TA_ClosestMatch_in_TB_index, 1]
    print(f'The motif is located at index {ClosestMatch_in_TB_index} of T_B time-series')

    plt.clf()
    temp_A = A_res_df[A_res_df.columns[1]]
    temp_B = B_res_df[B_res_df.columns[1]]
    plt.suptitle('Matching_Pattern_in_sub-sequence_A', fontsize='15', fontweight='bold')
    plt.plot(temp_A.iloc[TA_ClosestMatch_in_TB_index: TA_ClosestMatch_in_TB_index + m], label='Pattern A')
    plt.autoscale()
    plt.savefig('Graphs/Graphs_Motifs/PatternTechnique3/' + 'Matching_Pattern_in_sub-sequence_A' + datetime.now().strftime(
            "%Y%m%d-%H%M%S") + '.png')
    #plt.show()
    plt.clf()
    plt.suptitle('Matching_Pattern_in_sub-sequence_B', fontsize='15', fontweight='bold')
    plt.autoscale()
    plt.plot(temp_B.iloc[ClosestMatch_in_TB_index: ClosestMatch_in_TB_index + m], label='Pattern B')
    plt.savefig('Graphs/Graphs_Motifs/PatternTechnique3/' + 'Matching_Pattern_in_sub-sequence_B' + datetime.now().strftime(
        "%Y%m%d-%H%M%S") + '.png')

    plt.clf()
    plt.suptitle('Overlaying The Best Matching Motif for AB-Join', fontsize='15', fontweight='bold')
    plt.plot(temp_A.iloc[TA_ClosestMatch_in_TB_index: TA_ClosestMatch_in_TB_index + m], label='Pattern A')
    plt.plot(temp_B.iloc[ClosestMatch_in_TB_index: ClosestMatch_in_TB_index + m], label='Pattern B')
    plt.legend()
    plt.ylabel('Pattern Values')
    plt.autoscale()
    plt.savefig('Graphs/Graphs_Motifs/PatternTechnique3/' + 'Overlaying_A_B_sub-sequences_for_AB-join' + datetime.now().strftime(
            "%Y%m%d-%H%M%S") + '.png')

    # checking matrix profile data
    # chk_df = pd.DataFrame()
    # chk_df = pd.DataFrame(TA_ClosestMatch_in_TB)
    # #chk_df=TA_ClosestMatch_in_TB
    # chk_df.to_csv('DataAnalysis/MotifDataAnalysis/PatternTechnique3/' + 'Matrix_Profile_AB-Join' + datetime.now().strftime(
    #     "%Y%m%d-%H%M%S") + '.csv',
    #         index=False, header=True)

    appended_data = pd.DataFrame()
    data = A_res_df[TA_ClosestMatch_in_TB_index: TA_ClosestMatch_in_TB_index + m]
    data.columns = ['A_ts_'+str(TA_ClosestMatch_in_TB_index), 'index_'+ str(TA_ClosestMatch_in_TB_index)]
    data.reset_index(inplace=True)
    appended_data = data
    data = B_res_df[ClosestMatch_in_TB_index: ClosestMatch_in_TB_index + m]
    data.columns = ['B_ts_'+str(ClosestMatch_in_TB_index), 'index_'+ str(ClosestMatch_in_TB_index)]
    data.reset_index(inplace=True)
    appended_data = pd.concat([appended_data, data], axis=1)
    appended_data.to_csv('DataAnalysis/MotifDataAnalysis/PatternTechnique3/' +  'AB-Join_MotifData_ClosestMatch_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv',
                             index=False, header=True)



def get_mp_data_data_analysis(A_start_period, A_end_period, B_start_period, B_end_period, A_mp_ids, B_mp_ids,level, include_missing_mp= False):
    A_dfs, A_missing_mp_ids = data_service.get_data_by_ids_period_and_level(A_start_period, A_end_period, A_mp_ids,level, include_missing_mp)
    B_dfs, B_missing_mp_ids = data_service.get_data_by_ids_period_and_level(B_start_period, B_end_period, B_mp_ids,level, include_missing_mp)
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
    execute_data_analysis(data_analysis.A_stumpy_measuringpoint_var.value, data_analysis.B_stumpy_measuringpoint_var.value,data_analysis.A_start_period.value, data_analysis.A_end_period.value,data_analysis.B_start_period.value, data_analysis.B_end_period.value,data_analysis.granularity.value)
    print("\nEnd of program successful run with files  saved to DataAnalysis and Graph folders, inside project folder:services.")
    exit()
except Exception as e:
    print(e)