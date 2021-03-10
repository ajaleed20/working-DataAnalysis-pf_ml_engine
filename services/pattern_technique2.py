from utils.enumerations import GanualityLevel,data_analysis
from utils.oauth import config_oauth
from services.helper_service import get_mp_data, get_freq_by_level
from services import data_service
import pandas as pd
import config
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import seaborn as sns
import stumpy
from matplotlib.patches import Rectangle
from matplotlib.pyplot import cm
from datetime import datetime
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
from matplotlib.colors import LinearSegmentedColormap

import sys

def create_heatmap_bar_plot(df):
    plt.clf()
    #fig, ax = plt.subplots(figsize=(20, 20))
    # plt.figure(figsize=(50, 25))
   # plt.title('Heatmap for ' + str(df.columns[1]), fontsize=20, fontweight="bold")
    #ax = plt.axes()
    #ax.set_xticklabels(df['ts'].dt.strftime('%d-%m-%Y'))
    # fig, ax = plt.subplots(figsize=(20, 20))
    df['ts'] = df['ts'].dt.strftime('%d-%m-%Y-%hh-%mm-%ss')
    df['ts'] = df['ts'].astype('str')
    heatmap1_data = pd.pivot_table(df, values=df.columns[1], columns=df.columns[0],fill_value=1)
    # df['ts'] = df['ts'].dt.strftime('%d-%m-%Y')
    # df['ts'] = df['ts'].astype('str')
    df[df.columns[1]] = df[df.columns[1]].astype('str')
    #sns.set(font_scale=5.0)

    heat = sns.heatmap(heatmap1_data, fmt="g",cmap='YlOrRd') # cmap="YlGnBu")
    #ax.set_xticklabels(df['ts'].dt.strftime('%d-%m-%Y'))
    heat.set_title('Heatmap of '+ str(df.columns[1]), fontsize=5)
    # heat.set_xticklabels(heat.get_xticklabels(), rotation=30)
    # heat.set_xticklabels(heat.get_xmajorticklabels(), fontsize = 18, rotation=45)
    #heat.set_yticklabels(heat.get_ymajorticklabels(), fontsize = 18)
    # res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 18)
    # res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 18)
    #sns.heatmap(train_set1, annot=True, fmt='g', cmap='viridis')
    figure = heat.get_figure()
    figure.savefig('Graphs/Graphs_Motifs/PatternTechnique2/' + 'HeatMap_for_'+ str(df.columns[1]) +'_'+ datetime.now().strftime(
        "%Y%m%d-%H%M%S") + '.png')
    plt.show()

    #
    # plt.clf()
    # #plt.figure(figsize=(45, 45))
    # fig, ax = plt.subplots(figsize=(70, 30))  # Sample figsize in inches
    # plt.xticks(rotation=70, weight='bold', fontsize=15)
    # plt.yticks(rotation=70, weight='bold', fontsize=15)
    #
    # # sns.heatmap(df1.iloc[:, 1:6:], annot=True, linewidths=.5, ax=ax)
    # ##     #
    # heatmap1_data = pd.pivot_table(df, values='1493', columns='ts', fill_value=1)
    # df['ts'] = df['ts'].astype('str')
    # df['1493'] = df['1493'].astype('str')
    # heat = sns.heatmap(heatmap1_data, cmap="YlGnBu", fmt = "d", ax=ax)
    # # sns.heatmap(train_set1, annot=True, fmt='g', cmap='viridis')
    # figure = heat.get_figure()
    # figure.savefig('Graphs/Graphs_Motifs/PatternTechnique2/' + 'LineGraph_increasing_values' + datetime.now().strftime(
    #     "%Y%m%d-%H%M%S") + '.png')
    # plt.show()

    #####################################running##########################################
    # plt.clf()
    # heatmap1_data = pd.pivot_table(df, values='1493', columns='ts',fill_value=1)
    # df['ts'] = df['ts'].astype('str')
    # df['1493'] = df['1493'].astype('str')
    # heat = sns.heatmap(heatmap1_data, annot=True, fmt="g",cmap='viridis') # cmap="YlGnBu")
    # #sns.heatmap(train_set1, annot=True, fmt='g', cmap='viridis')
    # figure = heat.get_figure()
    # figure.savefig('Graphs/Graphs_Motifs/PatternTechnique2/' + 'LineGraph_increasing_values' + datetime.now().strftime(
    #     "%Y%m%d-%H%M%S") + '.png')
    # plt.show()
#####################################running##########################################


    plt.clf()
    sns.heatmap(df[df.columns[1]], annot = 'true' , cmap = 'coolwarm')

    #
    # plt.figure(figsize=(20, 20))
    # #df.set_index('ts', inplace=True)
    # plt.plot(df[df.columns[0]], df[df.columns[1]])
    # plt.xticks(rotation=70)

    plt.show()
    plt.clf()


def drop_nan(df,Q_df):
    res_df = df
    res_df.dropna(inplace=True)
    res_df[res_df.columns[1]] = res_df[res_df.columns[1]].replace([np.inf, -np.inf], np.nan)
    res_df[res_df.columns[1]] = res_df[res_df.columns[1]].replace(0, np.nan)
    res_df.dropna(inplace=True)

    # for query/input time series (sub-sequence)
    Q_res_df = Q_df
    Q_res_df.dropna(axis=0, how='all', inplace=True)
    Q_res_df.dropna(inplace=True)
    Q_res_df[Q_res_df.columns[1]] = Q_res_df[Q_res_df.columns[1]].replace([np.inf, -np.inf], np.nan)
    Q_res_df[Q_res_df.columns[1]] = Q_res_df[Q_res_df.columns[1]].replace(0, np.nan)
    Q_res_df.dropna(inplace=True)

    return res_df,Q_res_df


def color_fill(value,threshold):
  if value < threshold:
    color = 'green'
  elif value >= threshold:
    color = 'red'
  else:
    color = ''
  return 'color: %s' % color


def get_colors(v):
    if v > 1:
        return 'background-color: yellow'
    else:
        return 'background-color: green'


def threshold_diff(df,mpid):
    df.style.applymap(get_colors)
    df.to_csv('DataAnalysis/DataAnalysis/' + 'DataAbove_threshold_' + 'color_df' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv',
              index=False, header=True)

    # df.style.applymap(get_colors,subset=df.columns[i])
    # # for index, row in df.iterrows():
    # #     if row['Article_660'] is  not np.nan:
    # #         lookup = get_by_category_and_value('Article', row['Article_660']);


    for threshold, i in zip(mpid, range(2, len(df.columns[1:]))):
        df.style.applymap(get_colors)
    #
    # for threshold in mpid:
    #     for i in range(2, len(df.columns[1:])):
    #         df.style.applymap(color_fill, subset=df.columns[i])
            # if df.loc[df[df.columns[i]]] > threshold:
            #     color = 'red'
            # else:
            #     color = ''
            # df.style.applymap(color)
    df.to_csv('DataAnalysis/DataAnalysis/' + 'DataAbove_threshold_' + str(threshold)
                       + '_for_' + df.columns[i] + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv',
                       index=False, header=True)
            #
            # df_filter = df.loc[abs(df[df.columns[i]]) > threshold]
            # if(df_filter.shape[0] >  0):
            #     df_filter.to_csv('DataAnalysis/DataAnalysis/' + 'DataAbove_threshold_'+ str(threshold)
            #                      +'_for_'+df.columns[i]+ '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv', index=False, header=True)



    # for threshold in data_analysis.thresholds.value:
    #     for i in range(len(df_+1, len(df.columns[1:])):
    #         df_filter = df.loc[abs(df[df.columns[i]]) > threshold]
    #         if(df_filter.shape[0] >  0):
    #             df_filter.to_csv('DataAnalysis/DataAnalysis/' + 'DataAbove_threshold_'+ str(threshold)
    #                              +'_for_'+df.columns[i]+ '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv', index=False, header=True)


def get_stumpy_query_pattern_filling_time_relevant_kpis(df,Q_df):

    df.to_csv('DataAnalysis/MotifDataAnalysis/PatternTechnique2/' + 'FillingTime_RKPIs_Complete_Data_' + '_' + datetime.now().strftime(
                "%Y%m%d-%H%M%S") + '.csv',index=False, header=True)

    #for original time series sequence
    df.dropna(thresh = 4, inplace=True)
    df = df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.replace(np.nan, 'NA', regex=True)
    #print(df.replace(r'^\s*$', np.nan, regex=True))
    res_df = df
    #res_df = res_df.replace(r'^\s*$', np.NaN, regex=True)
    #res_df = res_df.replace('', np.nan, regex=True)
    #res_df = df.dropna(axis = 0 , how = 'all')
    #res_df.dropna(axis = 0 , how = 'all')
    #dropna(self, axis=0, how="any", thresh=None, subset=None, inplace=False)
    #res_df = res_df.replace(np.nan, 'NA', regex=True)

    # for query/input time series (sub-sequence)
    Q_res_df = Q_df
    Q_res_df.dropna(how='all', inplace = True)
    #Q_res_df = Q_res_df.replace(np.nan, 'NA', regex=True)

    df.to_csv('DataAnalysis/MotifDataAnalysis/PatternTechnique2/' + 'FillingTime_RKPIs_Data_' + '_' + datetime.now().strftime(
                "%Y%m%d-%H%M%S") + '.csv',index=False, header=True)

    #df, Q_res_df = drop_nan(df, Q_df)
    # threshold implementation
    threshold_diff(df,data_analysis.thresholds.value)

def get_stumpy_query_pattern(df,Q_df):

    days_dict={
        "Three-min": 6,
        "Five-min": 10,
        "Ten-min": 20,
        "Fifteen-min": 30,
        "Twenty-min": 40,
        "Half-Hour": 60,
    }

    res_df,Q_res_df = drop_nan(df,Q_df)

    # create_heatmap_bar_plot(res_df)
    plt.clf()
    # plotting query/input time series (sub-sequence) along datetime
    plt.suptitle('Query Subsequence, Q_df', fontsize='10')
    plt.xlabel('Time', fontsize='5', y=0)
    plt.ylabel('Values', fontsize='7')
    plt.plot(Q_res_df[Q_res_df.columns[1]], lw=1, color="green")
    plt.xticks(rotation=70, weight='bold', fontsize=5)
    plt.xticks(weight='bold', fontsize=5)
    plt.savefig('Graphs/Graphs_Motifs/PatternTechnique2/' + 'Query_Pattern_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')

    # Since MASS computes z-normalized Euclidean distances, we should z-normalize our subsequences before plotting
    distance_profile = stumpy.core.mass(Q_res_df[Q_res_df.columns[1]], res_df[res_df.columns[1]])
    idx = np.argmin(distance_profile)

    max_val_res_df = res_df[res_df.columns[1]].max()
    max_val_Q_df = Q_res_df[Q_res_df.columns[1]].max()
    print(f"The max value in original series is {max_val_res_df} ")
    print(f"The max value in Query series is {max_val_Q_df} ")
    data = res_df
    data.to_csv('DataAnalysis/MotifDataAnalysis/PatternTechnique2/' + 'Total_MotifData_for_max_val' + str(res_df.columns[1])
                         + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv',
                         index=False, header=True)


    if (distance_profile[idx] > 60):
        print(f"The global minimum is above considerable range of motif detection")
    else:
        print(f"The nearest neighbor to `Query Pattern` is located at index {idx} in `Original Data`")
        temp_ts_res_df = res_df[res_df.columns[0]]
        temp_res_df = res_df[res_df.columns[1]]
        # plotting found motif time series (sub-sequence) along datetime
        plt.clf()
        plt.suptitle('Motif Subsequence', fontsize='10')
        plt.xlabel('Time', fontsize='5', y=0)
        plt.ylabel('Values', fontsize='7')
        plt.plot(temp_res_df.values[idx:idx + len(Q_res_df)], lw=1, color="green")
        plt.xticks(rotation=70, weight='bold', fontsize=5)
        plt.xticks(weight='bold', fontsize=5)
        plt.autoscale()
        plt.savefig('Graphs/Graphs_Motifs/PatternTechnique2/' + 'Motif_Pattern_' + datetime.now().strftime(
            "%Y%m%d-%H%M%S") + '.png')
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
        appended_data.to_csv('DataAnalysis/MotifDataAnalysis/PatternTechnique2/' +  'MotifData_for_' + str(res_df.columns[1])
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
        plt.plot(res_df[res_df.columns[1]])
        plt.autoscale()
        for idx in idxs:
            plt.plot(range(idx, idx + len(Q_res_df)), temp_res_df.values[idx:idx + len(Q_res_df)], lw=2, color = idx)
        plt.savefig('Graphs/Graphs_Motifs/PatternTechnique2/' + 'k_Closest_Matching_Patterns' + datetime.now().strftime(
                "%Y%m%d-%H%M%S") + '.png')

def get_mp_data_data_analysis(start_period, end_period, Q_start_period, Q_end_period, mp_ids, Q_mp_ids,level, include_missing_mp= False):

    if (data_analysis.FillingTime.value):
        dfs, missing_mp_ids = data_service.get_data_by_ids_period_and_level_for_filling_time_relevant_kpis(start_period,end_period, mp_ids,level, include_missing_mp)
        Q_dfs, Q_missing_mp_ids = data_service.get_data_by_ids_period_and_level_for_filling_time_relevant_kpis(Q_start_period, Q_end_period, Q_mp_ids,level, include_missing_mp)
    else:
        dfs, missing_mp_ids = data_service.get_data_by_ids_period_and_level(start_period, end_period, mp_ids,
                                                                            level, include_missing_mp)
        Q_dfs, Q_missing_mp_ids = data_service.get_data_by_ids_period_and_level(Q_start_period, Q_end_period, Q_mp_ids,
                                                                                level, include_missing_mp)
    res_df = pd.DataFrame({'ts': pd.date_range(start=start_period, end=end_period, freq=get_freq_by_level(level))})

    #res_df['ts'] = res_df['ts'].dt.tz_convert('Europe/Berlin')

    res_df['ts'] = res_df['ts'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')

    Q_res_df = pd.DataFrame({'ts': pd.date_range(start=Q_start_period, end=Q_end_period, freq=get_freq_by_level(level))})

    Q_res_df['ts'] = Q_res_df['ts'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')

    #Q_res_df['ts'] = Q_res_df['ts'].dt.tz_convert('Europe/Berlin')

    for df in dfs:
        res_df = pd.merge(res_df, df, on='ts', how='outer')
    res_df.sort_values(by=['ts'], inplace=True)
    res_df = res_df.drop_duplicates().reset_index(drop=True)
    #res_df = res_df.fillna(method='ffill')
    #res_df.dropna(inplace=True)

    for Q_df in Q_dfs:
        Q_res_df = pd.merge(Q_res_df, Q_df, on='ts', how='outer')

    Q_res_df.sort_values(by=['ts'], inplace=True)
    Q_res_df = Q_res_df.drop_duplicates().reset_index(drop=True)
    #Q_res_df = Q_res_df.fillna(method='ffill')
    #Q_res_df.dropna(inplace=True)

    if (data_analysis.FillingTime.value):
        get_stumpy_query_pattern_filling_time_relevant_kpis(res_df, Q_res_df)
    else:
        get_stumpy_query_pattern(res_df, Q_res_df)

def execute_data_analysis(mpid_var, Q_mpid_var,start_period, end_period,granularity_level,Q_start_period, Q_end_period):
   get_mp_data_data_analysis(start_period, end_period, Q_start_period, Q_end_period, mpid_var,Q_mpid_var, granularity_level)

#instance = data_analysis.instance.value
config_oauth(config.get_current_config())

try:
    execute_data_analysis(data_analysis.stumpy_measuringpoint_var.value, data_analysis.Q_stumpy_measuringpoint_var.value, data_analysis.start_period.value,data_analysis.end_period.value,data_analysis.granularity.value,data_analysis.Q_start_period.value, data_analysis.Q_end_period.value)
    print("\nEnd of program successful run with files  saved to DataAnalysis and Graph folders, inside project folder:services.")
    exit()
except Exception as e:
    print(e)