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


def analysis_to_file(res_df,mp_ids,iter_num):
    string_ints = [str(int) for int in mp_ids]
    str_of_mpids = "_".join(string_ints)
    res_df.to_csv('DataAnalysis/'+ data_analysis.Filename_Analysis.value + '_mpid_' + str_of_mpids + '_iteration' + str(iter_num) + '.csv', index = False, header=True)


def threshold_diff(df,mpid,iter_num):
    for threshold in data_analysis.thresholds.value:
        for i in range(len(mpid)+1, len(df.columns)):
            df_filter = df.loc[abs(df[df.columns[i]]) > threshold]
            if(df_filter.shape[0] >  0):
                df_filter.to_csv('DataAnalysis/' + data_analysis.Filename_Threshold.value + '_threshold_'+ str(threshold)
                                 +'_for_'+df.columns[i]+'_iteration_'+str(iter_num) + '.csv', index=False, header=True)


def get_stumpy(df,num):
    df[df.columns[1]] = df[df.columns[1]].replace([np.inf, -np.inf], np.nan)
    df[df.columns[1]] = df[df.columns[1]].replace(0, np.nan)
    non_nan_df = df[df.columns[1]].dropna(inplace=False)
    df[df.columns[1]] = non_nan_df
    #non_nan_dfa = df[df.columns[1]][np.logical_not(np.isnan(df[df.columns[1]]))]

    days_dict = {
        "Three-min": 6,
        "Five-min": 10,
        "Ten-min": 20,
        "Fifteen-min": 30,
        "Twenty-min": 40,
        "Half-Hour": 60,
        # "1-Hour": 120,
        # "2-Hour": 240,
        # "3-Hour"  : 360,
        # "6-Hour"  : 720,
        # "9-Hour"  : 1080,
        # "12-Hour" : 1440,
    }
    # m = 120
    # mp = stumpy.stump(df[df.columns[1]], m)
    # print(mp)
    # plt.plot(df[df.columns[1]])
    # plt.show()
    # plt.plot(mp[:, 0])
    # plt.show()
    # print(mp[:, 0].min())


    #-----------------------------------------------------------------------------
    m = days_dict['Fifteen-min']
    mp = stumpy.stump(non_nan_df, m)
    #mp = stumpy.stump(df[df.columns[1]], m)
    mp = mp[mp[:, 1].argsort()]
    for index, value in np.ndenumerate(mp[:, 0]):
        mp[:, 0][index] = np.around(value, 2)

    a = mp[:, 0]
    min_value = mp[:, 0].min()
    # max_value_not_zero =   min(i for i in a if i > 0.9)
    min_index_row = np.where(a == min_value)
    #max_index_row_l = list(max_index_row)
    print("Position:", min_index_row)
    print("Value:", min_index_row)

    fig, axs = plt.subplots(2+len(min_index_row)+1, sharex=True, gridspec_kw={'hspace': 0})
    plt.suptitle('Motif (Pattern) Discovery', fontsize='20')

    color = iter(cm.rainbow(np.linspace(0, 1)))
    for i in range(len(min_index_row)+1):
        c = next(color)
        axs[0].plot(non_nan_df[min_index_row[0][i]:min_index_row[0][i] + m], color=c)
    #axs[0].plot(non_nan_df[min_index_row[0][0]:min_index_row[0][0] + m], color='C1')
    #axs[0].plot(non_nan_df[min_index_row[0][1]:min_index_row[0][1] + m], color='C2')
    #axs[0].plot(non_nan_df[max_index_row[0][1]:max_index_row[0][1] + m], color='C2')

    axs[1].plot(non_nan_df)
    axs[1].set_ylabel('Flow Valve 008', fontsize='08')
    #rect = Rectangle((643, 0), m, 40, facecolor='lightgrey')
    #axs[0].add_patch(rect)
    #rect = Rectangle((8724, 0), m, 40, facecolor='lightgrey')
    #axs[0].add_patch(rect)
    axs[2].set_xlabel('Time', fontsize='10')
    axs[2].set_ylabel('Matrix Profile', fontsize='10')
    #axs[1].axvline(x=643, linestyle="dashed")
    #axs[1].axvline(x=8724, linestyle="dashed")
    axs[2].plot(mp[:, 0])
    #plt.show()
    # axs[3].set_xlabel("Time", fontsize='10')
    # axs[3].set_ylabel("Zoomed ", fontsize='10')
    # axs[3].plot(non_nan_df[min_index_row[0]:min_index_row[0] + m], color='C1')
    # axs[3].plot(non_nan_df[min_index_row[1]:min_index_row[1] + m], color='C2')
    # axs[3].plot(non_nan_df[min_index_row[2]:min_index_row[2] + m], color='C3')

    plt.show()


    #print(mp[:, 0].min())
    #print(mp[:,0].max())

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
        mp = stumpy.stump(non_nan_df, varying_m)
        #mp = stumpy.stump(df[df.columns[1]], varying_m)
        axs[i].plot(mp[:, 0])

        axs[i].set_ylim(0, 9.5)
        axs[i].set_xlim(0, 3600)
        title = f"m = {varying_m}"
        axs[i].set_title(title, fontsize=10, y=.5)

    plt.xticks(np.arange(0, df.shape[0], 5000.0))
    #plt.xticks(np.arange(0, df.shape[0], (48 * DAY_MULTIPLIER) / 2), x_axis_labels)
    plt.xticks(rotation=75)
    plt.suptitle('STUMP with Varying Window Sizes', fontsize='10')
    plt.show()



def plot_data_analysis_graphs(df,iter):

    degrees = 70

    if(df.shape[0]>0):
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
            plt.savefig('Graphs_DataAnalysis/' + 'Analysis_Historic Data_' + 'mp_id_' + str(df.columns[i]) + ' _iteration' + str(iter))
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
            plt.savefig('Graphs_DataAnalysis/' + 'Analysis_Correlated_Data_' + 'mp_id_' + str(df.columns[1]) + '_' + str(df.columns[i]) + ' _iteration' + str(iter))
            plt.show()





def get_mp_data_data_analysis(start_period, end_period, mp_ids, level, iter_num, include_missing_mp= False):
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

    get_stumpy(res_df, iter_num)

    plot_data_analysis_graphs(res_df,iter_num)

    for i in range(1,len(res_df.columns)-1):
        res_df['diff_FlowValve008_1492 - ' + str(res_df.columns[i + 1])] = res_df[res_df.columns[1]] - res_df[res_df.columns[i + 1]]



    analysis_to_file(res_df,mp_ids,iter_num)

    threshold_diff(res_df,mp_ids,iter_num)




def execute_data_analysis(mpid_var, start_period, end_period,granularity_level,iter_num):
   get_mp_data_data_analysis(start_period, end_period, mpid_var, granularity_level,iter_num)


instance = data_analysis.instance.value
config_oauth(config.get_current_config())

try:
    execute_data_analysis(data_analysis.measuringpoint_var.value, data_analysis.start_period.value, data_analysis.end_period.value,data_analysis.granularity.value,data_analysis.iteration_num.value)
    print("end of program successful with file saved to DataAnalysis folder, inside project folder.")
except Exception as e:
    print(e)
