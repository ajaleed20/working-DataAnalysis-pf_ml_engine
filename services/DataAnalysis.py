import math

from utils.enumerations import GanualityLevel,data_analysis
from utils.oauth import config_oauth
from services.helper_service import get_mp_data, get_freq_by_level
from services import data_service
import pandas as pd
import config
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

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

def multiplot(data, features, plottype, nrows, ncols, figsize, y=None, colorize=False, fileName=None):
    """ This function draw a multi plot for 3 types of plots ["regplot","distplot","coutplot"]"""
    n = 0
    plt.figure(1)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    if len(axes.shape) == 1:
        axes = np.reshape(axes, (1, axes.shape[0]))

    if colorize:
        colors = sns.color_palette(n_colors=(nrows * ncols))
    else:
        colors = [None] * (nrows * ncols)

    for row in range(nrows):
        for col in range(ncols):

            if plottype == 'regplot':
                if y == None:
                    raise ValueError('y value is needed with regplot type')

                sns.regplot(data=data, x=features[n], y=y, ax=axes[row, col], color=colors[n])
                correlation = np.corrcoef(data[features[n]], data[y])[0, 1]
                axes[row, col].set_title("Correlation {:.2f}".format(correlation))

            elif plottype == 'distplot':
                sns.distplot(a=data[features[n]], ax=axes[row, col], color=colors[n])
                skewness = data[features[n]].skew()
                axes[row, col].legend(["Skew : {:.2f}".format(skewness)])

            elif plottype in ['countplot']:
                g = sns.countplot(x=data[features[n]], y=y, ax=axes[row, col], color=colors[n])
                g = plt.setp(g.get_xticklabels(), rotation=45)

            n += 1
            if n >= len(features):
                break
    plt.tight_layout()
    if fileName != None:
        plt.savefig('Graphs_DataAnalysis/'+fileName)
    plt.show()
    plt.gcf().clear()

def get_features(dependent_var, independent_var):
    return list(map(str, dependent_var))+ list(map(str, independent_var))

def plot_data_analysis_graphs(df,iter):

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
        plt.savefig('Graphs_DataAnalysis/' + 'Analysis_Historic Data_' + 'mp_id_' + str(df.columns[i]) + ' _iteration' + str(iter))
        plt.show()

    drawCorrelation(df)
    # for i in range(2, len(df.columns)):
    #     y = df[data_analysis.target_mpid.value]
    #     x = df[df.columns[i]]
    #     print(np.corrcoef(x, y))
    #     plt.figure(figsize=(22, 22))
    #     plt.scatter(x, y)
    #     plt.title('A plot to show the correlation between_' + str(df.columns[1]) + '_and_' + str(df.columns[i]),fontsize=24,fontweight="bold")
    #     plt.xlabel(str(data_analysis.target_mpid.value),fontsize=24,fontweight="bold")
    #     plt.ylabel(str(df.columns[i]),fontsize=24,fontweight="bold")
    #     plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='red')
    #     plt.xticks(rotation=degrees, weight = 'bold',fontsize=20)
    #     plt.yticks(weight = 'bold',fontsize=20)
    #     #plt.figure(figsize=(10,10))
    #     plt.savefig('Graphs_DataAnalysis/' + 'Analysis_Correlated_Data_' + 'mp_id_' + str(df.columns[1]) + '_' + str(df.columns[i]) + ' _iteration' + str(iter))
    #     plt.show()

def drawCorrelation(df):
    features = get_features([],data_analysis.measuringpoint_var.value)
    target = str(data_analysis.target_mpid.value[0])
    correlation_rows = int(math.ceil(len(features) / 2)) if len(features) > 1 else len(features)
    correlation_cols = 2 if len(features) > 1 else len(features)

    multiplot(data=df, features=features, plottype="regplot", nrows=correlation_rows, ncols=correlation_cols,
              figsize=(25, 20), y=target, colorize=True,
              fileName='mp_id_ ' + target + 'VScorrelation _iteration_' + str(data_analysis.iteration_num.value))


def get_mp_data_data_analysis(start_period, end_period, mp_ids, level, iter_num, include_missing_mp= False):
    dfs, missing_mp_ids = data_service.get_data_by_ids_period_and_level(start_period, end_period, mp_ids,
                                                                                      level, include_missing_mp)
    res_df = pd.DataFrame({'ts': pd.date_range(start=start_period, end=end_period, freq=get_freq_by_level(level))})

    for df in dfs:
        res_df = pd.merge(res_df, df, on='ts', how='outer')

    res_df.sort_values(by=['ts'], inplace=True)

    res_df = res_df.drop_duplicates().reset_index(drop=True)
    res_df = res_df.fillna(method='ffill')
    res_df.dropna(inplace=True)

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
