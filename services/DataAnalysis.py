from utils.enumerations import GanualityLevel,data_analysis
from utils.oauth import config_oauth
from services.helper_service import get_mp_data, get_freq_by_level
from services import data_service
import pandas as pd
import config


def analysis_to_file(res_df,mp_ids,iter_num):

    string_ints = [str(int) for int in mp_ids]
    str_of_mpids = "_".join(string_ints)
    res_df.to_csv('DataAnalysis/'+ data_analysis.Filename_Analysis.value + '_mpid_' + str_of_mpids + '_iteration' + str(iter_num) + '.csv', index = False, header=True)

def get_mp_data_data_analysis(start_period, end_period, mp_ids, level, iter_num, include_missing_mp= False):
    dfs, missing_mp_ids = data_service.get_data_by_ids_period_and_level(start_period, end_period, mp_ids,
                                                                                      level, include_missing_mp)
    res_df = pd.DataFrame({'ts': pd.date_range(start=start_period, end=end_period, freq=get_freq_by_level(level))})

    for df in dfs:
        res_df = pd.merge(res_df, df, on='ts', how='outer')

    res_df.sort_values(by=['ts'], inplace=True)

    for i in range(1,len(res_df.columns)-1):
        res_df['diff_FlowValve008_1492 - ' + str(res_df.columns[i + 1])] = res_df[res_df.columns[1]] - res_df[res_df.columns[i + 1]]

    analysis_to_file(res_df,mp_ids,iter_num)

    return res_df, missing_mp_ids



def execute_data_analysis(mpid_var, start_period, end_period,granularity_level,iter_num):
     data, missing_ids = get_mp_data_data_analysis(start_period, end_period, mpid_var, granularity_level,iter_num)


instance = data_analysis.instance.value
config_oauth(config.get_current_config())

try:
    execute_data_analysis(data_analysis.measuringpoint_var.value, data_analysis.start_period.value, data_analysis.end_period.value,data_analysis.granularity.value,data_analysis.iteration_num.value)
    print("end of program successful with file saved to DataAnalysis folder, inside project folder.")
except Exception as e:
    print(e)
