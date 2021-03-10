import enum


class Services(enum.Enum):
    info_service = 'INFO_SERVICE'
    data_service = 'DATA_SERVICE'
    analytical_service = 'ANALYTICAL_SERVICE'


class RemoteControllers(enum.Enum):
    lookup = 'Lookup'
    data_trend = 'DataTrend'
    use_case = 'UseCase'


class LookupCategory(enum.Enum):
    usecase_execution_status = 'UseCaseExecutionStatus'
    ml_models = 'MachineLearningModel'
    clustering_algos = 'ClusteringAlgos'
    ganuality_level = 'GanualityLevel'
    usecase_type = 'UseCaseType'

class UsecasTypeLookup(enum.Enum):
    univariate_regression = 1
    clustering = 2
    overhaul_date_prediction = 3



class UsecaseExecutionStatusLookup(enum.Enum):
    unscheduled = 1
    waiting =  2
    in_progress = 3
    completed = 4
    failed = 5

class MachineLeaningModelLookup(enum.Enum):
    lasso = 1
    lasso_cv = 2
    ridge = 3
    elastic_net = 4
    passive_aggressive_regressor = 5
    k_neighbors_regressor = 6
    svmr = 7
    ada_boost_regressor = 8
    bagging_regressor = 9
    random_forest_regressor = 10
    extra_trees_regressor = 11
    gradient_boosting_regressor = 12
    xgb = 13

class ClusteringAlgoLookup(enum.Enum):
    k_means = 1
    brich = 2
    db_scan = 3
    aglo = 4

class GanualityLevel(enum.Enum):
    one_sec = 1
    thirty_sec = 2
    ten_min = 3,
    one_hour = 4,
    three_hour = 5,
    one_day = 6,
    one_week = 7,
    one_min = 8,

class data_analysis(enum.Enum):
    # For Patterns.py Implementation
    stumpy_measuringpoint_var = [660,1637,1958,13749,1797,557,567]
    Q_stumpy_measuringpoint_var = [1637]
    # For DataAnalysis.py Implementation
    measuringpoint_var = [1492, 1491, 1493]


    start_period = '11-01-2020T00:00:00'  # 'mm-dd-yyyyTHH:MM:SS'
    end_period =   '11-30-2020T00:00:00'  # 'mm-dd-yyyyTHH:MM:SS'
    Filename_Analysis = 'DataAnalysis'
    Filename_Threshold = 'Data_Above_Threshold'
    thresholds = [20,16,4,2200,120,20]
    if len(stumpy_measuringpoint_var) >1 :
        FillingTime = True
    else:
        FillingTime = False

    #For pattern_technique2.py with Query pattern Technique
    #Q_start_period = '11-09-2020T14:00:00'  # 'mm-dd-yyyyTHH:MM:SS'
    #Q_end_period = '11-09-2020T14:30:00'  # 'mm-dd-yyyyTHH:MM:SS'

    Q_start_period = '11-09-2020T00:00:00'  # 'mm-dd-yyyyTHH:MM:SS'
    Q_end_period   = '11-11-2020T00:00:00'  # 'mm-dd-yyyyTHH:MM:SS'
    #
    # Q_start_period = '11-09-2020T22:50:00'  # 'mm-dd-yyyyTHH:MM:SS'
    # Q_end_period = '11-09-2020T23:30:00'  # 'mm-dd-yyyyTHH:MM:SS'


    # For pattern_technique3.py with AB JOIN Technique
    A_stumpy_measuringpoint_var = [1637]
    B_stumpy_measuringpoint_var = [1637]
    A_start_period = '11-09-2020T22:50:00'  # 'mm-dd-yyyyTHH:MM:SS'
    A_end_period = '11-09-2020T23:30:00'  # 'mm-dd-yyyyTHH:MM:SS'
    B_start_period = '11-08-2020T00:00:00'  # 'mm-dd-yyyyTHH:MM:SS'
    B_end_period = '11-11-2020T00:00:00'  # 'mm-dd-yyyyTHH:MM:SS'

    # General purpose variables for all implementations
    granularity = 2
    instance = 'hbc'
    iteration_num = 2


    # input_mpid = [1492, 1491, 1493]
    # target_mpid = [1492]
    # A_start_period = '11-10-2020T02:00:00'  # 'mm-dd-yyyyTHH:MM:SS'
    # A_end_period = '11-10-2020T04:00:00'  # 'mm-dd-yyyyTHH:MM:SS'
    # B_start_period = '11-10-2020T02:00:00'  # 'mm-dd-yyyyTHH:MM:SS'
    # B_end_period = '11-10-2020T04:00:00'  # 'mm-dd-yyyyTHH:MM:SS'