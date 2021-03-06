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
    measuringpoint_var = [1492, 1491, 1493]
    #measuringpoint_var = [1492,1461,1486,1488,1490,1463,1524,1476]  # Measuring Point Ids of dependent variables
    start_period = '11-10-2020T00:00:00'  # 'mm-dd-yyyyTHH:MM:SS'
    end_period = '11-11-2020T00:00:00'  # 'mm-dd-yyyyTHH:MM:SS'
    Filename_Analysis = 'DataAnalysis'
    Filename_Threshold = 'Data_Above_Threshold'
    granularity = 1
    instance = 'hbc'
    iteration_num = 2
    thresholds = [300, 200, 100]


