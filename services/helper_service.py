import os

from utils.enumerations import GanualityLevel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from . import data_service

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import numpy as np
from sklearn import metrics


import pickle

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LassoCV, LassoLarsCV, ElasticNetCV
from sklearn.cluster import Birch, KMeans, DBSCAN, AgglomerativeClustering



def truncateMinsPer10min(row):
    t = row['TS'] + pd.Timedelta('10 minutes')
    return t.replace(minute=t.minute // 10 * 10).replace(second=0, microsecond=0)


def truncateMins(row):
    return row['TS'].replace(minute=0, second=0, microsecond=0)


def RMSE(y_true, y_pred):
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    return rmse

def get_mp_data(start_period, end_period, mp_ids, level=GanualityLevel.one_hour.value, include_missing_mp= False):
    dfs, missing_mp_ids = data_service.get_data_by_ids_period_and_level(start_period, end_period, mp_ids,
                                                                                      level, include_missing_mp)
    res_df = pd.DataFrame({'ts': pd.date_range(start=start_period, end=end_period, freq=get_freq_by_level(level))})

    for df in dfs:
        res_df = pd.merge(res_df, df, on='ts', how='outer')

    res_df.sort_values(by=['ts'], inplace=True)
    res_df = res_df.drop_duplicates().reset_index(drop=True)

    res_df = res_df.fillna(method='ffill')

    res_df.dropna(inplace= True)

    return res_df, missing_mp_ids


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    columns = data.columns
    df = pd.DataFrame(data)
    cols, leadNames, lagNames = list(), list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        lagNames += [(columns[j] + '(t-%d)' % (i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        leadNames += [(columns[j] + '(t+%d)' % (i)) for j in range(n_vars)]

    res = pd.concat(cols, axis=1)
    res.columns = np.concatenate((lagNames, leadNames))

    # drop rows with NaN values
    if dropnan:
        res.dropna(inplace=True)

    return res[lagNames], res[leadNames]


def removeSkewness(df: pd.DataFrame, method: str, col: str, threshold: float):
    power_transform = None
    g = df.skew(axis=0)
    current_skewness = abs(df.skew(axis=0)[col])
    temp_copy = df.copy()

    if current_skewness > threshold:
        if method == 'yeo':

            power_transform = PowerTransformer(method='yeo-johnson', standardize=False)
            power_transform.fit(df[col].values.reshape(-1, 1))

            df[col] = power_transform.transform(df[col].values.reshape(-1, 1))
        elif method == 'log':
            df[col] = np.log1p(df[col])

    transformedSkewness = abs(df.skew(axis=0)[col])

    if transformedSkewness < current_skewness:
        return df, power_transform

    return temp_copy, None

def remove_target_skewness_and_normalization_clustering(data, scaler, power_transformers, cols):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    raw_data = scaler.inverse_transform(data)
    for col in cols:
        if power_transformers[col] != None:
            raw_data[col] = power_transformers[col].inverse_transform(raw_data[col])

        if np.isnan(raw_data[col]).any():
            imp.fit(raw_data[col])
            raw_data[col] = imp.transform(raw_data[col])


    return raw_data

def remove_target_skewness_and_normalization_univariate(predY, testY, scaler, powerTransformer):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    rawPredY = predY.reshape(-1, 1)
    if powerTransformer != None:
        rawPredY = powerTransformer.inverse_transform(rawPredY)
    rawPredY = scaler.inverse_transform(rawPredY)[:, -1].reshape(-1, 1)


    # rawPredY[rawPredY == -np.inf] = 0
    # rawPredY[rawPredY == np.inf] = 0

    if np.isnan(rawPredY).any():
        imp.fit(rawPredY)
        rawPredY = imp.transform(rawPredY)

    if testY != None:
        rawTestY = testY.values.reshape(-1, 1)
        rawTestY = scaler.inverse_transform(rawTestY)[:, -1].reshape(-1, 1)
        if powerTransformer != None:
            rawTestY = powerTransformer.inverse_transform(rawTestY)
        return rawPredY, pd.DataFrame(rawTestY)

    return rawPredY


# def removeTargetSkewnessAndNormalization(predY, testY, testX, scaler, powerTransformer):
#     imp = SimpleImputer(missing_values=np.nan, strategy='mean')
#
#     rawPredY = predY.reshape(-1, 1)
#     invertPredY = np.concatenate((testX, rawPredY), axis=1)
#     rawPredY = scaler.inverse_transform(invertPredY)[:, -1].reshape(-1, 1)
#     if powerTransformer != None:
#         rawPredY = powerTransformer[target[0]].inverse_transform(rawPredY)
#
#     if np.isnan(rawPredY).any():
#         imp.fit(rawPredY)
#         rawPredY = imp.transform(rawPredY)
#
#     if testY != None:
#         rawTestY = testY.values.reshape(-1, 1)
#         invertTestY = np.concatenate((testX, rawTestY), axis=1)
#         rawTestY = scaler.inverse_transform(invertTestY)[:, -1].reshape(-1, 1)
#         if powerTransformer != None:
#             rawTestY = powerTransformer[target[0]].inverse_transform(rawTestY)
#         return rawPredY, pd.DataFrame(rawTestY)
#
#     return rawPredY


def normalizeData(df):
    scaler = RobustScaler()
    scaler.fit(df)
    normalizedData = pd.DataFrame(scaler.transform(df), columns=df.columns)

    return normalizedData, scaler


def normalize_data_with_ts(df):
    scaler = RobustScaler()
    TransformedData = df.drop(['ts'], axis=1)
    scaler.fit(TransformedData)
    normalizedData = pd.DataFrame(scaler.transform(TransformedData), columns=TransformedData.columns)
    normalizedData['ts'] = df['ts']
    return normalizedData, scaler


def saveModel(model, fileName):
    pickle.dump(model, open(fileName, 'wb'))


def getModel(fileName):
    return pickle.load(open(fileName, 'rb'))


def getBestEstimator(estimator, param, trainX, trainY, cvCount, algo='random'):
    grid = RandomizedSearchCV(estimator=estimator, param_distributions=param, n_jobs=-1, scoring=rmseScorer())

    if algo == 'grid':
        grid = GridSearchCV(estimator, param, cv=cvCount, return_train_score=True, scoring=rmseScorer(), n_jobs=-1)

    grid.fit(trainX, trainY)
    return grid.best_estimator_


def RMSE(y_true, y_pred):
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    return rmse


def rmseScorer():
    return metrics.make_scorer(RMSE, greater_is_better=False)

def rmse_time_series(y_true, y_pred):
    rmses = []
    for i in range(y_true.shape[1]):
        rmse = np.sqrt(metrics.mean_squared_error(y_true[:,i], y_pred[:,i]))
        rmses.append(rmse)
    return sum(rmses)/len(rmses), rmses

def rmse_time_series_hori(y_true, y_pred):
    rmses = []
    for i in range(y_true.shape[0]):
        rmse = np.sqrt(metrics.mean_squared_error(y_true[i,:], y_pred[i,:]))
        rmses.append(rmse)
    return sum(rmses)/len(rmses)


def getModels():
    models = dict()
    models['lasso'] = Lasso(normalize=False, copy_X=True, max_iter=1000000)
    models['lasso_cv'] = LassoCV(normalize=False, copy_X=True, n_jobs=-1, max_iter=1000000)
    models['ridge'] = Ridge(normalize=False, copy_X=True)
    models['elastic_net'] = ElasticNet(normalize=False, copy_X=True, max_iter=1000000)
    models['llars'] = LassoLars(normalize=False, copy_X=True, max_iter=1000000)
    models['llarsCV'] = LassoLarsCV(normalize=False, copy_X=True, n_jobs=-1, max_iter=1000000)
    models['enCV'] = ElasticNetCV(normalize=False, copy_X=True, n_jobs=-1, max_iter=1000000)
    models['passive_aggressive_regressor'] = PassiveAggressiveRegressor(max_iter=10000000)

    models['k_neighbors_regressor'] = KNeighborsRegressor(n_jobs=-1)
    models['svmr'] = SVR()

    models['ada_boost_regressor'] = AdaBoostRegressor()
    models['bagging_regressor'] = BaggingRegressor(n_jobs=-1)
    models['random_forest_regressor'] = RandomForestRegressor(n_jobs=-1)
    models['extra_trees_regressor'] = ExtraTreesRegressor(n_jobs=-1)
    models['gradient_boosting_regressor'] = GradientBoostingRegressor(verbose=1, random_state=7, loss='lad', )
    models['mlp'] = MLPRegressor(max_iter=1000, hidden_layer_sizes=(100, 100, 1))
    models['xgb'] = XGBRegressor(n_jobs=-1)
    return models


def getModelParameters():
    parameters = dict()
    parameters['ridge'] = {
        'alpha': [0.000003, 0.00003, 0.0003, 0.003, 0.03, 0.1, 0.3, 0.7, 0.8, 0.6, 0.9, 1.0, 5.0, 10.0],
        'solver': ['auto', 'svd', 'cholesky', 'sparse_cg', 'lsqr'], 'fit_intercept': [True, False],
        'tol': [0.0000001, 0.000001, 0.00001, 0.001]}
    parameters['lasso'] = {
        'alpha': [0.000003, 0.00003, 0.0003, 0.003, 0.03, 0.1, 0.3, 0.7, 0.8, 0.6, 0.9, 1.0, 5.0, 10.0],
        'fit_intercept': [True, False], 'tol': [0.0000001, 0.000001, 0.00001, 0.001], 'precompute': [True, False],
        'selection': ['cyclic', 'random']}
    parameters['lasso_cv'] = {'eps': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
                             'n_alphas': [10, 50, 100, 200, 300, 400, 500, 1000], 'fit_intercept': [True, False],
                             'tol': [0.000001, 0.00001, 0.0001, 0.001]}
    parameters['llars'] = {
        'alpha': [0.000003, 0.00003, 0.0003, 0.003, 0.03, 0.1, 0.3, 0.7, 0.8, 0.6, 0.9, 1.0, 5.0, 10.0],
        'fit_intercept': [True, False], 'eps': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1], 'fit_path': [True, False]}
    parameters['llarsCV'] = {'eps': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
                             'max_n_alphas': [10, 50, 100, 200, 300, 400, 500, 1000, 2000, 5000],
                             }
    parameters['elastic_net'] = {'alpha': [0.000003, 0.00003, 0.0003, 0.003, 0.03, 0.1, 0.3, 0.7, 0.8, 0.6, 0.9, 1.0, 5.0, 10.0],
                        'l1_ratio': [0, 0.25, 0.5, 0.75, 1], 'fit_intercept': [True, False],
                        'tol': [0.0000001, 0.000001, 0.00001, 0.001], 'precompute': [True, False],
                        'warm_start': [True, False], 'selection': ['cyclic', 'random']}
    parameters['enCV'] = {'eps': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
                          'n_alphas': [10, 50, 100, 200, 300, 400, 500, 1000], 'fit_intercept': [True, False],
                          'tol': [0.000001, 0.00001, 0.0001, 0.001], 'selection': ['cyclic'],
                          'l1_ratio': [0.1, 0.25, 0.5, 0.75, 1]}
    parameters['passive_aggressive_regressor'] = {
        'C': [0.000003, 0.00003, 0.0003, 0.003, 0.03, 0.1, 0.3, 0.7, 0.8, 0.6, 0.9, 1.0, 5.0, 10.0],
        'validation_fraction': [0.1, 0.2, 0.5, 0.7, 0.8], 'fit_intercept': [True, False],
        'tol': [0.0000001, 0.000001, 0.00001, 0.001], 'n_iter_no_change': [5, 7, 10], 'shuffle': [True, False],
        'average': [True, 1, 10, 50, 100, 1000], 'warm_start': [True, False]}
    parameters['ada_boost_regressor'] = {'n_estimators': [25, 50, 100, 150, 200],
                         'learning_rate': [0.0000003, 0.00003, 0.0003, 0.003, 0.03, 0.3, 0.5, 0.7, 1.0, 5.0, 10.0],
                         'loss': ['linear', 'square', 'exponential']}
    parameters['bagging_regressor'] = [
        {'n_estimators': [25, 50, 100, 150, 200], 'bootstrap': [True, False], 'bootstrap_features': [True, False]},
        {'n_estimators': [25, 50, 100, 150, 200], 'bootstrap_features': [True, False], 'oob_score': [True, False]}]
    parameters['random_forest_regressor'] = {'n_estimators': [25, 50, 100, 150, 200], 'criterion': ['mse', 'mae'],
                        'min_samples_split': [2, 3, 5, 0.01, 0.02, 0.1, 0.3, 0.7],
                        'min_samples_leaf': [1, 3, 5, 0.01, 0.02, 0.1, 0.3, 0.5], 'bootstrap': [True, False]}

    parameters['gradient_boosting_regressor'] = {'learning_rate': [0.0000003, 0.00003, 0.0003, 0.003, 0.03, 0.3, 0.5, 0.7, 1.0, 5.0, 10.0],
                         'n_estimators': [25, 50, 100, 150, 200],
                         'min_samples_leaf': [1, 3, 5, 0.01, 0.02, 0.1, 0.3, 0.5],
                         'min_samples_split': [2, 3, 5, 0.01, 0.02, 0.1, 0.3, 0.7], 'max_depth': [1, 3, 6, 9],
                         'alpha': [0.3, 0.6, 0.9], 'tol': [0.000001, 0.00001, 0.0001, 0.001]}
    parameters['mlp'] = {
        'hidden_layer_sizes': [(5,), (100,), (5, 100), (100, 200), (100, 200, 50), (100, 200, 500, 50)],
        'solver': ['lbfgs', 'sgd', 'adam'], 'alpha': [0.0000003, 0.000003, 0.00003, 0.0003, 0.003, 0.03, 0.1, 0.3, 0.7],
        'learning_rate': ['constant', 'invscaling', 'adaptive']}
    parameters['extra_trees_regressor'] = {'n_estimators': [25, 50, 100, 150, 200],
                        'min_samples_split': [2, 3, 5, 0.01, 0.02, 0.1, 0.3, 0.7],
                        'min_samples_leaf': [1, 3, 5, 0.01, 0.02, 0.1, 0.3, 0.5], 'bootstrap': [True, False],
                        'min_impurity_split': [0.0000001, 0.00000001, 0.000000001, 0.0000000001]}
    parameters['k_neighbors_regressor'] = {'n_neighbors': [3, 6, 9, 15, 21], 'weights': ['uniform', 'distance'],
                         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
    parameters['svmr'] = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], 'degree': [1, 3, 4, 5, 9],
                          'tol': [0.0000001, 0.000001, 0.00001, 0.001], 'C': [1, 3, 6, 0.7, 0.5, 0.3, 0.1],
                          'shrinking': [True, False]}
    # parameters['xgb'] = {   'colsample_bytree':[0.4,0.6,0.8],'gamma':[0,0.03,0.1,0.3], 'min_child_weight':[1.5,6,10], 'learning_rate':[0.1,0.07,0.008], 'max_depth':[3,5], 'n_estimators':[10000], 'reg_alpha':[1e-5, 1e-2,  0.75], 'reg_lambda':[1e-5, 1e-2, 0.45], 'subsample':[0.6,0.95] , 'booster': ['gbtree', 'gblinear', 'dart'] }
    parameters['xgb'] = {'colsample_bytree': [0.4, 0.6, 0.8], 'gamma': [0, 0.03, 0.1, 0.3],
                         'min_child_weight': [1.5, 6, 10], 'learning_rate': [0.1, 0.07, 0.008], 'max_depth': [3, 5],
                         'n_estimators': [10000], 'reg_alpha': [1e-5, 1e-2, 0.75], 'reg_lambda': [1e-5, 1e-2, 0.45],
                         'subsample': [0.6, 0.95]}

    return parameters

def get_clustering_models():
    models = dict()

    models['brich'] = Birch(threshold=0.3)
    models['k_means'] = KMeans()
    models['db_scan'] =DBSCAN()
    models['aglo'] = AgglomerativeClustering()

    return models

def get_freq_by_level(ganuality_level_value):
    if GanualityLevel.one_hour.value[0] == ganuality_level_value:
        return '60T'
    elif GanualityLevel.three_hour.value[0] == ganuality_level_value:
        return '180T'
    elif GanualityLevel.one_day.value[0] == ganuality_level_value:
        return '1440T'
    elif GanualityLevel.one_week.value[0] == ganuality_level_value:
        return '10080T'
    elif GanualityLevel.ten_min.value[0] == ganuality_level_value:
        return '10T'
    elif GanualityLevel.one_min.value[0] == ganuality_level_value:
        return '1T'
    elif GanualityLevel.thirty_sec.value == ganuality_level_value:
        return '0.5T'
    elif GanualityLevel.one_sec.value == ganuality_level_value:
        return '0.016666666667T'


def remove_skewness_and_normalize(data, skew_transformers, normalizer, cols):
    normalized_data = pd.DataFrame(normalizer.transform(data[cols]), columns=cols)
    normalized_data['ts'] = data['ts']
    for col in cols:
        if skew_transformers[col] != None:
            normalized_data[col] = skew_transformers[col].transform(normalized_data[col].values.reshape(-1, 1))
    return normalized_data

