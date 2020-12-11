from datetime import timedelta

import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import Input, Dense, Dropout, concatenate, Conv1D, Flatten, AveragePooling1D
from keras.models import Model
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import math

import config
from services.helper_service import get_mp_data, normalize_data_with_ts, removeSkewness, series_to_supervised, rmse_time_series_hori, rmse_time_series
from utils.enumerations import GanualityLevel
from utils.oauth import config_oauth


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
        plt.savefig('Graphs/'+fileName)
    plt.show()
    plt.gcf().clear()


def get_normalized_data(start_date, end_date, level, mp_ids):
    data, missing_ids = get_mp_data(start_date, end_date, mp_ids, level)

    normalized_data, normalizer = normalize_data_with_ts(data)

    skew_transformers = {}

    for id in mp_ids:
        normalized_data, transformer = removeSkewness(normalized_data, 'yeo', str(id), 0.2)
        if transformer != None:
            skew_transformers[str(id)] = (transformer)

    return data, normalized_data, normalizer, skew_transformers


def exploratoryAnalysis(df, target, features, iteration):
    correlation_rows = int(math.ceil(len(features) / 2)) if len(features) > 1 else len(features)
    skewness_graphs_rows = int(math.ceil(len(features) / 4)) if len(features) > 3 else 1
    correlation_cols = 2 if len(features) > 1 else len(features)
    skewness_graphs_cols = 4 if len(features) > 3 else len(features)

    multiplot(data=df, features=features, plottype="regplot", nrows=correlation_rows, ncols=correlation_cols,
              figsize=(25, 20), y=target, colorize=True, fileName='mp_id_ ' + target + 'VScorrelation _iteration_' + iteration)

    multiplot(data=df, features=features, plottype="distplot",
              nrows=skewness_graphs_rows, ncols=skewness_graphs_cols, figsize=(25, 20), colorize=True,
              fileName= 'mp_id_ ' + target + 'VS ALL skew _iteration_' + iteration)


def get_forecast_input(data, mp_ids, lag_time_steps):
    X, y = series_to_supervised(data, lag_time_steps, 0)
    values = X.values
    X = values.reshape((values.shape[0], lag_time_steps, len(mp_ids)))
    x_inputs = []

    for i in range(X.shape[2]):
        x_inputs.append(np.reshape(X[-1, :, i], (1, X.shape[1])))
    return x_inputs


def plot_forecast(original_data, independent_var, forecast, ganuality_level_value, lead_time_steps, iteration, rmse_list):
    time_delta, freq = get_time_delta_and_freq(ganuality_level_value)
    ref_time = original_data.iloc[-1]['ts']
    start_time = ref_time + timedelta(hours=time_delta)
    end_time = ref_time + timedelta(hours=lead_time_steps * time_delta)
    pred_ts = pd.date_range(start=start_time, end=end_time, freq=freq)
    forecast = forecast.reshape(forecast.shape[1])
    plt.plot(pred_ts.values, forecast, 'g')
    plt.fill_between(pred_ts.values,forecast - rmse_list, forecast + rmse_list,alpha=0.1, color="g")
    plt.xlabel('ts')
    plt.ylabel(independent_var)
    plt.title('prediction')
    plt.savefig('Graphs/'+'Prediction_ ' + 'mp_id_ ' +independent_var +' _iteration ' + iteration)
    plt.show()
    ori_ts = original_data['ts'].values
    ori_data = original_data[independent_var].values
    # k= np.concatenate([ori_ts, [pred_ts[0]]])
    # plt.plot(np.concatenate([ori_ts, [pred_ts[0]]]) , np.concatenate([ori_data, [forecast[0]]]), 'r')
    plt.plot(ori_ts, ori_data, 'r')
    plt.xlabel('ts')
    plt.ylabel(independent_var)
    plt.title('Historic Data')
    plt.savefig('Graphs/'+'Historic Data_ ' + 'mp_id_ ' + independent_var + ' _iteration ' + iteration)
    plt.show()


def prepare_data_mlp(data, mp_ids, target_mp_id, lag_time_steps, lead_time_steps,
                     test_split_size):
    X, y = series_to_supervised(data, lag_time_steps, lead_time_steps)

    leadColumns = [col for col in y.columns if target_mp_id == col[0:col.find('(')]]
    y = y[leadColumns]
    values = X.values
    X = values.reshape((values.shape[0], lag_time_steps, len(mp_ids)))
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=test_split_size, shuffle=False)

    x_inputs = []
    x_outputs = []

    for i in range(trainX.shape[2]):
        x_inputs.append(trainX[:, :, i])
    for i in range(testX.shape[2]):
        x_outputs.append(testX[:, :, i])

    return x_inputs, x_outputs, testX, testY, trainY

def prepare_data_cnn(data, mp_ids, target_mp_id, lag_time_steps, lead_time_steps,
                     test_split_size):
    X, y = series_to_supervised(data, lag_time_steps, lead_time_steps)

    leadColumns = [col for col in y.columns if target_mp_id == col[0:col.find('(')]]
    y = y[leadColumns]
    values = X.values
    X = values.reshape((values.shape[0], lag_time_steps, len(mp_ids)))
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=test_split_size, shuffle=False)

    x_inputs = []
    x_outputs = []

    for i in range(trainX.shape[2]):
        x_inputs.append(trainX[:, :, i].reshape(trainX.shape[0], trainX.shape[1], 1))
    for i in range(testX.shape[2]):
        x_outputs.append(testX[:, :, i].reshape(testX.shape[0], testX.shape[1], 1))

    return x_inputs, x_outputs, testX, testY, trainY


def get_mlp_model(x_inputs, lag_time_steps, lead_time_steps):
    input_models = []
    dense_layers = []
    for i in range(len(x_inputs)):
        visible = Input(shape=(lag_time_steps,))
        hidden0 = Dense(36, activation='relu')(visible)  # sigmoid #tanh #relu
        dropout0 = Dropout(0.2)(hidden0)
        dense = Dense(16, activation='relu')(dropout0)  # relu #sigmoid #tanh
        input_models.append(visible)
        dense_layers.append(dense)

    if len(x_inputs) > 1:
        merge = concatenate(dense_layers)
    else:
        merge = dense_layers[0]

    hidden1 = Dense(len(x_inputs) * 16, activation='relu')(merge)  # sigmoid #tanh #relu
    dropout1 = Dropout(0.2)(hidden1)
    hidden2 = Dense(len(x_inputs) * 4, activation='relu')(dropout1)  # relu #sigmoid #tanh
    dropout2 = Dropout(0.3)(hidden2)
    hidden3 = Dense(len(x_inputs), activation='relu')(dropout2)  # relu #sigmoid #tanh
    dropout3 = Dropout(0.1)(hidden3)
    output = Dense(lead_time_steps)(dropout3)

    model = Model(inputs=input_models, outputs=output)

    model.compile(optimizer='adam', loss='mse')
    return model

def get_cnn_model(x_inputs, lag_time_steps, lead_time_steps):
    input_models = []
    dense_layers = []
    for i in range(len(x_inputs)):
        visible = Input(shape=(lag_time_steps,1))
        cnn1 = Conv1D(filters=64, kernel_size=2, activation='relu')(visible)
        cnn1 = AveragePooling1D(pool_size=2)(cnn1)
        cnn1 = Flatten()(cnn1)
        dense_layers.append(cnn1)
        input_models.append(visible)

    if len(x_inputs) > 1:
        merge = concatenate(dense_layers)
    else:
        merge = dense_layers[0]

    hidden1 = Dense(len(x_inputs) * 16, activation='relu')(merge)  # sigmoid #tanh #relu
    dropout1 = Dropout(0.2)(hidden1)
    hidden2 = Dense(len(x_inputs) * 4, activation='relu')(dropout1)  # relu #sigmoid #tanh
    dropout2 = Dropout(0.2)(hidden2)
    hidden3 = Dense(len(x_inputs), activation='relu')(dropout2)  # relu #sigmoid #tanh
    dropout3 = Dropout(0.2)(hidden3)
    output = Dense(lead_time_steps)(dropout3)

    model = Model(inputs=input_models, outputs=output)

    model.compile(optimizer='adam', loss='mse')
    return model


def get_time_delta_and_freq(ganuality_level_value):
    if GanualityLevel.one_hour.value[0] == ganuality_level_value:
        return 1, '60T'
    elif GanualityLevel.three_hour.value[0] == ganuality_level_value:
        return 3, '180T'
    elif GanualityLevel.one_day.value[0] == ganuality_level_value:
        return 24, '1440T'
    elif GanualityLevel.one_week.value[0] == ganuality_level_value:
        return 168, '10080T'
    elif GanualityLevel.ten_min.value[0] == ganuality_level_value:
        return 0.16666666667, '10T'


def inverse_transform_forecast(pred_y, test_x, scaler, power_transformers, features):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    raw_pred_y = pred_y.reshape(-1, 1)
    test_x = np.concatenate((test_x, raw_pred_y), axis=1)
    for i in range(len(features)):
        if features[i] in power_transformers:
            test_x[:, i] = power_transformers[features[i]].inverse_transform(test_x[:, i].reshape(-1, 1)).reshape(test_x.shape[0])
    raw_pred_y = scaler.inverse_transform(test_x)[:, -1].reshape(-1, 1)

    if np.isnan(raw_pred_y).any():
        imp.fit(raw_pred_y)
        raw_pred_y = imp.transform(raw_pred_y)

    return raw_pred_y




def cal_rmse(model, x_outputs, testX, testY, lead_time_steps, features, normalizer, power_transformers, normalize_data=True):
    forecast = model.predict(x_outputs)
    testY_transformed = testY.values

    if normalize_data:
        tempTestX = testX.reshape(testX.shape[0], lag_time_steps * len(features))[:,
                        -(len(features)):-1]
        if len(tempTestX.shape) == 1:
            tempTestX = np.reshape(tempTestX, (1, tempTestX.shape[0]))

        for i in range(lead_time_steps):
            raw_forecast = inverse_transform_forecast(forecast[:, i], tempTestX, normalizer, power_transformers,
                                                          features)
            forecast[:, i] = raw_forecast.reshape(1, -1)

        for i in range(lead_time_steps):
            raw_test_data = inverse_transform_forecast(testY_transformed[:, i], tempTestX, normalizer, power_transformers,
                                                          features)
            testY_transformed[:, i] = raw_test_data.reshape(1, -1)

    rmse, rmse_list = rmse_time_series(testY_transformed, forecast)

    return rmse, rmse_list

def get_learning_curve_and_forecast(model, x_inputs, x_outputs, trainY, testX, testY, lead_time_step, lag_time_steps,
                                    features, forecast_input, normalizer, power_transformers, iteration, target,confidence_interval_multiple_factor ,normalize_data= True):
    history = model.fit(x_inputs, trainY, epochs=100, verbose=2, validation_data=(x_outputs, testY))

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig('Graphs/'+'mp_id_ '+target+'_learning curve_Lag_' + str(lag_time_steps) + '_Lead_' + str(lead_time_step) + '_iteration' + iteration)
    plt.show()

    forecast = model.predict(forecast_input)

    if normalize_data:
        tempTestX = testX.reshape(testX.shape[0], lag_time_steps * len(features))[-1,
                    -(len(features)):-1]
        tempTestX = np.reshape(tempTestX, (1, tempTestX.shape[0]))

        for i in range(lead_time_steps):
            raw_forecast = inverse_transform_forecast(forecast[:, i], tempTestX, normalizer, power_transformers, features)
            forecast[:, i] = raw_forecast.reshape(1, -1)

        rmse, rmse_list = cal_rmse(model, x_outputs, testX, testY, lead_time_steps, features, normalizer,
                                   power_transformers)

        rmse_list = [rmse * confidence_interval_multiple_factor for rmse in rmse_list]

    else:
        rmse, rmse_list = cal_rmse(model, x_outputs, testX, testY, lead_time_steps, features, normalizer,
                                   power_transformers,normalize_data=False)

        rmse_list = [rmse * confidence_interval_multiple_factor for rmse in rmse_list]


    print(rmse)

    return forecast, model, rmse_list

def get_features(dependent_var, independent_var):
    return list(map(str, dependent_var))+ list(map(str, independent_var))

def execute_mlp(dependent_var, independent_var, start_period, end_period, lag_time_steps, lead_time_steps, ganaulity_level,
                test_train_split, iteration, confidence_interval_multiple_factor, algo='MLP' ,normalize_data = True):
    original_data, normalized_data, normalizer, power_transformers = get_normalized_data(start_period, end_period, ganaulity_level,
                                                                                          dependent_var + independent_var )

    if original_data.values.shape[0] <= (lag_time_steps + lead_time_steps):
        print("Not enough Data, please change the configurations")
        return
    features = get_features(dependent_var, independent_var)
    if normalize_data:

        forecast_input = get_forecast_input(normalized_data[features], features, lag_time_steps)

        exploratoryAnalysis(normalized_data, str(independent_var[0]), features, iteration)

        if algo == 'MLP':
            x_inputs, x_outputs, testX, testY, trainY = prepare_data_mlp(normalized_data[features], features,
                                                                     str(independent_var[0]),
                                                                     lag_time_steps, lead_time_steps, test_train_split)
            model = get_mlp_model(x_inputs, lag_time_steps, lead_time_steps)
        elif algo == 'CNN':
            x_inputs, x_outputs, testX, testY, trainY = prepare_data_cnn(normalized_data[features], features,
                                                                         str(independent_var[0]),
                                                                         lag_time_steps, lead_time_steps,
                                                                         test_train_split)
            model = get_cnn_model(x_inputs, lag_time_steps, lead_time_steps)

        forecast, model, rmse_list = get_learning_curve_and_forecast(model, x_inputs, x_outputs, trainY, testX, testY, lead_time_steps,
                                                          lag_time_steps, features, forecast_input, normalizer,
                                                          power_transformers, iteration, str(independent_var[0]), confidence_interval_multiple_factor)
        plot_forecast(original_data, str(independent_var[0]), forecast, ganaulity_level, lead_time_steps, iteration, rmse_list)

    else:

        forecast_input = get_forecast_input(original_data[features], features, lag_time_steps)

        exploratoryAnalysis(original_data, str(independent_var[0]), features, iteration)

        if algo == 'MLP':

            x_inputs, x_outputs, testX, testY, trainY = prepare_data_mlp(original_data[features], features,
                                                                         str(independent_var[0]),
                                                                         lag_time_steps, lead_time_steps, test_train_split)
            model = get_mlp_model(x_inputs, lag_time_steps, lead_time_steps)
        elif algo == 'CNN':
            x_inputs, x_outputs, testX, testY, trainY = prepare_data_cnn(original_data[features], features,
                                                                         str(independent_var[0]),
                                                                         lag_time_steps, lead_time_steps,
                                                                         test_train_split)
            model = get_cnn_model(x_inputs, lag_time_steps, lead_time_steps)

        forecast, model, rmse_list = get_learning_curve_and_forecast(model, x_inputs, x_outputs, trainY, testX, testY,
                                                          lead_time_steps,
                                                          lag_time_steps, features, forecast_input, normalizer,
                                                          power_transformers, iteration, str(independent_var[0]), confidence_interval_multiple_factor, normalize_data=False)
        plot_forecast(original_data, str(independent_var[0]), forecast, ganaulity_level, lead_time_steps, iteration, rmse_list)


dependent_var = [1491] # Measuring Point Ids of dependent variables
independent_var = [] # Measuring Point Ids of Independent variable i.e. target variable
start_period = '11-01-2020T05:00' #'mm-dd-yyyyTHH:MM'
end_period = '11-03-2020T05:00' #'mm-dd-yyyyTHH:MM'
lag_time_steps = 200
lead_time_steps = 100
ganaulity_level = 6  # 3 for 10 mins, 4 for hour, 6 for days, 7 for weeks
test_train_split_size = 0.3
iteration = 3 #for each iteration unique graph file are saved
normalize_data = True
confidence_interval_multiple_factor = 1
algorithm = 'CNN' # MLP for multi layer preceptron, CNN for convolutional neural network
instance = 'hbc'

config_oauth(config.get_current_config())

try:
    execute_mlp(dependent_var, independent_var, start_period, end_period, lag_time_steps, lead_time_steps, ganaulity_level, test_train_split_size, str(iteration), confidence_interval_multiple_factor, algo= algorithm ,normalize_data= normalize_data)
except Exception as e:
    print(e)
