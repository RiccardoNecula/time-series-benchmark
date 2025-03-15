import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
#from keras_tuner import HyperModel, RandomSearch
from tensorflow.keras.optimizers import Adam
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from metrics.metrics import evaluate_model


from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from datetime import date, datetime



from datasets import load_rainfall_data, load_maxtemperature_data, load_nvidia_data

import warnings
warnings.filterwarnings("ignore")

np.random.seed(7)


def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    non_zero_idx = y_true != 0  # Filtra solo valori dove y_true != 0
    return 100 * np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx]))

df = load_nvidia_data()


if df.columns[0] != "date":
    df.columns.values[0] = 'date'
    #print(rainfall_data.columns[0])


df['date'] = pd.to_datetime(df['date'], format = '%Y-%m-%d')
#df = df.sort_values(by="date")

#df["delta"] = df["date"] - df["date"].shift(1)

#print(rainfall_data[["date", "delta"]].head()) #how many days pass from one measure to the other
#print(df["delta"].sum(), df["delta"].count()) #counts num of days and how many measures we got

#df = df.drop("delta", axis=1)
#print(df.isna().sum()) #in our case no need to interpolate, fill nan with 0 or fill nan mean value



#Finito Prophet

#UNVARIATE
if len(df.columns) == 2:

    #time series forecasting
    train_size = int(0.85 * len(df))
    test_size = len(df) - train_size

    target = df.columns[1]
    print(target)
    univariate_df = df[['date', target]].copy()
    univariate_df.columns = ['ds', 'y']

    print(univariate_df)

    train = univariate_df.iloc[:train_size, :]

    x_train, y_train = pd.DataFrame(univariate_df.iloc[:train_size, 0]), pd.DataFrame(univariate_df.iloc[:train_size, 1])
    x_valid, y_valid = pd.DataFrame(univariate_df.iloc[train_size:, 0]), pd.DataFrame(univariate_df.iloc[train_size:, 1])


    #print(len(train), len(x_valid))

    print(x_valid.head())
    print(x_valid.columns)

    # Train the model
    model = Prophet(growth='linear',
                    seasonality_prior_scale=0.2,
                    changepoint_prior_scale=12,
                    interval_width=0.95,
                    yearly_seasonality=True,
                    weekly_seasonality=True
    )
    model.fit(train)

    # x_valid = model.make_future_dataframe(periods=test_size, freq='w')

    # Predict on valid set
    y_pred = model.predict(x_valid)



    print(y_pred.tail(test_size)["yhat"])

    # Calculate metrics
    score_mae, score_mse, score_rmse, score_mape = evaluate_model(y_valid, y_pred.tail(test_size)["yhat"])

    print('MAE: {}'.format(score_mae))
    print('MSE: {}'.format(score_mse))
    print('RMSE: {}'.format(score_rmse))
    print('MAPE: {}'.format(score_mape))

    #Plot the forecast
    f, ax = plt.subplots(1)
    f.set_figheight(8)
    f.set_figwidth(15)

    model.plot(y_pred, ax=ax)
    sns.lineplot(x=x_valid['ds'], y=y_valid['y'], ax=ax, color='#FF7F50', label='Ground truth')
    model.plot(y_pred, ax=ax)

    ax.set_title(f'Univariate Dataset Prophet Prediction \n MAE: {score_mae:.2f}, RMSE: {score_rmse:.2f}, MSE: {score_mse:.2f}, MAPE: {score_mape:.2f}', fontsize=14)
    ax.set_xlabel(xlabel='Periodo', fontsize=14)
    ax.set_ylabel(f'{df.columns.values[1]}', fontsize=14)

    plt.show()

#MULTIVARIATE
else:
    # Define feature columns and target column
    feature_columns = [
        'Open',
        'High',
        'Low',
        'Adj Close',
        'Volume'
    ]

    target_column = ['Close']

    train_size = int(0.85 * len(df))
    test_size = len(df) - train_size

    multivariate_df = df[['date'] + target_column + feature_columns].copy()
    multivariate_df.columns = ['ds', 'y'] + feature_columns

    train = multivariate_df.iloc[:train_size, :]
    x_train, y_train = pd.DataFrame(multivariate_df.iloc[:train_size, [0, 2, 3, 4, 5, 6]]), pd.DataFrame(
        multivariate_df.iloc[:train_size, 1])
    x_valid, y_valid = pd.DataFrame(multivariate_df.iloc[train_size:, [0, 2, 3, 4, 5, 6]]), pd.DataFrame(
        multivariate_df.iloc[train_size:, 1])

    print(train.head())

    # Initialize the Prophet model
    model = Prophet(
        growth='linear',
        seasonality_prior_scale=0.2,
        changepoint_prior_scale=12,
        interval_width=0.95, #viene automaticamente disabilitato quando si aggiungono regressori perchè dati per certi
        yearly_seasonality=True,
        weekly_seasonality=True
    )

    # Add regressors to the Prophet model
    for feature in feature_columns:
        model.add_regressor(feature)

    # Fit the model with the train setr
    model.fit(train)

    y_pred = model.predict(x_valid)

    # Calcolo del MAPE con gestione dei valori uguali a zero o vicini a zero
    def calculate_mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
        non_zero_idx = y_true != 0  # Filtra solo valori dove y_true != 0
        return 100 * np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx]))

    # Calculate metrics
    score_mae = mean_absolute_error(y_valid, y_pred.tail(test_size)['yhat'])  # yhat è colonna in prophet
    score_mse = mean_squared_error(y_valid, y_pred.tail(test_size)['yhat'])
    score_rmse = math.sqrt(score_mse)
    score_mape = calculate_mape(y_valid, y_pred.tail(test_size)[
        'yhat'])  # non lavora bene con dataset con molti valori piccoli o 0

    print('MAE: {}'.format(score_mae))
    print('MSE: {}'.format(score_mse))
    print('RMSE: {}'.format(score_rmse))
    print('MAPE: {}'.format(score_mape))

    # Plot the results
    f, ax = plt.subplots(1)
    f.set_figheight(6)
    f.set_figwidth(15)

    model.plot(y_pred, ax=ax)
    sns.lineplot(x=x_valid['ds'], y=y_valid['y'], ax=ax, color='#FF7F50', label='Ground truth')
    model.plot(y_pred, ax=ax)

    ax.set_title(f'Multivariate Dataset Prophet Prediction \n MAE: {score_mae:.2f}, RMSE: {score_rmse:.2f}, MSE: {score_mse:.2f}, MAPE: {score_mape:.2f}', fontsize=14)
    ax.set_xlabel(xlabel='Periodo', fontsize=14)
    ax.set_ylabel(f'{target_column[0]}', fontsize=14)

    plt.show()
"""

print("--------------------------------------------------------------------------------------------")

#ARIMA


if len(df.columns) == 2:
    train_size = int(0.85 * len(df))
    test_size = len(df) - train_size

    univariate_df = df[['date', 'Temperature']].copy()
    univariate_df.columns = ['ds', 'y']

    train = univariate_df.iloc[:train_size, :]

    x_train, y_train = pd.DataFrame(univariate_df.iloc[:train_size, 0]), pd.DataFrame(univariate_df.iloc[:train_size, 1])
    x_valid, y_valid = pd.DataFrame(univariate_df.iloc[train_size:, 0]), pd.DataFrame(univariate_df.iloc[train_size:, 1])


else:
    # Define feature columns and target column
    feature_columns = [
        'Open',
        'High',
        'Low',
        'Adj Close',
        'Volume'
    ]

    target_column = ['Close']

    train_size = int(0.85 * len(df))
    test_size = len(df) - train_size

    multivariate_df = df[['date'] + target_column + feature_columns].copy()
    multivariate_df.columns = ['ds', 'y'] + feature_columns

    train = multivariate_df.iloc[:train_size, :]
    x_train, y_train = pd.DataFrame(multivariate_df.iloc[:train_size, [0, 2, 3, 4, 5, 6]]), pd.DataFrame(
        multivariate_df.iloc[:train_size, 1])
    x_valid, y_valid = pd.DataFrame(multivariate_df.iloc[train_size:, [0, 2, 3, 4, 5, 6]]), pd.DataFrame(
        multivariate_df.iloc[train_size:, 1])


from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(y_train, lags=20)
plt.show()

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(y_train, lags=20)
plt.show()

from pmdarima import auto_arima


# Calcolo automatico dei parametri ottimali (p, d, q)
auto_model = auto_arima(y_train,
                        test="adf",
                        seasonal=False,   # No Seasonality
                        trace=True,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True)


best_order = auto_model.order  # Estrarre i migliori (p, d, q)
print(f"Migliori parametri trovati da auto_arima: {best_order}")
print(auto_model.summary())

# Fit model
model = ARIMA(y_train, best_order)
model_fit = model.fit()

# Prediction with ARIMA
y_pred = model_fit.forecast(steps=len(y_valid))



score_mae = mean_absolute_error(y_valid, y_pred)
score_mse = mean_squared_error(y_valid, y_pred)
score_rmse = math.sqrt(score_mse)
score_mape = calculate_mape(y_valid, y_pred)

print('MAE: {}'.format(score_mae))
print('MSE: {}'.format(score_mse))
print('RMSE: {}'.format(score_rmse))
print('MAPE: {}'.format(score_mape))

# Creazione delle date corrette per le previsioni
last_date = x_train.iloc[-1, 0]  # Ultima data disponibile nel training set
forecast_index = pd.date_range(start=last_date, periods=len(y_valid), freq='D')  # Frequenza giornaliera

# Ottenere le previsioni con intervalli di confidenza
forecast = model_fit.get_forecast(steps=len(y_valid))
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()
print(forecast_ci.head())

# Plot dei risultati
plt.figure(figsize=(15, 8))

# Serie storica
sns.lineplot(x=x_train['ds'], y=y_train['y'], label="Dati storici", color="black")

# Ground Truth
sns.lineplot(x=x_valid['ds'], y=y_valid['y'], label="Ground Truth", color="#FF7F50")

# Previsioni ARIMA
sns.lineplot(x=forecast_index, y=forecast_mean, label="Previsione ARIMA", color="dodgerblue")

# Intervalli di confidenza
plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='blue', alpha=0.2, label="Intervallo di Confidenza")

plt.title(f"Previsione con ARIMA \n MAE: {score_mae:.2f}, RMSE: {score_rmse:.2f}", fontsize=14)
plt.xlabel("Data", fontsize=12)
plt.ylabel("Quantità di Pioggia (mm)", fontsize=12)
plt.legend()
plt.show()



print("----------------------------------------------------------------------------------------------------------------------")


"""
if len(df.columns) == 2:
    train_size = int(0.85 * len(df))
    test_size = len(df) - train_size

    univariate_df = df[['date', 'rainfall']].copy()
    univariate_df.columns = ['ds', 'y']

    train = univariate_df.iloc[:train_size, :]

    x_train, y_train = pd.DataFrame(univariate_df.iloc[:train_size, 0]), pd.DataFrame(univariate_df.iloc[:train_size, 1])
    x_valid, y_valid = pd.DataFrame(univariate_df.iloc[train_size:, 0]), pd.DataFrame(univariate_df.iloc[train_size:, 1])

    data = univariate_df.filter(['y'])

else:
    # Define feature columns and target column
    feature_columns = [
        'Open',
        'High',
        'Low',
        'Adj Close',
        'Volume'
    ]

    target_column = ['Close']

    train_size = int(0.85 * len(df))
    test_size = len(df) - train_size

    multivariate_df = df[['date'] + target_column + feature_columns].copy()
    multivariate_df.columns = ['ds', 'y'] + feature_columns

    train = multivariate_df.iloc[:train_size, :]
    x_train, y_train = pd.DataFrame(multivariate_df.iloc[:train_size, [0, 2, 3, 4, 5, 6]]), pd.DataFrame(
        multivariate_df.iloc[:train_size, 1])
    x_valid, y_valid = pd.DataFrame(multivariate_df.iloc[train_size:, [0, 2, 3, 4, 5, 6]]), pd.DataFrame(
        multivariate_df.iloc[train_size:, 1])

    data = multivariate_df.filter(['y'])

#print(len(train), len(x_valid))



#Convert the dataframe to a numpy array
dataset = data.values

scaler = MinMaxScaler(feature_range=(-1, 0))
scaled_data = scaler.fit_transform(dataset)

print(scaled_data[:10])

# Defines the rolling window
look_back = 14 #imposto 14 perchè ho un dataset piccolo per rainfall

# Split into train and test sets
train, test = scaled_data[:train_size-look_back,:], scaled_data[train_size-look_back:,:]

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(look_back, len(dataset)):
        a = dataset[i-look_back:i, 0]
        X.append(a)
        Y.append(dataset[i, 0])
    return np.array(X), np.array(Y)

x_train, y_train = create_dataset(train, look_back)
x_test, y_test = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

print(len(x_train), len(x_test))


#Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
keras.layers.Dropout(0.3)
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

optimizer = Adam(learning_rate=0.001)
# Compile the model
model.compile(optimizer=optimizer, loss='mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=5, validation_data=(x_test, y_test))
model.summary()

#Fare previsioni sui dati di test con x_test
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

#Ridimensionare le previsioni e i dati veri per riportarli alla scala originale
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])

test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Calcolo del MAPE con gestione dei valori uguali a zero o vicini a zero
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    non_zero_idx = y_true != 0  # Filtra solo valori dove y_true != 0
    return 100 * np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx]))


score_mae = mean_absolute_error(y_test[0], test_predict[:,0])
score_mse = mean_squared_error(y_test[0], test_predict[:,0])
score_rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
score_mape = calculate_mape(y_test[0], test_predict[:,0])

print('MAE: {}'.format(score_mae))
print('MSE: {}'.format(score_mse))
print('RMSE: {}'.format(score_rmse))
print('MAPE: {}'.format(score_mape))


# Plot the forecast
f, ax = plt.subplots(1)
f.set_figheight(6)
f.set_figwidth(15)

#print(len(y_train))
#print(y_train)
#y_train_flat = y_train[0].flatten()
#print(y_train_flat)

if len(df.columns) == 2:
    x_train_ticks = univariate_df.head(train_size)['ds']
    y_train = univariate_df.head(train_size)['y']
    x_test_ticks = univariate_df.tail(test_size)['ds']
else:
    x_train_ticks = multivariate_df.head(train_size)['ds']
    y_train = multivariate_df.head(train_size)['y']
    x_test_ticks = multivariate_df.tail(test_size)['ds']

sns.lineplot(x=x_train_ticks, y=y_train, ax=ax, color='dodgerblue', label='Train Set')
sns.lineplot(x=x_test_ticks, y=y_test[0], ax=ax, color='#FF7F50', label='Ground Truth')
sns.lineplot(x=x_test_ticks, y=test_predict[:,0], ax=ax, color='darkgreen', label='Prediction')

ax.set_title(f'LSTM Prediction \n MAE: {score_mae:.2f}, RMSE: {score_rmse:.2f}, MSE: {score_mse:.2f}, MAPE: {score_mape:.2f}', fontsize=14)
ax.set_xlabel(xlabel='Periodo', fontsize=14)
ax.set_ylabel(f'{df.columns.values[1]}', fontsize=14)

plt.show()
"""


print("-----------------------------------------------------------------------------------------------------------------------------------------------")



#Auto - ARIMA

from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm

model = pm.auto_arima(y_train, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)

print(model.summary())

model.plot_diagnostics(figsize=(16,8))
plt.show()
"""