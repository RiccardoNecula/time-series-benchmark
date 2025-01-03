import warnings
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from datasets import load_nvidia_data

#mancano in requirements
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# from datetime import datetime

warnings.filterwarnings("ignore")

nvidia_data = load_nvidia_data()
print(nvidia_data.head())  # 7 columns, including the Date.

# Separate dates for future plotting
train_dates = pd.to_datetime(nvidia_data['Date'])
print(train_dates.tail(15))  # Check last few dates.

# Variables for training
cols = list(nvidia_data)[1:6]
# Date and volume columns are not used in training.
print(cols)  # ['Open', 'High', 'Low', 'Close', 'Adj Close']

# New dataframe with only training data - 5 columns
df_for_training = nvidia_data[cols].astype(float)

# df_for_plot=df_for_training.tail(500)
# df_for_plot.plot.line()

# LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

# As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
# In this example, the n_features is 5. We will make timesteps = 14 (past days data used for training).

# Empty lists to be populated using formatted training data
trainX = []
trainY = []

n_future = 1  # Number of days we want to look into the future based on the past days.
n_past = 14  # Number of past days we want to use to predict the future.

# Reformat input data into a shape: (n_samples x timesteps x n_features)
# In my example, my df_for_training_scaled has a shape (1260, 5)
# 1246 refers to the number of data points and 5 refers to the columns (multi-variables). #mine 1260
for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

# In my case, trainX has a shape (1260, 14, 5).
# 12809 because we are looking back 14 days (1260 - 14 = 1248).
# Remember that we cannot look back 14 days until we get to the 15th day.
# Also, trainY has a shape (1248, 1). Our model only predicts a single value, but
# it needs multiple variables (5 in my example) to make this prediction.
# This is why we can only predict a single day after our training, the day after where our data ends.
# To predict more days in future, we need all the 5 variables which we do not have.
# We need to predict all variables if we want to do that.

# define the Autoencoder model

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()

# fit the model
history = model.fit(trainX, trainY, epochs=5, batch_size=16, validation_split=0.1, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

# Predicting...
# Libraries that will help us extract only business days in the US.
# Otherwise our dates would be wrong when we look back (or forward).
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
# Remember that we can only predict one day in future as our model needs 5 variables
# as inputs for prediction. We only have all 5 variables until the last day in our dataset.
n_past = 16
n_days_for_prediction = 15  # let us predict past 15 days

predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq=us_bd).tolist()
print(predict_period_dates)

# Make prediction
prediction = model.predict(trainX[-n_days_for_prediction:])  # shape = (n, 1) where n is the n_days_for_prediction

# Perform inverse transformation to rescale back to original range
# Since we used 5 variables for transform, the inverse expects same dimensions
# Therefore, let us copy our values 5 times and discard them after inverse transform
prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:, 0]

# Convert timestamp to date
forecast_dates = []
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())

df_forecast = pd.DataFrame({'Date': np.array(forecast_dates), 'Open': y_pred_future})
df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])

original = nvidia_data[['Date', 'Open']]
original['Date'] = pd.to_datetime(original['Date'])

# Filtrare i dati per il range di date specifico
start_date = '2016-01-01'
end_date = '2022-12-31'
original_filtered = original[(original['Date'] >= start_date) & (original['Date'] <= end_date)]
df_forecast_filtered = df_forecast[(df_forecast['Date'] >= start_date) & (df_forecast['Date'] <= end_date)]

# Plot dei dati filtrati
sns.lineplot(x=original_filtered['Date'], y=original_filtered['Open'], label='Original')
sns.lineplot(x=df_forecast_filtered['Date'], y=df_forecast_filtered['Open'], label='Forecast')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title('Original vs Forecasted Open Prices (2016-2022)')
plt.show()

#Teoricamente questa parte dovrenne disegnare solo tra 2016 e 2023
# Filtrare i dati per il range di date specifico
start_date = '2017-01-01'
end_date = '2023-01-01'
original_filtered = original[(original['Date'] >= start_date) & (original['Date'] <= end_date)]
df_forecast_filtered = df_forecast[(df_forecast['Date'] >= start_date) & (df_forecast['Date'] <= end_date)]

# Creare il plot con Seaborn
plt.figure(figsize=(12, 6))  # Impostare dimensioni del grafico
sns.lineplot(x=original_filtered['Date'], y=original_filtered['Open'], label='Original', color='blue')
sns.lineplot(x=df_forecast_filtered['Date'], y=df_forecast_filtered['Open'], label='Forecast', color='orange')

# Personalizzazione del grafico
plt.xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))
plt.xlabel('Date', fontsize=12)
plt.ylabel('Open Price', fontsize=12)
plt.title('Original vs Forecasted Open Prices NVIDIA (2016-2023)', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()


"""

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout

#from tensorflow.python.keras.models import Sequential -> così compila ma l'errore rimane in shell
#from tensorflow.python.keras.layers import Dense,LSTM,Dropout
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(10, 5)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

print("Import e configurazione corretti!")
"""