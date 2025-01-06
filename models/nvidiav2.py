import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from datasets import load_nvidia_data

#This actual state, model predicts on historical text data. Working on predicting on future data [part]

nvidia_data = load_nvidia_data()

"""
print(nvidia_data.shape, "\n")
print(nvidia_data.head(), "\n")

#checks for null values
print(nvidia_data.isna().sum(), "\n")

print(nvidia_data.info(), "\n")
"""

#converting "Date" col into "datetime"
nvidia_data["Date"] = pd.to_datetime(nvidia_data["Date"])

#making "Date" new index
nvidia_data.set_index("Date", inplace=True) #inplace, for memory optimization

print(nvidia_data.info())

#sort the indexes
nvidia_data.sort_index(inplace=True)
#print(nvidia_data.head())

#apply log transformation to Volume to stabilize variance having a heteroscendastic dataset
nvidia_data['Log_Volume'] = np.log(nvidia_data['Volume'])

#delete old volume col
nvidia_data.drop(columns=["Volume"], inplace=True)
#print(nvidia_data.head())

#normalizing the data -> all data between 0 & 1
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(nvidia_data[nvidia_data.columns])
print(scaled_values)

#converting the array into dataframe
nvidia_data_scaled = pd.DataFrame(scaled_values, columns=nvidia_data.columns, index=nvidia_data.index)
print(nvidia_data.head())


#plot the columns
plt.rcParams["figure.figsize"] = (20, 20)
figure, axes = plt.subplots(6)

for ax, col in zip(axes, nvidia_data_scaled.columns):
    ax.plot(nvidia_data_scaled[col])
    ax.set_title(col)
    ax.axes.xaxis.set_visible(True)

plt.show()


#create the sliding door
window_size = 5
def create_sequence(data, window_size):
    X = []
    y = []
    for i in range(window_size, len(data)):
        X.append(data.iloc[i - window_size:i].values)
        y.append(data.iloc[i].values)
    return np.array(X), np.array(y)

X, y = create_sequence(nvidia_data_scaled, window_size)
print("dimensioni X e y dopo sliding window: ", X.shape, y.shape)

#train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
print("dimensioni X e y del training:", X_train.shape, y_train.shape, "\n")
print("dimensioni di X e y nel test:", X_test.shape, y_test.shape)

#LSTM model
model = keras.Sequential([
    #Adding the first LSTM Layer wth Dropout
    keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    keras.layers.Dropout(0.3),

    #Adding the second LSTM model with Dropout
    keras.layers.LSTM(units=50, return_sequences=True),
    keras.layers.Dropout(0.3),

    #Adding the third LSTM model Layer with Dropout
    keras.layers.LSTM(units=50, return_sequences=False),
    keras.layers.Dropout(0.3),

    #Adding a Dense output Layer
    keras.layers.Dense(y_train.shape[1])
])

#compilation
model.compile(optimizer=Adam(learning_rate=0.01),
              loss="mean_squared_error",
              metrics=["RootMeanSquaredError"])
model.summary()

#Early stopping condition
early_stopping = EarlyStopping(monitor="val_loss",
                               patience=10,
                               restore_best_weights=True)

history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    epochs=20,
                    batch_size=3,
                    callbacks=[early_stopping]) #early stopping prevents overfitting
#Making predictions on the test data
predictions = model.predict(X_test)

#Inverse scaling to get the original values
predictions = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test)

#Plotting the results
plt.figure(figsize=(14, 7))

#plotting on test data, existing days
for i, col in enumerate(nvidia_data_scaled.columns):
    plt.subplot(2, 3, i+1)
    plt.plot(y_test_rescaled[:, i], color="blue", label=f"Actual {col}")
    plt.plot(predictions[:, i], color="red", label=f"Predicted {col}")
    plt.title(f"{col} Price Prediction")
    plt.xlabel("Time")
    plt.ylabel(f"{col} Price")
    plt.legend()

plt.tight_layout()
plt.show()



#to resolve: visualization of future predictions


#future predictions function
def predict_future(model, data, scaler, window_size, future_steps):
    future_predictions = []
    current_sequence = data[- window_size:]  #starts with last available window

    for _ in range(future_steps):
        current_sequence_scaled = scaler.transform(pd.DataFrame(current_sequence, columns=data.columns))
        prediction = model.predict(np.expand_dims(current_sequence_scaled, axis=0))
        prediction_rescaled = scaler.inverse_transform(prediction)
        future_predictions.append(prediction_rescaled[0])

        # Update the current sequence by adding the new prediction and removing the oldest entry
        current_sequence = np.append(current_sequence[1:], prediction_rescaled, axis=0)

    return np.array(future_predictions)

# Number of future steps to predict
future_steps = 10

# Make future predictions
future_predictions = predict_future(model, nvidia_data_scaled, scaler, window_size, future_steps)

# Print future predictions
print(future_predictions)

# Generate future dates starting from the last available date
future_dates = pd.date_range(start=nvidia_data_scaled.index[-1], periods=future_steps + 1, freq='B')[1:]

# Convert future predictions (numpy array) into a pandas DataFrame with the future dates as the index
future_predictions_df = pd.DataFrame(future_predictions, columns=nvidia_data.columns, index=future_dates)

# Plot actual and predicted values
plt.figure(figsize=(14, 7))

for i, col in enumerate(nvidia_data_scaled.columns):
    plt.subplot(2, 3, i + 1)

    # Plot actual values in green
    plt.plot(nvidia_data.index, nvidia_data[col], color="green", label=f"Actual {col}")

    # Plot predicted values in yellow starting from the last actual date
    plt.plot(future_predictions_df.index, future_predictions_df[col], color="brown", label=f"Predicted {col}", linestyle='--')

    # Add labels and title
    plt.title(f"{col} Future Price Over Time")
    plt.xlabel("Time")
    plt.ylabel(f"{col} Predicted Price")
    plt.legend()

    plt.xlim([nvidia_data.index.min(), pd.Timestamp("2023-12-31")])

plt.tight_layout()
plt.show()
