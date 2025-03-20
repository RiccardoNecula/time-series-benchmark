import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

from metrics.metrics import evaluate_model, seleziona_metriche
from hparams.hparams import LSTM_grid_params

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i - look_back:i, 0])
        Y.append(dataset[i, 0])
    return np.array(X), np.array(Y)


def LSTM_model(df):

    # utente seleziona la metrica che gli interessa
    metrica_sel = seleziona_metriche()#utente seleziona la metrica che gli interessa

    if len(df.columns) == 2:
        train_size = int(0.85 * len(df))
        test_size = len(df) - train_size

        target = df.columns[1]
        univariate_df = df[['date', target]].copy()
        univariate_df.columns = ['ds', 'y']

        data = univariate_df.filter(['y'])

    else:
        feature_columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
        target_column = ['Close']

        train_size = int(0.85 * len(df))
        test_size = len(df) - train_size

        multivariate_df = df[['date'] + target_column + feature_columns].copy()
        multivariate_df.columns = ['ds', 'y'] + feature_columns

        data = multivariate_df.filter(['y'])

    dataset = data.values
    scaler = MinMaxScaler(feature_range=(-1, 0))
    scaled_data = scaler.fit_transform(dataset)

    look_back = 14

    train, test = scaled_data[:train_size - look_back, :], scaled_data[train_size - look_back:, :]

    x_train, y_train = create_dataset(train, look_back)
    x_test, y_test = create_dataset(test, look_back)

    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))



    #GRID SEARCH CON numero di unità nel aye TUNER
    def build_model(hp):
        model = Sequential()
        model.add(LSTM(
            units=hp.Int('units_1', min_value=LSTM_grid_params['unit1_min'], max_value=LSTM_grid_params['unit1_max'], step=LSTM_grid_params['unit1_step']),
            return_sequences=True,
            input_shape=(x_train.shape[1], x_train.shape[2])
        ))
        model.add(Dropout(hp.Float('dropout_1', LSTM_grid_params['dropout_min'], LSTM_grid_params['dropout_max'], step=LSTM_grid_params['dropout_step'])))

        model.add(LSTM(hp.Int('units_2', min_value=LSTM_grid_params['unit2_min'], max_value=LSTM_grid_params['unit2_max'], step=LSTM_grid_params['unit2_step']), return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=LSTM_grid_params['learning_rate_min'], max_value=LSTM_grid_params['learning_rate_max'], step=LSTM_grid_params['learning_rate_step']))
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=10,  # Numero di combinazioni da testare
        executions_per_trial=1,
        #directory='lstm_tuning', #serve per salvare il modello
        #project_name='lstm_gridsearch'
    )

    tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=1)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Migliori iperparametri trovati: {best_hps.values}")

    model = tuner.hypermodel.build(best_hps)

    model.fit(x_train, y_train, batch_size=16, epochs=5, validation_data=(x_test, y_test))
    model.summary()

    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)


    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform([y_test])

    score_mae, score_mse, score_rmse, score_mape = evaluate_model(y_test[0], test_predict[:, 0])

    metric_scores = {
        "MAE": score_mae,
        "MSE": score_mse,
        "RMSE": score_rmse,
        "MAPE": score_mape
    }

    # Check if the user selected 'all' or one specific metric
    if metrica_sel == 0:
        print("Hai selezionato la metrica standard:\n")
        print('📊 MAE:', score_mae)

    elif metrica_sel == 5:  # '5' means "all metrics"
        print("Hai selezionato tutte le metriche:\n")
        print('📊 MAE:', score_mae)
        print('📊 MSE:', score_mse)
        print('📊 RMSE:', score_rmse)
        print('📊 MAPE:', score_mape)

    else:
        # Retrieve the metric name from the mapping and print its score
        nome_metrica = ["MAE", "MSE", "RMSE", "MAPE"][metrica_sel - 1]  # Correct index by subtracting 1
        print(f"📊 La metrica che hai selezionato:\n{nome_metrica}: {metric_scores[nome_metrica]}")

    f, ax = plt.subplots(1)
    f.set_figheight(6)
    f.set_figwidth(15)

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
    sns.lineplot(x=x_test_ticks, y=test_predict[:, 0], ax=ax, color='darkgreen', label='Prediction')

    ax.set_title('LSTM Prediction', fontsize=14)
    ax.set_xlabel(xlabel='Periodo', fontsize=14)
    ax.set_ylabel(f'{df.columns.values[1] if len(df.columns) == 2 else target_column[0]}', fontsize=14)
    plt.show()
