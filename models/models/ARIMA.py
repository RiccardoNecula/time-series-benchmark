import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from pmdarima import auto_arima

from metrics.metrics import evaluate_model, seleziona_metriche
from hparams.hparams import Auto_arima_params

def ARIMA_model(df):

    # utente seleziona la metrica che gli interessa
    metrica_sel = seleziona_metriche()  # utente seleziona la metrica che gli interessa

    if len(df.columns) == 2:
        train_size = int(0.85 * len(df))
        test_size = len(df) - train_size

        target = df.columns[1]
        univariate_df = df[['date', target]].copy()
        univariate_df.columns = ['ds', 'y']

        train = univariate_df.iloc[:train_size, :]

        x_train, y_train = pd.DataFrame(univariate_df.iloc[:train_size, 0]), pd.DataFrame(
            univariate_df.iloc[:train_size, 1])
        x_valid, y_valid = pd.DataFrame(univariate_df.iloc[train_size:, 0]), pd.DataFrame(
            univariate_df.iloc[train_size:, 1])


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


    plot_pacf(y_train, lags=20)
    plt.show()

    plot_acf(y_train, lags=20)
    plt.show()



    # Calcolo automatico dei parametri ottimali (p, d, q)
    auto_model = auto_arima(y_train, seasonal=Auto_arima_params["seasonal"], stepwise=Auto_arima_params["stepwise"], trace=Auto_arima_params["trace"],
                            suppress_warnings=Auto_arima_params["suppress_warning"])

    best_order = auto_model.order  # Estrarre i migliori (p, d, q)
    print(f"Migliori parametri trovati da auto_arima: {best_order}")

    # Fit model
    model = ARIMA(y_train, order=best_order, trend='n', concentrate_scale=True) #HPARAMS

    model_fit = model.fit()

    # Prediction with ARIMA
    y_pred = model_fit.forecast(steps=len(y_valid))

    # Calculate metrics
    score_mae, score_mse, score_rmse, score_mape = evaluate_model(y_valid, y_pred)

    # Dictionary of available scores
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


    # Creazione delle date corrette per le previsioni
    last_date = x_train.iloc[-1, 0]  # Ultima data disponibile nel training set
    forecast_index = pd.date_range(start=last_date, periods=len(y_valid), freq='D')  # Frequenza giornaliera

    # Ottenere le previsioni con intervalli di confidenza
    forecast = model_fit.get_forecast(steps=len(y_valid))
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    #print(forecast_ci.head())

    # Plot dei risultati
    plt.figure(figsize=(15, 8))

    # Serie storica
    sns.lineplot(x=x_train['ds'], y=y_train['y'], label="Dati storici", color="dodgerblue")

    # Ground Truth
    sns.lineplot(x=x_valid['ds'], y=y_valid['y'], label="Ground Truth", color="#FF7F50")

    # Previsioni ARIMA
    sns.lineplot(x=forecast_index, y=forecast_mean, label="Previsione ARIMA", color="darkgreen")

    # Intervalli di confidenza
    plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='blue', alpha=0.2,
                     label="Intervallo di Confidenza")

    plt.title("Previsione con ARIMA", fontsize=14)
    plt.xlabel("Periodo", fontsize=14)
    if len(df.columns) == 2:
        plt.ylabel(f'{df.columns.values[1]}', fontsize=14)
    else:
        plt.ylabel(f'{target_column[0]}', fontsize=14)
    plt.legend()
    plt.show()