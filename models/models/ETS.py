import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error
from itertools import product

from metrics.metrics import evaluate_model, seleziona_metriche
from hparams.hparams import ETS_smoothing_params


def ETS_model(df):
    # Selezione metrica utente
    metrica_sel = seleziona_metriche()

    # Controllo se il dataset è univariato o multivariato
    if len(df.columns) == 2:
        train_size = int(0.85 * len(df))

        target = df.columns[1]
        univariate_df = df[['date', target]].copy()
        univariate_df.columns = ['ds', 'y']

        x_train, y_train = univariate_df.iloc[:train_size, 0], univariate_df.iloc[:train_size, 1]
        x_valid, y_valid = univariate_df.iloc[train_size:, 0], univariate_df.iloc[train_size:, 1]

    else:
        feature_columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
        target_column = ['Close']

        train_size = int(0.85 * len(df))

        multivariate_df = df[['date'] + target_column + feature_columns].copy()
        multivariate_df.columns = ['ds', 'y'] + feature_columns

        x_train, y_train = multivariate_df.iloc[:train_size, 0], multivariate_df.iloc[:train_size, 1]
        x_valid, y_valid = multivariate_df.iloc[train_size:, 0], multivariate_df.iloc[train_size:, 1]

    # Stima della stagionalità con seasonal_decompose
    decomposition = seasonal_decompose(y_train, period=12, model='additive')
    seasonal_period = 12  # Valore predefinito, può essere ottimizzato

    # Genera tutte le combinazioni di iperparametri
    param_grid = list(product(ETS_smoothing_params["trend"],
                              ETS_smoothing_params["seasonal"],
                              ETS_smoothing_params["damped_trend"],
                              ETS_smoothing_params["seasonal_periods"]))

    # Trova la migliore combinazione di iperparametri
    best_score = float("inf")  # Inizializza con un valore molto alto
    best_params = None
    best_model = None

    for params in param_grid:
        trend, seasonal, damped_trend, seasonal_periods = params
        try:
            # Modello Exponential Smoothing con iperparametri attuali
            model = ExponentialSmoothing(
                y_train,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                damped_trend=damped_trend
            ).fit()

            # Previsione
            y_pred = model.forecast(steps=len(y_valid))

            # Calcolo delle metriche (es. MAE come criterio di selezione)
            mae = mean_absolute_error(y_valid, y_pred)

            # Aggiorna il miglior modello se il punteggio corrente è migliore
            if mae < best_score:
                best_score = mae
                best_params = params
                best_model = model

        except Exception as e:
            print(f"Errore con i parametri {params}: {e}")

    # Stampa dei migliori parametri e della migliore metrica
    print(
        f"Migliori parametri trovati: Trend: {best_params[0]}, Stagionalità: {best_params[1]}, Damped: {best_params[2]}, "
        f"Stagionalità periodi: {best_params[3]}")
    #print(f"MAE migliore: {best_score:.4f}")


    # Calcolo delle metriche di valutazione
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
    last_date = x_train.iloc[-1]  # Ultima data disponibile nel training set
    forecast_index = pd.date_range(last_date, periods=len(y_valid), freq='D')


    """
    # Plot dei risultati
    plt.figure(figsize=(15, 8))

    sns.lineplot(x=x_train, y=y_train, label="Dati storici", color="dodgerblue")
    sns.lineplot(x=x_valid, y=y_valid, label="Ground Truth", color="#FF7F50")
    sns.lineplot(x=forecast_index, y=y_pred, label="Previsione ETS", color="darkgreen")

    plt.title("Previsione con ETS", fontsize=14)
    plt.xlabel("Periodo", fontsize=14)
    plt.ylabel("Valore", fontsize=14)
    plt.legend()
    plt.show()
    """