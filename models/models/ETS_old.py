import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from metrics.metrics import evaluate_model, seleziona_metriche
from itertools import product  # Per generare combinazioni di iperparametri

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from metrics.metrics import evaluate_model, seleziona_metriche

from datasets import load_rainfall_data, load_maxtemperature_data, load_nvidia_data
df = load_rainfall_data()

def ETS_model(df):

    if df.columns[0] != "date":
        df.columns.values[0] = 'date'
        # print(rainfall_data.columns[0])

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')


    # Selezione metrica utente
    metrica_sel = seleziona_metriche()

    # Controllo se il dataset è univariato o multivariato
    if len(df.columns) == 2:
        train_size = int(0.85 * len(df))
        test_size = len(df) - train_size

        target = df.columns[1]
        univariate_df = df[['date', target]].copy()
        univariate_df.columns = ['ds', 'y']

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

    # Stima della stagionalità con seasonal_decompose
    decomposition = seasonal_decompose(y_train, period=12, model='additive')
    seasonal_period = 12  # Valore predefinito, può essere ottimizzato

    # Scelta del modello ETS (Additivo o Moltiplicativo)
    ets_type = 'add' if decomposition.seasonal.abs().mean() > 0 else 'mul'

    # Definizione e addestramento del modello ETS
    model = ExponentialSmoothing(
        y_train, trend=ets_type, seasonal=ets_type, seasonal_periods=seasonal_period
    ).fit()

    # Previsione ETS
    y_pred = model.forecast(steps=len(y_valid))

    # Calcolo delle metriche di valutazione
    score_mae, score_mse, score_rmse, score_mape = evaluate_model(y_valid, y_pred)

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

    last_date = pd.to_datetime(x_train.iloc[-1, 0])  # Converti l'ultima data in datetime
    forecast_index = pd.date_range(start=last_date, periods=len(y_valid), freq='D')

    # Combina train, valid e forecast per evitare errori
    train_data = pd.DataFrame({"date": x_train.iloc[:, 0], "value": y_train.iloc[:, 0]})
    valid_data = pd.DataFrame({"date": x_valid.iloc[:, 0], "value": y_valid.iloc[:, 0]})
    forecast_data = pd.DataFrame({"date": forecast_index, "value": y_pred})

    # Plot
    plt.figure(figsize=(15, 8))

    sns.lineplot(data=train_data, x="date", y="value", label="Dati storici", color="dodgerblue")
    sns.lineplot(data=valid_data, x="date", y="value", label="Ground Truth", color="#FF7F50")
    sns.lineplot(data=forecast_data, x="date", y="value", label="Previsione ETS", color="darkgreen")

    plt.title("Previsione con ETS", fontsize=14)
    plt.xlabel("Periodo", fontsize=14)
    plt.ylabel("Valore", fontsize=14)
    plt.legend()
    plt.show()

ETS_model(df)