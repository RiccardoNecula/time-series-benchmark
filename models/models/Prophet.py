import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error

from metrics.metrics import evaluate_model, seleziona_metriche
from hparams.hparams import Prophet_grid_params

def Prophet_model(df):

    #utente seleziona la metrica che gli interessa
    metrica_sel = seleziona_metriche()

    if df.columns[0] != "date":
        df.columns.values[0] = 'date'

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

    train_size = int(0.85 * len(df))
    test_size = len(df) - train_size

    # UNIVARIATE CASE
    if len(df.columns) == 2:
        target = df.columns[1]
        univariate_df = df[['date', target]].copy()
        univariate_df.columns = ['ds', 'y']

        train = univariate_df.iloc[:train_size, :]
        valid = univariate_df.iloc[train_size:, :]

    # MULTIVARIATE CASE
    else:
        feature_columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
        target_column = ['Close']

        multivariate_df = df[['date'] + target_column + feature_columns].copy()
        multivariate_df.columns = ['ds', 'y'] + feature_columns

        train = multivariate_df.iloc[:train_size, :]
        valid = multivariate_df.iloc[train_size:, :]


    grid = ParameterGrid(Prophet_grid_params)
    best_model = None
    best_mae = float("inf")
    best_params = None

    for params in grid:
        model = Prophet(**params)

        # Aggiungere regressori se è un problema multivariato
        if len(df.columns) > 2:
            for feature in feature_columns:
                model.add_regressor(feature)

        # Fit del modello
        model.fit(train)

        # Previsione usando il validation set**
        forecast = model.predict(valid)

        # Calcolo errore (MAE)
        y_pred = forecast['yhat'].values
        y_true = valid['y'].values
        mae = mean_absolute_error(y_true, y_pred)

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_params = params

    print(f"\nMigliori iperparametri trovati: {best_params}")
    #print(f"MAE migliore: {best_mae:.4f}")

    # Ricalcoliamo la previsione finale con il miglior modello**
    y_pred = best_model.predict(valid)

    # Calcoliamo tutte le metriche**
    score_mae, score_mse, score_rmse, score_mape = evaluate_model(valid['y'], y_pred.tail(test_size)["yhat"])

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

    # Grafico delle previsioni**
    f, ax = plt.subplots(1, figsize=(15, 6))
    best_model.plot(y_pred, ax=ax)
    sns.lineplot(x=valid['ds'], y=valid['y'], ax=ax, color='#FF7F50', label='Ground truth')
    best_model.plot(y_pred, ax=ax)

    ax.set_title('Prophet Prediction', fontsize=14)
    ax.set_xlabel('Periodo', fontsize=14)
    ax.set_ylabel('Valore Predetto', fontsize=14)
    plt.show()
