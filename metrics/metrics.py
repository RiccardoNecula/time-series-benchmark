from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import math
import numpy as np

# Calcolo del MAPE con gestione dei valori uguali a zero o vicini a zero per rainfall
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    non_zero_idx = y_true != 0  # Filtra solo valori dove y_true != 0
    return 100 * np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx]))


def evaluate_model(labels, predictions):

    # Calcolo degli errori
    mae = mean_absolute_error(labels, predictions)  # Mean Absolute Error
    mse = mean_squared_error(labels, predictions)  # Mean Squared Error
    rmse = math.sqrt(mse)  # Root Mean Squared Error
    mape = calculate_mape(labels, predictions)  # Mean Absolute Percentage Error (MAPE)

    return mae, mse, rmse, mape


def seleziona_metriche():

    metriche = {
        0: "MAE",
        1: "MAE",
        2: "MSE",
        3: "RMSE",
        4: "MAPE",
        5: "ALL"
    }

    while True:
        try:
            metrica_sel = int(input("Seleziona la metrica (0-> Predefinita, 1 -> MAE, 2 -> MSE, 3 -> RMSE, 4-> MAPE, 5-> TUTTE): "))
            if metrica_sel in metriche:
                return metrica_sel

            else:
                print("Input invalido. Scegli un numero tra 0 e 5.")
        except ValueError:
            print("Input invalido. Devi inserire un numero.")