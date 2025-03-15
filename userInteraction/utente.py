import os
from datasets import load_nvidia_data, load_rainfall_data, load_maxtemperature_data
from models.models.ARIMA import ARIMA_model
from models.models.Prophet import Prophet_model
from models.models.LSTM import LSTM_model
from models.models.ETS import ETS_model

#selezione dei dataset
def seleziona_dataset():
    #carico i dataset
    nvidia_data = load_nvidia_data()
    rainfall_data = load_rainfall_data()
    temperature_data = load_maxtemperature_data()

    datasets = [
        nvidia_data,
        rainfall_data,
        temperature_data
    ]

    #effettivamente richiedo la scelta all'utente
    while True:
        try:
            sel_dataset = int(input("Seleziona il dataset (1 -> Nvidia, 2 -> Rainfall, 3 -> Temperature): "))
            if 1 <= sel_dataset <= 3:
                return sel_dataset, datasets
            else:
                print("Input invalido. Scegli un numero tra 1 e 3.")
        except ValueError:
            print("Input invalido. Devi inserire un numero.")


def chiedi_visualizzazione_iniziale():
    risposta = input("Desideri la visualizzazione dei dati iniziale (S o N): ").strip().upper()
    while risposta not in ["S", "N"]:
        print("Input invalido. Per favore inserisci 'S' o 'N'.")
        risposta = input("Desideri la visualizzazione dei dati iniziale (S o N): ").strip().upper()
    return risposta


def chiedi_preprocessing():
    risposta = input("Desideri visualizzare il preprocessing del dataset (S o N): ").strip().upper()
    while risposta not in ["S", "N"]:
        print("Input invalido. Per favore inserisci 'S' o 'N'.")
        risposta = input("Desideri visualizzare il preprocessing del dataset (S o N): ").strip().upper()
    return risposta


def chiedi_ADF():
    risposta = input("Desideri visualizzare l'ADF del dataset scelto (S o N): ").strip().upper()
    while risposta not in ["S", "N"]:
        print("Input invalido. Per favore inserisci 'S' o 'N'.")
        risposta = input("Desideri visualizzare l'ADF del dataset scelto (S o N): ").strip().upper()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    adf_nvidia_path = os.path.join(current_dir, "../EDA", "ADF", "ADFCloseNvidia.py")
    adf_rainfall_path = os.path.join(current_dir, "../EDA", "ADF", "ADFRainfall.py")
    adf_maxtemp_path = os.path.join(current_dir, "../EDA", "ADF", "ADFMaxTemp.py")

    ADF = {
        1: adf_nvidia_path,
        2: adf_rainfall_path,
        3: adf_maxtemp_path
    }

    return risposta, ADF


def seleziona_modello(df):
    #ordine dei dataset

    modelli = {
        1 : ARIMA_model,
        2: Prophet_model,
        3: LSTM_model,
        4: ETS_model
    }

    #effettivamente richiedo la scelta all'utente
    while True:
        try:
            sel_modello = int(input("Seleziona il dataset (1 -> ARIMA, 2 -> Prophet, 3 -> LSTM, 4 -> ETS): "))
            if sel_modello in modelli:
                return modelli[sel_modello](df)  # Esegue il modello selezionato e lo ritorna

            else:
                print("Input invalido. Scegli un numero tra 1 e 4.")
        except ValueError:
            print("Input invalido. Devi inserire un numero.")
