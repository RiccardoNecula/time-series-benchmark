import time

import numpy as np
import warnings

#built
from visualization.initialVis import plot_iniziale
from preprocessing.preprocessing import Preprocessing
from userInteraction.utente import seleziona_dataset, chiedi_visualizzazione_iniziale, chiedi_preprocessing, chiedi_ADF, \
    seleziona_modello
from visualization.timeSeriesSplit import  plot_timeSeriesSplit

warnings.filterwarnings("ignore")
np.random.seed(7)

#main

if __name__ == "__main__":

    #Selezione dataset
    sel_dataset, datasets = seleziona_dataset()
    df = datasets[sel_dataset - 1]

    #Visualizzazione iniziale
    risposta = chiedi_visualizzazione_iniziale()
    if risposta == 'S':
        plot_iniziale(df)

    #Preprocessing
    risposta = chiedi_preprocessing()
    if risposta == 'S':
        Preprocessing(df)

    #visualizzazione di ADF e Time Series Decomposition
    risposta, ADF = chiedi_ADF()
    if risposta == "S":
        with open(ADF[sel_dataset]) as f:
            exec(f.read())



    time.sleep(3)
    plot_timeSeriesSplit(df, sel_dataset)

    seleziona_modello(df)

    #Prophet_model(df)
    #ARIMA_model(df)
    #LSTM_model(df)