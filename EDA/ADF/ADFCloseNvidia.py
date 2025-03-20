import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from datasets import load_nvidia_data
from statsmodels.tsa.stattools import adfuller

# funzione per il Test di stazionarietà - Dickey-Fuller aumentato
def test_stationarity(timeseries):
    result = adfuller(timeseries)
    print('Statistiche ADF:', result[0])
    print('p-value:', result[1])
    print('Valori Critici:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')

# Caricamento dei dati
nvidia_data = load_nvidia_data()

# Applicazione della trasformazione logaritmica a Volume per stabilizzare la varianza
nvidia_data["Date"] = pd.to_datetime(nvidia_data["Date"])
nvidia_data.set_index("Date", inplace=True)

# Visualizzazione della serie temporale
plt.figure(figsize=(15, 8))
plt.plot(nvidia_data['Close'], label='Valore Azione', color='dodgerblue')
plt.title("Valore della singlola Azione NVIDIA in Chiusura di Borsa")
plt.xlabel("Periodo")
plt.ylabel("Valore in Chiusura ($)")
plt.legend(loc='upper left')
plt.show()


# Decomposizione della serie temporale
decomposition = seasonal_decompose(nvidia_data['Adj Close'], model='additive', period=365)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Visualizzazione della decomposizione
plt.figure(figsize=(15, 8))
plt.subplot(411)
plt.plot(nvidia_data['Close'], label='Originale', color='dodgerblue')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend', color='dodgerblue')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Stagionalità', color='dodgerblue')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuo', color='dodgerblue')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Creazione della serie adjusted rimuovendo la stagionalità
adjusted_nvidia = nvidia_data['Close'] - seasonal

# Visualizzazione della serie depurata dalla stagionalità
plt.figure(figsize=(15, 8))
plt.plot(adjusted_nvidia, label='Valore Azione Adjusted', color='#FF7F50') #corallo
plt.title('Valore della singola Azione NVIDIA in Chiusura di Borsa senza Componente Stagionale')
plt.xlabel('Periodo')
plt.ylabel('Valore in Chiusura ($)')
plt.legend()
plt.show()



print("Test di Stazionarietà del dataset Nvidia:")
test_stationarity(nvidia_data['Close'])

# Apply log transformation (to stabilize variance)
nvidia_data['log_Close'] = np.log(nvidia_data['Close'])

# Visualizzazione post trasformazione logaritmica
plt.figure(figsize=(15,8))
plt.plot(nvidia_data['log_Close'], color='darkgreen')
plt.title("Valore della singola Azione in Chiusura di Borsa - dopo trasformazione logaritmica\n(dataset non stazionario)")
plt.xlabel('Periodo')
plt.ylabel('Valore in Chiusura trasformato ($)')
plt.show()