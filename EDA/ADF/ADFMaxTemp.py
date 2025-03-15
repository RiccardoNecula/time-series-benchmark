import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from datasets import load_maxtemperature_data
import seaborn as sns
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
maxtemperature_data = load_maxtemperature_data()
maxtemperature_data["Date"] = pd.to_datetime(maxtemperature_data["Date"])
maxtemperature_data.set_index("Date", inplace=True)

#istogramma frequenza
plt.figure(figsize=(15, 8))
sns.histplot(maxtemperature_data['Temperature'], bins=30, kde=True, color='dodgerblue')
plt.title("Distribuzione delle Temperature Giornaliere", fontsize=14)
plt.xlabel("Temperatura (°C)", fontsize=12)
plt.ylabel("Frequenza", fontsize=12)
plt.show()


# Visualizzazione della serie temporale
plt.figure(figsize=(15, 8))
plt.plot(maxtemperature_data['Temperature'], label='Temperatura Originale', color='dodgerblue')
plt.title("Temperature nel Tempo")
plt.xlabel("Periodo")
plt.ylabel("Temperatura (°C)")
plt.legend()
plt.show()


# Decomposizione della serie temporale
decomposition = seasonal_decompose(maxtemperature_data['Temperature'], model='additive', period=365)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Visualizzazione della decomposizione
plt.figure(figsize=(15, 8))
plt.subplot(411)
plt.plot(maxtemperature_data['Temperature'], label='Originale', color='dodgerblue')
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
adjusted_temperature = maxtemperature_data['Temperature'] - seasonal

# Visualizzazione della serie depurata dalla stagionalità
plt.figure(figsize=(15, 8))
plt.plot(adjusted_temperature, label='Temperatura Adjusted', color='#FF7F50') #corallo
plt.title('Temperatura senza Componente Stagionale')
plt.xlabel('Periodo')
plt.ylabel('Temperatura (° C)')
plt.legend()
plt.show()


print("Test di Stazionarietà:")
test_stationarity(maxtemperature_data['Temperature'])