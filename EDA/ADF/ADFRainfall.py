import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from datasets import load_rainfall_data
import matplotlib.pyplot as plt
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
rainfall_data = load_rainfall_data()
rainfall_data["date"] = pd.to_datetime(rainfall_data["date"])
rainfall_data.set_index("date", inplace=True)

#istogramma frequenza
plt.figure(figsize=(15, 8))
sns.histplot(rainfall_data['rainfall'], bins=30, kde=True, color='dodgerblue')
plt.title("Distribuzione delle Precipitazioni Giornaliere", fontsize=14)
plt.xlabel("Precipitazioni (mm)", fontsize=12)
plt.ylabel("Frequenza", fontsize=12)
plt.show()


# Visualizzazione della serie temporale
plt.figure(figsize=(15, 8))
plt.plot(rainfall_data['rainfall'], label='Precipitazioni')
plt.title("Precipitazioni nel Tempo")
plt.xlabel("Periodo")
plt.ylabel("Precipitazioni (mm)")
plt.legend(loc='upper right')
plt.show()


# Decomposizione della serie temporale
decomposition = seasonal_decompose(rainfall_data['rainfall'], model='additive', period=30)  # Periodo di 1 mese
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Visualizzazione della decomposizione
plt.figure(figsize=(15, 8))
plt.subplot(411)
plt.plot(rainfall_data['rainfall'], label='Originale', color='dodgerblue')
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
adjusted_rainfall = rainfall_data['rainfall'] - seasonal

# Visualizzazione della serie depurata dalla stagionalità
plt.figure(figsize=(15, 8))
plt.plot(adjusted_rainfall, label='Precipitazioni Adjusted', color='#FF7F50') #corallo
plt.title('Precipitazioni senza componente stagionale')
plt.xlabel('Data')
plt.ylabel('Precipitazioni (mm)')
plt.legend()
plt.show()


# Test di stazionarietà - Dickey-Fuller aumentato
print("Test di Stazionarietà del dataset rainfall:")
test_stationarity(rainfall_data['rainfall'])