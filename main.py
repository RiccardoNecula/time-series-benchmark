#testing datasets
import warnings
import pandas as pd
from datasets import load_nvidia_data, load_rainfall_data, load_romaniaTourism_data
from utils import normalize

warnings.filterwarnings("ignore")

nvidia_data = load_nvidia_data()
rainfall_data = load_rainfall_data()
tourism_data = load_romaniaTourism_data()


#verify if data is correctly read
print(nvidia_data.head(), "\n")
print(rainfall_data.head(), "\n")
print(tourism_data.head(), "\n")

#general data info
nvidia_data.info()
print("\n------------\n")
rainfall_data.info()
print("\n------------\n")
tourism_data.info()

#data dim
print("\nNvida:", nvidia_data.shape)
print("Rainfall:", rainfall_data.shape)
print("Rou Tourism:", tourism_data.shape, "\n")

#Separate dates for future plotting because multivariate
train_dates = pd.to_datetime(nvidia_data['Date'])
print(train_dates.tail(15)) #Check last few dates.

#normalize(nvidia_data)
#normalize(rainfall_data)
#normalize(tourism_data)