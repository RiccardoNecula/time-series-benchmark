# testing datasets
import warnings
from datasets import load_nvidia_data, load_rainfall_data, load_romaniaTourism_data
from utils import normalize

warnings.filterwarnings("ignore")

nvidia_data = load_nvidia_data()
rainfall_data = load_rainfall_data()
tourism_data = load_romaniaTourism_data()


#verify if  data is correctly read
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
print("Rou Tourism:", tourism_data.shape)



#normalize(nvidia_data)
#normalize(rainfall_data)
#normalize(tourism_data)