#testing datasets

from datasets import load_nvidia_data, load_rainfall_data

nvidia_data = load_nvidia_data()
rainfall_data = load_rainfall_data()

print(nvidia_data.head())
print()
print(rainfall_data.head())
