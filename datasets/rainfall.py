import pandas as pd

def load_rainfall_data():
    file_path = "datasets/csvdatasets/rainfall.csv"
    data = pd.read_csv(file_path)
    return data
