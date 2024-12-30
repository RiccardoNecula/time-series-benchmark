import pandas as pd

def load_nvidia_data():
    file_path = "datasets/csvdatasets/nvidia.csv"
    data = pd.read_csv(file_path)
    return data
