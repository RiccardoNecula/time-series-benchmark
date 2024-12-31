import pandas as pd

def load_romaniaTourism_data():
    file_path = "datasets/csvdatasets/romaniaTourism.csv"
    data = pd.read_csv(file_path)
    return data
