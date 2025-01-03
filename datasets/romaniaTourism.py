"""
#old version, works for main
import pandas as pd

def load_romaniaTourism_data():
    file_path = "datasets/csvdatasets/romaniaTourism.csv"
    data = pd.read_csv(file_path)
    return data
"""

import os
import pandas as pd

#new one, works for model as well
def load_romaniaTourism_data():
    #creates absolute or relative path
    base_dir = os.path.dirname(os.path.abspath(__file__))  #gives current directory
    file_path = os.path.join(base_dir, "csvdatasets", "romaniaTourism.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Il file {file_path} non esiste. Controlla il percorso o copia il file nella posizione corretta.")

    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise RuntimeError(f"Errore durante la lettura del file CSV: {e}")