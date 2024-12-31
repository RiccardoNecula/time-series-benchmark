import pandas as pd

#nessun dato mancante nei dataset
"""
def fill_missing_values(data, method='mean'):
    if method == 'mean':
        return data.fillna(data.mean())
    elif method == 'median':
        return data.fillna(data.median())
    elif method == 'ffill':
        return data.fillna(method='ffill')
    elif method == 'bfill':
        return data.fillna(method='bfill')
    else:
        raise ValueError("Metodo sconosciuto")
"""

def normalize(data):
    return (data - data.min()) / (data.max() - data.min())
