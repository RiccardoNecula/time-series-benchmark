import pandas as pd

#no missing values in the datasets
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

#if lstm model is used (multivariate)
def standardize(data):
    return (data - data.mean()) / data.std()
