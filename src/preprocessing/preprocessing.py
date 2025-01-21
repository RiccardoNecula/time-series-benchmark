
#no missing values in the datasets
"""
def fill_missing_values(preprocessing, method='mean'):
    if method == 'mean':
        return preprocessing.fillna(preprocessing.mean())
    elif method == 'median':
        return preprocessing.fillna(preprocessing.median())
    elif method == 'ffill':
        return preprocessing.fillna(method='ffill')
    elif method == 'bfill':
        return preprocessing.fillna(method='bfill')
    else:
        raise ValueError("Metodo sconosciuto")
"""

def normalize(data):
    return (data - data.min()) / (data.max() - data.min())

#if lstm model is used (multivariate)
def standardize(data):
    return (data - data.mean()) / data.std()

