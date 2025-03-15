from sklearn.model_selection import TimeSeriesSplit
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_timeSeriesSplit(df, numero_dataset):

    if df.columns[0] != "date":
        df.columns.values[0] = 'date'

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

    lista_target = ["Close", "rainfall", "Temperature"]
    window_size = 30
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 12))

    target = df[lista_target[numero_dataset-1]]
    #print(target)

    sns.lineplot(x=df['date'], y=target, ax=ax, color='dodgerblue')
    sns.lineplot(x=df['date'], y=target.rolling(window_size).mean(), ax=ax, color='black', label='media')
    sns.lineplot(x=df['date'], y=target.rolling(window_size).std(), ax=ax, color='#FF7F50', label='deviazione standard')
    ax.set_title(f'{lista_target[numero_dataset-1]}: Non-Stazionarietà \nLa varianza è dipendente dal tempo (seasonality)', fontsize=14)
    ax.set_xlabel(xlabel='Periodo analizzato', fontsize=14)
    ax.set_ylabel(ylabel=lista_target[numero_dataset-1], fontsize=14)
    plt.legend(title=f"rolling window utilizzato = {window_size} gg", loc='upper right', framealpha=0.5)
    plt.tight_layout()
    plt.show() #rolling mean and rolling standard aren't constant but they are close to each other

    N_SPLITS = 3

    X = df['date']
    y = target

    x_min = X.min()
    x_max = X.max()
    y_min = y.min()
    y_max = y.max()

    folds = TimeSeriesSplit(n_splits=N_SPLITS)

    N_SPLITS = 5  # Numero di split
    folds = TimeSeriesSplit(n_splits=N_SPLITS)

    # Creazione di una singola colonna di subplot
    f, ax = plt.subplots(nrows=N_SPLITS, ncols=1, figsize=(15, 8))


    for i, (train_index, valid_index) in enumerate(folds.split(X)):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        # Training set in blu
        sns.lineplot(
            x=X_train,
            y=y_train,
            ax=ax[i],
            color='dodgerblue',
            label='train'
        )

        # Validation set in arancione
        sns.lineplot(
            x=X_valid,
            y=y_valid,
            ax=ax[i],
            color='#FF7F50',
            label='validation'
        )

        # Titolo del grafico
        ax[i].set_title(f"Rolling Window con Divisione e Dimensione dei Dati di Traning Constante (Split {i+1})", fontsize=14)
        ax[i].set_ylabel(lista_target[numero_dataset-1], fontsize=8)
        ax[i].set_ylim([y_min, y_max])
        ax[i].set_xlim([x_min, x_max])

    plt.tight_layout()
    plt.show()