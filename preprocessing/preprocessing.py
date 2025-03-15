import pandas as pd
def Preprocessing(df):

    print("\nLe prossime scritture ti daranno delle informazioni iniziali sul dataset scelto.")
    print("\nprimi 5 record:\n", df.head(5))
    print("numero di record: \n", len(df))
    print(df.info)
    print("dimensione dataset: \n", df.shape,"\n")

    if df.columns[0] != "date":
        df.rename(columns={df.columns[0]: 'date'}, inplace=True)
        #print("\n",df.columns[0])

    df['date'] = pd.to_datetime(df['date'])

    #Missing Values

    print("\nVisualizzazione di eventuali valori nulli per colonna:\n", df.isna().sum())

    """
    df = df.sort_values(by='date')
    # Check time intervals
    df['delta'] = df['date'] - df['date'].shift(1)
    print(df[['date', 'delta']].head(10))
    print("\nSe il dataset non è uniforme nelle misurazioni, i due valori seguenti risulteranno diversi:"
          "\n--->", df['delta'].sum(), "|", (df['delta'].count())) #primo valore ha delta NaN
    """