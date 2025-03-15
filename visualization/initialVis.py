import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#used to visualize data graphs

def plot_iniziale(df):
    if df.columns[0] != "date":
        df.columns.values[0] = 'date'
        #print(df.columns[0])

    date_column = df.columns[0]  # First column (date)
    value_column = df.columns[1]  # Second column (rainfall)

    x_max = df["date"].max()
    x_min = df["date"].min()

    if len(df.columns) == 2:  # data + target

        plt.figure(figsize=(15, 8))

        ax = sns.lineplot(x=df[date_column], y=df[value_column], color='dodgerblue')

        ax.set_xlim([x_min, x_max])

        plt.title(f'Feature target: {value_column}', fontsize=14)
        plt.xlabel(date_column, fontsize=14)
        plt.ylabel(value_column, fontsize=14)

        step = len(df['date']) // 10
        plt.xticks(ticks=np.arange(1, len(df['date']), step))
        plt.show()

    else:

        f, ax = plt.subplots(nrows=(len(df.columns) - 1), ncols=1, figsize=(15, 25))

        df['date'] = pd.to_datetime(df['date'])

        for i, column in enumerate(df.drop('date', axis=1).columns):
            sns.lineplot(x=df['date'], y=df[column], ax=ax[i], color='dodgerblue')
            ax[i].set_title(f'Feature: {column}', fontsize=14)
            ax[i].set_ylabel(ylabel=column, fontsize=14)

            ax[i].set_xticks([])

    plt.show()