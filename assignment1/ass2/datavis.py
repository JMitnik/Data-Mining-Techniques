import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def countplot(df):
    ## remove single valued categories that went through the preprocessing and unknowns
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    for column, subplot in zip(['gender', 'class', 'port_of_departure'], ax.flatten()):
        sns.countplot(df[column], ax=subplot)
        for label in subplot.get_xticklabels():
            label.set_rotation(90)
    plt.suptitle('Overview of categorical columns of the Titanic dataset')
    plt.show()
    return None

def heatmap(df):

    df.port_of_departure.replace(('S', 'C', 'Q'), (0, 1, 2), inplace=True)
    df.gender.replace(('male', 'female'), (0, 1), inplace=True)
    df['class'].replace((1, 2, 3), (0, 1, 2), inplace=True )
    correlation_matrix = df.corr()
    sns.heatmap(data=correlation_matrix, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.2f', cmap='coolwarm')
    plt.show()
    return None

def countplots(df, cols):
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    df['survived'] = df['survived'].map({
        0: 'Died',
        1: 'Survived'
    })
    for column, subplot in zip(cols, ax.flatten()):
        if column == 'passenger_fare':
            interval = (0, 20, 40, 60, 80, 100)
            categories = ['0-20', '20-40' '40-60', '60-80', '80-100', '100+']
            df['passenger_fare_categories'] = pd.cut(df.passenger_fare, interval, labels=categories)
            sns.countplot(x='passenger_fare_categories', data=df, hue='survived', palette="pastel", ax=subplot)
        else:
            sns.countplot(x=column, hue=df['survived'], palette="pastel", data=df, ax=subplot)

    plt.suptitle('Frequency plots of the Titanic survived/passed entries for 3 high correlated categories')
    plt.show()
    return None

def distplots(df, cols):
    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    for column, subplot in zip(cols, ax.flatten()):
        sns.distplot(df[column], ax=subplot, bins=20, color='red')
    plt.suptitle(
        'Overview of numerical columns of the Titanic dataset')
    plt.show()
    return None

