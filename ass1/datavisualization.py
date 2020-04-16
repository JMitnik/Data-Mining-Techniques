import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


## categorical variables ##
def countplot(df, categorical_cols):
    ## remove single valued categories that went through the preprocessing and unknowns
    for column in df:
        value_counts = df[column].value_counts()
        to_remove = value_counts[value_counts <= 1].index
        df[column].replace(to_remove, np.nan, inplace=True)
        df[column].replace(-1, np.nan, inplace=True)
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    for column, subplot in zip(categorical_cols, ax.flatten()):
        sns.countplot(df[column], ax=subplot)
        for label in subplot.get_xticklabels():
            label.set_rotation(90)
    plt.show()
    return None

def histogram(df, numerical_cols):
    for column in df:
        value_counts = df[column].value_counts()
        to_remove = value_counts[value_counts <= 1].index
        df[column].replace(to_remove, np.nan, inplace=True)
        df[column].replace(-1, np.nan, inplace=True)
    df[numerical_cols].hist(bins=10, figsize=(15, 6), layout=(1, 4))
    plt.show()
    return None

def boxplot(df, categorical_cols):
    for column in df:
        value_counts = df[column].value_counts()
        to_remove = value_counts[value_counts <= 1].index
        df[column].replace(to_remove, np.nan, inplace=True)
        df[column].replace(-1, np.nan, inplace=True)

    fig, ax = plt.subplots(2, 4, figsize=(15, 10))
    for var, subplot in zip(categorical_cols, ax.flatten()):
        sns.boxplot(x=var, y='date_of_birth', data=df, ax=subplot)
        for label in subplot.get_xticklabels():
            label.set_rotation(90)
    plt.show()
    return None