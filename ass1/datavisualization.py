import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_preprocess(df):
    for column in df:
        value_counts = df[column].value_counts()
        to_remove = value_counts[value_counts <= 1].index
        df[column].replace(to_remove, np.nan, inplace=True)
        df[column].replace(-1, np.NaN, inplace=True)
        df[column].replace('unknown', 'other', inplace=True)
    return df

## categorical variables ##
def countplot(df, categorical_cols):
    ## remove single valued categories that went through the preprocessing and unknowns
    df = plot_preprocess(df)
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    for column, subplot in zip(categorical_cols, ax.flatten()):
        sns.countplot(df[column], ax=subplot)
        for label in subplot.get_xticklabels():
            label.set_rotation(90)
    plt.show()
    return None

def histogram(df, numerical_cols):
    df = plot_preprocess(df)
    df[numerical_cols].hist(bins=10, figsize=(15, 6), layout=(1, 4))
    plt.show()
    return None

def boxplot(df, categorical_cols):
    df = plot_preprocess(df)
    fig, ax = plt.subplots(2, 4, figsize=(15, 10))
    for var, subplot in zip(categorical_cols, ax.flatten()):
        sns.boxplot(x=var, y='date_of_birth', data=df, ax=subplot)
        for label in subplot.get_xticklabels():
            label.set_rotation(90)
    plt.show()
    return None

def heatmap(df):
    df = plot_preprocess(df)
    sns.set(style="white")
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0, square=True,
                linewidth=.5, cbar_kws={'shrink': .5})
    ax.set_title('Multi-Collinearity of Features')
    plt.show()
    #plt.savefig('correlation2.png')
    return None

# ['programme', 'did_ml', 'did_stats', 'did_db', 'did_ir', 'did_stand', 'gender']
def heatmap2(df):
    df.programme.replace(('cs', 'ai', 'other', 'computational_sci', 'qrm', 'ba', 'dbi'), (0, 1, 2, 3, 4, 5, 6), inplace=True)
    df.did_ml.replace((-1, 0, 1, 'other'), (0, 1, 2, 3), inplace=True)
    df.did_stats.replace((-1, 0, 1, 'other'), (0, 1, 2, 3), inplace=True)
    df.did_db.replace((-1, 0, 1, 'other'), (0, 1, 2, 3), inplace=True)
    df.did_ir.replace((-1, 0, 1, 'other'), (0, 1, 2, 3), inplace=True)
    df.did_stand.replace((-1, 0, 1, 'other'), (0, 1, 2, 3), inplace=True)
    df.gender.replace((-1, 0, 1, 'other'), (0, 1, 2, 3), inplace=True)
    print (df.info())
    correlation_matrix = df.corr()
    print (correlation_matrix)
    sns.heatmap(data=correlation_matrix, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.2f', cmap='coolwarm')
    plt.show()
    return None

def stacked_bars(df):
    df = plot_preprocess(df)
    fig, ax = plt.subplots(1, 4, figsize=(15, 10))
    for var, subplot in zip(['did_ml', 'did_stats', 'did_ir', 'did_db'], ax.flatten()):
        sns.countplot(x="programme", hue=var, palette="pastel", edgecolor=".6", data=df, ax=subplot)
        for label in subplot.get_xticklabels():
            label.set_rotation(90)
    plt.show()
    return None
