import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def single_freqtable(df, columnname):
    value_counts = df[columnname].value_counts()
    to_remove = value_counts[value_counts <= 1].index
    df[columnname].replace(to_remove, np.nan, inplace=True)
    value_counts = df[columnname].value_counts()
    plt.figure(figsize=(10, 5))
    sns.barplot(value_counts.index, value_counts.values, alpha=0.8)
    plt.title('Frequency of ' + columnname + ' of students following DMT')
    plt.ylabel('Amount of Students', fontsize=12)
    plt.xlabel(columnname, fontsize=12)
    plt.show()
    return None

def histogram(df, columnname):
    print (df['stress_level'].value_counts())
    plt.hist(df[columnname].hist(bins=10)) #make bins variable?
    plt.xlabel(columnname, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xlim([min(df[columnname]), max(df[columnname])])
    plt.show()
    return None
