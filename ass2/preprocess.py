import pandas as pd
import re
from sklearn.compose import make_column_selector
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import nltk
import numpy as np

def transform_titanic_dataset(df):
    # Original columns ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
    #        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

    new_columns = [
        'ID',
        'survived',
        'class',
        'name',
        'gender',
        'age',
        'nr_siblings_spouses',
        'nr_parents_children',
        'ticket_nr',
        'passenger_fare',
        'cabin_nr',
        'port_of_departure'
    ]
    if len(df.columns) == 11:
        new_columns.remove('survived')

    df.columns = new_columns

    # [1/12] ID: Int stays int
    # Note: Id will be ignored in feature-selection

    # [2/12] Survived: Target label, will be removed

    # [3/12] Class: Becomes categorical
    df['class'] = df['class'].replace({ np.NaN: 'unknown' }).astype('category')

    # [4/12] Name: string stays string
    # Note: name will be ignored in feature-selection
    # TODO: Maybe extract surnames?

    # [5/12] Gender: categorical variable
    df['gender'] = df['gender'].astype('category')

    # [6/12] Age: replace Nans with -1, make int
    df['age'] = df['age'].replace({np.NaN : -1}).astype('int')

    # [7/12] nr_siblings_spouses: int stays int
    # [8/12] nr_parents_children: int stays int

    # [9/12] ticket_nr: string stays string
    # Note: ticket_nr will be ignored in feature-selection
    # TODO: What can we do here?

    # [10/12] passenger_fare: float stays float
    # Note: kept for now, maybe high correlation with other features, we dont need multiple features saying the same thing
    # TODO: Maybe clean this up a bit?

    # [11/12] cabin_nr: split and nans
    df['cabin_nr'] = df['cabin_nr'].str.split(' ').replace({np.NaN : -1})

    # [12/12] Port of departure: categorical variable
    df['port_of_departure'] = df['port_of_departure'].replace({ np.NaN: 'unknown' }).astype('category')

    return(df)
