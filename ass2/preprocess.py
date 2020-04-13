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
    # columns ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
    #        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    # todo? ticketnr, name, passenger_fare

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

    df['class'] = df['class'].astype('category')
    df['class'] = df['port_of_departure'].astype('category')
    df['gender'] = df['gender'].replace({'male':0, 'female':1})
    df['age'] = df['age'].replace({np.NaN : -1}).astype('int')
    df['cabin_nr'] = df['cabin_nr'].str.split(' ').replace({np.NaN : -1})

    return(df)
