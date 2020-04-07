import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import nltk
import numpy as np
from utils import alias_item

def clean_date_of_birth(df):
    date_col = pd.DataFrame(df['date_of_birth'])
    i=0
    for date in date_col.values:
        df.loc[i, 'date_of_birth'] = -1
        date = str(date)
        seperator = ('/', ' ', '-', '.', ',', '_')
        date = re.split("[%s]" % ("".join(seperator)), date)
        for values in date:
            if len(values) == 4 and values.isdigit():
                df.loc[i, 'date_of_birth'] = np.array(values)
        i+=1
    return df

def transform_ODI_dataset(df):
    """
    Transforms existing dataframe by renaming columns, change their code, and imputes missing values.

    Arguments:
        df

    Returns:
        df
    """
    # TODO:
    # - Transform chocolate
    # - Transform deserves_money into numbers, put rest to unknown (-1)
    # - Random_nr (allow only Ints, remove the drop table command)
    # - Stress level

    # New readable column names
    new_columns = [
        'programme',
        'did_ml',
        'did_ir',
        'did_stats',
        'did_db',
        'gender',
        'chocolate',
        'date_of_birth',
        'nr_neighbours',
        'did_stand',
        'stress_level',
        'deserves_money',
        'random_nr',
        'bedtime_yesterday',
        'good_day_text_1',
        'good_day_text_2'
    ]
    df.columns = new_columns

    # Format categoricals first
    # First column has many different aliases, get the most out of them
    programme_alias_map = {
        'ai': ['ai', 'artificial intelligence'],
        'cs': ['computer science', 'cs'],
        'computational_sci': ['computational science'],
        'ba': ['ba', 'business analytics'],
        'bioinformatics': ['bioinformatics'],
        'qrm': ['qrm', 'quantitative risk management'],
        'info_sci': ['information sciences', 'information science'],
        'dbi': ['digital business & innovation', 'digital business and innovation'],
        'ft': ['finance & technology', 'finance and technology']
    }

    df['programme'] = df['programme'].apply(alias_item, args=(programme_alias_map,)).astype('category')
    df['gender'] = df['gender'].astype('category')
    df['chocolate'] = df['chocolate'].astype('category')

    # Format booleans
    df['did_ml'] = df['did_ml'].replace({ 'no': 0, 'yes': 1, 'unknown': -1 })
    df['did_ir'] = df['did_ir'].replace({ 'no': 0, 'yes': 1, 'unknown': -1 })
    df['did_stats'] = df['did_stats'].replace({ 'sigma': 0, 'mu': 1, 'unknown': -1 })
    df['did_db'] = df['did_db'].replace({ 'nee': 0, 'ja': 1, 'unknown': -1 })
    df['did_stand'] = df['did_stand'].replace({ 'no': 0, 'yes': 1, 'unknown': -1 })

    # Format year of birth
    df = clean_date_of_birth(df)

    # Format nr neighbours
    neighbour_alias_map = {
        0: ['none', 'zero', 'right now', 'quarantining'],
        -1: ['can I know this'],
        100: ['>100']
    }

    df['nr_neighbours'] = df['nr_neighbours'].apply(alias_item, args=(neighbour_alias_map,)).astype('int')

    # Tokenize open text
    # TODO: Stem words, remove stop-words
    df['good_day_text_1'] = df['good_day_text_1'].apply(lambda sentence: ' '.join([word.lower() for word in nltk.word_tokenize(sentence)]))
    df['good_day_text_2'] = df['good_day_text_2'].apply(lambda sentence: ' '.join([word.lower() for word in nltk.word_tokenize(sentence)]))

    # TODO: Check for empty values / Np.Nans and such
    return df


def make_preprocessing_pipeline():
    """
    Creates an SKLearn pipeline which converts a transformed
     Pandas dataframe into a classifier-ready numpy tensor.

    Arguments:
        df
    """
    # Define transformations for open text (good_day_text_1 and good_day_text_2)
    count_trans = CountVectorizer()
    tf_idf_trans = TfidfTransformer()
    text_trans = make_pipeline(count_trans, tf_idf_trans)

    # Connect transformations for all columns
    col_trans = make_column_transformer(
        (text_trans, ['good_day_text_1']),
        remainder='drop'
    )

    return col_trans
