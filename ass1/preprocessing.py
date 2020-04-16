import pandas as pd
import re
from sklearn.compose import make_column_selector
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.utils import safe_sqr
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
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
                values = int(values)
                if values > 1920 and values < 2004:
                    df.loc[i, 'date_of_birth'] = values
        i+=1
    return df.dropna()

def transform_ODI_dataset(df, programme_threshold=5):
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
    # - Combine BoW counts for good_day_text_1/2
    # - Bedtime :(

    # New readable column names
    new_columns = [
        'programme',        #categorical value; done
        'did_ml',           #categorical value; Yes, No, Unknown
        'did_ir',           #categorical value; Yes, No, Unknown
        'did_stats',        ##categorical value; Yes, No, Unknown
        'did_db',           #categorical value; Yes, No, Unknown
        'gender',           #categorical value; Male, Female, Unknown
        'chocolate',        #categorical value; Slim, Fat, Neither, No Idea, Unknown
        'date_of_birth',    #numerical value; 1920-2004 / done
        'nr_neighbours',    #numerical value; 0-10
        'did_stand',        #categorical value; Yes, No, Unknown
        'stress_level',     #numerical value; 0-100
        'deserves_money',   #numerical value; 0-100
        'random_nr',        #numerical value; just a random int
        'bedtime_yesterday',#numerical value; HH:MM values?
        'good_day_text_1',  #categorical value;
        'good_day_text_2'   #categorical value;
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

    df['programme'] = df['programme'].apply(alias_item, args=(programme_alias_map,))

    if programme_threshold is not None:
        df = df.groupby('programme').filter(lambda x: len(x) > programme_threshold).dropna()
        df['programme'] = df['programme'].astype('category')

    # - Random_nr (allow only Ints, remove the drop table command)
    df['random_nr']=df['random_nr'].str.replace('four','1')
    df['random_nr']=df['random_nr'].str.replace('nine','1')
    df['random_nr']=df['random_nr'].str.replace('1E-36','1')
    df['random_nr']=df['random_nr'].str.replace('a random number','1')
    df['random_nr']=df['random_nr'].str.replace('; DROP ALL TABLES ;','')
    df['random_nr']=df['random_nr'].str.replace(',\d','1',regex=True)
    prevent_overflow(df['random_nr'])
    df['random_nr']=df['random_nr'].astype(np.int64)
    # - Transforms deserves_money into numbers, put rest to unknown (-1)
    df['deserves_money']=df['deserves_money'].replace({
        '-':-1,
        'the amount of money you want to distribute/the number of people':-1,
        'Depends on the number of people it is divided between. When there are 10 people - say 15 euros':15,
        '100/470':0.212,
        '100/N':0.356,
        '100/num of students':0.356,
        '1/470':0.00212,
        'Based on 500 students: €0,20':0.2,
        '(1/participents)*total amount given':-1,
        '10 euro de neus':10,
        'Less':-1,
        'Depends on the amount of participants ':-1,
        'a fiver':5,
        'all the money you have':-1,
        'amount of money / number of people':-1,
        '100/n':0.356,
        '100/#students':0.356,
        'equally divided':-1,
        'i deserve the dopamine that ill get solving the assignments. I dont want to use money as a metric. If you really need a number have the number 29.':29,
        'Everyone 1 euro':1,
        '100 euros + 10 more euros for good luck.':110,
        '💯':100,
        '100/participants - 10%':0.32,
        'Very much.':-1,
        'I do not deserve them.':-1,
        '2 euros for being present hehe':2,
        'Less than 100':-1,
        'Not at all, you should win it':-1,
        '1/500':0.002,
        '0%  100%':-1
    })

    df['deserves_money']=df['deserves_money'].str.replace(',','.')
    df['deserves_money']=df['deserves_money'].str.replace('?','-1')
    df['deserves_money']=df['deserves_money'].str.replace('euros','')
    df['deserves_money']=df['deserves_money'].str.replace('euro','')
    df['deserves_money']=df['deserves_money'].str.replace('\u20AC','',regex=True)
    df['deserves_money']=df['deserves_money'].str.replace('%','',regex=True)



    # Format booleans
    df['did_ml'] = df['did_ml'].replace({ 'no': 0, 'yes': 1, 'unknown': -1 })
    df['did_stats'] = df['did_stats'].replace({ 'sigma': 0, 'mu': 1, 'unknown': -1 })
    df['did_db'] = df['did_db'].replace({ 'nee': 0, 'ja': 1, 'unknown': -1 })
    df['did_stand'] = df['did_stand'].replace({ 'no': 0, 'yes': 1, 'unknown': -1 })
    df['gender'] = df['gender'].replace({'male':0, 'female':1,'unknown':-1})


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


def make_encoding_pipeline(
    data,
    text_min_count
):
    """
    Creates an SKLearn pipeline which converts a transformed
     Pandas dataframe into a classifier-ready numpy tensor.

    Arguments:
        df
    """
    # Check which one-hot columns are used as predictors
    default_oh_columns = [
        'programme',
        'gender',
        'did_stand',
        'did_ml',
        'did_ir',
        'did_db',
        'did_stats'
    ]
    present_oh_columns = [column for column in default_oh_columns if column in data.columns]

    # Make BOW, include only words with at least `text_min_count`
    bow_encoder = CountVectorizer(min_df=text_min_count)

    # One-hot-encoder for categorical and boolean features
    oh_encoder = OneHotEncoder(handle_unknown='ignore')

    # Make feature-engineering transformer
    col_encoders = make_column_transformer(
        (bow_encoder, 'good_day_text_1'),
        (bow_encoder, 'good_day_text_2'),
        (oh_encoder, [
            *present_oh_columns
            # 'did_ir', # TODO: Something weird with did_ir going on
        ]),
        remainder='drop'
    )

    return col_encoders


def prevent_overflow(df):
    j=-1
    for i in df.map(len):
        j += 1
        if i > 15:
            df.iloc[j] = -1
    return df

def preprocess_target(target):
    # Assume for now target is oh encoded.
    oh_encoder = LabelEncoder()
    target = oh_encoder.fit_transform(target)

    return oh_encoder, target


def read_selected_features_from_pipeline(classification_pipeline, is_sorted=True):
    """
    Given a classification pipeline, sort all of the features
    from the 'selector', as well as return the selection
    of features

    Arguments:
        classification_pipeline
    """
    rfe_step = classification_pipeline.named_steps.selection.named_steps.rfe

    # Get selected features
    sorted_idxs = np.argsort(safe_sqr(rfe_step.estimator_.coef_).sum(axis=0))
    mask = rfe_step.support_
    selected_features = np.array(read_all_features_from_pipeline(classification_pipeline))[mask]

    if not is_sorted:
        return selected_features

    # Sort selected features from bottom (worst) to highest (best)
    return selected_features[sorted_idxs]

def read_all_features_from_pipeline(classification_pipeline):
    """
    Given the classification pipeline, reads all features from the engineering part

    Arguments:
        classification_pipeline
    """

    return classification_pipeline.named_steps.engineering.get_feature_names()
