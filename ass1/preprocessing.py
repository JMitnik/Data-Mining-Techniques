import pandas as pd
from utils import alias_item

def transform_ODI_dataset(df):
    # TODO:
    # - Transform deserves_money into numbers, put rest to unknown (-1)
    # - Extract fun text content from good_day_text_1
    # - Extract fun text content from good_day_text_2
    # - Random_nr (allow only Ints, remove the drop table command)
    # - Add date_of_birth from @Thomasdegier code

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

    # Format nr neighbours
    neighbour_alias_map = {
        0: ['none', 'zero', 'right now', 'quarantining'],
        -1: ['can I know this'],
        100: ['>100']
    }

    df['nr_neighbours'] = df['nr_neighbours'].apply(alias_item, args=(neighbour_alias_map,)).astype('int')

    return df
