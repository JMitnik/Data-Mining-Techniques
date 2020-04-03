import pandas as pd
from utils import alias_item

def transform_ODI_dataset(df):
    # Transform column names
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

    # Format first column programme
    # First column has many different aliases
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

    # Format booleans
    df['did_ml'] = df['did_ml'].replace({ 'no': 0, 'yes': 1, 'unknown': -1 })
    df['did_ir'] = df['did_ir'].replace({ 'no': 0, 'yes': 1, 'unknown': -1 })
    df['did_stats'] = df['did_stats'].replace({ 'sigma': 0, 'mu': 1, 'unknown': -1 })
    df['did_db'] = df['did_db'].replace({ 'nee': 0, 'ja': 1, 'unknown': -1 })
    df['did_stand'] = df['did_stand'].replace({ 'no': 0, 'yes': 1, 'unknown': -1 })

    # Format categoricals
    df['gender'] = df['gender'].astype('category')
    df['chocolate'] = df['chocolate'].astype('category')

    # Format nr neighbours
    neighbour_alias_map = {
        0: ['none', 'zero', 'right now', 'quarantining'],
        -1: ['can I know this'],
        100: ['>100']
    }

    df['nr_neighbours'] = df['nr_neighbours'].apply(alias_item, args=(neighbour_alias_map,)).astype('int')

    return df
