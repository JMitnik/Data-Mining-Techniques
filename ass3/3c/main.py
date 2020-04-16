# %%
from io import StringIO
import html
import pandas as pd

df = pd.read_csv(
    'data/SmsCollection.csv',
    sep=';',
    usecols=range(2),
    names=['label', 'text']
)[1:]


# %%
df

# %%
