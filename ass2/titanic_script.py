# %%
# Imports
import pandas as pd
from scipy.sparse import find
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split
import nltk
import matplotlib.pyplot as plt


from preprocess import transform_titanic_dataset

#%% Part 1, Data Preparation
training_df = pd.read_csv("data/train.csv", sep=",", encoding="utf-8")
test_df = pd.read_csv("data/test.csv", sep=",", encoding="utf-8")

training_df = transform_titanic_dataset(training_df)
test_df = transform_titanic_dataset(test_df)

print (training_df.info())
# %%
# Use describe to get some generic statistics
training_df.describe()



# %%
###
### Visualizations
###
import seaborn as sns
import importlib

# Let's select some interesting data we might want to visualize in a grid
interesting_data_df = training_df[[
    'class',
    'port_of_departure',
    'gender',
    'survived',
    'age',
    'nr_siblings_spouses',
    'passenger_fare',
    'nr_parents_children'
]]
grid = sns.PairGrid(interesting_data_df, hue="class").add_legend()
grid.map_diag(sns.distplot, hist=False, rug=True)
grid.map_offdiag(sns.scatterplot)
plt.show()

# %%
###
### Feature engineering
###
# Let's start with defining the one-hot encoding the categorical variables
oh_encoder = OneHotEncoder()
oh_columnns = ['gender', 'class', 'port_of_departure']

# We define numerial operations by a scaling process, where we remove mean and scale to unit variance
# -> Better performance
# TODO: Check if these features look normally distributed, maybe in plots, and remove the ones not
num_scale_encoder = StandardScaler()
num_scale_columns = ['age', 'nr_siblings_spouses', 'nr_parents_children', 'passenger_fare']

# We ignore for the moment `id`, `name`, `cabin_nr`, `ticket_nr`

# We define a transformer which can apply these encoders to their respective columns, ignoring the rest
df_transformer = ColumnTransformer([
    ('oh', oh_encoder, oh_columnns),
    ('num', num_scale_encoder, num_scale_columns),
], remainder='drop')

# We fit this transformer on our training data, and transform our training data into this new format
encoded_X = df_transformer.fit_transform(training_df)

# As a sanity check, we check what our data looks like right now
new_oh_columns = df_transformer.named_transformers_.oh.get_feature_names(oh_columnns)
encoded_columns = [ *new_oh_columns, *num_scale_columns]
encoded_df = pd.DataFrame(encoded_X, columns=encoded_columns)
print(encoded_df.head(5))
# %%
###
### Feature selection
###

pd.DataFrame()
