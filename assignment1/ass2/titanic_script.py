# %%
# Imports
import pandas as pd
from scipy.sparse import find
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate, train_test_split
import nltk
import matplotlib.pyplot as plt

from training import train_model

from preprocess import transform_titanic_dataset
from datavis import heatmap, countplots, distplots, countplot
import utils

#%% Part 1, Data Preparation
training_df = pd.read_csv("data/train.csv", sep=",", encoding="utf-8")
test_df = pd.read_csv("data/test.csv", sep=",", encoding="utf-8")

training_df = transform_titanic_dataset(training_df)
test_df = transform_titanic_dataset(test_df)

# %%
# Use describe to get some generic statistics
# training_df.describe()
#
#
# print (training_df.info())

# %%
###
### Visualizations
###
import seaborn as sns
import importlib
#distplots(training_df, ['passenger_fare', 'age', 'nr_siblings_spouses', 'nr_parents_children'])

#countplot(training_df)

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
# grid = sns.PairGrid(interesting_data_df, hue="class").add_legend()
# # grid.map_diag(sns.distplot, hist=False, rug=True)
# # grid.map_offdiag(sns.scatterplot)
# # plt.show()
#heatmap(interesting_data_df)
# high_corr_col = ['class', 'gender', 'passenger_fare']
# countplots(interesting_data_df, high_corr_col)

# %%
###
### Feature engineering and selection
###

# Manually define here the columns we choose <-@tho, hiermee kunnen we het handmatig aanpassen
chosen_columns = ['gender', 'passenger_fare', 'class']

train_X = training_df.pop('survived')

# Let's start with defining the one-hot encoding the categorical variables
oh_encoder = OneHotEncoder()
oh_columns = ['gender', 'class', 'port_of_departure']

# We define numerial operations by a scaling process, where we remove mean and scale to unit variance
# -> Better performance
num_scale_encoder = StandardScaler()
num_scale_columns = ['age', 'nr_siblings_spouses', 'nr_parents_children', 'passenger_fare']

# We ignore for the moment `id`, `name`, `cabin_nr`, `ticket_nr`

# We define a transformer which can apply these encoders to their respective columns, ignoring the rest

# TODO: for both `oh_columns` and `num_scale_columns`, check if the features are in 'chosen_columns'
chosen_oh_cols = list(set(chosen_columns) & set(oh_columns))
chosen_num_cols = list(set(chosen_columns) & set(num_scale_columns))


df_transformer = ColumnTransformer([
    ('oh', oh_encoder, chosen_oh_cols),
    ('num', num_scale_encoder, chosen_num_cols),
], remainder='drop')

# We fit this transformer on our training data, and transform our training data into this new format
encoded_X = df_transformer.fit_transform(training_df)

# As a sanity check, we check what our data looks like right now
new_oh_columns = df_transformer.named_transformers_.oh.get_feature_names(chosen_oh_cols)
encoded_columns = [ *new_oh_columns, *chosen_num_cols]
encoded_df = pd.DataFrame(encoded_X, columns=encoded_columns)
print(encoded_df.head(5))

# %%
###
### Modeling: Train
###
models = [
    DecisionTreeClassifier(),
    RandomForestClassifier()
]

results_df = pd.DataFrame()

for model in models:
    trained_model, results = train_model(model, encoded_X, train_X)
    results_df = results_df.append(results, ignore_index = True)

utils.save_results(results_df, 'results/training_results.csv')

# %%
###
### Modeling: Predict
###

# With our trained model, let's predict the rest

# First we apply the same transformation on our test-data
test_X = df_transformer.transform(test_df)

# Then we predict
predictions = model.predict(test_X)

kaggle_result_df = pd.DataFrame(columns=['Survived'], index=test_df.ID)
kaggle_result_df.Survived = predictions
utils.save_results(kaggle_result_df, 'results/kaggle_result_df.csv')

# %%
