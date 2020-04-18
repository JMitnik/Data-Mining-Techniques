# %%
# Imports
import pandas as pd
from scipy.sparse import find
from sklearn.compose import make_column_selector
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
### Feature engineering and selection
###



# %%
