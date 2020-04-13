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


from preprocess import transform_titanic_dataset

#%% Part 1, Data Preparation
training_df = pd.read_csv("data/train.csv", sep=",", encoding="utf-8")
test_df = pd.read_csv("data/test.csv", sep=",", encoding="utf-8")

training_df = transform_titanic_dataset(training_df)
test_df = transform_titanic_dataset(test_df)

print (test_df['cabin_nr'].value_counts())


