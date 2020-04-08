# %%
# Imports
import pandas as pd
from scipy.sparse import find
from sklearn.compose import make_column_selector
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split
import nltk
from config import Config

# Script for notebooks to reload dependency
import importlib
import preprocessing
importlib.reload(preprocessing)
from preprocessing import transform_ODI_dataset, make_ODI_preprocess_pipeline, preprocess_target

# Config
config = Config(
    min_word_count=4,
    random_state=0
)

#%% Part 1 of preprocessing: reading data and cleaning it

# Read ODI dataset and transform it, clean it, etc
df = pd.read_csv("data/ODI-2020.csv", sep=";", encoding="utf-8")
df = transform_ODI_dataset(df)

#%% Part 2 of preprocessing: Make it ready for ML algorithms
working_data = df.copy() # Copy as we will mutate the original dataframe otherwise
target = working_data.pop('programme') #Pop removes programme from

# Creates a pipeline for the input which will transform our input to ndarrays
preprocessing_pipeline_ODI = make_ODI_preprocess_pipeline(
    working_data,
    config.min_word_count
)
target_encoder, target = preprocess_target(target)

X_train, X_test, y_train, y_test = train_test_split(working_data, target, test_size=0.2)

# Create a few classification pipelines for comparisons
classifier_svc = make_pipeline(preprocessing_pipeline_ODI, SVC())
classifier_tree = make_pipeline(preprocessing_pipeline_ODI, DecisionTreeClassifier())

# Apply cross-validation to get representation of score
svc_performance = cross_val_score(classifier_svc, X_train, y_train).mean()
tree_performance = cross_val_score(classifier_tree, X_train, y_train).mean()

# Now actually do training on complete training data
classifier_svc.fit(X_train, y_train)
classifier_tree.fit(X_train, y_train)

# %%
# Check feature importances of decision tree
classifier_tree.named_steps.columntransformer.transformers[-1][1].get_feature_names()
# classifier_tree.named_steps.decisiontreeclassifier.n_features_
# %%
