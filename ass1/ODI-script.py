# %%
# Imports
import pandas as pd
from scipy.sparse import find
from sklearn.compose import make_column_selector
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
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
inp = df.copy() # Copy as we will mutate the original dataframe otherwise
target = inp.pop('programme') #Pop removes programme from

# Creates a pipeline for the input which will transform our input to ndarrays
preprocessing_pipeline_ODI = make_ODI_preprocess_pipeline(
    inp,
    config.min_word_count
)

target_encoder, target = preprocess_target(target)

# Add to the preprocessing step a classification, and make this our final pipeline
svc_model = SVC()
classifier_svc = make_pipeline(preprocessing_pipeline_ODI, model)

# Apply cross-validation to apply classifier to input and target
performances = cross_val_score(classifier, inp, target)

# %%
