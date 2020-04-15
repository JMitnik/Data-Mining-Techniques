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
from config import Config
import seaborn as sns

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


#%% Part 2 Visualizing cleaned data
categorical_cols = ['programme', 'did_ml', 'did_stats', 'did_db', 'did_ir', 'did_stand', 'gender', 'chocolate'] #todo gd text?
numerical_cols = ['random_nr', 'stress_level', 'nr_neighbours', 'bedtime_yesterday', 'date_of_birth']   #fixed by cleaning data? date_of_birth here? bedtime?

from datavisualization import countplot, histogram, boxplot
#countplot(df[categorical_cols], categorical_cols)
#histogram(df[numerical_cols], numerical_cols)

#possible interesting multivariable scatter and boxplots:
# date_of_birth with stress_level, nr_neighbours, bedtime_yesterday?
# programme with courses followed?:
# date_of_birth with bow

#boxplot(df, categorical_cols)

#%% Part 3 of preprocessing: Make it ready for ML algorithms
working_data = df.copy() # Copy as we will mutate the original dataframe otherwise
target = working_data.pop('programme') #Pop removes programme from



# Creates a pipeline for the input which will transform our input to ndarrays
preprocessing_pipeline_ODI = make_ODI_preprocess_pipeline(
    working_data,
    config.min_word_count
)
target_encoder, target = preprocess_target(target)

X_train, X_test, y_train, y_test = train_test_split(working_data, target, test_size=0.2)

# Instantiate a few classification pipelines for comparisons
algorithms = [
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
]

fitted_pipelines = []

# Try each algorithm out
for algo in algorithms:
    classification_pipeline = make_pipeline(preprocessing_pipeline_ODI, algo)

    # Measure general performance
    avg_cv_score = cross_val_score(classification_pipeline, X_train, y_train).mean()
    print(avg_cv_score)
    # Fit model
    classification_pipeline.fit(X_train, y_train)
    fitted_pipelines.append(classification_pipeline)

# %%
