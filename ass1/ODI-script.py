# %%
# Imports
import pandas as pd
from scipy.sparse import find
from sklearn.compose import make_column_selector
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Lasso
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split
import nltk
from config import Config

# Script for notebooks to reload dependency
import importlib
import preprocessing
importlib.reload(preprocessing)
from preprocessing import transform_ODI_dataset, make_ODI_preprocess_pipeline, preprocess_target

# Configgg
config = Config(
    min_word_count=4,
    random_state=0
)

#%% Part 1 of preprocessing: reading data and cleaning it

# Read ODI dataset and transform it, clean it, etc
df = pd.read_csv("data/ODI-2020.csv", sep=";", encoding="utf-8")
df = transform_ODI_dataset(df)

#%% Part 2 Visualizing cleaned data
from datavisualization import single_freqtable, histogram
#single_freqtable(df, 'programme')
#histogram(df, 'stress_level') #fix stress level first ints from 0-100 in 10 bins?

#%% Part 3 of preprocessing: Make it ready for ML algorithms
working_data = df.copy().dropna() # Copy as we will mutate the original dataframe otherwise
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

# Try each algorithm out
for algo in algorithms:
    nr_features = 30
    classification_pipeline = Pipeline([
        ('engineering', preprocessing_pipeline_ODI),
        ('selection', SelectKBest(k=nr_features)),
        ('mode', algo)
    ])

    top_features_idx = classification_pipeline.named_steps.selection.scores_.argpartition(-nr_features)[-nr_features:]
    top_features = classification_pipeline.named_steps.engineering.named_steps.feature_engineering.get_feature_names()[top_features_idx]

    print(top_features)
    avg_train_accuracy = 0

    print(avg_train_accuracy)
    # Measure general performance

    classification_pipeline.fit(X_train, y_train)
    test_predictions = classification_pipeline.predict(X_test)
    test_accuracy = accuracy_score(test_predictions, y_test)
    print(f"For model {type(algo).__name__}, the avg performance for training was {avg_train_accuracy}, and for test {test_accuracy} \n",
          f"\t Predictions were: {test_predictions} \n",
          f"\t Truth is: ${y_test} \n"
    )

# %%


# %%
