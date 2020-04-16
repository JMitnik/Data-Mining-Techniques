# %%
# Imports
import pandas as pd
from scipy.sparse import find
from sklearn.compose import make_column_selector
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.linear_model import Lasso
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, RFE
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate
import nltk
from tempfile import mkdtemp
from config import Config

# Script for notebooks to reload dependency
import importlib
import preprocessing
importlib.reload(preprocessing)
from preprocessing import transform_ODI_dataset, make_encoding_pipeline, preprocess_target

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
X = df.copy().dropna() # Copy as we will mutate the original dataframe otherwise
y = X.pop('programme') #Pop removes programme from

# Creates a pipeline for the input which will transform our input to ndarrays
encoding_pipeline = make_encoding_pipeline(
    X,
    config.min_word_count
)
target_encoder, y_encode = preprocess_target(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encode, test_size=0.2)

# Instantiate a few classification pipelines for comparisons
algorithms = [
    SVC(kernel='linear',
            probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
]

selection_pipeline = Pipeline([('rfe', RFE(SVC(kernel='linear')))])

# Try each algorithm out
for algo in algorithms:
    nr_features = 30
    classification_pipeline = Pipeline([
        ('engineering', encoding_pipeline),
        ('selection', selection_pipeline),
        ('mode', algo)
    ])

    # Measure general performance
    avg_train_accuracy = cross_validate(classification_pipeline,
        X_train,
        y_train,
        scoring=['accuracy', 'balanced_accuracy']
    )
    classification_pipeline.fit(X_train, y_train)
    test_predictions = classification_pipeline.predict(X_test)
    test_accuracy = accuracy_score(test_predictions, y_test)
    test_auc = balanced_accuracy_score(test_predictions, y_test)
    print(f"For model {type(algo).__name__}, the avg performance for training was {avg_train_accuracy}, and for test {test_accuracy} \n",
          f"\t Predictions were: {test_predictions} \n",
          f"\t Truth is: ${y_test} \n"
    )

# %%


# %%
