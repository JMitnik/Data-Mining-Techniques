# %%
# Imports
import pandas as pd
from scipy.sparse import find
from sklearn.compose import make_column_selector
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.linear_model import Lasso, RidgeClassifier, LogisticRegression
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
from preprocessing import transform_ODI_dataset, read_all_features_from_pipeline, make_encoding_pipeline, preprocess_target, read_selected_features_from_pipeline

# Configgg
config = Config(
    min_word_count=4,
    random_state=0
)

#%% Part 1 of preprocessing: reading data and cleaning it

# Read ODI dataset and transform it, clean it, etc
df = pd.read_csv("data/ODI-2020.csv", sep=";", encoding="utf-8")
df = transform_ODI_dataset(df, programme_threshold=5)

#%% Part 2 Visualizing cleaned data
categorical_cols = ['programme', 'did_ml', 'did_stats', 'did_db', 'did_ir', 'did_stand', 'gender', 'chocolate'] #todo gd text?
numerical_cols = ['random_nr', 'stress_level', 'nr_neighbours', 'bedtime_yesterday', 'date_of_birth']   #fixed by cleaning data? date_of_birth here? bedtime?

from datavisualization import countplot, histogram, boxplot, heatmap, heatmap2, stacked_bars

### distribution of our columns are here:
#countplot(df[categorical_cols], categorical_cols)
#histogram(df[numerical_cols], numerical_cols)


### correlation between multivariable plots are here

#boxplot(df, categorical_cols)
#heatmap2(df)
stacked_bars(df)
#
#%% Part 3 of preprocessing: Make it ready for ML algorithms
X = df.copy().dropna() # Copy as we will mutate the original dataframe otherwise
y = X.pop('programme') #Pop removes programme from



# Creates a pipeline for the input which will transform our input to ndarrays
encoding_pipeline = make_encoding_pipeline(
    X,
    config.min_word_count
)
target_encoder, y_encode = preprocess_target(y)

# TODO: Figure out train/test-size disparacy
X_train, X_test, y_train, y_test = train_test_split(X, y_encode, test_size=0.2)

# Instantiate a few classification pipelines for comparisons
algorithms = [
    SVC(kernel='linear'),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
]

selection_pipeline = Pipeline([('rfe', RFE(RidgeClassifier()))])

# Initialize empty results dataframe
results_df = pd.DataFrame(columns=[
    'model_name',
    'avg_cv_train_acc',
    'var_cv_train_acc',
    'avg_cv_train_balanced_acc',
    'var_cv_train_balanced_acc',
    'test_acc',
    'test_balanced_acc',
    'nr_features',
    'top_5_features',
    'bottom_5_features'
])
results_df['top_5_features'] = results_df['top_5_features'].astype('object')
results_df['bottom_5_features'] = results_df['bottom_5_features'].astype('object')

trained_pipelines = {}


# Try each algorithm out
for algo in algorithms:
    classification_pipeline = Pipeline([
        ('engineering', encoding_pipeline),
        ('selection', selection_pipeline),
        ('model', algo)
    ])

    # Measure general performance
    fold_scores = cross_validate(classification_pipeline,
        X_train,
        y_train,
        scoring=['accuracy', 'balanced_accuracy']
    )

    # Perform predictions and such
    classification_pipeline.fit(X_train, y_train)

    trained_pipelines[type(algo).__name__] = classification_pipeline

    test_predictions = classification_pipeline.predict(X_test)

    # Measure final metrics on test-set
    test_accuracy = accuracy_score(test_predictions, y_test)
    test_balanced_accuracy = balanced_accuracy_score(test_predictions, y_test)

    # Store the results of this current run
    results_df = results_df.append({
        'model_name': type(algo).__name__,
        'avg_cv_train_acc': fold_scores['test_accuracy'].mean(),
        'var_cv_train_acc': fold_scores['test_accuracy'].var(),
        'avg_cv_train_balanced_acc': fold_scores['test_balanced_accuracy'].mean(),
        'var_cv_train_balanced_acc': fold_scores['test_balanced_accuracy'].var(),
        'test_acc': test_accuracy,
        'test_balanced_accuracy': test_balanced_accuracy,
        'top_5_features': [read_selected_features_from_pipeline(classification_pipeline)[::-1][0:5]],
        'bottom_5_features': [read_selected_features_from_pipeline(classification_pipeline)[0:5]]
    }, ignore_index=True)


results_df.to_csv('results/run.csv')

# %%
from sklearn.tree import plot_tree, export_graphviz
import matplotlib.pyplot as plt

# Plot and #savetreelives
plt.figure(figsize=(8,6))
tree_pipeline = trained_pipelines['DecisionTreeClassifier']
tree = trained_pipelines['DecisionTreeClassifier'].named_steps.model

# Explore tree
# Option 1: Plot tree, bit slow though
# plot_tree(
#     tree,
#     feature_names=read_all_features_from_pipeline(tree_pipeline),
#     class_names=y.cat.categories,
#     filled=True
# )

# Option 2: Export tree to dot file
export_graphviz(
    tree,
    feature_names=read_selected_features_from_pipeline(tree_pipeline, is_sorted=False),
    class_names=y.cat.categories,
    filled=True,
    out_file='results/tree.dot'
)

# %%
