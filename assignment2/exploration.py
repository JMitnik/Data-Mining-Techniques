# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Introduction: Setting up
# ---

# %%
# Imports
import sklearn as sk
import pandas as pd
import lightgbm as lgb
import numpy as np

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Lasso, RidgeClassifier, LogisticRegression
from sklearn.feature_selection import SelectFromModel, SelectKBest, RFE
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC, SVC
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA, TruncatedSVD


# %% [markdown]
# Config parameters:
#
# nrows: default dataframelength: 4958347
#
# classifiers: (SVC, linearSVC, RidgeClassifier)
#     - SVC params: C (0-1), kernel (rbf, linear, poly), max_iter (-1, int), random_state (int)
#     - linearSVC params: C (0-1), penalty (l1, l2), max_iter (1000, int), random_state (int)
#     - RidgeClassifier params: max_iter (1000, int), random_state (int)
#
# feature_selection: (SelectFromModel, SelectKBest, RFE)
#     - SelectFromModel params: threshold (0-int), max_features (0-int)
#     - SelectKBest params: threshold (0-int), k (0-int)
#     - RFE params: n_features_to_select (0-int), step

# %%
import importlib
import config
from args import ARGS
importlib.reload(config)
from config import Config
import config_presets
importlib.reload(config_presets)
from config_presets import all_numerical_config, all_numerical_early_nana_removal_config, no_engineer_all_numerical_and_categorical_early_nana_removal_config

# Config Settings
config = no_engineer_all_numerical_and_categorical_early_nana_removal_config
print(f"Config ran with label is {config.label}")

# %%
if config.nrows is not None:
    print(f"Reading data using {config.nrows} rows")
    train_data = pd.read_csv('data/training_set_VU_DM.csv', nrows=config.nrows)
else:
    print(f"Reading all data")
    train_data = pd.read_csv('data/training_set_VU_DM.csv')

    original_columns = train_data.columns
train_data.head(5) # Show top 5

print(f"We read train-data with shape {train_data.shape}")

# %% [markdown]
# # Manual Column exploration
# ---
# ## Main columns
# - `search_id` seems to represent each individual 'user'.
# - `booking_bool` is essentially the answer.
#
# ## Categorical features
# The following features are categorical (to be onehot-encoded):
#
# User-specific
# - `site_id`: category of website Expedia used
# - `visitor_location_country_id`: categories of which country user is from
# - `srch_destination_id`: where did the user search from
# - `srch_saturday_night_bool`: boolean if stay includes staturday
#
# Hotel-specific:
# - `prop_id`: categories of associated hotels
# - `prop_brand_bool`: boolean if hotel is part of chain or not
# - `promotion_flag`: displaying promotion or not
#
# Expedia-specific vs competitors 1_8:
# - `comp{i}_rate`: if expedia has a lower price, do +1, 0 if same, -1 price is higher, null if no competitive data
# - `comp{i}_inv`: if competitor has no availability, +1, 0 if both have availability, null if no competitive data
#
# ## Numerical features
#
# User-specific
# - `visitor_hist_starrating`: average of previous stars of associated user
# - `visitor_hist_adr_usd`: average price per night of hotels of associated user
# - `srch_length_of_stay`: number of nights stays **searched**
# - `srch_booking_window`: number of days ahead the start of booking window **searched**
# - `srch_adults_count`: number of adults **searched**
# - `srch_children_count`: number of children **searched**
# - `srch_room_count`: number of rooms **searched**
# - `random_bool`: if sort was random at time of search
# - `gross_booking_usd`: ❗Training-only❗ payment includign taxes, etc for hotel
#
# Hotel-specific
# - `prop_starrating`: star rating of hotel (1-5)
# - `prop_review_score`: average review score of hotel (1-5)
# - `prop_location_score_1`: score1 of hotel's location desirability
# - `prop_location_score_2`: score2 of hotel's location desirability
# - `prop_log_historical_price`: logarithm of average price of hotel lately (0 == not sold)
# - `price_usd`: displayed price of hotel.
#     - ❗ Important: Different countries have different conventions.
#     - Value can change per night
# - `srch_query_affinity_score`: log probability a hotel is clicked in internet searches
#
# User-hotel coupled:
# - `orig_destination_distance`: distance between hotel and customer at search-time (null means no distance calculated)
#
# Expedia-specific vs competitors 1_8:
# - `comp{i}_rate_percent_diff`: absolute difference between expedia and competitor's price, with null being no competitive data
#
#
# ## Unknown type
# - `date_time`

# %% [markdown]
# Putting all columns except competitor columns in either categorical list or numerical list.

# %%
categorical_cols = ['srch_id', 'date_time', 'site_id', 'visitor_location_country_id', 'prop_country_id',
                    'prop_id', 'prop_brand_bool', 'promotion_flag', 'position',
                    'srch_destination_id', 'srch_saturday_night_bool', 'random_bool',
                    'click_bool', 'booking_bool'
                   ]
numerical_cols = ['visitor_hist_starrating', 'visitor_hist_adr_usd',
                  'prop_starrating', 'prop_review_score', 'prop_location_score1', 'prop_location_score2',
                  'prop_log_historical_price', 'price_usd',
                  'srch_length_of_stay', 'srch_booking_window',
                  'srch_adults_count', 'srch_children_count',
                  'srch_room_count', 'srch_query_affinity_score',
                  'orig_destination_distance', 'gross_bookings_usd'
                  ]

# %% [markdown]
# # Some analysis

# %%
# Plot number of unique values for categorical
import matplotlib.pyplot as plt

# Plot frequencies
np.log(train_data.loc[:, categorical_cols].nunique()).sort_values().plot(kind='barh')
plt.title('Logarithmic Frequency of categorical features')
plt.xlabel('Logarithmic frequency')
plt.show()

# %%
train_data[numerical_cols].boxplot()

# %%
train_data_correlations = train_data.corr()

# %%
import seaborn as sns
ax = sns.heatmap(
    train_data_correlations,
    vmin=-1,
    vmax=1,
    center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
#     square=True
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

# %%
from utils import remove_null_features
train_data_correlations_without_null = remove_null_features(train_data).corr()

# %%
sns.set(rc={'figure.figsize':(11.7,8.27)})

ax = sns.heatmap(
    train_data_correlations_without_null,
    vmin=-1,
    vmax=1,
    center=0,
    yticklabels=True,
    cmap=sns.diverging_palette(20, 220, n=200),
#     square=True
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

# %% [markdown]
# # Feature Pnumerical_cols--

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# When engineering we want to add them to either the numerical or categorical columnlist and append them to the dataframe

# %%
numerical_engineered = []
categorical_engineered = []

# TODO: Make th
def transform_features(df):
        # Do some date feature engineering
    if 'date_time' in df.columns:
        print("Setting month and year")
        time = pd.to_datetime(df['date_time'])
        df['month'] = time.dt.month
        df['year'] = time.dt.year

    # Check if a user comes from same country
    if 'visitor_location_country_id' in df.columns and 'prop_country_id' in df.columns:
        print("Setting same-country-visitor")
        df['same_country_visitor_prop'] = np.where(
            df['visitor_location_country_id'] == df['prop_country_id'],
            1, 0
        )

    try:
        print("Setting viable comp score")

        # Check if Expedia has no competitors that deal better prices.
        df['viable_comp'] = np.where(
                          (df['comp1_rate']== -1)& (df['comp1_inv']== 0) |
                          (df['comp2_rate']== -1)& (df['comp2_inv']== 0) |
                          (df['comp3_rate']== -1)& (df['comp3_inv']== 0) |
                          (df['comp4_rate']== -1)& (df['comp4_inv']== 0) |
                          (df['comp5_rate']== -1)& (df['comp5_inv']== 0) |
                          (df['comp6_rate']== -1)& (df['comp6_inv']== 0) |
                          (df['comp7_rate']== -1)& (df['comp7_inv']== 0) |
                          (df['comp8_rate']== -1)& (df['comp8_inv']== 0)
                          ,1,0)
    except:
        print("Cant find comprate probably")

    # Average location scores of property
    try:
        print("Setting mean score of location scores")
        mcol = df.loc[:,['prop_location_score1', 'prop_location_score2']]
        df['prop_mean_score'] = mcol.mean(axis=1)
    except:
        print("Cant find prop_location columns")

    try:
        print("Removing our columns from our dataframe")

        df = df.drop(columns=['date_time', 'visitor_location_country_id', 'prop_country_id',
                                     'prop_location_score1', 'prop_location_score2'])

        for i in range(8):
            df = df.drop(columns=['comp' + str(i+1) + '_rate'])
            df = df.drop(columns=['comp' + str(i+1) + '_inv'])
            df = df.drop(columns=['comp' + str(i+1) + '_rate_percent_diff'])
    except:
        print("Probably dropped these columns already")

    return df

if config.feature_engineering:
    print("We are doing feature engineering now")
    train_data = transform_features(train_data)

    # Add categorical features to our list of categorical columns
    categorical_engineered = ['same_country_visitor_prop', 'viable_comp']
    for col in categorical_engineered:
        categorical_cols.append(col)

    # Add numerical features to our list of numerical columns
    numerical_engineered = ['prop_mean_score', 'month', 'year']
    for col in numerical_engineered:
        numerical_cols.append(col)

# %% [markdown]
# If we engineer new features, we might want to remove their old columns from the dataframe and columnlists.

# %%
from utils import remove_null_features

if config.remove_null_features_early:
    print("\t Removing null features early!")
    train_data = remove_null_features(train_data, 0.5)

# %%
if config.feature_engineering:
    try:
        train_data = train_data.drop(columns=['date_time', 'visitor_location_country_id', 'prop_country_id',
                                 'prop_location_score1', 'prop_location_score2'])
        for i in range(8):
            train_data = train_data.drop(columns=['comp' + str(i+1) + '_rate'])
            train_data = train_data.drop(columns=['comp' + str(i+1) + '_inv'])
            train_data = train_data.drop(columns=['comp' + str(i+1) + '_rate_percent_diff'])
        categorical_to_remove = ['visitor_location_country_id', 'date_time', 'prop_country_id']
        numerical_to_remove = ['prop_location_score1', 'prop_location_score2']

        for col in categorical_to_remove:
            categorical_cols.remove(col)
        for col in numerical_to_remove:
            numerical_cols.remove(col)
    except:
        print("We probably removed these features already")

# %% [markdown]
# ## Data cleanup: Imputing missing values

# %%
# We will have to cleanup our data next up. Let's first impute the missing columns.
# To do this we search for the columns with nans
print("\n FEATURE IMPUTING: \n")

na_cols = train_data.isna().any()
nan_cols = train_data.columns[na_cols]
print("\tThese are columns with NaN:")
print(nan_cols.to_list())

# %% [markdown]
# Aside from `comp{i}_rate` and `comp2_inv`, all of these columns are numerical features. We could, initially,
# simply replace all these values with -1 for the moment.
#
# ❗ Important: Note, this is actually incorrect, but might work for the moment.

# %%
# Simple numerical impute: select numerical data, fill it with -1

if config.naive_imputing:
    print("\tWe will naively impute and set numericals to -1")
    imputed_numerical_data = train_data[nan_cols].filter(regex='[^comp\d_(rate|inv)$]')
    imputed_numerical_data = imputed_numerical_data.fillna(-1)
    train_data.update(imputed_numerical_data)

    # Manual cleanup to ensure no problem with space
    del imputed_numerical_data
    train_data.head(5)

# %%
# Simple naive categorical impute
if config.naive_imputing:
    print("\tWe will naively impute and set categoricals to -2")
    na_cols = train_data.columns[train_data.isna().any()]
    imputed_categorical_data = train_data[na_cols].fillna(-2)
    train_data.update(imputed_categorical_data)

    # Cleanup
    del imputed_categorical_data
    train_data.head(5)

# %% [markdown]
# A second, less naive approach is to average numerical values grouped by either their hotel (prop_id) or the user (srch_id).
# On top of that we would want to remove columns with over 50% null Values (refence for this?)

# %%
#remove columns with over 50% nans
from utils import remove_null_features
                
if not config.naive_imputing:
    train_data = remove_null_features(train_data, 0.5)

train_data.isnull().sum()/len(train_data)

# %%
not config.naive_imputing

# %% [markdown]
# ### slow method of averaging mean values

# %%
from sklearn.preprocessing import RobustScaler, MinMaxScaler

if not config.naive_imputing:
    #fill in nans with mean values:
    na_cols = train_data.isna().any()
    nan_cols = train_data.columns[na_cols]
    for column in nan_cols:
        print (column)
        if column in ['visitor_hist_starrating', 'visitor_hist_adr_usd',
                         'srch_length_of_stay', 'srch_booking_window',
                         'srch_adults_count', 'srch_children_count',
                         'srch_room_count'
                        ]:
            train_data[column] = train_data.groupby('srch_id').transform(lambda x: x.fillna(x.mean()))
        elif column in ['prop_starrating', 'prop_review_score',
                           'prop_location_score1', 'prop_location_score2',
                           'prop_log_historical_price', 'price_usd',
                           'search_', 'orig_destination_distance',
                           'srch_query_affinity_score'
                          ]:
            train_data[column] = train_data.groupby('prop_id').transform(lambda x: x.fillna(x.mean()))

    train_data.isnull().sum()/len(train_data)

# if config.naive_imputing:
    # Here we definehow we would like to encode

# For One-Hot Encoding
# Onehot encode the categorical variables
oh_impute = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-2)
oh_encoder = OneHotEncoder(handle_unknown='ignore')
oh_pipeline = Pipeline([
    ('impute', oh_impute),
    ('encode', oh_encoder)
])

# Encode the numerical values
num_scale_encoder = StandardScaler()
num_impute = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1)
num_pipeline = Pipeline([
    ('impute', num_impute),
    ('encode', num_scale_encoder)
])

num_outlier_scale_encoder = RobustScaler()
num_scale_impute = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1)
num_scale_outliers_pipeline = Pipeline([
    ('impute', num_scale_impute),
    ('encode', num_outlier_scale_encoder)
])

# %% [markdown]
# ## Feature encoding

# %%
# Imports for feature transformation
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# %% [markdown]
# In the next part, we decide on our final columns (`chosen_columns`) based on our initial config file, before we encode them.
#
# This consists of the following decisions:
# - Pre-feature selection: Do we restrict ourselves to a select amount of features?
# - Feature-engineering: Do we apply feature engineering and add it on top?
# - Are these columns in our training data?

# %%
chosen_columns = categorical_cols + numerical_cols

print("\n FEATURE SELECTION: \n")

# Manual feature-selection
if config.pre_feature_selection:
    print("Setting pre feature selection")
    chosen_columns = config.pre_selection_cols

# Manual feature engineering
if config.feature_engineering:
    print("We will incl feature engineering")
    chosen_columns = [*chosen_columns, *categorical_engineered, *numerical_engineered]

for column in categorical_cols:
    try:
        train_data[column]=train_data[column].astype('category')
    except:
        print(f"Error setting column {column} to categorical!")

# Ensure only intersection of chosen columns and present columns are in data now
chosen_columns = list(set(chosen_columns) & set(train_data.columns))

# Split chosen columns into numerical and categorical
chosen_oh_cols = list(set(chosen_columns) & set(categorical_cols))
chosen_num_cols = list(set(chosen_columns) & set(numerical_cols))

print(f"\tWe have {len(chosen_columns)} columns now!")
print(f"\tThey are: {chosen_columns}")
print(f"\t\t Out of those, {len(chosen_oh_cols)} are categorical now!")
print(f"\t\t Out of those, {len(chosen_num_cols)} are numerical now!")

# %%
print("\n Feature ENCODING: \n")

chosen_train_data = train_data[chosen_columns]

chosen_num_cols = [i for i in chosen_num_cols if 'price' not in i]
chosen_num_scale_cols = ['price_usd']

df_transformer = ColumnTransformer([
    ('oh', oh_pipeline, chosen_oh_cols),
    ('num', num_pipeline, chosen_num_cols),
    ('num_scale', num_scale_outliers_pipeline, chosen_num_scale_cols),
], remainder='drop')

# We fit this transformer on our training data, and transform/encode our training data
encoded_X = df_transformer.fit_transform(chosen_train_data)

# We also represent this same X using the original columns.
new_oh_columns = df_transformer.named_transformers_.oh.named_steps.encode.get_feature_names(chosen_oh_cols)
encoded_columns = [ *new_oh_columns, *chosen_num_cols, *chosen_num_scale_cols]
print(f"\tEncoded our X to have shape {encoded_X.shape}")
print(f"\tEncoded columns are {encoded_columns}")

# %% [markdown]
# ## Feature selection

# %%
# We extract the y-target in general
X_only = train_data.copy()
y = X_only.pop('booking_bool')

# %%
bool_vec = np.ones(encoded_X.shape[1], dtype=bool)

# %%
##### We apply feature selection using the model from our config
print("\n AUTOMATED FEATURE SELECTION: \n")

# Use PCA feature selection method
if config.dimensionality_reduc_selection and not config.algo_feature_selection:
    print("Applying feature selection using TruncatedSVD")
    pca = TruncatedSVD(n_components=config.dimension_features)
    encoded_X = pca.fit_transform(encoded_X)

# Use linear feature-selection methods
if config.algo_feature_selection and not config.dimensionality_reduc_selection:
    # If classifier is not None, we use a classifier as feature-selection helper
    if config.classifier is not None:
        classifier = config.classifier(**config.classifier_dict)
        feature_selector = config.feature_selection(classifier, **config.feature_selection_dict)
        print(f"Applying feature selection using linear feature selection methods \n,"f"{type(feature_selector).__name__}, using {type(classifier).__name__} as classifier.")
        feature_encoded_X = feature_selector.fit_transform(encoded_X, y)
        bool_vec = feature_selector.support_

    # Else we use a feature scoring method
    else:
        scoring_func = config.feature_selection_scoring_func
        feature_selector = config.feature_selection(scoring_func, **config.feature_selection_dict)
        encoded_X = feature_selector.fit_transform(encoded_X, y)
        bool_vec = feature_selector.get_support()


# %%
assert encoded_X.shape[1] == len(bool_vec), "Bool vec mismatch with encoded shape"

# %% [markdown]
# # Training a model

# %% [markdown]
# We will now try a various amount of models with parameters.

# %%
# Utility functions

# Gets the sizes of same search-id chunks.
get_user_groups_from_df = lambda df: df.groupby('srch_id', observed=True).size().tolist()

# %%
# Reassign `srch_id`
srch_id_col = train_data['srch_id'].astype(int)
srch = np.array(srch_id_col)

# %% [markdown]
# ### Learn-to-rank with LGBMRanker

# %% [markdown]
# If we decide to split our data into train/val, we can do it this way.

# %%
from sklearn.model_selection import GroupShuffleSplit
print("\n TRAINING THE MODEL: \n")

# feature_encoded_X = encoded_X
# Split data into (default) 80% train and 20% validation, maintaining the groups however.
print("\tGoing to split data now!")
train_inds, val_inds = next(GroupShuffleSplit(
    test_size=0.05,
    n_splits=2,
    random_state = 7
).split(encoded_X, groups=srch_id_col))

print(f"\tWill train with {len(train_inds)} and validate with {len(val_inds)} amount of data.")

# Split train / validation by their indices
X_train = encoded_X[train_inds]
y_train = np.array(y[train_inds])
X_val = encoded_X[val_inds]
y_val = np.array(y[val_inds])

# Get the groups related to `srch_id`
query_train = get_user_groups_from_df(train_data.iloc[train_inds])
query_val = get_user_groups_from_df(train_data.iloc[val_inds])

print("\tReady to rank!")

# %%
# Number of sanity checks
assert len(list(y_train)) == (X_train.shape[0]), "Mismatch in sample-size between feature-length and labels for training"
assert len(list(y_train)) == np.sum(query_train), "Mismatch in sample-size sum query-train and number of labels for training"

assert len(list(y_val)) == X_val.shape[0], "Mismatch in sample-size between feature-length and labels for validation"
assert len(list(y_val)) == np.sum(query_val), "Mismatch in sample-size sum query-train and number of labels for validation"

# Ensure no 0's in query train

assert (0 not in query_train), "There is a 0 in query train! This will crash your LightBGM!"

print("\t Passed our sanity checks!")

# %%
import lightgbm as lgb

# %%
# book_w = 10
# click_w = 3

# y_pos = train_data['position']
# y_book = train_data['booking_bool']
# y_click = train_data['click_bool']
# y_combined = book_w * np.array(y_book) + click_w * np.array(y_click) + np.round(2 * (1 / np.array(y_pos)))
# y_train = y_combined[train_inds]
# y_val = np.array(y_book[val_inds])

# %%
# 221879
# We define our ranker (default parameters)
eval_results = []
gbm = lgb.LGBMRanker(n_estimators=250, num_leaves=50, min_data_in_leaf=200)

def store_results(results):
    callb = lgb.print_evaluation(dict)
    eval_results.append(results.evaluation_result_list)
    return callb

print("\t Training now!")

gbm.fit(X_train, y_train, callbacks=[store_results],group=query_train,
        eval_set=[(X_val, y_val)], eval_group=[query_val],
        eval_at=[5, 10, 20], early_stopping_rounds=50)

# %%
import json

print("\t Finished training!")

# Extract feature importances, normalize them, and store them in the config
feature_importances = gbm.feature_importances_
feature_importances_norm = np.linalg.norm(feature_importances)
feature_importances_normalized = (feature_importances / feature_importances_norm)

# Sort the features and get their names
ranking_features_idx = feature_importances_normalized.argsort()[::-1]
feature_importances_normalized_sorted = feature_importances_normalized[ranking_features_idx]
feature_names = np.array(encoded_columns)[ranking_features_idx]

# Combine features with the names
features_by_score = list(zip(feature_names, feature_importances_normalized_sorted))[:10]

print(f"Best features by score!: \n {features_by_score}")
# Store it in the config
config.mutable_feature_importances_from_learner = json.dumps(features_by_score)

# %%
import os
def ensure_path(path_to_file):
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)


# %%
# Get validation performance metrics and store along with config.
import datetime
now = datetime.datetime.now()

eval_results_reformatted = {
    'valid_ndcg_5': [i[0][2]for i in eval_results],
    'valid_ndcg_10': [i[1][2] for i in eval_results],
    'valid_ndcg_20': [i[2][2] for i in eval_results],
}

df_eval_results = pd.DataFrame.from_dict({**eval_results_reformatted, **config.to_dict()})
df_eval_results['timestamp_end'] = now
df_eval_results = df_eval_results.set_index('timestamp_end')

# Save results
if not os.path.isfile(config.path_to_eval_results):
    ensure_path(config.path_to_eval_results)
    df_eval_results.to_csv(config.path_to_eval_results)
    print("Saved new file!")
else: # else it exists so append without writing the header
    df_eval_results.to_csv(config.path_to_eval_results, mode='a', header=False)
    print("Saved by appending!")

# %%
# Save model

ensure_path('storage/best_gbm.txt')
gbm.booster_.save_model('storage/best_gbm.txt')

# %%
try:
    del train_data, encoded_X, feature_encoded_X, X_only
except:
    pass

# %%
# Garbage collection and show
try:
    import gc
    from guppy import hpy
    gc.collect()
    h=hpy()
    print(h.heap())
except:
    print("Unable to show heap, probably missing guppy")


# %% [markdown]
# # Testing
# ---

# %% [markdown]
# ## Testing with LGBM-Ranker

# %%
print("\n Testing Time!: \n")

# Read test data, and use the same columns as was used for training
df_test_data = pd.read_csv('data/test_set_VU_DM.csv')

if config.feature_engineering:
    df_test_data = transform_features(df_test_data)

chosen_test_data = df_test_data[chosen_columns]

# Apply transformations (encoding + selection)
encoded_test_data = df_transformer.transform(chosen_test_data)

if config.dimensionality_reduc_selection:
    filtered_test_data = pca.transform(encoded_test_data)
else:
    filtered_test_data = encoded_test_data[:, bool_vec]

X_test = filtered_test_data

print(f"Test data now shaped like {X_test.shape}")


# %% [markdown]
# #### Predicting all at once: More performant

# %%
pred_all = gbm.predict(X_test)
df_test_data['pred'] = pred_all

print("\t Finished prediction!")

# %%
# Sort predictions based on srch_id and pred
sorted_preds = df_test_data[['srch_id', 'prop_id', 'pred']].sort_values(by=['srch_id', 'pred'], ascending=[True, False]).reset_index()

# Save
sorted_preds[['srch_id', 'prop_id']].to_csv(f'results/predictions_{config.label}.csv', index=False)

print(f"\t Stored final prediction in results_{config.label}.csv! ")
