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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Lasso, RidgeClassifier, LogisticRegression
from sklearn.feature_selection import SelectFromModel, SelectKBest, RFE
from sklearn.svm import LinearSVC, SVC


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
importlib.reload(config)
from config import Config

# Config Settings
config = Config(
    nrows=1000,
    pre_feature_selection=True, #todo Bug in prefeature selection = False
    train_data_subset=0.8,
    classifier=RidgeClassifier,
    classifier_dict={'max_iter' : 1000, 'random_state' : 2},
    feature_selection=RFE,
    feature_selection_dict={'n_features_to_select' : 5, 'step' : 1} 
)

# %%
train_data = pd.read_csv('data/training_set_VU_DM.csv', nrows=config.nrows)
original_columns = train_data.columns
train_data.head(5) # Show top 5

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
# # Feature Preprocessing
# ---

# %% [markdown]
# ## Data cleanup: Imputing missing values

# %%
# We will have to cleanup our data next up. Let's first impute the missing columns. 
# To do this we search for the columns with nans
na_cols = train_data.isna().any()
nan_cols = train_data.columns[na_cols]
nan_cols

# %% [markdown]
# Aside from `comp{i}_rate` and `comp2_inv`, all of these columns are numerical features. We could, initially,
# simply replace all these values with -1 for the moment.
#
# ❗ Important: Note, this is actually incorrect, but might work for the moment.

# %%
# Simple numerical impute: select numerical data, fill it with -1
imputed_numerical_data = train_data[nan_cols].filter(regex='[^comp\d_(rate|inv)$]')
imputed_numerical_data = imputed_numerical_data.fillna(-1)
train_data.update(imputed_numerical_data)

# Manual cleanup to ensure no problem with space
del imputed_numerical_data
train_data.head(5)

# %%
# Simple naive categorical impute
na_cols = train_data.columns[train_data.isna().any()]
imputed_categorical_data = train_data[na_cols].fillna(-2)
train_data.update(imputed_categorical_data)

# Cleanup
del imputed_categorical_data
train_data.head(5)

# %% [markdown]
# ## Feature encoding

# %%
# Imports for feature transformation
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# %%
import numpy as np
# Here we definehow we would like to encode

# For One-Hot Encoding
# Onehot encode the categorical variables
oh_columns = ['site_id', 'visitor_location_country_id', 'prop_country_id', 
              'prop_id', 'prop_brand_bool', 'promotion_flag', 
              'srch_destination_id', 'srch_saturday_night_bool', 'random_bool', 'click_bool'
             ]
oh_impute = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-2)
oh_encoder = OneHotEncoder(handle_unknown='ignore')
oh_pipeline = Pipeline([
    ('impute', oh_impute),
    ('encode', oh_encoder)
])
# TODO: competitor columns
for column in oh_columns:
    train_data[column]=train_data[column].astype('category')


# Encode the numerical values
num_scale_columns = ['visitor_hist_starrating', 'visitor_hist_adr_usd', 
                     'prop_starrating', 'prop_review_score', 
                     'prop_location_score1', 'prop_location_score2', 
                     'prop_log_historical_price', 'price_usd', 
                     'srch_length_of_stay', 'srch_booking_window', 
                     'srch_adults_count', 'srch_children_count',
                     'srch_room_count', 'srch_query_affinity_score', 
                     'orig_destination_distance' 
                    ]
num_impute = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1)
num_scale_encoder = StandardScaler()
num_pipeline = Pipeline([
    ('impute', num_impute),
    ('encode', num_scale_encoder)
])

# %%
# Manual feature-selection
# We do a preselection of columns that we feel will become useful features after encoding
if config.pre_feature_selection == True:
    chosen_columns = ['prop_starrating', 'prop_review_score', 'prop_location_score1', 'prop_location_score2', 
                      'prop_log_historical_price', 'price_usd', 'srch_query_affinity_score',  'promotion_flag']
else:
    chosen_columns = oh_columns + num_scale_columns

# Select the chosen columns, and 
# define the corresponding transformer's transformations to their columns
chosen_oh_cols = list(set(chosen_columns) & set(oh_columns))
chosen_num_cols = list(set(chosen_columns) & set(num_scale_columns))

# %%
chosen_train_data = train_data[chosen_columns]

df_transformer = ColumnTransformer([
    ('oh', oh_pipeline, chosen_oh_cols),
    ('num', num_pipeline, chosen_num_cols),
], remainder='drop')

# We fit this transformer on our training data, and transform/encode our training data
encoded_X = df_transformer.fit_transform(chosen_train_data)

# We also represent this same X using the original columns.
new_oh_columns = df_transformer.named_transformers_.oh.named_steps.encode.get_feature_names(chosen_oh_cols)
encoded_columns = [ *new_oh_columns, *chosen_num_cols]
df_encoded_X = pd.DataFrame(encoded_X, columns=encoded_columns)

# %% [markdown]
# ## Feature selection

# %%
# We extract the y-target in general
X_only = train_data.copy()
y = X_only.pop('booking_bool')

# %%
# We apply feature selection using the model from our config
feature_selector = config.feature_selection(classifier, **config.feature_selection_dict)
encoded_df = feature_selector.fit_transform(encoded_df, y)

# %%
# Utility cell to investigate data elements
# Data elements we have available
# Encoded data:
    # - df_encoded_X: Dataframe that contains the preprocessed features
    # - encoded_X: numpy version of `encoded_df`
# Original data:
    # - train_data: training data, but cleaned up
    # - X_only: `train_data` without `booking_bool`

# %% [markdown]
# # Training a model

# %% [markdown]
# We will now try a various amount of models with parameters.

# %%
# Utility functions

# Gets the sizes of same search-id chunks.
get_user_groups_from_df = lambda df: df.groupby('srch_id').size().tolist()

# %%
# Reassign `srch_id`
df_encoded_X['srch_id'] = train_data['srch_id'].astype(int)

# %% [markdown]
# ### Learn-to-rank with LGBMRanker

# %% [markdown]
# If we decide to split our data into train/val, we can do it this way.

# %%
from sklearn.model_selection import GroupShuffleSplit

# Split data into 80% train and 20% validation, maintaining the groups however.
train_inds, val_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 7).split(df_encoded_X, groups=df_encoded_X['srch_id']))

# Split train / validation by their indices
df_X_train = df_encoded_X.iloc[train_inds]
y_train = y[train_inds]
df_X_val = df_encoded_X.iloc[val_inds]
y_val = y[val_inds]

# Get the groups related to `srch_id`
query_train = get_user_groups_from_df(df_X_train)
query_val = get_user_groups_from_df(df_X_val)

# %%
sum(query_train)

# %%
# We define our ranker (default parameters)
gbm = lgb.LGBMRanker()

gbm.fit(df_X_train, y_train, group=query_train,
        eval_set=[(df_X_val, y_val)], eval_group=[query_val],
        eval_at=[5, 10, 20], early_stopping_rounds=50)

# %% [markdown]
# # Testing
# ---

# %% [markdown]
# ## Testing with LGBM-Ranker

# %%
test_data = pd.read_csv('data/test_set_VU_DM.csv')

# %%
chosen_test_data = test_data[chosen_columns]

# %%
df_transformer.transform(chosen_test_data)

# %%
chosen_test_data['srch_query_affinity_score'].to_numpy().shape

# %%
