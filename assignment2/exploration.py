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
importlib.reload(config)
from config import Config

# Config Settings
config = Config(
    label='TrialAndError',
    path_to_eval_results='results/eval_results.csv',
    nrows=None,
    valid_size=0.2,
    pre_feature_selection=True,
    algo_feature_selection=True,
    train_data_subset=0.8,
    classifier=SVC,
    classifier_dict={'C' : 1, 'kernel' : 'rbf', 'random_state' : 2},
    feature_selection=SelectFromModel,
    feature_selection_dict={'threshold' : 1},
    dimensionality_reduc_selection=False,
    pre_selection_cols=[
        'srch_saturday_night_bool', 'prop_starrating', 'prop_review_score', 'prop_location_score1', 'prop_location_score2', 
        'prop_log_historical_price', 'price_usd', 'srch_query_affinity_score', 'promotion_flag'
    ],
    dimension_features=25,
    feature_engineering=True,  
    naive_imputing=True #todo faster method for averaging nan values if naive=False
)

# %%
if config.nrows is not None:
    train_data = pd.read_csv('data/training_set_VU_DM.csv', nrows=config.nrows)
else:
    train_data = pd.read_csv('data/training_set_VU_DM.csv')

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
# # Feature Preprocessing
# ---

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# When engineering we want to add them to either the numerical or categorical columnlist and append them to the dataframe

# %%
numerical_engineered = []
categorical_engineered = []

if config.feature_engineering:
    # Do some date feature engineering     
    time = pd.to_datetime(train_data['date_time'])
    train_data['month'] = time.dt.month
    train_data['year'] = time.dt.year
    
    # Check if a user comes from same country
    train_data['same_country_visitor_prop'] = np.where(
        train_data['visitor_location_country_id'] == train_data['prop_country_id'], 
        1, 0
    )
    
    # Check if Expedia has no competitors that deal better prices.     
    train_data['viable_comp'] = np.where(
                      (train_data['comp1_rate']== -1)& (train_data['comp1_inv']== 0) |
                      (train_data['comp2_rate']== -1)& (train_data['comp2_inv']== 0) |
                      (train_data['comp3_rate']== -1)& (train_data['comp3_inv']== 0) |
                      (train_data['comp4_rate']== -1)& (train_data['comp4_inv']== 0) |
                      (train_data['comp5_rate']== -1)& (train_data['comp5_inv']== 0) |
                      (train_data['comp6_rate']== -1)& (train_data['comp6_inv']== 0) |
                      (train_data['comp7_rate']== -1)& (train_data['comp7_inv']== 0) |
                      (train_data['comp8_rate']== -1)& (train_data['comp8_inv']== 0) 
                      ,1,0)

    # Average location scores of property    
    mcol = train_data.loc[:,['prop_location_score1', 'prop_location_score2']]
    train_data['prop_mean_score'] = mcol.mean(axis=1)
    
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
na_cols = train_data.isna().any()
nan_cols = train_data.columns[na_cols]
print("These are columns with NaN:")
print(nan_cols.to_list())

# %% [markdown]
# Aside from `comp{i}_rate` and `comp2_inv`, all of these columns are numerical features. We could, initially,
# simply replace all these values with -1 for the moment.
#
# ❗ Important: Note, this is actually incorrect, but might work for the moment.

# %%
# Simple numerical impute: select numerical data, fill it with -1
if config.naive_imputing:    
    imputed_numerical_data = train_data[nan_cols].filter(regex='[^comp\d_(rate|inv)$]')
    imputed_numerical_data = imputed_numerical_data.fillna(-1)
    train_data.update(imputed_numerical_data)

    # Manual cleanup to ensure no problem with space
    del imputed_numerical_data
    train_data.head(5)

# %%
# Simple naive categorical impute
if config.naive_imputing:    
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
if not config.naive_imputing:
    for column in train_data.columns:
        if train_data[column].isnull().sum()/len(train_data) > 0.5:
            train_data = train_data.drop(columns=column, axis=1)

train_data.isnull().sum()/len(train_data)

# %% [markdown]
# ### slow method of averaging mean values

# %%
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
num_impute = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1)
num_scale_encoder = StandardScaler()
num_pipeline = Pipeline([
    ('impute', num_impute),
    ('encode', num_scale_encoder)
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

# Manual feature-selection
if config.pre_feature_selection:
    print("Setting pre feature selection")
    chosen_columns = config.pre_selection_cols

# Manual feature engineering
if config.feature_engineering:
    print("Including feature engineering")
    chosen_columns = [*chosen_columns, *categorical_engineered, *numerical_engineered]

for column in categorical_cols:
    train_data[column]=train_data[column].astype('category')

# Ensure only intersection of chosen columns and present columns are in data now
chosen_columns = list(set(chosen_columns) & set(train_data.columns))

# Split chosen columns into numerical and categorical
chosen_oh_cols = list(set(chosen_columns) & set(categorical_cols))
chosen_num_cols = list(set(chosen_columns) & set(numerical_cols))

print(f"We have {len(chosen_columns)} columns now!")
print(f"\t Out of those, {len(chosen_oh_cols)} are categorical now!")
print(f"\t Out of those, {len(chosen_num_cols)} are numerical now!")

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

# %% [markdown]
# ## Feature selection

# %%
# We extract the y-target in general
X_only = train_data.copy()
y = X_only.pop('booking_bool')

# %%
##### We apply feature selection using the model from our config

# Use PCA feature selection method
if config.dimensionality_reduc_selection and not config.algo_feature_selection:
    print("Applying feature selection using TruncatedSVD")
    pca = TruncatedSVD(n_components=config.dimension_features)
    feature_encoded_X = pca.fit_transform(encoded_X)

# Use linear feature-selection methods
if config.algo_feature_selection and not config.dimensionality_reduc_selection:
    classifier = config.classifier(**config.classifier_dict)
    feature_selector = config.feature_selection(classifier, **config.feature_selection_dict)
    print(f"Applying feature selection using linear feature selection methods \n,"
          f"{type(feature_selector).__name__}, using {type(feature_selector).__name__} as classifier.")
    feature_encoded_X = feature_selector.fit_transform(encoded_X, y)
    # TODO: support_ method might not work on every featureselector we choose, test this
    bool_vec = feature_selector.support_

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

# feature_encoded_X = encoded_X
# Split data into (default) 80% train and 20% validation, maintaining the groups however.
print("Going to split data now!")
train_inds, val_inds = next(GroupShuffleSplit(
    test_size=config.valid_size, 
    n_splits=2, 
    random_state = 7
).split(feature_encoded_X, groups=srch_id_col))

print(f"Will train with {len(train_inds)} and validate with {len(val_inds)} amount of data.")

# Split train / validation by their indices
X_train = feature_encoded_X[train_inds]
y_train = np.array(y[train_inds])
X_val = feature_encoded_X[val_inds]
y_val = np.array(y[val_inds])

# Get the groups related to `srch_id`
query_train = get_user_groups_from_df(train_data.iloc[train_inds])
query_val = get_user_groups_from_df(train_data.iloc[val_inds])

print("Ready to rank!")

# %%
# Number of sanity checks
assert len(list(y_train)) == len(X_train), "Mismatch in sample-size between feature-length and labels for training"
assert len(list(y_train)) == np.sum(query_train), "Mismatch in sample-size sum query-train and number of labels for training"

assert len(list(y_val)) == len(X_val), "Mismatch in sample-size between feature-length and labels for validation"
assert len(list(y_val)) == np.sum(query_val), "Mismatch in sample-size sum query-train and number of labels for validation"

# Ensure no 0's in query train

assert (0 not in query_train), "There is a 0 in query train! This will crash your LightBGM!"

# %%
config.feature_selection.__name__

# %%
import lightgbm as lgb
lgb.record_evaluation(dict)

# %%
# We define our ranker (default parameters)
eval_results = []
gbm = lgb.LGBMRanker(n_jobs=1)

def store_results(results):
    callb = lgb.print_evaluation(dict)
    eval_results.append(results.evaluation_result_list)
    return callb    

gbm.fit(X_train, list(y_train), callbacks=[store_results],group=query_train,
        eval_set=[(X_val, list(y_val))], eval_group=[query_val],
        eval_at=[5, 10, 20], early_stopping_rounds=50)

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
else: # else it exists so append without writing the header
    df_eval_results.to_csv(config.path_to_eval_results, mode='a', header=False)

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
# Read test data, and use the same columns as was used for training
df_test_data = pd.read_csv('data/test_set_VU_DM.csv')
chosen_test_data = df_test_data[chosen_columns]

# Apply transformations (encoding + selection)
encoded_test_data = df_transformer.transform(chosen_test_data)

if config.PCA_use:
    filtered_test_data = pca.transform(encoded_test_data)
else:
    filtered_test_data = encoded_test_data[:, bool_vec]
                                    
X_test = filtered_test_data


# %%
filtered_test_data


# %% [markdown]
# #### Predicting on a per-group basis: slow as hell (skip section for faster method)

# %%
# # Split test-data into groups based on the original data
# groups = df_test_data.groupby('srch_id').indices
# groups_by_idxs = list(groups.values())
# print (groups_by_idxs)

# %%
# Predictions
def predict_for_group(X_test, group_idxs, df_test_data):
    # Use gbm to predict
    X_test_group = X_test[group_idxs]
    preds = gbm.predict(X_test_group)
    preds = preds.argsort()[::-1] # Reverses
    
    # Get th
    pred_idxs = group_idxs[preds]
    pred_props = df_test_data.loc[pred_idxs, ['srch_id', 'prop_id']]
    
    return pred_props

# %%
# Doing it on a 'per-group basis'
# Commented because it is slow.
# # Perform the prediction (Can take a while, shitton of predictions)
# result = []

# for i, idx_group in enumerate(groups_by_idxs):
#     preds = predict_for_group(X_test, idx_group, df_test_data)
#     result.append(preds)
    
#     if i % 10000 == 0:
#         print(f"Doing group {i + 1} / {len(groups_by_idxs)} now")


# %%
# len(result)

# %% [markdown]
# #### Predicting all at once: More performant

# %%
pred_all = gbm.predict(X_test)
df_test_data['pred'] = pred_all

# %%
pred_all

# %%
# Sort predictions based on srch_id and pred
sorted_preds = df_test_data[['srch_id', 'prop_id', 'pred']].sort_values(by=['srch_id', 'pred'], ascending=[True, False]).reset_index()

# Save
sorted_preds[['srch_id', 'prop_id']].to_csv('results.csv', index=False)

# %%
sorted_preds

# %%
pd.read_csv('results.csv', nrows=100)

# %%
