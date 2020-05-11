#!/usr/bin/env python
# coding: utf-8

# # Start of Data Exploration

# In[1]:


# Imports
import sklearn as sk
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, RFE
from sklearn.svm import LinearSVC, SVC


# In[2]:


from config import Config
# Config Settings
config = Config(
    pre_feature_selection=True,
    train_data_subset=0.8,
    #classifier=SVC(),
    #feature_selection=SelectFromModel(lsvc, prefit=True),
    feature_selection_dict={'threshold' : 2, 'max_features' : 2}

)


# In[3]:


train_data = pd.read_csv('data/training_set_VU_DM.csv')
original_columns = train_data.columns
train_data.head(5) # Show top 5


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

# # Initial data cleanup

# ## Impute missing value

# In[4]:


# We will have to cleanup our data next up. Let's first impute the missing columns. 
# To do this we search for the columns with nans
na_cols = train_data.isna().any()
nan_cols = train_data.columns[na_cols]
nan_cols


# Aside from `comp{i}_rate` and `comp2_inv`, all of these columns are numerical features. We could, initially,
# simply replace all these values with -1 for the moment.
# 
# ❗ Important: Note, this is actually incorrect, but might work for the moment.

# In[5]:


# Simple numerical impute: select numerical data, fill it with -1
imputed_numerical_data = train_data[nan_cols].filter(regex='[^comp\d_(rate|inv)$]')
imputed_numerical_data = imputed_numerical_data.fillna(-1)
train_data.update(imputed_numerical_data)

# Manual cleanup to ensure no problem with space
del imputed_numerical_data
train_data.head(5)


# In[6]:


# Simple naive categorical impute
na_cols = train_data.columns[train_data.isna().any()]
imputed_categorical_data = train_data[na_cols].fillna(-2)
train_data.update(imputed_categorical_data)

# Cleanup
del imputed_categorical_data
train_data.head(5)


# # Initial feature transformation

# In[7]:


# Imports for feature transformation
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# In[8]:


#Onehot encode the categorical variables
oh_encoder = OneHotEncoder()
oh_columns = ['site_id', 'visitor_location_country_id', 'prop_country_id', 'prop_id', 'prop_brand_bool', 'promotion_flag', 
              'srch_destination_id', 'srch_saturday_night_bool', 'random_bool', 'click_bool'
             ]
#todo competitor columns

for column in oh_columns:
    train_data[column]=train_data[column].astype('category')


#encode the numerical values
num_scale_encoder = StandardScaler()
num_scale_columns = ['visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_starrating', 'prop_review_score', 
                     'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 
                     'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count',
                     'srch_room_count', 'srch_query_affinity_score', 'orig_destination_distance' 
                    ]

#we do a preselection of columns that we feel will become useful features after encoding
if config.pre_feature_selection == True:
    chosen_columns = ['prop_starrating', 'prop_review_score', 'prop_location_score1', 'prop_location_score2', 
                      'prop_log_historical_price', 'price_usd', 'srch_query_affinity_score',  'promotion_flag']
else:
    chosen_columns = oh_columns + num_scale_columns


chosen_oh_cols = list(set(chosen_columns) & set(oh_columns))
chosen_num_cols = list(set(chosen_columns) & set(num_scale_columns))
df_transformer = ColumnTransformer([
    ('oh', oh_encoder, chosen_oh_cols),
    ('num', num_scale_encoder, chosen_num_cols),
], remainder='drop')

# We fit this transformer on our training data, and transform our training data into this new format
encoded_X = df_transformer.fit_transform(train_data)
new_oh_columns = df_transformer.named_transformers_.oh.get_feature_names(chosen_oh_cols)
encoded_columns = [ *new_oh_columns, *chosen_num_cols]
encoded_df = pd.DataFrame(encoded_X, columns=encoded_columns)
print (encoded_df.shape)


# # Initial model feature selection

# In[ ]:


prediction_target = train_data.pop('booking_bool')
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(encoded_df, prediction_target)
model = SelectFromModel(lsvc, config.feature_selection_dict, prefit=True)
encoded_df_new = model.transform(encoded_df)
print (encoded_df_new.shape)  


# In[ ]:


print ('hey')


# In[ ]:





# In[ ]:




