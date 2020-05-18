# Imports
import sklearn as sk
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np

train_data = pd.read_csv('data/training_set_VU_DM.csv',nrows=1000
                        )

original_columns = train_data.columns

time = pd.to_datetime(train_data['date_time'])

train_data['month']=time.dt.month
train_data['year']=time.dt.year
train_data['same_country_visitor_prop']=np.where(train_data['visitor_location_country_id'] == train_data['prop_country_id'],1,0)
train_data['viable_comp']= np.where(
                  (train_data['comp1_rate']== -1)& (train_data['comp1_inv']== 0) |
                  (train_data['comp2_rate']== -1)& (train_data['comp2_inv']== 0) |
                  (train_data['comp3_rate']== -1)& (train_data['comp3_inv']== 0) |
                  (train_data['comp4_rate']== -1)& (train_data['comp4_inv']== 0) |
                  (train_data['comp5_rate']== -1)& (train_data['comp5_inv']== 0) |
                  (train_data['comp6_rate']== -1)& (train_data['comp6_inv']== 0) |
                  (train_data['comp7_rate']== -1)& (train_data['comp7_inv']== 0) |
                  (train_data['comp8_rate']== -1)& (train_data['comp8_inv']== 0) 
                  ,1,0)
#Onehot encode the categorical variables
oh_encoder = OneHotEncoder()
oh_columns = ['site_id', 'visitor_location_country_id', 'prop_country_id', 'viable_comp', 'same_country_visitor_prop','prop_id', 'prop_brand_bool', 'promotion_flag', 
              'srch_destination_id', 'srch_saturday_night_bool', 'random_bool', 'booking_bool', 'click_bool'
             ]
#todo competitor columns

print(train_data['viable_comp'])
for column in oh_columns:
    train_data[column]=train_data[column].astype('category')

mcol=train_data.loc[:,['prop_location_score1', 'prop_location_score2']]
train_data['prop_mean_score'] = mcol.mean(axis=1)


#encode the numerical values
num_scale_encoder = StandardScaler()
num_scale_columns = ['visitor_hist_starrating', 'visitor_hist_adr_usd', 
                     'prop_starrating', 'prop_review_score', 
                     'prop_mean_score',  
                     'prop_log_historical_price', 'price_usd', 
                     'srch_length_of_stay', 'srch_booking_window', 
                     'srch_adults_count', 'srch_children_count',
                     'srch_room_count', 'srch_query_affinity_score', 
                     'orig_destination_distance','year','month'
                    ]
#we do a preselection of columns that we feel will become useful features after encoding
chosen_columns = ['prop_starrating', 'prop_review_score', 'prop_mean_score', 'viable_comp','same_country_visitor_prop' 
                  'prop_log_historical_price', 'price_usd', 'srch_query_affinity_score',  'promotion_flag']







