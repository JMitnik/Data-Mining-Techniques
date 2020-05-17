#imports
import numpy as np



train_data['same_country_visitor_prop']=np.where(train_data['visitor_location_country_id'] == train_data['prop_country_id'],1,0)

#competitor combine nog niet aan de praat gekregen
ccol=train_data.loc[:,['comp1_rate',
                       'comp2_rate',
                       'comp3_rate',
                       'comp4_rate',
                       'comp5_rate',
                       'comp6_rate',
                       'comp7_rate',
                       'comp8_rate']]
icol=train_data.loc[:,['comp1_inv',
                       'comp2_inv',
                       'comp3_inv',
                       'comp4_inv',
                       'comp5_inv',
                       'comp6_inv',
                       'comp7_inv',
                       'comp8_inv']]

train_data['viable_comp'] = np.where(ccol== -1 & icol == 0,1,0)

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
                     'orig_destination_distance' 
                    ]
#we do a preselection of columns that we feel will become useful features after encoding
chosen_columns = ['prop_starrating', 'prop_review_score', 'prop_mean_score', 'viable_comp','same_country_visitor_prop' 
                  'prop_log_historical_price', 'price_usd', 'srch_query_affinity_score',  'promotion_flag']







