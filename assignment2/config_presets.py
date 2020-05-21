from config import Config
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Lasso, RidgeClassifier, LogisticRegression
from sklearn.feature_selection import SelectFromModel, SelectKBest, RFE, mutual_info_classif
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC, SVC
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA, TruncatedSVD

# Numerical-only
numerical_config = Config(
    label='AllNumerical',
    nrows=None,
    valid_size=0.2,
    pre_feature_selection=True,
    algo_feature_selection=False,
    train_data_subset=0.8,
    classifier=None,
    classifier_dict={'C' : 1, 'kernel' : 'rbf', 'random_state' : 2},
    feature_selection=SelectKBest,
    feature_selection_scoring_func=mutual_info_classif,
    feature_selection_dict={'k' : 10},
    dimensionality_reduc_selection=False,
    pre_selection_cols=[
        'visitor_hist_starrating', 'visitor_hist_adr_usd',
        'prop_starrating', 'prop_review_score', 'prop_location_score1', 'prop_location_score2',
        'prop_log_historical_price', 'price_usd',
        'srch_length_of_stay', 'srch_booking_window',
        'srch_adults_count', 'srch_children_count',
        'srch_room_count', 'srch_query_affinity_score',
        'orig_destination_distance'
    ],
    dimension_features=25,
    feature_engineering=True,
    path_to_eval_results='results/eval_results.csv',
    naive_imputing=True #todo faster method for averaging nan values if naive=False
)

# Categorical-no-propId
categorical_no_propid_config = Config(
    label='Categorical-no-propId',
    nrows=None,
    valid_size=0.2,
    pre_feature_selection=True,
    algo_feature_selection=False,
    train_data_subset=0.8,
    classifier=None,
    classifier_dict={'C' : 1, 'kernel' : 'rbf', 'random_state' : 2},
    feature_selection=SelectKBest,
    feature_selection_dict={'threshold' : 1},
    dimensionality_reduc_selection=False,
    pre_selection_cols=[
        'site_id', 'visitor_location_country_id', 'prop_country_id',
        'prop_brand_bool', 'promotion_flag', 'position',
        'srch_destination_id', 'srch_saturday_night_bool', 'random_bool',
        'click_bool', 'booking_bool'
    ],
    dimension_features=25,
    feature_engineering=True,
    path_to_eval_results='results/eval_results.csv',
    naive_imputing=True #todo faster method for averaging nan values if naive=False
)
