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

all_numerical_columns = [
    'prop_mean_score', 'price_usd', 'visitor_hist_adr_usd',
        'prop_starrating', 'prop_review_score', 'prop_location_score1', 'prop_location_score2',
        'prop_log_historical_price',
        'srch_length_of_stay', 'srch_booking_window',
        'srch_adults_count', 'srch_children_count',
        'srch_room_count', 'srch_query_affinity_score',
        'orig_destination_distance']

all_categorical = ['date_time', 'site_id', 'visitor_location_country_id', 'prop_country_id',
    'prop_id', 'prop_brand_bool', 'promotion_flag',
    'srch_destination_id', 'srch_saturday_night_bool', 'random_bool',
]

all_cat_without_prop_and_datetime = [
    col for col in all_categorical if 'prop_id' not in col and 'date_time' not in col
]

best_numerical_columns = [
    'prop_mean_score', 'price_usd', 'prop_log_historical_price', 'prop_starrating', 'prop_review_score',
    'srch_query_affinity_score', 'orig_destination_distance', 'srch_booking_window'
]

categorical_somewhat_high_freq = ['srch_destination_id']

categorical_mid_freq = [
    'visitor_location_country_id', 'prop_country_id', 'site_id'
]

categorical_low_freq = [
    'random_bool', 'srch_saturday_night_bool', 'promotion_flag', 'prop_brand_bool'
]

all_numerical_config = Config(
    label='AllNumericalRun',
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
    pre_selection_cols=[*all_numerical_columns],
    dimension_features=25,
    feature_engineering=True,
    path_to_eval_results='results/eval_results_2.csv',
    naive_imputing=True #todo faster method for averaging nan values if naive=False
)

all_numerical_early_nana_removal_config = Config(
    label='AllNumericalEarlyNANRemoveRun',
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
    pre_selection_cols=[*all_numerical_columns],
    dimension_features=25,
    feature_engineering=True,
    remove_null_features_early=True,
    path_to_eval_results='results/eval_results_2.csv',
    naive_imputing=True #todo faster method for averaging nan values if naive=False
)

all_numerical_and_categorical_early_nana_removal_config = Config(
    label='AllNumericalAndAllCategoricalEarlyNANRemoveRun',
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
    pre_selection_cols=[*all_numerical_columns, *all_categorical],
    dimension_features=25,
    feature_engineering=True,
    remove_null_features_early=True,
    path_to_eval_results='results/eval_results_2.csv',
    naive_imputing=True #todo faster method for averaging nan values if naive=False
)

no_engineer_all_numerical_and_nocrazycategorical_early_nana_removal_config = Config(
    label='NoEngineeringV2AllNumAndNoCrazyCategoricalEarlyNANRemoveRun',
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
    pre_selection_cols=[*all_numerical_columns, *all_cat_without_prop_and_datetime],
    dimension_features=25,
    feature_engineering=False,
    remove_null_features_early=True,
    path_to_eval_results='results/eval_results_2.csv',
    naive_imputing=True #todo faster method for averaging nan values if naive=False
)

no_engineer_all_numerical_and_early_nana_removal_config = Config(
    label='NoEngineeringAllNumericalEarlyAndRandomBoolNANRemoveRun',
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
    pre_selection_cols=[*all_numerical_columns, 'random_bool'],
    dimension_features=25,
    feature_engineering=False,
    remove_null_features_early=True,
    path_to_eval_results='results/eval_results_2.csv',
    naive_imputing=True #todo faster method for averaging nan values if naive=False
)

pca25_no_engineer_all_numerical_and_early_nana_removal_config = Config(
    label='PCA25NoEngineeringAllNumericalAndNoCrazyCategoricalEarlyNANRemoveRun',
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
    dimensionality_reduc_selection=True,
    pre_selection_cols=[*all_numerical_columns, *all_cat_without_prop_and_datetime],
    dimension_features=9,
    feature_engineering=False,
    remove_null_features_early=True,
    path_to_eval_results='results/eval_results_2.csv',
    naive_imputing=True #todo faster method for averaging nan values if naive=False
)

pca25_no_engineer_all_numerical_and_early_nana_removal_config = Config(
    label='PCA3NoEngineeringAllNumericalAndNoCrazyCategoricalEarlyNANRemoveRun',
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
    dimensionality_reduc_selection=True,
    pre_selection_cols=[*all_numerical_columns, *all_cat_without_prop_and_datetime],
    dimension_features=9,
    feature_engineering=False,
    remove_null_features_early=True,
    path_to_eval_results='results/eval_results_2.csv',
    naive_imputing=True #todo faster method for averaging nan values if naive=False
)

pca25_no_engineer_all_numerical_and_early_nana_removal_config = Config(
    label='PCA25NoEngineeringAllNumericalAndNoCrazyCategoricalEarlyNANRemoveRun',
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
    dimensionality_reduc_selection=True,
    pre_selection_cols=[*all_numerical_columns, *all_cat_without_prop_and_datetime],
    dimension_features=25,
    feature_engineering=False,
    remove_null_features_early=True,
    path_to_eval_results='results/eval_results_2.csv',
    naive_imputing=True #todo faster method for averaging nan values if naive=False
)

pca4_no_engineer_all_numerical_and_early_nana_removal_config = Config(
    label='PCA4NoEngineeringAllNumericalAndNoCrazyCategoricalEarlyNANRemoveRun',
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
    dimensionality_reduc_selection=True,
    pre_selection_cols=[*all_numerical_columns, *all_cat_without_prop_and_datetime],
    dimension_features=4,
    feature_engineering=False,
    remove_null_features_early=True,
    path_to_eval_results='results/eval_results_2.csv',
    naive_imputing=True #todo faster method for averaging nan values if naive=False
)

no_engineer_all_numerical_and_early_nana_removal_config = Config(
    label='NoEngineeringAllNumericalEarlyAndRandomBoolNANRemoveRun',
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
    pre_selection_cols=[*all_numerical_columns, 'random_bool'],
    dimension_features=25,
    feature_engineering=False,
    remove_null_features_early=True,
    path_to_eval_results='results/eval_results_2.csv',
    naive_imputing=True #todo faster method for averaging nan values if naive=False
)
