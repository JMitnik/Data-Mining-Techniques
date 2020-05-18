from dataclasses import dataclass
from typing import Any

@dataclass
class Config:
    nrows: int
    pre_feature_selection: bool
    train_data_subset: int
    classifier: any
    classifier_dict: dict
    feature_selection: any
    feature_selection_dict: dict
    dimensionality_reduc: bool
    dimension_features: int
    feature_engineering: bool
    naive_imputing: bool
