from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class Config:
    label: str
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
    valid_size: float

    def to_dict(self):
        return asdict(self)
