from dataclasses import dataclass, asdict, field
from typing import Any, List

@dataclass
class Config:
    label: str
    nrows: int
    pre_feature_selection: bool
    algo_feature_selection: bool
    train_data_subset: int
    classifier: any
    classifier_dict: dict
    feature_selection: any
    feature_selection_dict: dict
    dimensionality_reduc_selection: bool
    dimension_features: int
    feature_engineering: bool
    naive_imputing: bool
    valid_size: float
    pre_selection_cols: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)
