from dataclasses import dataclass, asdict, field
import json
from typing import Any, List

@dataclass
class Config:
    label: str
    nrows: int
    path_to_eval_results: str
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
    most_important_metric: str = 'ndcg_10'

    def to_dict(self):
        config_dict = asdict(self)
        scalar_types = [bool, int, float, str]

        reformatted_dict = {}

        for key, item in config_dict.items():
            if type(item) == dict or type(item) == list:
                reformatted_dict[key] = json.dumps(item)
            elif type(item) not in scalar_types:
                try:
                    reformatted_dict[key] = item.__name__
                except:
                    reformatted_dict[key] = str(item)
            else:
                reformatted_dict[key] = item

        return reformatted_dict
