from dataclasses import dataclass

@dataclass
class Config:
    pre_feature_selection: bool
    train_data_subset: int
    #classifier: object
    #feature_selection: object
    feature_selection_dict: dict
