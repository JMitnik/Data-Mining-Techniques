# %%
import pandas as pd
from scipy.sparse import find
# Script for notebooks to reload dependency
import importlib
import preprocessing
importlib.reload(preprocessing)

from preprocessing import transform_ODI_dataset, make_preprocessing_pipeline
# Read ODI dataset and transform it
df = pd.read_csv("data/ODI-2020.csv", sep=";", encoding="utf-8")
df = transform_ODI_dataset(df)


# %%
# # Create Preprocessing Pipeline for prediction
# preprocesser = make_preprocessing_pipeline()
# inp = preprocesser.fit_transform(df)
# inp.shape



# %%
