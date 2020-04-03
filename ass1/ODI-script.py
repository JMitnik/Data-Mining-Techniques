# %%
import pandas as pd
from preprocessing import transform_ODI_dataset
# %%
# Read ODI dataset and transform it
df = pd.read_csv("data/ODI-2020.csv", sep=";", encoding="utf-8")
df = transform_ODI_dataset(df)
df.head(10)
