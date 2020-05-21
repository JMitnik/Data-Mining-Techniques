# %%
from config import Config
import pandas as pd

df = pd.read_csv('results/eval_results.csv')


import json
feature_importances = json.loads(df.iloc[0].mutable_feature_importances_from_learner)

sorted_df = df.sort_values(by='valid_ndcg_5', ascending=False)
sorted_df.head()
# %%
