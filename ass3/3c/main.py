# %%
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_validate, train_test_split
import html
import pandas as pd

df = pd.read_csv(
    'data/SmsCollection.csv',
    sep=';',
    usecols=range(2),
    names=['label', 'text']
)[1:]

df['label'] = df['label'].astype('category')

# %%
import nltk
# Get unique word count
all_text = df['text'].str.lower().str.cat(sep=' ')
words = nltk.tokenize.word_tokenize(all_text)
word_dist = nltk.FreqDist(words)
word_dist.most_common(100)

# %%
# Transform target first
labeler = LabelEncoder()
y = labeler.fit_transform(df['label'])
# %%
import re
# Possible transformations: count | tf-idf
transformation = 'count'
map_integers = False
df_x = df['text']
df_x = df_x.fillna('UNKNOWN')
tfidf_encoder = TfidfVectorizer()
count_encoder = TfidfVectorizer()

if transformation == 'count':
    encoder = count_encoder
elif transformation == 'tf-idf':
    encoder = tfidf_encoder
else:
    encoder = count_encoder

# Thanks to stackoverflow
def any_curr(s, curr="¥$€£"):
    return any(c in s for c in curr)

def map_str_to_NUMBER_token(row):
    result = []

    for word in nltk.tokenize.word_tokenize(row):
        if word.isdigit():
            result.append('NUMB')
        if any_curr(word):
            result.append('MONEY')
        else:
            result.append(word)

    return ' '.join(result)

if map_integers:
    df_x = df_x.apply(map_str_to_NUMBER_token)

x = encoder.fit_transform(df_x)

# Naive approach
model = DecisionTreeClassifier()

cv_performances = cross_validate(model, x, y, scoring=['balanced_accuracy', 'accuracy', 'precision', 'recall'])
bal_acc = cv_performances['test_balanced_accuracy'].mean()
acc = cv_performances['test_accuracy'].mean()
pr = cv_performances['test_precision'].mean()
rc = cv_performances['test_recall'].mean()

model.fit(x, y)

print(f"Method: {transformation}, model: {type(model).__name__}, maps_integer: {map_integers}\n"
      f"\t - balanced_accuracy: {bal_acc:.3f}, accuracy: {acc:.3f}, precision: {pr:.3f}, rc: {rc:.3f}")

# %%
import os
# 4825 ham, 747 spam
df['label'].value_counts()

from sklearn.tree import plot_tree, export_graphviz
path_to_tree = 'results/tree.dot'
os.makedirs(os.path.dirname(path_to_tree), exist_ok=True)

export_graphviz(
    model,
    feature_names=encoder.get_feature_names(),
    class_names=df['label'].cat.categories,
    filled=True,
    out_file='results/tree.dot'
)

# %%
# Try: Putting all numerical features into 1 column: numerical
encoder.get_feature_names()

# %%
# Could get all monetary values as well
# Could count
