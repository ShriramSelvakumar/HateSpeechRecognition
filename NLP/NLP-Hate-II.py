# Processing Hate Data - Part 2
# Separating into two parts for more readability & less processing
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Setting display width for panda outputs
from py.path import local
from sympy.abc import alpha

pd.set_option('display.max_colwidth', 200)
# PorterStemmer
ps = nltk.PorterStemmer()


def stem_words(text):
    return_text = [ps.stem(word) for word in text]
    return return_text


def finding_length(text):
    return len(text)


def finding_non_words(text):
    regex = re.compile(r'[\d][\d][\d][\d][\d][\d]')
    sum_non_words = sum([1 for word in text if word in regex.findall(word)])
    regex = re.compile(r'[\W]')
    sum_non_words = sum_non_words + sum([1 for word in text if word in regex.findall(word)])
    return sum_non_words


def create_clean_text(text):
    return " ".join(text)


def clean_text(text):
    return text.split()


def save_to_pickle(file_name="Data-Hate-Stemmed-DF"):
    data_Hate.to_pickle('../Data/'+file_name+'.pkl')
    print('Saved clean data to Pickle')
    return


# Loading pickle file
def load_pickle():
    data = pd.read_pickle('../Data/Data-Hate-Stemmed-DF.pkl')
    return data


# Importing Pickle file created in other file
# data_Hate = pd.read_pickle('../Data/Data-Hate-DF.pkl')
# Finding non words
# words = set(nltk.corpus.words.words())
# test = " ".join(w for w in data_Hate.text_no_stop_words[0] if w in words or not w.isalpha())

# Stemming words
# data_Hate['cleaned_text_tokens'] = data_Hate['text_no_stop_words_tokens'].apply(lambda x: stem_words(x))

# Save dataframe to pickle file
# save_to_pickle()

# Load pickle file
data_Hate = load_pickle()
# Create clean text attribute/feature
data_Hate['cleaned_text'] = data_Hate['cleaned_text_tokens'].apply(lambda x: create_clean_text(x))
# Find length of non stemmed tokens and create it as new attribute/feature
data_Hate['length'] = data_Hate['text_tokens'].apply(lambda x: finding_length(x))
# Find number of non words and create it as new attribute/feature
data_Hate['number_non_words'] = data_Hate['cleaned_text_tokens'].apply(lambda x: finding_non_words(x))

tfidf_vectorizer = TfidfVectorizer(analyzer=clean_text)
X_tfidf = tfidf_vectorizer.fit_transform(data_Hate['cleaned_text'])

X_features = pd.concat([data_Hate['length'], data_Hate['number_non_words'], pd.DataFrame(X_tfidf.toarray())], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_features, data_Hate['final_label'], test_size=0.2)

random_forest = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1)
rf_model = random_forest.fit(X_train, y_train)

# Visualizing distribution of data
# fig = plt.figure(figsize=(8, 6))
# data_Hate.groupby('final_label').cleaned_text.count().plot.bar(ylim=0)
# plt.show()

y_pred = rf_model.predict(X_test)
precision, recall, fscore, support = score(y_test, y_pred, average='micro')












