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
import pickle


# Setting display width for panda outputs
# from py.path import local
# from sympy.abc import alpha

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


def train():
    tfidf_vectorizer = TfidfVectorizer(analyzer=clean_text)
    X_tfidf = tfidf_vectorizer.fit_transform(data_Hate['cleaned_text'])
    X_features = pd.concat([data_Hate['length'], data_Hate['number_non_words'], pd.DataFrame(X_tfidf.toarray())],
                           axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_features, data_Hate['final_label'], test_size=0.2)

    random_forest = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1)
    rf_model = random_forest.fit(X_train, y_train)
    print('Training Completed')
    y_pred = rf_model.predict(X_test)
    precision, recall, fscore, support = score(y_test, y_pred, average='micro')
    print('Precision : '+precision)
    # Saving Vocabulary
    pickle.dump(tfidf_vectorizer.vocabulary_, open("../Data/Hate-TFIDF-Vocabulary.pkl", "wb"))
    # Saving RandomForest Model
    pickle.dump(rf_model, open("../Data/Hate-RF-Model.pkl", "wb"))
    return tfidf_vectorizer.vocabulary_, rf_model


try:
    data_Hate = pd.read_pickle('../Data/Data-Hate-TFIDF-DF.pkl')
except FileNotFoundError:
    try:
        data_Hate = pd.read_pickle('../Data/Data-Hate-Stemmed-DF.pkl')
        # Create clean text attribute/feature
    except FileNotFoundError:
        # Importing Pickle file created in other file
        data_Hate = pd.read_pickle('../Data/Data-Hate-DF.pkl')
        # Stemming words
        data_Hate['cleaned_text_tokens'] = data_Hate['text_no_stop_words_tokens'].apply(lambda x: stem_words(x))
        # Save stemmed dataframe to pickle file
        save_to_pickle('Data-Hate-Stemmed-DF')
    # Create clean text attribute/feature
    data_Hate['cleaned_text'] = data_Hate['cleaned_text_tokens'].apply(lambda x: create_clean_text(x))
    # Find length of non stemmed tokens and create it as new attribute/feature
    data_Hate['length'] = data_Hate['text_tokens'].apply(lambda x: finding_length(x))
    # Find number of non words and create it as new attribute/feature
    data_Hate['number_non_words'] = data_Hate['cleaned_text_tokens'].apply(lambda x: finding_non_words(x))
    # Save TFIDF dataframe to pickle file
    save_to_pickle('Data-Hate-TFIDF-DF')

# Visualizing distribution of data
# fig = plt.figure(figsize=(8, 6))
# data_Hate.groupby('final_label').cleaned_text.count().plot.bar(ylim=0)
# plt.show()
try:
    random_forest_model = pickle.load(open("../Data/Hate-RF-Model.pkl", "rb"))
    trained_tfidf_vocabulary = pickle.load(open("../Data/Hate-TFIDF-Vocabulary.pkl", "rb"))
except FileNotFoundError:
    trained_tfidf_vocabulary, random_forest_model = train()

# Testing model with existing data
test_tfidf_vectorizer = TfidfVectorizer(analyzer=clean_text, vocabulary=trained_tfidf_vocabulary)
test_X_tfidf = test_tfidf_vectorizer.fit_transform(data_Hate.iloc[-200:-100, 10])
length = data_Hate.iloc[-200:-100, 11]
non_words = data_Hate.iloc[-200:-100, 12]
real_y = data_Hate.iloc[-200:-100, 1]
test_X_features = pd.concat([length.reset_index(drop=True), non_words.reset_index(drop=True),
                             pd.DataFrame(test_X_tfidf.toarray())], axis=1)
y = random_forest_model.predict(test_X_features)
print('Precision : ', (y == real_y).sum()/len(y))
y = pd.DataFrame(y)

# Testing model with new data
word = pd.DataFrame(['hello world', 'fuck off'])
x_tfidf = test_tfidf_vectorizer.fit_transform(word[0])
x = pd.concat([pd.DataFrame([2, 2], columns=['length']), pd.DataFrame([0, 0], columns=['number_non_words']),
               pd.DataFrame(x_tfidf.toarray())], axis=1)
yy = random_forest_model.predict(x)






