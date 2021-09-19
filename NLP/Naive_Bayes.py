import pandas as pd
from HateSpeechNLP import HateSpeechNLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import os
import datetime
import pickle


path = '../Data/'

# Import HateSpeech DataFrame
try:
    data_Hate = pd.read_pickle('../Data/Data-Hate-Stemmed-DF.pkl')
except FileNotFoundError:
    # Import HS_Data
    data_Hate_HS = pd.read_csv('../Data/HS_DATA_TRAIN.csv', sep=',')
    hs_NLP = HateSpeechNLP(data_Hate_HS, save=True, default_name=True)
    data_Hate = hs_NLP.fit_transform()


def clean_text(text):
    return text.split()


# Method to train RF model
def train(X, y, model):
    # Vectorizer and Scaler
    tfidf_vectorizer = TfidfVectorizer(analyzer=clean_text)
    standard_scaler = StandardScaler()

    # Vectorization of text data into numerical data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X.cleaned_stemmed_text)
    # Scaling length, number_non_words features
    X_train_scaled = standard_scaler.fit_transform(X.loc[:, ['length', 'number_non_words']])
    # Saving tfidf and scaler objects to use the same when testing
    save_tfidf_scaler(tfidf_vectorizer, standard_scaler)

    # Combining tfidf and scaler outputs into single dataset - for training
    X_train_features = pd.concat([pd.DataFrame(X_train_scaled, columns=['length', 'number_non_words']),
                                  pd.DataFrame(X_train_tfidf.toarray())], axis=1)
    model.fit(X_train_features, y)
    save_RF_model(model)
    return model


def test(X, y):
    # Load trained model
    try:
        trained_tfidf_vocabulary = pickle.load(open(path + "xx", "rb"))
        trained_scaler = pickle.load(open(path + "xx", "rb"))
        trained_NB_model = pickle.load(open(path + "xx", "rb"))

        tfidf_vectorizer = TfidfVectorizer(analyzer=clean_text, vocabulary=trained_tfidf_vocabulary)
        X_test_tfidf = tfidf_vectorizer.fit_transform(X.cleaned_stemmed_text)
        X_test_scaled = trained_scaler.transform(X.loc[:, ['length', 'number_non_words']])
        X_test_features = pd.concat([pd.DataFrame(X_test_scaled, columns=['length', 'number_non_words']),
                                     pd.DataFrame(X_test_tfidf.toarray())], axis=1)
        y_pred = trained_NB_model.predict(X_test_features)
        print('Micro Values -----')
        print("Precision : ", precision_score(y, y_pred, average="micro"))
        print("Recall : ", recall_score(y, y_pred, average='micro'))
        print('Macro Values -----')
        print("Precision : ", precision_score(y, y_pred, average="macro"))
        print("Recall : ", recall_score(y, y_pred, average='macro'))
        print('Weighted Values -----')
        print("Precision : ", precision_score(y, y_pred, average="weighted"))
        print("Recall : ", recall_score(y, y_pred, average='weighted'))
        print('Confusion Matrix -----')
        print(confusion_matrix(y, y_pred))
        return y_pred

    except FileNotFoundError:
        print('Run train method before test method')


def save_tfidf_scaler(tfidf, scaler):
    os.makedirs(path, exist_ok=True)
    pickle.dump(tfidf.vocabulary_, open(path + 'TFIDF-Vocabulary-NB_' +
                                        datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.pkl', "wb"))
    print('Saved TFIDF-NB to Pickle')
    pickle.dump(scaler, open(path + 'StandardScaler-NB_' +
                             datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.pkl', "wb"))
    print('Saved Scaler-NB to Pickle')
    return


def save_RF_model(model):
    os.makedirs(path, exist_ok=True)
    pickle.dump(model, open(path + 'NB-Model_' +
                            datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.pkl', "wb"))
    print('Saved NB-Model to Pickle')
    return


# Splitting data into train and test sets
X_train, X_val, y_train, y_val = train_test_split(data_Hate.loc[:, ['length', 'number_non_words',
                                                                    'cleaned_stemmed_text']], data_Hate.final_label,
                                                  random_state=42, test_size=0.1, stratify=data_Hate.final_label)
# Resetting index for train and test sets
X_train.reset_index(drop=True, inplace=True)
X_val.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_val.reset_index(drop=True, inplace=True)

# Creating RF object
nb_clf = MultinomialNB()

# Calling training method to start training RF model
train(X_train, y_train, nb_clf)

# Calling test method to test the accuracy of the trained RF model
print('Test Result on VAL set')
pred_val = test(X_val, y_val)

print('Test Result on TEST set')
test_data = pd.read_csv(path + 'HS_DATA_TEST.csv', sep=',')
hs_test_NLP = HateSpeechNLP(test_data)
data_test = hs_test_NLP.fit_transform()
pred_test = test(data_test.loc[:, ['length', 'number_non_words', 'cleaned_stemmed_text']], data_test.final_label)












