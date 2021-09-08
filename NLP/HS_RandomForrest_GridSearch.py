import pandas as pd
from HateSpeechNLP import HateSpeechNLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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
    data_Hate_HS = pd.read_csv('../Data/HS_DATA.csv', sep=',', index_col=0)
    hs_NLP = HateSpeechNLP(data_Hate_HS, save=True, default_name=True)
    data_Hate = hs_NLP.fit_transform()


def clean_text(text):
    return text.split()


# Method to train RF model
def train(X, y, X_te, y_te, grid):
    # Vectorizer and Scaler
    tfidf_vectorizer = TfidfVectorizer(analyzer=clean_text)
    standard_scaler = StandardScaler()

    # Vectorization of text data into numerical data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X.cleaned_stemmed_text)
    # Scaling length, number_non_words features
    X_train_scaled = standard_scaler.fit_transform(X.loc[:, ['length', 'number_non_words']])

    # Combining tfidf and scaler outputs into single dataset - for training
    X_train_features = pd.concat([pd.DataFrame(X_train_scaled, columns=['length', 'number_non_words']),
                                  pd.DataFrame(X_train_tfidf.toarray())], axis=1)

    X_test_tfidf = tfidf_vectorizer.transform(X_te.cleaned_stemmed_text)
    X_test_scaled = standard_scaler.transform(X_te.loc[:, ['length', 'number_non_words']])
    X_test_features = pd.concat([pd.DataFrame(X_test_scaled, columns=['length', 'number_non_words']),
                                 pd.DataFrame(X_test_tfidf.toarray())], axis=1)

    models = []
    performance = []
    for estimator in grid['n_estimators']:
        for split in grid['min_samples_split']:
            print("-------------------------------------------------------------------------------------")
            print('Training Model : n_estimators={0} max_features={1} bootstrap={2} min_sample_split={3}'.format(
                estimator, 'auto', 'True', split))
            model = RandomForestClassifier(n_estimators=estimator, max_depth=None, min_samples_split=split, n_jobs=-1)
            model.fit(X_train_features, y)
            y_pred = model.predict(X_test_features)
            precision = precision_score(y_te, y_pred, average="micro")
            recall = recall_score(y_te, y_pred, average='micro')
            print("Precision : {0} ------ Recall : {1}".format(precision, recall))
            performance.append([precision, recall])
            models.append(model)
    return models, performance


def test(X, y, model):
    # Load trained model
    try:
        trained_tfidf_vocabulary = pickle.load(open(path + "TFIDF-Vocabulary-RF_02-09-2021_16-50-27.pkl", "rb"))
        trained_scaler = pickle.load(open(path + "StandardScaler-RF_02-09-2021_16-50-27.pkl", "rb"))

        tfidf_vectorizer = TfidfVectorizer(analyzer=clean_text, vocabulary=trained_tfidf_vocabulary)
        X_test_tfidf = tfidf_vectorizer.fit_transform(X.cleaned_stemmed_text)
        X_test_scaled = trained_scaler.transform(X.loc[:, ['length', 'number_non_words']])
        X_test_features = pd.concat([pd.DataFrame(X_test_scaled, columns=['length', 'number_non_words']),
                                     pd.DataFrame(X_test_tfidf.toarray())], axis=1)
        y_pred = model.predict(X_test_features)
        print("Precision : ", precision_score(y, y_pred, average="micro"))
        print("Recall : ", recall_score(y, y_pred, average='micro'))
        print(confusion_matrix(y, y_pred))
        return y_pred

    except FileNotFoundError:
        print('Run train method before test method')


def save_tfidf_scaler(tfidf, scaler):
    os.makedirs(path, exist_ok=True)
    pickle.dump(tfidf.vocabulary_, open(path + 'TFIDF-Vocabulary-RF_' +
                                        datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.pkl', "wb"))
    print('Saved TFIDF-RF to Pickle')
    pickle.dump(scaler, open(path + 'StandardScaler-RF_' +
                             datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.pkl', "wb"))
    print('Saved Scaler-RF to Pickle')
    return


def save_RF_model(model):
    os.makedirs(path, exist_ok=True)
    pickle.dump(model, open(path + 'RF-GridSearchCV-Model_' +
                            datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.pkl', "wb"))
    print('Saved GridSearchCV-RF-Model to Pickle')
    return


# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data_Hate.loc[:, ['length', 'number_non_words',
                                                                      'cleaned_stemmed_text']],
                                                    data_Hate.final_label, random_state=42, test_size=0.2)
# Resetting index for train and test sets
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# Hyper parameters for RandomForrest model
n_estimators = [10, 30, 50, 70]
max_depth = [None]
min_samples_split = [2, 5]
max_features = ['auto', 'sqrt']
bootstrap = [True, False]

# Random grid
rf_grid = {'n_estimators': n_estimators,
           'max_depth': max_depth,
           'min_samples_split': min_samples_split,
           'max_features': max_features,
           'bootstrap': bootstrap}

# Creating Grid Search RF object
# gs_rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=rf_grid, n_jobs=5, cv=2, verbose=3)

# Calling training method to start training RF model
gs_models, gs_score = train(X_train, y_train, X_test, y_test, rf_grid)

# Calling test method to test the accuracy of the trained RF model
# pred = test(X_test, y_test, gs_model.best_estimator_)
