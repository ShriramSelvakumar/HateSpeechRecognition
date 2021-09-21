import pandas as pd
import os
import datetime
import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
from HateSpeechNLP import HateSpeechNLP
import pickle


# Setting display width for panda outputs
pd.set_option('display.max_colwidth', 200)
# Constants
data_path = '../Data/'
model_path = '../Models/'
input_features = ['text', 'cleaned_stemmed_text', 'length', 'length_original_tokens', 'length_original_text',
                  'number_non_words']
output_features = ['final_label', 'binary_label']

# Import HateSpeech DataFrame
try:
    data_Hate = pd.read_pickle(data_path + 'HateSpeech_DataFrame_19-09-2021_12-57-00.pkl').loc[:, input_features +
                                                                                                output_features]
except FileNotFoundError:
    # Import HS_Data
    data_Hate_HS = pd.read_csv(data_path + 'HS_DATA_NEW_TRAIN.csv', sep=',')
    hs_NLP = HateSpeechNLP(data_Hate_HS, save=True, default_name=False, features=input_features+output_features)
    data_Hate = hs_NLP.fit_transform()


def clean_text(text):
    return text.split()


def train(X, y, X_valid, y_valid):
    # Vectorizer and Scaler
    tfidf_vectorizer = TfidfVectorizer(analyzer=clean_text, ngram_range=(1, 1))
    standard_scaler = StandardScaler()

    # Vectorization of text data into numerical data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X.cleaned_stemmed_text)
    X_valid_tfidf = tfidf_vectorizer.transform(X_valid.cleaned_stemmed_text)
    # Scaling integer features
    X_train_scaled = standard_scaler.fit_transform(X.loc[:, ['length', 'length_original_tokens',
                                                             'length_original_text', 'number_non_words']])
    X_valid_scaled = standard_scaler.transform(X_valid.loc[:, ['length', 'length_original_tokens',
                                                               'length_original_text', 'number_non_words']])
    # Saving tfidf and scaler objects to use the same when testing
    save_tfidf_scaler(tfidf_vectorizer, standard_scaler)

    # Combining tfidf and scaler outputs into single dataset - for training
    X_train_features = pd.concat([pd.DataFrame(X_train_scaled, columns=['length', 'length_original_tokens',
                                                                        'length_original_text', 'number_non_words']),
                                  pd.DataFrame(X_train_tfidf.toarray())], axis=1)
    X_valid_features = pd.concat([pd.DataFrame(X_valid_scaled, columns=['length', 'length_original_tokens',
                                                                        'length_original_text', 'number_non_words']),
                                  pd.DataFrame(X_valid_tfidf.toarray())], axis=1)

    # Keras model
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=X_train_features.shape))
    model.add(keras.layers.Dense(5, activation="relu"))
    model.add(keras.layers.Dense(5, activation="relu"))
    model.add(keras.layers.Dense(4, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(X_train_features, y['final_label'], epochs=30, validation_data=(X_valid_features,
                                                                                        y_valid['final_label']))

    return history, model


def test(X, y):
    # Load trained model
    try:
        trained_tfidf_vocabulary = pickle.load(open(model_path + "TFIDF-Vocabulary-NN_02-09-2021_21-26-27.pkl", "rb"))
        trained_scaler = pickle.load(open(model_path + "StandardScaler-NN_02-09-2021_21-26-27.pkl", "rb"))
        trained_NN_model = keras.models.load_model(model_path + 'Keras-Model_' + '.h5')

        tfidf_vectorizer = TfidfVectorizer(analyzer=clean_text, vocabulary=trained_tfidf_vocabulary,
                                           ngram_range=(1, 1))
        X_test_tfidf = tfidf_vectorizer.fit_transform(X.cleaned_stemmed_text)
        X_test_scaled = trained_scaler.transform(X.loc[:, ['length', 'length_original_tokens', 'length_original_text',
                                                           'number_non_words']])
        X_test_features = pd.concat([pd.DataFrame(X_test_scaled, columns=['length', 'length_original_tokens',
                                                                          'length_original_text', 'number_non_words']),
                                     pd.DataFrame(X_test_tfidf.toarray())], axis=1)
        y_pred = trained_NN_model.predict_classes(X_test_features)
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
        print("")
        plot_confusion_matrix(trained_NN_model, X_test_features, y)
        plt.show()
        return y_pred

    except FileNotFoundError:
        print('Run train method before test method')


def save_tfidf_scaler(tfidf, scaler):
    os.makedirs(model_path, exist_ok=True)
    pickle.dump(tfidf.vocabulary_, open(model_path + 'TFIDF-Vocabulary-Keras_' +
                                        datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.pkl', "wb"))
    print('Saved TFIDF-Keras to Pickle')
    pickle.dump(scaler, open(model_path + 'StandardScaler-Keras_' +
                             datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.pkl', "wb"))
    print('Saved Scaler-Keras to Pickle')
    return


def save_NN_model(keras_model):
    os.makedirs(model_path, exist_ok=True)
    keras_model.save(model_path + 'Keras-Model_' + datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.h5')
    print('Saved Keras-Model to Pickle')
    return


X_train, X_val, y_train, y_val = train_test_split(data_Hate.loc[:, input_features], data_Hate.loc[:, output_features],
                                                  random_state=42, test_size=0.1, stratify=data_Hate.final_label)
# Resetting index for train and val sets
X_train.reset_index(drop=True, inplace=True)
X_val.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_val.reset_index(drop=True, inplace=True)

# Train Keras Sequential model
seq_history, seq_model = train(X_train, y_train, X_val, y_val)
pd.DataFrame(seq_history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()




















