import pandas as pd
import numpy as np
import glob
import os
import datetime
import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from HateSpeechNLP import HateSpeechNLP
import pickle
import logging


# Setting display width for panda outputs
pd.set_option('display.max_colwidth', 200)
# Dont show tensorflow messages
tf.get_logger().setLevel(logging.ERROR)
# Constants
data_path = '../Data/'
model_path = '../Models/'
train_file_name = 'HS_DATA_BINARY_TRAIN.csv'
test_file_name = 'HS_DATA_BINARY_TEST.csv'
input_features = ['text', 'cleaned_stemmed_text', 'length', 'length_original_tokens', 'length_original_text',
                  'number_non_words']
output_features = ['final_label', 'binary_label', 'NONE_label', 'HATE_label', 'OFFN_label', 'PRFN_label']

# Import HateSpeech DataFrame
try:
    data_Hate = pd.read_pickle(data_path + 'HateSpeech_DataFrame_30-09-2021_20-14-56.pkl').loc[:, input_features +
                                                                                                  output_features]
except FileNotFoundError:
    # Import HS_Data
    data_Hate_HS = pd.read_csv(data_path + train_file_name, sep=',')
    hs_NLP = HateSpeechNLP(data_Hate_HS, save=True, default_name=False, features=input_features+output_features)
    data_Hate = hs_NLP.fit_transform()


def clean_text(text):
    return text.split()


def train(X, y, X_valid, y_valid, save_name):
    # Vectorizer and Scaler
    tfidf_vectorizer = TfidfVectorizer(analyzer=clean_text, ngram_range=(1, 1))

    # Vectorization of text data into numerical data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X.cleaned_stemmed_text)
    X_valid_tfidf = tfidf_vectorizer.transform(X_valid.cleaned_stemmed_text)
    # Saving tfidf and scaler objects to use the same when testing
    save_tfidf(tfidf_vectorizer)

    X_train_features = X_train_tfidf.toarray()
    X_valid_features = X_valid_tfidf.toarray()

    # Computing Class Weights
    y_classes = list(y.unique())
    y_classes.sort()
    class_weights_list = class_weight.compute_class_weight('balanced',  y_classes, y)
    # Calculated class weights - [3.14126016, 0.89438657, 0.43002226, 4.19972826] - assuming for [Hate, None, Off, Prf]
    class_weights = {}
    for i in range(0, len(class_weights_list)):
        class_weights[i] = class_weights_list[i]

    # Implementing Callbacks - Saving checkpoints & Early Stopping
    checkpoint_cb = keras.callbacks.ModelCheckpoint(model_path + 'Keras_' + save_name +
                                                    '-{epoch:02d}-{val_accuracy:.3f}' + ".h5", save_best_only=False,
                                                    monitor='val_accuracy')
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    # Keras model
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=X_train_features.shape[1:]))
    model.add(keras.layers.Dense(16, activation="relu"))
    model.add(keras.layers.Dense(16, activation="relu"))
    model.add(keras.layers.Dense(16, activation="relu"))
    model.add(keras.layers.Dense(16, activation="relu"))
    model.add(keras.layers.Dense(16, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(X_train_features, y, epochs=30, validation_data=(X_valid_features, y_valid),
                        callbacks=[checkpoint_cb, early_stopping_cb], class_weight=class_weights)
    save_NN_model(model, save_name)
    return history, model


def test(X, y, save_name):
    # Load trained model
    try:
        tfidf_path = model_path + 'TFIDF/'
        file_type = '\*pkl'
        tfidf_files = glob.glob(tfidf_path + file_type)
        try:
            tfidf_file = max(tfidf_files, key=os.path.getctime)
        except ValueError:
            # Added this exception because try statements are throwing errors in Linux
            tfidf_file = model_path + 'TFIDF/' + 'TFIDF-Vocabulary-Keras_21-09-2021_18-46-42.pkl'

        trained_tfidf_vocabulary = pickle.load(open(tfidf_file, "rb"))
        # trained_NN_model = keras.models.load_model(model_path + 'Keras_' + save_name + '.h5')
        trained_NN_model = keras.models.load_model(model_path + 'Keras_HATE_label-01-0.92' + '.h5')
        print("Loaded TFIDF Model -", tfidf_file)

        tfidf_vectorizer = TfidfVectorizer(analyzer=clean_text, vocabulary=trained_tfidf_vocabulary,
                                           ngram_range=(1, 1))
        X_test_tfidf = tfidf_vectorizer.fit_transform(X.cleaned_stemmed_text)
        X_test_features = X_test_tfidf.toarray()
        y_pred = (trained_NN_model.predict(X_test_features) > 0.5).astype("int32")

        # y_pred = trained_NN_model.predict_classes(X_test_features)
        print('Binary Values -----')
        print("Precision : ", precision_score(y, y_pred, average="binary"))
        print("Recall : ", recall_score(y, y_pred, average='binary'))
        print('Confusion Matrix -----')
        print(confusion_matrix(y, y_pred))
        return y_pred

    except FileNotFoundError:
        print('Run train method before test method')


def save_tfidf(tfidf):
    os.makedirs(model_path + 'TFIDF/', exist_ok=True)
    pickle.dump(tfidf.vocabulary_, open(model_path + 'TFIDF/' + 'TFIDF-Vocabulary-Keras_' +
                                        datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.pkl', "wb"))
    print('Saved TFIDF-Keras to Pickle')
    return


def save_NN_model(keras_model, save_name):
    os.makedirs(model_path, exist_ok=True)
    keras_model.save(model_path + 'Keras-Model_' + save_name +
                     datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.h5')
    print('Saved Keras-Model to Pickle')
    return


X_train, X_val, y_train, y_val = train_test_split(data_Hate.loc[:, input_features], data_Hate.loc[:, output_features],
                                                  random_state=42, test_size=0.1, stratify=data_Hate.final_label)
# Resetting index for train and val sets
X_train.reset_index(drop=True, inplace=True)
X_val.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_val.reset_index(drop=True, inplace=True)


X_train = pd.concat([X_train, y_train], axis=1)
for i in range(0, 1):
    X_train = pd.concat([X_train, X_train[X_train['HATE_label'] == 1], X_train[X_train['PRFN_label'] == 1]])
    X_train.reset_index(drop=True, inplace=True)


y_train = X_train.loc[:, output_features]
X_train = X_train.loc[:, input_features]

# label_encoder = LabelEncoder()
# label_encoder.fit(['NONE', 'PRFN', 'OFFN', 'HATE'], )
# y_train['final_label_int'] = label_encoder.transform(y_train['final_label'])
# y_val['final_label_int'] = label_encoder.transform(y_val['final_label'])

# binary_labels = ['NONE_label', 'HATE_label', 'OFFN_label', 'PRFN_label']
binary_labels = ['HATE_label']
# Train Keras Sequential model
print("==================================================================================================")
print("Training Starts")
for label in binary_labels:
    print("Training", label)
    seq_history, seq_model = train(X_train, y_train[label], X_val, y_val[label], label)
print("Training Ends")
print("==================================================================================================")

# Testing the validation set
# print("==================================================================================================")
# print("Evaluating Validation Set")
# for label in binary_labels:
#     test(X_val, y_val[label], label)
#     print('---------------------------------------------')
#
# # Testing the test set
# # Importing HS_DATA - Test set
# print("==================================================================================================")
# data_Hate_Test = pd.read_csv(data_path + test_file_name, sep=',')
# hs_NLP = HateSpeechNLP(data_Hate_Test, save=False, default_name=False, features=input_features+output_features)
# data_Test = hs_NLP.fit_transform()
# print("Evaluating Test Data")
# for label in binary_labels:
#     y_test = test(data_Test, data_Test[label], label)
#     print('---------------------------------------------')
# print("==================================================================================================")


















