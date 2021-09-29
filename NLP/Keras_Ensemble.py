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
    data_Hate = pd.read_pickle(data_path + 'HateSpeech_DataFrame_21-09-2021_15-25-53.pkl').loc[:, input_features +
                                                                                                  output_features]
except FileNotFoundError:
    # Import HS_Data
    data_Hate_HS = pd.read_csv(data_path + 'HS_DATA_NEW_TRAIN.csv', sep=',')
    hs_NLP = HateSpeechNLP(data_Hate_HS, save=True, default_name=False, features=input_features+output_features)
    data_Hate = hs_NLP.fit_transform()


def clean_text(text):
    return text.split()


def prepare_input_one_gram(X):
    tfidf_file = model_path + 'TFIDF/' + 'TFIDF-Vocabulary-Keras_23-09-2021_20-48-04.pkl'
    standard_scaler_file = model_path + 'StandardScaler/' + 'StandardScaler-Keras_23-09-2021_20-48-04.pkl'
    print("Loaded TFIDF Model -", tfidf_file)
    print("Loaded StandardScaler Model -", standard_scaler_file)
    trained_tfidf_vocabulary = pickle.load(open(tfidf_file, "rb"))
    trained_scaler = pickle.load(open(standard_scaler_file, "rb"))

    tfidf_vectorizer = TfidfVectorizer(analyzer=clean_text, vocabulary=trained_tfidf_vocabulary,
                                       ngram_range=(1, 1))
    X_tfidf = tfidf_vectorizer.fit_transform(X.cleaned_stemmed_text)
    X_scaled = trained_scaler.transform(X.loc[:, ['length', 'length_original_tokens', 'length_original_text',
                                                  'number_non_words']])
    X_features = pd.concat([pd.DataFrame(X_scaled, columns=['length', 'length_original_tokens',
                                                            'length_original_text', 'number_non_words']),
                                 pd.DataFrame(X_tfidf.toarray())], axis=1)
    return X_features


def predict(X, y, model_names=None):
    y_pred = {}
    models = [keras.models.load_model(model_path + model + '.h5') for model in model_names]
    for i in range(0, len(model_names)):
        predictions = models[i].predict(X)
        y_pred[str(i)] = predictions
    y_pred['addition'] = y_pred['0']
    for i in range(1, len(y_pred)-1):
        y_pred['addition'] = y_pred['addition'].__add__(y_pred[str(i)])
    return y_pred


# Importing training dataset
X_train, X_val, y_train, y_val = train_test_split(data_Hate.loc[:, input_features], data_Hate.loc[:, output_features],
                                                  random_state=42, test_size=0.1, stratify=data_Hate.final_label)
# Resetting index for train and val sets
X_train.reset_index(drop=True, inplace=True)
X_val.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_val.reset_index(drop=True, inplace=True)

data_Hate_Test = pd.read_csv(data_path + 'HS_DATA_NEW_TEST.csv', sep=',')
hs_NLP = HateSpeechNLP(data_Hate_Test, save=False, default_name=False, features=input_features+output_features)
data_Test = hs_NLP.fit_transform()


label_encoder = LabelEncoder()
label_encoder.fit(['NONE', 'PRFN', 'OFFN', 'HATE'], )
y_train['final_label_int'] = label_encoder.transform(y_train['final_label'])
y_val['final_label_int'] = label_encoder.transform(y_val['final_label'])
data_Test['final_label_int'] = label_encoder.transform(data_Test['final_label'])

keras_model_names = ['Keras_sgd', 'Keras_DR_1g', 'Keras_2']
keras_model_names_binary = ['Keras_binary_1', 'Keras_binary_2', 'Keras_binary_3']

# Validation set predictions
val_predictions = predict(prepare_input_one_gram(X_val), y_val.final_label_int, keras_model_names)
val_predictions_binary = predict(prepare_input_one_gram(X_val), y_val.binary_label, keras_model_names_binary)

test_predictions = predict(prepare_input_one_gram(data_Test), data_Test.final_label_int, keras_model_names)
test_predictions_binary = predict(prepare_input_one_gram(data_Test), data_Test.binary_label, keras_model_names_binary)

val_add = np.argmax(val_predictions['addition'], axis=-1)
test_add = np.argmax(test_predictions['addition'], axis=-1)
























