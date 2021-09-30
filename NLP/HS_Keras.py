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

    # Computing Class Weights
    y_classes = list(y['final_label_int'].unique())
    y_classes.sort()
    class_weights_list = class_weight.compute_class_weight('balanced',  y_classes, y['final_label_int'])
    # Calculated class weights - [3.14126016, 0.89438657, 0.43002226, 4.19972826] - assuming for [Hate, None, Off, Prf]
    class_weights = {}
    for i in range(0, len(class_weights_list)):
        class_weights[i] = class_weights_list[i]

    # Implementing Callbacks - Saving checkpoints & Early Stopping
    checkpoint_cb = keras.callbacks.ModelCheckpoint(model_path + "Keras_test.h5", save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    # Keras model
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=X_train_features.shape[1:]))
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(4, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(X_train_features, y['final_label_int'], epochs=30, validation_data=(X_valid_features,
                                                                                            y_valid['final_label_int']),
                        callbacks=[checkpoint_cb, early_stopping_cb], class_weight=class_weights)

    return history, model


def test(X, y):
    # Load trained model
    try:
        tfidf_path = model_path + 'TFIDF/'
        standard_scaler_path = model_path + 'StandardScaler/'
        file_type = '\*pkl'
        tfidf_files = glob.glob(tfidf_path + file_type)
        standard_scaler_files = glob.glob(standard_scaler_path + file_type)
        try:
            tfidf_file = max(tfidf_files, key=os.path.getctime)
            standard_scaler_file = max(standard_scaler_files, key=os.path.getctime)
        except ValueError:
            # Added this exception because try statements are throwing errors in Linux
            tfidf_file = model_path + 'TFIDF/' + 'TFIDF-Vocabulary-Keras_21-09-2021_18-46-42.pkl'
            standard_scaler_file = model_path + 'StandardScaler/' + 'StandardScaler-Keras_21-09-2021_18-46-42.pkl'

        trained_tfidf_vocabulary = pickle.load(open(tfidf_file, "rb"))
        trained_scaler = pickle.load(open(standard_scaler_file, "rb"))
        trained_NN_model = keras.models.load_model(model_path + 'Keras_test' + '.h5')
        print("Loaded TFIDF Model -", tfidf_file)
        print("Loaded StandardScaler Model -", standard_scaler_file)

        tfidf_vectorizer = TfidfVectorizer(analyzer=clean_text, vocabulary=trained_tfidf_vocabulary,
                                           ngram_range=(1, 1))
        X_test_tfidf = tfidf_vectorizer.fit_transform(X.cleaned_stemmed_text)
        X_test_scaled = trained_scaler.transform(X.loc[:, ['length', 'length_original_tokens', 'length_original_text',
                                                           'number_non_words']])
        X_test_features = pd.concat([pd.DataFrame(X_test_scaled, columns=['length', 'length_original_tokens',
                                                                          'length_original_text', 'number_non_words']),
                                     pd.DataFrame(X_test_tfidf.toarray())], axis=1)
        y_pred = np.argmax(trained_NN_model.predict(X_test_features), axis=-1)

        # y_pred = trained_NN_model.predict_classes(X_test_features)
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
    os.makedirs(model_path + 'TFIDF/', exist_ok=True)
    pickle.dump(tfidf.vocabulary_, open(model_path + 'TFIDF/' + 'TFIDF-Vocabulary-Keras_' +
                                        datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.pkl', "wb"))
    print('Saved TFIDF-Keras to Pickle')
    os.makedirs(model_path + 'StandardScaler/', exist_ok=True)
    pickle.dump(scaler, open(model_path + 'StandardScaler/' + 'StandardScaler-Keras_' +
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


# X_train = pd.concat([X_train, y_train], axis=1)
# for i in range(0, 1):
#     X_train = pd.concat([X_train, X_train[X_train['HATE_label'] == 1], X_train[X_train['PRFN_label'] == 1]])
#     X_train.reset_index(drop=True, inplace=True)
#
#
# y_train = X_train.loc[:, output_features]
# X_train = X_train.loc[:, input_features]
# X_train.reset_index(drop=True, inplace=True)
# y_train.reset_index(drop=True, inplace=True)

label_encoder = LabelEncoder()
label_encoder.fit(['NONE', 'PRFN', 'OFFN', 'HATE'], )
y_train['final_label_int'] = label_encoder.transform(y_train['final_label'])
y_val['final_label_int'] = label_encoder.transform(y_val['final_label'])

# Train Keras Sequential model
print("==================================================================================================")
print("Training starts")
seq_history, seq_model = train(X_train, y_train, X_val, y_val)
pd.DataFrame(seq_history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

# Testing the validation set
print("==================================================================================================")
print("Evaluating Validation Set")
test(X_val, y_val['final_label_int'])

# Testing the test set
# Importing HS_DATA - Test set
print("==================================================================================================")
data_Hate_Test = pd.read_csv(data_path + test_file_name, sep=',')
hs_NLP = HateSpeechNLP(data_Hate_Test, save=False, default_name=False, features=input_features+output_features)
data_Test = hs_NLP.fit_transform()
print("Evaluating Test Data")
print(data_Test.info())
data_Test['final_label_int'] = label_encoder.transform(data_Test['final_label'])
y_test = test(data_Test, data_Test['final_label_int'])



















