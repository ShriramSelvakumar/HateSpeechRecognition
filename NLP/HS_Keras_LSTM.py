import numpy as np
import pandas as pd
import os
import glob
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import string
import re
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


def train(X, y, X_valid, y_valid):
    max_features = 22000
    max_len = 300

    # Vectorization using Keras preprocessing
    vectorize_layer = TextVectorization(max_tokens=max_features, standardize=None, ngrams=1,
                                        output_mode='int', output_sequence_length=max_len)
    vectorize_layer.adapt(np.asarray(X.cleaned_stemmed_text))
    pickle.dump({'config': vectorize_layer.get_config(), 'weights': vectorize_layer.get_weights()},
                open(model_path + "tv_layer_LSTM.pkl", "wb"))
    print('Saved TextVectorizer - In Pickle===========================================')

    X_train_vectorized = vectorize_layer(np.asarray(X.cleaned_stemmed_text))
    X_valid_vectorized = vectorize_layer(np.asarray(X_valid.cleaned_stemmed_text))

    # Computing Class Weights
    y_classes = list(y['final_label_int'].unique())
    y_classes.sort()
    class_weights_list = class_weight.compute_class_weight('balanced', y_classes, y['final_label_int'])
    # Calculated class weights - [3.14126016, 0.89438657, 0.43002226, 4.19972826] - assuming for [Hate, None, Off, Prf]
    class_weights = {}
    for i in range(0, 4):
        class_weights[i] = class_weights_list[i]

    # Implementing Callbacks - Saving checkpoints & Early Stopping
    checkpoint_cb = keras.callbacks.ModelCheckpoint(model_path + "Keras_LSTM.h5", save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)

    # Input for variable-length sequences of integers
    inputs = layers.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x_embedding = layers.Embedding(max_features, 128)(inputs)
    # Add 2 bidirectional LSTMs
    x_LSTM = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x_embedding)
    x_LSTM = layers.Bidirectional(layers.LSTM(64))(x_LSTM)
    # Multi Class classifier
    outputs = layers.Dense(4, activation='softmax')(x_LSTM)

    # Model
    model = keras.Model(inputs, outputs)
    # Printing summary
    print(model.summary())

    # Compile
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Fit the model
    history = model.fit(X_train_vectorized, y['final_label_int'], epochs=15, batch_size=32,
                        validation_data=(X_valid_vectorized, y_valid['final_label_int']),
                        callbacks=[checkpoint_cb, early_stopping_cb], class_weight=class_weights)

    return history, model


def test(X, y, model_name, vectorizer_name):
    # Load trained model
    try:
        # Load TextVectorization Pickle
        from_disk = pickle.load(open(model_path + vectorizer_name + ".pkl", "rb"))
        loaded_vectorized = TextVectorization.from_config(from_disk['config'])
        loaded_vectorized.set_weights(from_disk['weights'])

        X_test_vectorized = loaded_vectorized(np.asarray(X.cleaned_stemmed_text))

        # Load trained model
        trained_NN_model = keras.models.load_model(model_path + model_name + '.h5')
        # Print model summary
        print(trained_NN_model.summary())

        y_pred = np.argmax(trained_NN_model.predict(X_test_vectorized), axis=-1)
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


# Main function
X_train, X_val, y_train, y_val = train_test_split(data_Hate.loc[:, input_features], data_Hate.loc[:, output_features],
                                                  random_state=42, test_size=0.1, stratify=data_Hate.final_label)
# Resetting index for train and val sets
X_train.reset_index(drop=True, inplace=True)
X_val.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_val.reset_index(drop=True, inplace=True)

label_encoder = LabelEncoder()
label_encoder.fit(['NONE', 'PRFN', 'OFFN', 'HATE'], )
y_train['final_label_int'] = label_encoder.transform(y_train['final_label'])
y_val['final_label_int'] = label_encoder.transform(y_val['final_label'])

print("==================================================================================================")
print("Training starts")
# LSTM_history, LSTM_model = train(X_train, y_train, X_val, y_val)
# LSTM_model.save(model_path + 'Keras_LSTM.h5', save_format='h5')

print("==================================================================================================")
print("Validation")
test(X_val, y_val['final_label_int'], model_name='Keras_LSTM', vectorizer_name='tv_layer_LSTM')

# Testing the test set
# Importing HS_DATA - Test set
print("==================================================================================================")
data_Hate_Test = pd.read_csv(data_path + 'HS_DATA_NEW_TEST.csv', sep=',')
hs_NLP = HateSpeechNLP(data_Hate_Test, save=False, default_name=False, features=input_features+output_features)
data_Test = hs_NLP.fit_transform()
print("Evaluating Test Data")
print(data_Test.info())
data_Test['final_label_int'] = label_encoder.transform(data_Test['final_label'])
y_test = test(data_Test, data_Test['final_label_int'], model_name='Keras_LSTM', vectorizer_name='tv_layer_LSTM')























