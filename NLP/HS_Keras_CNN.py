import pandas as pd
import numpy as np
import glob
import os
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from HateSpeechNLP import HateSpeechNLP
import pickle

# Added Comments
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


def train(X, y, X_valid, y_valid):

    # Computing Class Weights
    y_classes = list(y['final_label_int'].unique())
    y_classes.sort()
    class_weights_list = class_weight.compute_class_weight('balanced',  y_classes, y['final_label_int'])
    # Calculated class weights - [3.14126016, 0.89438657, 0.43002226, 4.19972826] - assuming for [Hate, None, Off, Prf]
    class_weights = {}
    for i in range(0, len(class_weights_list)):
        class_weights[i] = class_weights_list[i]

    # Implementing Callbacks - Saving checkpoints & Early Stopping
    checkpoint_cb = keras.callbacks.ModelCheckpoint(model_path + "Keras_CNN_2.h5", save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    # Constants to train the model
    # max_features = 183000
    max_features = 22000
    embedding_dim = 128

    # Declare Text Vectorization
    vectorize_layer = TextVectorization(max_tokens=max_features, standardize=None, split='whitespace', ngrams=2,
                                        output_mode='tf-idf')
    # Learn vocabulary from train data
    vectorize_layer.adapt(np.asarray(X.cleaned_stemmed_text))
    pickle.dump({'config': vectorize_layer.get_config(), 'weights': vectorize_layer.get_weights()},
                open(model_path + "tv_layer_2.pkl", "wb"))
    print('Saved TextVectorizer - In Pickle===========================================')
    # Load Pickle
    # from_disk = pickle.load(open(model_path + "tv_layer.pkl", "rb"))
    X_train_vectorized = vectorize_layer(np.asarray(X.cleaned_stemmed_text))
    X_valid_vectorized = vectorize_layer(np.asarray(X_valid.cleaned_stemmed_text))

    # TF-IDF from scikit learn
    # tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), dtype=np.float32)
    # X_train_tfidf = tfidf_vectorizer.fit_transform(X_train.cleaned_stemmed_text)
    # X_train_feature = pd.DataFrame(X_train_tfidf.toarray())
    # os.makedirs(model_path + 'TFIDF/', exist_ok=True)
    # pickle.dump(tfidf_vectorizer.vocabulary_, open(model_path + 'TFIDF/' + 'TFIDF-Vocabulary-Keras_' +
    #                                            datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.pkl', "wb"))

    # Building model
    inputs = keras.layers.Input(shape=(None,))

    # Embedding layer - To map input to certain dimensionality
    x_embedding = layers.Embedding(max_features, embedding_dim)(inputs)
    x_embedding = layers.Dropout(0.5)(x_embedding)

    # Conv1D + global max pooling
    x_CNN = layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3)(x_embedding)
    x_CNN = layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3)(x_CNN)
    x_CNN = layers.GlobalMaxPool1D()(x_CNN)

    # Hidden Layers
    x_hidden = layers.Dense(128, activation='relu')(x_CNN)
    x_hidden = layers.Dropout(0.5)(x_hidden)

    # Output layer
    output = layers.Dense(4, activation='softmax')(x_hidden)

    # Model
    model = keras.Model(inputs, output)

    # Compile
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Fitting the model
    history = model.fit(X_train_vectorized, y['final_label_int'], epochs=3,
                        validation_data=(X_valid_vectorized, y_valid['final_label_int']),
                        callbacks=[checkpoint_cb, early_stopping_cb], class_weight=class_weights)

    return history, model


def test(X, y, model_name, vectorizer_name):
    # Load trained model
    try:
        from_disk = pickle.load(open(model_path + vectorizer_name + ".pkl", "rb"))
        loaded_vectorized = TextVectorization.from_config(from_disk['config'])
        loaded_vectorized.set_weights(from_disk['weights'])

        X_test_vectorized = loaded_vectorized(np.asarray(X.cleaned_stemmed_text))

        # Load trained model
        trained_NN_model = keras.models.load_model(model_path + model_name + '.h5')
        # Print model summary
        print(trained_NN_model.summary())

        y_pred = np.argmax(trained_NN_model.predict(np.asarray(X_test_vectorized)), axis=-1)
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

label_encoder = LabelEncoder()
label_encoder.fit(['NONE', 'PRFN', 'OFFN', 'HATE'])
y_train['final_label_int'] = label_encoder.transform(y_train['final_label'])
y_val['final_label_int'] = label_encoder.transform(y_val['final_label'])

print("==================================================================================================")
print("Training starts")
# fun_history, fun_model = train(X_train, y_train, X_val, y_val)
# fun_model.save(model_path + 'Keras_CNN_2.h5', save_format='h5')

print("==================================================================================================")
print("==================================================================================================")
print("==========================================CNN-1===================================================")
print("Validation")
test(X_val, y_val['final_label_int'], model_name='Keras_CNN', vectorizer_name='tv_layer')

# Testing the test set
# Importing HS_DATA - Test set
print("==================================================================================================")
data_Hate_Test = pd.read_csv(data_path + 'HS_DATA_NEW_TEST.csv', sep=',')
hs_NLP = HateSpeechNLP(data_Hate_Test, save=False, default_name=False, features=input_features+output_features)
data_Test = hs_NLP.fit_transform()
print("Evaluating Test Data")
print(data_Test.info())
data_Test['final_label_int'] = label_encoder.transform(data_Test['final_label'])
y_test = test(data_Test, data_Test['final_label_int'], model_name='Keras_CNN',
              vectorizer_name='tv_layer')

print("==================================================================================================")
print("==================================================================================================")
print("==========================================CNN-2===================================================")
print("Validation")
test(X_val, y_val['final_label_int'], model_name='Keras_CNN_2', vectorizer_name='tv_layer_2')

# Testing the test set
# Importing HS_DATA - Test set
print("==================================================================================================")
data_Hate_Test = pd.read_csv(data_path + 'HS_DATA_NEW_TEST.csv', sep=',')
hs_NLP = HateSpeechNLP(data_Hate_Test, save=False, default_name=False, features=input_features+output_features)
data_Test = hs_NLP.fit_transform()
print("Evaluating Test Data")
print(data_Test.info())
data_Test['final_label_int'] = label_encoder.transform(data_Test['final_label'])
y_test_2 = test(data_Test, data_Test['final_label_int'], model_name='Keras_CNN_2',
                vectorizer_name='tv_layer_2')


print("End of File")













