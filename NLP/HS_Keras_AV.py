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
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import string
import re
from tensorflow.keras import layers


# Setting display width for panda outputs
pd.set_option('display.max_colwidth', 200)
# Constants
data_path = '../Data/'
model_path = '../Models/'
train_file_name = 'HS_DATA_BINARY_TRAIN.csv'
test_file_name = 'HS_DATA_BINARY_TEST.csv'
input_features_NW = ['lemmatized_text_NW1', 'lemmatized_text_NW2', 'cleaned_stemmed_text_NW1',
                     'cleaned_stemmed_text_NW2']
input_features = ['text', 'lemmatized_text', 'cleaned_stemmed_text', 'length',
                  'length_original_tokens', 'length_original_text',
                  'number_non_words'] + input_features_NW
output_features = ['final_label', 'binary_label', 'NONE_label', 'HATE_label', 'OFFN_label', 'PRFN_label']
train_feature = 'cleaned_stemmed_text'

# Import HateSpeech DataFrame
try:
    data_Hate = pd.read_pickle(data_path + 'HateSpeech_DataFrame_AV.pkl').loc[:,
                input_features + output_features]
except FileNotFoundError:
    # Import HS_Data
    data_Hate_HS = pd.read_csv(data_path + train_file_name, sep=',')
    hs_NLP = HateSpeechNLP(data_Hate_HS, save=True, default_name=False, features=input_features+output_features)
    data_Hate = hs_NLP.fit_transform()


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

label_encoder = LabelEncoder()
label_encoder.fit(['NONE', 'PRFN', 'OFFN', 'HATE'], )
y_train['final_label_int'] = label_encoder.transform(y_train['final_label'])
y_val['final_label_int'] = label_encoder.transform(y_val['final_label'])

# Keras constants
max_features = 40000
embedding_dim = 128
sequence_length = 500

vectorize_layer = TextVectorization(
    standardize=None,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
    ngrams=3
)

vectorize_layer.adapt(np.array(X_train[train_feature]))


def vectorize_text(text):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text)


# text_train = X_train['cleaned_stemmed_text'].apply(lambda val: vectorize_text(val))
# text_val = X_val['cleaned_stemmed_text'].apply(lambda val: vectorize_text(val))
text_train = vectorize_layer(np.array(X_train[train_feature]))
text_val = vectorize_layer(np.array(X_val[train_feature]))


# A integer input for vocab indices.
inputs = tf.keras.Input(shape=(None,), dtype='int64')

# Next, we add a layer to map those vocab indices into a space of dimensionality
# 'embedding_dim'.
x = layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)

# Conv1D + global max pooling
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

# We add a vanilla hidden layer:
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# We project onto a single unit output layer, and squash it with a sigmoid:
predictions = layers.Dense(4, activation="softmax", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

save_name = 'final_label_int'

# Computing Class Weights
y_classes = list(y_train[save_name].unique())
y_classes.sort()
class_weights_list = class_weight.compute_class_weight('balanced', y_classes, y_train[save_name])
# Calculated class weights - [3.14126016, 0.89438657, 0.43002226, 4.19972826] - assuming for [Hate, None, Off, Prf]
class_weights = {}
for i in range(0, len(class_weights_list)):
    class_weights[i] = class_weights_list[i]

# Implementing Callbacks - Saving checkpoints & Early Stopping
checkpoint_cb = keras.callbacks.ModelCheckpoint(model_path + '40k--Keras_BINARY_CNN' + save_name +
                                                '-{epoch:02d}-{val_accuracy:.3f}' + ".h5", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# Fit the model using the train and test datasets.
history = model.fit(text_train, y_train[save_name], epochs=20, validation_data=(text_val, y_val[save_name]),
                    callbacks=[checkpoint_cb, early_stopping_cb], class_weight=class_weights)

y_pred = np.argmax(model.predict(text_val), axis=-1)

print('Micro Values -----')
print("Precision : ", precision_score(y_val[save_name], y_pred, average="micro"))
print("Recall : ", recall_score(y_val[save_name], y_pred, average='micro'))
print('Macro Values -----')
print("Precision : ", precision_score(y_val[save_name], y_pred, average="macro"))
print("Recall : ", recall_score(y_val[save_name], y_pred, average='macro'))
print('Weighted Values -----')
print("Precision : ", precision_score(y_val[save_name], y_pred, average="weighted"))
print("Recall : ", recall_score(y_val[save_name], y_pred, average='weighted'))
print('Confusion Matrix -----')
print(confusion_matrix(y_val[save_name], y_pred))

# model.save(model_path + '30k--Keras-Model_' + save_name + '__' +
#            datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.h5')



















