import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from HateSpeechNLP import HateSpeechNLP


# Setting display width for panda outputs
pd.set_option('display.max_colwidth', 200)
# Paths
data_path = '../Data/'
model_path = '../Models/'

# Import HateSpeech DataFrame
try:
    data_Hate = pd.read_pickle(data_path + 'HateSpeech_DataFrame_18-09-2021_15-10-16.pkl').loc[:, ['text',
                'cleaned_stemmed_text', 'length', 'length_original_tokens', 'length_original_text', 'number_non_words',
                                                                                         'final_label']]
except FileNotFoundError:
    # Import HS_Data
    data_Hate_HS = pd.read_csv(data_path + 'HS_DATA_TRAIN.csv', sep=',')
    hs_NLP = HateSpeechNLP(data_Hate_HS, save=True, default_name=False)
    data_Hate = hs_NLP.fit_transform()


def train(X, y):
    model = keras.models.Sequential()
    return model


X_train, X_val, y_train, y_val = train_test_split(data_Hate.loc[:, ['length', 'length_original_tokens',
                                                                    'length_original_text','number_non_words',
                                                                    'cleaned_stemmed_text']], data_Hate.final_label,
                                                  random_state=42, test_size=0.1, stratify=data_Hate.final_label)
# Resetting index for train and val sets
X_train.reset_index(drop=True, inplace=True)
X_val.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_val.reset_index(drop=True, inplace=True)

# Train Keras Sequential model
seq_model = train(X_train, y_train)





















