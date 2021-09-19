import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from DisasterSituationNLP import DisasterSituationNLP


# Setting display width for panda outputs
pd.set_option('display.max_colwidth', 200)
# Paths
data_path = '../Data/'
model_path = '../Models/'

# Import DisasterSituation Train DataFrame
try:
    data_DS = pd.read_pickle(data_path + 'DisasterSituation_DataFrame.pkl').loc[:, ['text', 'keyword',
                'cleaned_stemmed_text', 'length', 'length_original_tokens', 'length_original_text', 'number_non_words',
                                                                                         'target']]
except FileNotFoundError:
    # Import DS_DATA_TRAIN
    data_Disaster = pd.read_csv(data_path + 'DS_DATA_TRAIN.csv', sep=',')
    ds_NLP = DisasterSituationNLP(data_Disaster, save=True, default_name=False)
    data_DS = ds_NLP.fit_transform()


def train(X, y):
    model = keras.models.Sequential()
    return model


X_train, X_val, y_train, y_val = train_test_split(data_DS.loc[:, ['length', 'length_original_tokens', 'keyword',
                                                                  'length_original_text', 'number_non_words',
                                                                  'cleaned_stemmed_text']], data_DS.target,
                                                  random_state=42, test_size=0.15, stratify=data_DS.target)
# Resetting index for train and val sets
X_train.reset_index(drop=True, inplace=True)
X_val.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_val.reset_index(drop=True, inplace=True)

# Train Keras Sequential model
seq_model = train(X_train, y_train)


















