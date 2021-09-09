import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


path = '../Data/'

data_HS = pd.read_csv(path + 'HS_DATA.csv', sep=',', index_col=0)
X_train_, X_test, y_train_, y_test = train_test_split(
    data_HS.text,
    data_HS.final_label,
    test_size=0.10,
    random_state=42,
    stratify=data_HS.final_label
)
X_train_.reset_index(drop=True, inplace=True)
y_train_.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

HS_train = pd.concat([X_train_, y_train_], axis=1)
HS_test = pd.concat([X_test, y_test], axis=1)
HS_train.to_csv(path + 'HS_DATA_TRAIN.csv', sep=',', header=True, index=False)
HS_test.to_csv(path + 'HS_DATA_TEST.csv', sep=',', header=True, index=False)


# Reading the exported csv file
train = pd.read_csv(path + 'HS_DATA_TRAIN.csv', sep=',')
test = pd.read_csv(path + 'HS_DATA_TEST.csv', sep=',')
