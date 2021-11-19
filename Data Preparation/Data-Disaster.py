# # Preparing Disaster Situation Training Data

import pandas as pd
from sklearn.model_selection import train_test_split


path = './Data/'

data_1 = pd.read_csv(path + 'Disaster-1.csv', sep=',')
# 'target' column is the label - 0 is not Disaster Related Tweet - 1 is Disaster Related Tweet

data_2 = pd.read_csv(path + 'Disaster-2.csv', sep=',')
# 'target' column is the label - 0 is not Disaster Related Tweet - 1 is Disaster Related Tweet


data = data_1.loc[:, ['keyword', 'text', 'target']]
data = pd.concat([data, data_2.loc[:, ['keyword', 'text', 'target']]])
data.reset_index(drop=True, inplace=True)

# Fill NaN in keyword column
index = pd.isna(data.keyword)
index_new = list(data[index].index)
for i in index_new:
    if data.loc[i, 'target'] == 1:
        data.loc[i, 'keyword'] = 'Disaster'
    else:
        data.loc[i, 'keyword'] = 'Not_Disaster'

# Write data to csv
data.to_csv(path + 'DS_DATA.csv', sep=',')

# Read data from csv
data_DS = pd.read_csv(path + 'DS_DATA.csv', sep=',', index_col=0)
X_train_, X_test, y_train_, y_test = train_test_split(
    data_DS.loc[:, ['text']],
    data_DS.loc[:, ['target']],
    test_size=0.15,
    random_state=42,
    stratify=data_DS.target
)
X_train_.reset_index(drop=True, inplace=True)
y_train_.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

DS_train = pd.concat([X_train_, y_train_], axis=1)
DS_test = pd.concat([X_test, y_test], axis=1)
DS_train.to_csv(path + 'DS_DATA_TRAIN.csv', sep=',', header=True, index=False)
DS_test.to_csv(path + 'DS_DATA_TEST.csv', sep=',', header=True, index=False)


# Reading the exported csv file
train = pd.read_csv(path + 'DS_DATA_TRAIN.csv', sep=',')
test = pd.read_csv(path + 'DS_DATA_TEST.csv', sep=',')







