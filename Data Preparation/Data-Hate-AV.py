# Preparing Hate Speech Training Data

import pandas as pd
from sklearn.model_selection import train_test_split


data_kaggle = pd.read_csv('../Data/Kaggle.csv', sep=',')
# 'Class' column is the label - 0 is Hate Speech - 1 is Offensive Language - 2 is Neither

data_2019 = pd.read_csv('../Data/hasoc_2019_en_train.tsv', sep='\t')
# 'task_1' column is the label - NOT is not Hate Speech - HOF is Hate Speech
# 'task_2' PRFN - profane, OFFN = offensive, HATE - hate speech

data_2020 = pd.read_excel('../Data/hasoc_2020_en_train.xlsx', sheet_name=0)
# 'task_1' column is the label - NOT is not Hate Speech - HOF is Hate Speech
# 'task_2' PRFN - profane, OFFN = offensive, HATE - hate speech

data_AV = pd.read_csv('../Data/AV-train.csv', sep=',', index_col=0)
# 0 - None, 1 - Hate

# data_AV_2 = pd.read_csv('../Data/AV-test.csv', sep=',', index_col=0)
# 0 - None, 1 - Hate

# Convert labels into 0, 1, 2, 3 format
# 0:NONE ----- 1:HATE & OFFN & PRFN


def label_converter(value):
    if value in ['HATE', 'OFFN', 'PRFN']:
        return 1
    elif value == 'NONE':
        return 0


def binary_label(value, class_value):
    if value == class_value:
        return 1
    else:
        return 0


# Final Label - HATE & OFFN & PRFN & NONE
# Binary Label - 0:NONE ----- 1:HATE & OFFN & PRFN
data_kaggle['final_label'] = data_kaggle['class'].replace([0, 1, 2], ['HATE', 'OFFN', 'NONE'])
data_kaggle['binary_label'] = data_kaggle['final_label'].apply(lambda x: label_converter(x))
data_kaggle.insert(0, 'expert', 0)
data_2019['final_label'] = data_2019['task_2'].copy()
data_2019['binary_label'] = data_2019['final_label'].apply(lambda x: label_converter(x))
data_2019.insert(0, 'expert', 1)
data_2020['final_label'] = data_2020['task2'].copy()
data_2020['binary_label'] = data_2020['final_label'].apply(lambda x: label_converter(x))
data_2020.insert(0, 'expert', 1)
data_AV['text'] = data_AV['tweet'].copy()
data_AV['final_label'] = data_AV['label'].replace([0, 1], ['NONE', 'HATE'])
data_AV['binary_label'] = data_AV['final_label'].apply(lambda x: label_converter(x))
data_AV.insert(0, 'expert', 0)

data = data_kaggle.loc[:, ['tweet', 'final_label', 'binary_label', 'expert']]
data.columns = ['text', 'final_label', 'binary_label', 'expert']
data = pd.concat([data, data_2019.loc[:, ['text', 'final_label', 'binary_label', 'expert']]])
data = pd.concat([data, data_2020.loc[:, ['text', 'final_label', 'binary_label', 'expert']]])
data = pd.concat([data, data_AV.loc[:, ['text', 'final_label', 'binary_label', 'expert']]])
data.reset_index(drop=True, inplace=True)


# Creating binary labels for future use
data['NONE_label'] = data['final_label'].apply(lambda x: binary_label(x, 'NONE'))
data['HATE_label'] = data['final_label'].apply(lambda x: binary_label(x, 'HATE'))
data['OFFN_label'] = data['final_label'].apply(lambda x: binary_label(x, 'OFFN'))
data['PRFN_label'] = data['final_label'].apply(lambda x: binary_label(x, 'PRFN'))
data.reset_index(drop=True, inplace=True)

data.to_csv('../Data/HS_DATA_AV.csv', sep=',')

path = '../Data/'
output_features = ['final_label', 'binary_label', 'NONE_label', 'HATE_label', 'OFFN_label', 'PRFN_label']

data_HS = pd.read_csv(path + 'HS_DATA_AV.csv', sep=',')
X_train_, X_test, y_train_, y_test = train_test_split(
    data_HS.loc[:, ['text', 'expert']],
    data_HS.loc[:, output_features],
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
HS_train.to_csv(path + 'HS_DATA_AV_TRAIN.csv', sep=',', header=True, index=False)
HS_test.to_csv(path + 'HS_DATA_AV_TEST.csv', sep=',', header=True, index=False)


# Reading the exported csv file
train = pd.read_csv(path + 'HS_DATA_AV_TRAIN.csv', sep=',')
test = pd.read_csv(path + 'HS_DATA_AV_TEST.csv', sep=',')









