# Preparing Training Data

import pandas as pd

data_kaggle = pd.read_csv('./Data/Hate Speech - Kaggle.csv', sep=',')
# 'Class' column is the label - 0 is Hate Speech - 1 is Offensive Language - 2 is Neither

data_2019 = pd.read_csv('./Data/hasoc_2019_en_train.tsv', sep='\t')
# 'task_1' column is the label - NOT is not Hate Speech - HOF is Hate Speech
# 'task_2' PRFN - profane, OFFN = offensive, HATE - hate speech

data_2020 = pd.read_excel('./Data/hasoc_2020_en_train.xlsx', sheet_name=0)
# 'task_1' column is the label - NOT is not Hate Speech - HOF is Hate Speech
# 'task_2' PRFN - profane, OFFN = offensive, HATE - hate speech

# Convert labels into 0, 1, 2, 3 format
# 0: HATE--- 1:OFFN ---- 2:PRFN ---- 3:NONE


def label_converter(value):
    if value == 'HATE':
        return 0
    elif value == 'OFFN':
        return 1
    elif value == 'PRFN':
        return 2
    else:
        return 3


data_kaggle['final_label'] = data_kaggle['class'].apply(lambda x: 3 if x == 2 else x)
data_2019['final_label'] = data_2019['task_2'].apply(lambda x: label_converter(x))
data_2020['final_label'] = data_2020['task2'].apply(lambda x: label_converter(x))
data = data_kaggle.iloc[:, 6:]
data.columns = ['text', 'final_label']
data = pd.concat([data, data_2019.iloc[:, [1, 5]]])
data = pd.concat([data, data_2020.iloc[:, [1, 5]]])
data = data.reset_index(drop=True)
data.to_csv('./Data/HS_DATA.csv', sep=',')
