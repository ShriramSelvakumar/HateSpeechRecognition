# Preparing Hate Speech Training Data

import pandas as pd

data_kaggle = pd.read_csv('../Data/Kaggle.csv', sep=',')
# 'Class' column is the label - 0 is Hate Speech - 1 is Offensive Language - 2 is Neither

data_2019 = pd.read_csv('../Data/hasoc_2019_en_train.tsv', sep='\t')
# 'task_1' column is the label - NOT is not Hate Speech - HOF is Hate Speech
# 'task_2' PRFN - profane, OFFN = offensive, HATE - hate speech

data_2020 = pd.read_excel('../Data/hasoc_2020_en_train.xlsx', sheet_name=0)
# 'task_1' column is the label - NOT is not Hate Speech - HOF is Hate Speech
# 'task_2' PRFN - profane, OFFN = offensive, HATE - hate speech

# Convert labels into 0, 1, 2, 3 format
# 0:NONE ----- 1:HATE & OFFN & PRFN


def label_converter(value):
    if value in ['HATE', 'OFFN', 'PRFN']:
        return 1
    elif value == 'NONE':
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

data = data_kaggle.loc[:, ['tweet', 'final_label', 'binary_label', 'expert']]
data.columns = ['text', 'final_label', 'binary_label', 'expert']
data = pd.concat([data, data_2019.loc[:, ['text', 'final_label', 'binary_label', 'expert']]])
data = pd.concat([data, data_2020.loc[:, ['text', 'final_label', 'binary_label', 'expert']]])
data.reset_index(drop=True, inplace=True)
data.to_csv('../Data/HS_DATA_NEW.csv', sep=',')
