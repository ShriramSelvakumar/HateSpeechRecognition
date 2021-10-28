# Preparing Hate Speech Training Data

import pandas as pd

# Import training and testing data of Hate Speech Recognition
data_Hate_New_Train = pd.read_csv('../Data/HS_DATA_NEW_TRAIN.csv', sep=',')
data_Hate_New_Test = pd.read_csv('../Data/HS_DATA_NEW_TEST.csv', sep=',')


def binary_label(value, class_value):
    if value == class_value:
        return 1
    else:
        return 0


# Final Label - HATE & OFFN & PRFN & NONE
# Binary Label - 0:NONE ----- 1:HATE & OFFN & PRFN
for dataframe in [data_Hate_New_Train, data_Hate_New_Test]:
    dataframe['NONE_label'] = dataframe['final_label'].apply(lambda x: binary_label(x, 'NONE'))
    dataframe['HATE_label'] = dataframe['final_label'].apply(lambda x: binary_label(x, 'HATE'))
    dataframe['OFFN_label'] = dataframe['final_label'].apply(lambda x: binary_label(x, 'OFFN'))
    dataframe['PRFN_label'] = dataframe['final_label'].apply(lambda x: binary_label(x, 'PRFN'))
    dataframe.reset_index(drop=True, inplace=True)


data_Hate_New_Test.to_csv('../Data/HS_DATA_BINARY_TEST.csv', sep=',')
data_Hate_New_Train.to_csv('../Data/HS_DATA_BINARY_TRAIN.csv', sep=',')

train = pd.read_csv('../Data/HS_DATA_BINARY_TRAIN.csv', sep=',')
test = pd.read_csv('../Data/HS_DATA_BINARY_TEST.csv', sep=',')












