# Preparing Disaster Situation Training Data

import pandas as pd

data_1 = pd.read_csv('./Data/Disaster-1.csv', sep=',')
# 'target' column is the label - 0 is not Disaster Related Tweet - 1 is Disaster Related Tweet

data_2 = pd.read_csv('./Data/Disaster-2.csv', sep=',')
# 'target' column is the label - 0 is Hate Speech - 1 is Offensive Language - 2 is Neither


data = data_1.iloc[:, 3:]
data = pd.concat([data, data_2.iloc[:, 3:]])
data = data.reset_index(drop=True)
data.to_csv('./Data/DS_DATA.csv', sep=',')




