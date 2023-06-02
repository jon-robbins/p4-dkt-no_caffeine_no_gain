import numpy as np
import pandas as pd
from tqdm import tqdm

# dtype = {
#     'userID': 'int16',
#     'answerCode': 'int8',
#     'KnowledgeTag': 'int16'
# }   
dtype = { #This is added to fit our actual data
    'userID': 'object',
    'answerCode': 'bool',
    'KnowledgeTag': 'object',
    'assessmentItemID': 'object',
    'testId': 'object'
}


#changed data_path
DATA_PATH = '../archive/caffeine_data/df_train_make_elapsed.csv'
train_org_df = pd.read_csv(DATA_PATH, dtype=dtype, parse_dates=['Timestamp'])
train_org_df = train_org_df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

train_df = train_org_df.copy()

diff = train_df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().shift(-1).fillna(pd.Timedelta(seconds=-1))
diff = diff.fillna(pd.Timedelta(seconds=-1))
diff = diff['Timestamp'].apply(lambda x: x.total_seconds())

train_df['elapsed'] = diff

tmp = ""
idx = []

for i in tqdm(train_df.index):
    if tmp == train_df.loc[i, "testId"]:
        continue
    else:
        tmp = train_df.loc[i, "testId"]
        idx.append(i)

train_df.loc[idx, "elapsed"] = -1
train_df.loc[train_df.elapsed > 250, "elapsed"] = -1
train_df.loc[train_df.elapsed == -1, "elapsed"] = train_df.loc[train_df.elapsed != -1, "elapsed"].mean()
#changed data save location
train_df.to_csv("../archive/caffeine_data/train_data_add_elapsed.csv", index=False)

# dtype = {
#     'userID': 'int16',
#     'answerCode': 'int8',
#     'KnowledgeTag': 'int16'
# }   
dtype = { #This is added to fit our actual data
    'userID': 'object',
    'answerCode': 'bool',
    'KnowledgeTag': 'object',
    'assessmentItemID': 'object',
    'testId': 'object'
}
# 데이터 경로 맞춰주세요!
DATA_PATH = '../archive/caffeine_data/df_test_make_elapsed.csv'
test_org_df = pd.read_csv(DATA_PATH, dtype=dtype, parse_dates=['Timestamp'])
test_org_df = test_org_df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

test_df = test_org_df.copy()

diff = test_df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().shift(-1).fillna(pd.Timedelta(seconds=-1))
diff = diff.fillna(pd.Timedelta(seconds=-1))
diff = diff['Timestamp'].apply(lambda x: x.total_seconds())

test_df['elapsed'] = diff

tmp = ""
idx = []

for i in tqdm(test_df.index):
    if tmp == test_df.loc[i, "testId"]:
        continue
    else:
        tmp = test_df.loc[i, "testId"]
        idx.append(i)

test_df.loc[idx, "elapsed"] = -1
test_df.loc[test_df.elapsed > 250, "elapsed"] = -1
test_df.loc[test_df.elapsed == -1, "elapsed"] = test_df.loc[test_df.elapsed != -1, "elapsed"].mean()
# Changed directory
#test_df.to_csv("/opt/ml/input/data/train_dataset/test_data_add_elapsed.csv", index=False) ../archive/caffeine_data/
test_df.to_csv("../archive/caffeine_data/test_data_add_elapsed.csv", index=False)