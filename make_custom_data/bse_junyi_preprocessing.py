import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import random 


def junyi_preprocessing(problems_csv, users_csv, content_csv, nrows=0):
    if nrows == 0:
        df_problems = pd.read_csv(problems_csv)
    else:
        df_problems = pd.read_csv(problems_csv, nrows=nrows)
    print('read problems')
    df_users = pd.read_csv(users_csv)
    print('read users')
    df_content = pd.read_csv(content_csv)
    print('read content')

    map_dict = {
        'uuid_int': 'userID',
        'upid_int': 'assessmentItemID',
        'level4_id_int':'testId',
        'ucid_int': 'KnowledgeTag',
        'is_correct': 'answerCode',
        'timestamp_TW': 'Timestamp',
        'total_sec_taken': 'elapsed',
    }

    #create new ID's that are only integers, then merge
    df_problems['upid_int'] = random.sample(range(100_000_000, 200_000_000 + 1), len(df_problems))
    df_content['ucid_int'] = random.sample(range(1_000, 10_000 + 1), len(df_content))
    df_users['uuid_int'] = random.sample(range(50_000, 200_000 + 1), len(df_users))
    df_content['level4_id_int'] = random.sample(range(20_000, 40_000 + 1), len(df_content))

    df_merged = pd.merge(df_problems, df_content[['ucid', 'ucid_int', 'level4_id_int']], on='ucid', how='left')
    df_merged = pd.merge(df_merged, df_users[['uuid', 'uuid_int']], on='uuid', how='left')

    #drop unnecessary features
    to_drop =  ['problem_number','exercise_problem_repeat_session', 'total_attempt_cnt', 'used_hint_cnt', 'is_hint_used', 'is_downgrade','is_upgrade', 'level','uuid', 'ucid', 'upid']    
    print(f"cols before drop: {df_merged.columns}")
    df_merged = df_merged.drop(columns=to_drop)
    print(f"cols after drop: {df_merged.columns}")
    df_merged = df_merged.rename(columns=map_dict)
    print(f"colnames after rename: {df_merged.columns}")
    df_train, df_test = train_test_split(df_merged, test_size=0.1, random_state=42069)
    print('done with train test split')
    return df_train, df_test
