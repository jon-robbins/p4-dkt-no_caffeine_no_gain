import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import random 


def junyi_preprocessing(problems_csv, users_csv, content_csv):
    df_problems = pd.read_csv(problems_csv)
    print('read problems')
    df_users = pd.read_csv(users_csv)
    print('read users')
    df_content = pd.read_csv(content_csv)
    print('read content')

df_problems = pd.read_csv('archive/Log_Problem.csv')

seed = 42069
def junyi_preprocessing(n=10_000):
    df_problems = df_problems.sample(frac=0.2, random_state=seed)
    df_users = pd.read_csv('archive/Info_UserData.csv').sample(frac=0.2, random_state=seed)
    df_content = pd.read_csv('archive/Info_Content.csv').sample(frac=0.2, random_state=seed)

    map_dict = {
        'uuid_int': 'userID',
        'upid_int': 'assessmentItemID',
        'level4_id_int':'testId',
        'ucid_int': 'KnowledgeTag',
        'is_correct': 'answerCode',
        'timestamp_TW': 'Timestamp'
    }

    df_problems_merged = pd.merge(df_problems, df_content[['ucid', 'level4_id', 'subject']], on='ucid', how='left')
    df_problems_merged = df_problems_merged.rename(columns=map_dict)
    to_drop =  ['problem_number','exercise_problem_repeat_session', 'total_sec_taken','total_attempt_cnt', 'used_hint_cnt', 'is_hint_used', 'is_downgrade','is_upgrade', 'level',]
    df_problems_merged = df_problems_merged.drop(columns=to_drop)
    df_problems_merged_sample = df_problems_merged.sample(n=n)
    df_train, df_test = train_test_split(df_problems_merged_sample, test_size=0.1, random_state=42069)

    df_train.to_csv('../archive/caffeine_data/df_train_make_elapsed.csv', index=False)
    df_test.to_csv('../archive/caffeine_data/df_test_make_elapsed.csv', index=False)
