# Built-in imports
from datetime import datetime
from datetime import timedelta

# external imports
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
from config import *

df = pd.read_csv("WearableComputing_weight_lifting_exercises_biceps_curl_variations.csv", low_memory=False)

###############################################################################
# Process timestamp
###############################################################################
def parse_timestamp(row):
    """
    Create a timestamp object out of the two given timestamp parts
    """
    ts0 = pd.Timestamp(row['raw_timestamp_part_1'], unit='s')
    ts1 = pd.Timedelta(row['raw_timestamp_part_2'], unit='microseconds')
    return ts0 + ts1

df['timestamp'] = df.apply(parse_timestamp, axis=1)

df.drop(columns=['raw_timestamp_part_1','raw_timestamp_part_2',
                 'cvtd_timestamp'], inplace=True)

df.set_index('timestamp', inplace=True)
df.sort_index(inplace=True)

###############################################################################
# Select features columns to use
###############################################################################

feature_columns = BELT_COLS + ARM_COLS + DUMBBELL_COLS + FOREARM_COLS
columns = feature_columns + ['user_name', 'classe']

df = df.loc[:, df.columns.isin(columns)].copy()

final_df_train, final_df_test = pd.DataFrame(), pd.DataFrame()


###############################################################################
# Train / Test splitting
###############################################################################
for group_name, subdf in df.groupby(['user_name', 'classe']):

    print(group_name)

    train_fraction = 0.7
    train_size = int(subdf.shape[0] * train_fraction)
    subdf_train, subdf_test = subdf[:train_size].copy(), subdf[train_size:].copy()

    new_subdf_train = subdf_train.copy()
    new_subdf_test  = subdf_test.copy()

    for col in feature_columns:
        new_subdf_train['mean_'+col] = subdf_train[col].rolling('1s', min_periods=2).mean()
        new_subdf_train['var_'+col]  = subdf_train[col].rolling('1s', min_periods=2).var()
        new_subdf_train['std_'+col]  = subdf_train[col].rolling('1s', min_periods=2).std()
        new_subdf_train['max_'+col]  = subdf_train[col].rolling('1s', min_periods=2).max()
        new_subdf_train['min_'+col]  = subdf_train[col].rolling('1s', min_periods=2).min()
        new_subdf_train['amp_'+col]  = subdf_train[col].rolling('1s', min_periods=2).apply(lambda x: np.abs(np.min(x)) + np.abs(np.max(x)))
        new_subdf_train['kurt_'+col] = subdf_train[col].rolling('1s', min_periods=2).kurt()
        new_subdf_train['skew_'+col] = subdf_train[col].rolling('1s', min_periods=2).skew()

    for col in feature_columns:
        new_subdf_test['mean_'+col] = subdf_test[col].rolling('1s', min_periods=2).mean()
        new_subdf_test['var_'+col]  = subdf_test[col].rolling('1s', min_periods=2).var()
        new_subdf_test['std_'+col]  = subdf_test[col].rolling('1s', min_periods=2).std()
        new_subdf_test['max_'+col]  = subdf_test[col].rolling('1s', min_periods=2).max()
        new_subdf_test['min_'+col]  = subdf_test[col].rolling('1s', min_periods=2).min()
        new_subdf_test['amp_'+col]  = subdf_test[col].rolling('1s', min_periods=2).apply(lambda x: np.abs(np.min(x)) + np.abs(np.max(x)))
        new_subdf_test['kurt_'+col] = subdf_test[col].rolling('1s', min_periods=2).kurt()
        new_subdf_test['skew_'+col] = subdf_test[col].rolling('1s', min_periods=2).skew()

    new_subdf_train = new_subdf_train.drop(columns=feature_columns)
    final_df_train = pd.concat([final_df_train, new_subdf_train], axis=0)
    final_df_train = final_df_train.drop(columns=['user_name']).copy()

    new_subdf_test = new_subdf_test.drop(columns=feature_columns)
    final_df_test  = pd.concat([final_df_test, new_subdf_test], axis=0)
    final_df_test = final_df_test.drop(columns=['user_name']).copy()

final_df_train.to_csv("train.csv")
final_df_test.to_csv("test.csv")