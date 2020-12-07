# Built-in imports
from datetime import datetime
from datetime import datetimetimedelta
from datetime import datetimetimezone

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
final_df = pd.DataFrame()

for group_name, subdf in df.groupby('user_name'):
    print(group_name)
    new_subdf = subdf.copy()

    for col in feature_columns:
        new_subdf['mean_'+col] = subdf[col].rolling('1s', min_periods=2).mean()
        new_subdf['var_'+col]  = subdf[col].rolling('1s', min_periods=2).var()
        new_subdf['std_'+col]  = subdf[col].rolling('1s', min_periods=2).std()
        new_subdf['max_'+col]  = subdf[col].rolling('1s', min_periods=2).max()
        new_subdf['min_'+col]  = subdf[col].rolling('1s', min_periods=2).min()
        new_subdf['amp_'+col]  = subdf[col].rolling('1s', min_periods=2).apply(lambda x: np.abs(np.min(x)) + np.abs(np.max(x)))
        new_subdf['kurt_'+col] = subdf[col].rolling('1s', min_periods=2).kurt()
        new_subdf['skew_'+col] = subdf[col].rolling('1s', min_periods=2).skew()

    new_subdf = new_subdf.drop(columns=feature_columns)
    final_df = pd.concat([final_df, new_subdf], axis=0)

final_df.to_csv("processed.csv")