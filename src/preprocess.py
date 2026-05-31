import pandas as pd
import numpy as np

#LOAD THE DATASET AND STRIP THE COLUMN NAMES OF ANY LEADING OR TRAILING WHITESPACE

df = pd.read_csv('data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
df.columns = df.columns.str.strip()

#Counts how many missing values are in each column and drops columns with more than 50% missing values

missing = df.isnull().sum()
cols_to_drop = missing[missing > len(df) * 0.5].index
df = df.drop(columns=cols_to_drop)

#Replace infinite values with NaN and then fill NaN values with the median of each column

df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.median(numeric_only=True))

#Select only numeric columns and the label column for further analysis but Label stays cause we need it for now

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Label' not in numeric_cols:
    numeric_cols.append('Label')
df = df[numeric_cols]

#Convert the label column to binary values (0 for benign and 1 for attack)

df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

#Shows how many 0s and 1s we have

print(df['Label'].value_counts())
