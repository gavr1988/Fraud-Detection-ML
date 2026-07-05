import pandas as pd
import numpy as np


import pandas as pd

#Firstly we create a function to check the data and get a better understanding of it.
# This function will print out the first 5 rows, random 10 rows, shape, columns, info, missing values, duplicate rows, target counts and percentages, numeric summary and unique values per column.
def check_data(file_path, target_col="Fraud_Label"):
    # Checking the data
    df = pd.read_csv(file_path)

    print("First 5 rows:")
    print(df.head())

    print("\nRandom 10 rows:")
    print(df.sample(10, random_state=42))

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    for col in df.columns:
        print(col)

    print("\nInfo:")
    print(df.info())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nTotal missing values:")
    print(df.isna().sum().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nTarget counts:")
    print(df[target_col].value_counts())

    print("\nTarget percentages:")
    print(df[target_col].value_counts(normalize=True))

    print("\nNumeric summary:")
    print(df.describe().T)

    print("\nUnique values per column:")
    for col in df.columns:
        print(col, df[col].nunique())

    return df

df = check_data("synthetic_fraud_dataset.csv")