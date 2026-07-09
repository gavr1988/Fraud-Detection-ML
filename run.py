import pandas as pd
import numpy as np

# Used to convert categorical/text data into numerical 0/1 columns
from sklearn.preprocessing import OneHotEncoder

# Used to apply different preprocessing steps to numeric and categorical columns
from sklearn.compose import ColumnTransformer

# Used to combine preprocessing and model training into one workflow
from sklearn.pipeline import Pipeline

# The classification model used to predict whether a transaction is fraud or not
from sklearn.tree import DecisionTreeClassifier

# SimpleImputer is included as a safeguard in case future data contains missing values.
# Although the current cleaned dataset has no missing values, this keeps the pipeline robust.
from sklearn.impute import SimpleImputer

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
#Creating a function now to clean the data based on the insights the check data function has provided. 
#This function must be run and undertaken before the data is used for training and testing the model.
#This will return a cleaned dataframe ready for the modelling stage. 

#Now it is time to train the models


def clean_data(df):
    
    # Make a copy so the original dataframe is not changed
    df = df.copy()

    print("Starting cleaning...")
    print("Original shape:", df.shape)

    # 1. Remove exact duplicate rows, if any
    duplicate_count = df.duplicated().sum()
    print("Duplicate rows found:", duplicate_count)

    if duplicate_count > 0:
        df = df.drop_duplicates()
        print("Shape after removing duplicates:", df.shape)

    # 2. Convert Timestamp into a proper datetime column
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # 3. Check if any timestamps failed to convert
    invalid_timestamps = df["Timestamp"].isna().sum()
    print("Invalid timestamps:", invalid_timestamps)

    if invalid_timestamps > 0:
        df = df.dropna(subset=["Timestamp"])
        print("Shape after removing invalid timestamps:", df.shape)

    # 4. Create useful time-based features
    df["Transaction_Hour"] = df["Timestamp"].dt.hour
    df["Transaction_DayOfWeek"] = df["Timestamp"].dt.dayofweek
    df["Transaction_Month"] = df["Timestamp"].dt.month

    # 5. Clean text columns by removing accidental spaces
    text_columns = df.select_dtypes(include="object").columns

    for col in text_columns:
        df[col] = df[col].str.strip()

    # 6. Check target column exists
    if "Fraud_Label" not in df.columns:
        raise ValueError("Fraud_Label column is missing.")

    # 7. Check target values
    print("\nFraud label counts:")
    print(df["Fraud_Label"].value_counts())

    print("\nFraud label percentages:")
    print(df["Fraud_Label"].value_counts(normalize=True))

    print("\nCleaning complete.")
    print("Final shape:", df.shape)

    return df




#Calling the functions to check and clean the data. The cleaned data will be used for training and testing the model.
df = check_data("synthetic_fraud_dataset.csv")

cleaned_df = clean_data(df)

# Save cleaned data as a new CSV file
cleaned_df.to_csv("cleaned_fraud_dataset.csv", index=False)

print("Cleaned CSV file has been saved.")