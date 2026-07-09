import pandas as pd

# Used for group-aware train/test splitting and hyperparameter tuning
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV

# Used to convert categorical/text data into numerical 0/1 columns
from sklearn.preprocessing import OneHotEncoder

# Used to apply different preprocessing steps to numeric and categorical columns
from sklearn.compose import ColumnTransformer

# Used to combine preprocessing and model training into one workflow
from sklearn.pipeline import Pipeline

# The classification model used to predict whether a transaction is fraud or not
from sklearn.tree import DecisionTreeClassifier

# Used to evaluate the model's performance
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

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

def train_and_predict():
    """
    Train the ML pipeline and make predictions using the cleaned fraud dataset.

    Loads cleaned fraud data, creates features and target,
    uses group-aware splitting based on User_ID,
    trains a Decision Tree Classifier,
    tunes the model using GridSearchCV,
    evaluates performance, and returns the best model.
    """

    # ---------------------------------------------------------------------
    # LOAD CLEANED DATA
    # ---------------------------------------------------------------------

    df = pd.read_csv("cleaned_fraud_dataset.csv")

    print("\nDataset loaded successfully.")
    print("Shape:", df.shape)

    # ---------------------------------------------------------------------
    # TARGET VARIABLE
    # ---------------------------------------------------------------------
    # Fraud_Label is the value the model is trying to predict.
    #
    # Fraud_Label:
    # 1 = fraud
    # 0 = not fraud
    # ---------------------------------------------------------------------

    y = df["Fraud_Label"]

    # ---------------------------------------------------------------------
    # GROUP-AWARE SPLITTING
    # ---------------------------------------------------------------------
    # We use User_ID as the grouping variable.
    #
    # This helps stop transactions from the same user appearing in both:
    # - training data
    # - testing data
    #
    # This gives a more honest test of how well the model generalises to
    # unseen users/transactions.
    #
    # User_ID is used for splitting, but it will NOT be used as a model feature.
    # ---------------------------------------------------------------------

    groups = df["User_ID"]

    print("\nNumber of rows:", len(df))
    print("Number of unique users:", groups.nunique())

    # ---------------------------------------------------------------------
    # FEATURE SET
    # ---------------------------------------------------------------------
    # These columns are dropped before training the model:
    #
    # Fraud_Label is dropped because it is the answer/target column.
    # Transaction_ID is dropped because it is a unique identifier.
    # User_ID is dropped because it is used for grouping, not prediction.
    # Timestamp is dropped because useful time features have already been created.
    # Risk_Score is dropped because it may cause data leakage.
    # ---------------------------------------------------------------------

    X = df.drop(columns=[
        "Fraud_Label",
        "Transaction_ID",
        "User_ID",
        "Timestamp",
        "Risk_Score"
    ])

    print("\nFeature columns used for modelling:")
    for col in X.columns:
        print(col)

    # ---------------------------------------------------------------------
    # GROUP-AWARE TRAIN/TEST SPLIT
    # ---------------------------------------------------------------------

    group_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=101
    )

    train_index, test_index = next(group_splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    groups_train = groups.iloc[train_index]
    groups_test = groups.iloc[test_index]

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Training users:", groups_train.nunique())
    print("Testing users:", groups_test.nunique())

    print("\nOriginal target balance:")
    print(y.value_counts(normalize=True))

    print("\nTraining target balance:")
    print(y_train.value_counts(normalize=True))

    print("\nTesting target balance:")
    print(y_test.value_counts(normalize=True))

    # ---------------------------------------------------------------------
    # IDENTIFY NUMERIC AND CATEGORICAL FEATURES
    # ---------------------------------------------------------------------
    # Decision trees can use numeric data directly.
    # Text/category columns need to be converted into numeric columns.
    # ---------------------------------------------------------------------

    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X_train.select_dtypes(include=["object"]).columns

    print("\nNumeric features:")
    print(numeric_features)

    print("\nCategorical features:")
    print(categorical_features)

    # ---------------------------------------------------------------------
    # PREPROCESSING
    # ---------------------------------------------------------------------
    # We do not need StandardScaler because decision trees do not require
    # feature scaling.
    #
    # Numeric columns are passed through unchanged.
    # Categorical columns are converted into 0/1 columns using OneHotEncoder.
    # ---------------------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # ---------------------------------------------------------------------
    # MODEL PIPELINE
    # ---------------------------------------------------------------------
    # We use DecisionTreeClassifier because this is a binary classification problem.
    #
    # The pipeline applies preprocessing first, then trains the classifier.
    # ---------------------------------------------------------------------

    fraud_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(random_state=101))
        ]
    )

    # ---------------------------------------------------------------------
    # PARAMETER GRID
    # ---------------------------------------------------------------------
    # These are the decision tree settings we want GridSearchCV to test.
    # ---------------------------------------------------------------------

    param_grid = {
        "model__max_depth": [3, 5, 7, 10, None],
        "model__min_samples_leaf": [1, 10, 25, 50, 100],
        "model__criterion": ["gini", "entropy"],
        "model__class_weight": [None, "balanced"]
    }

    # ---------------------------------------------------------------------
    # GROUP-AWARE CROSS-VALIDATION
    # ---------------------------------------------------------------------
    # GroupKFold makes sure the same user does not appear in both the training
    # and validation part of a fold during GridSearchCV.
    # ---------------------------------------------------------------------

    number_of_training_groups = groups_train.nunique()

    if number_of_training_groups < 2:
        raise ValueError(
            "There are not enough unique users for group-aware cross-validation."
        )

    n_splits = min(5, number_of_training_groups)

    cv = GroupKFold(n_splits=n_splits)

    grid_search = GridSearchCV(
        fraud_pipeline,
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    # Fitting the model.
    # The groups=groups_train part is essential because GroupKFold needs to know
    # which rows/users belong together.
    grid_search.fit(X_train, y_train, groups=groups_train)

    # Best model after hyperparameter tuning
    best_model = grid_search.best_estimator_

    print("\nBest Settings:")
    print(grid_search.best_params_)

    # ---------------------------------------------------------------------
    # MODEL EVALUATION
    # ---------------------------------------------------------------------

    print(f"\nTraining Accuracy: {best_model.score(X_train, y_train):.2f}")
    print(f"Testing Accuracy: {best_model.score(X_test, y_test):.2f}")

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Predict probabilities for ROC AUC
    # ROC AUC measures how well the model separates fraud from non-fraud cases.
# It uses the predicted probability of fraud rather than just the final 0/1 prediction.
# A score of 0.5 means the model is no better than random guessing, while a score of 1.0 means perfect separation.
# This is useful for fraud detection because it shows whether the model generally gives higher fraud probabilities to actual fraud cases.
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    print("\nROC AUC Score:")
    print(roc_auc_score(y_test, y_pred_proba))

    return best_model

# ---------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------
# This section runs the project in order:
# 1. Load and check the original dataset
# 2. Clean the dataset
# 3. Save the cleaned dataset as a new CSV file
# 4. Train and evaluate the Decision Tree model
# 5. Use the trained model to make predictions on sample test data

df = check_data("synthetic_fraud_dataset.csv")

cleaned_df = clean_data(df)

cleaned_df.to_csv("cleaned_fraud_dataset.csv", index=False)

print("Cleaned CSV file has been saved.")

best_model = train_and_predict()
# ---------------------------------------------------------------------
# CREATE SAMPLE TEST DATA
# ---------------------------------------------------------------------
# This data is made up manually to test the trained model.
# It must contain the same feature columns that were used to train the model.
#
# We do NOT include:
# Fraud_Label - because this is what we want the model to predict
# Transaction_ID - because it was dropped before training
# User_ID - because it was used for grouping, not prediction
# Timestamp - because we use Transaction_Hour, Transaction_DayOfWeek and Transaction_Month instead
# Risk_Score - because it was dropped to avoid possible data leakage


test_data = pd.DataFrame([
    {
        "Transaction_Amount": 25.99,
        "Transaction_Type": "POS",
        "Account_Balance": 7500.00,
        "Device_Type": "Mobile",
        "Location": "London",
        "Merchant_Category": "Groceries",
        "IP_Address_Flag": 0,
        "Previous_Fraudulent_Activity": 0,
        "Daily_Transaction_Count": 2,
        "Avg_Transaction_Amount_7d": 30.50,
        "Failed_Transaction_Count_7d": 0,
        "Card_Type": "Visa",
        "Card_Age": 120,
        "Transaction_Distance": 5.2,
        "Authentication_Method": "PIN",
        "Is_Weekend": 0,
        "Transaction_Hour": 14,
        "Transaction_DayOfWeek": 2,
        "Transaction_Month": 6
    },
    {
        "Transaction_Amount": 950.00,
        "Transaction_Type": "Online",
        "Account_Balance": 1200.00,
        "Device_Type": "Laptop",
        "Location": "Tokyo",
        "Merchant_Category": "Electronics",
        "IP_Address_Flag": 1,
        "Previous_Fraudulent_Activity": 1,
        "Daily_Transaction_Count": 13,
        "Avg_Transaction_Amount_7d": 450.00,
        "Failed_Transaction_Count_7d": 4,
        "Card_Type": "Mastercard",
        "Card_Age": 12,
        "Transaction_Distance": 4200.0,
        "Authentication_Method": "Password",
        "Is_Weekend": 1,
        "Transaction_Hour": 3,
        "Transaction_DayOfWeek": 6,
        "Transaction_Month": 12
    },
    {
        "Transaction_Amount": 180.00,
        "Transaction_Type": "Bank Transfer",
        "Account_Balance": 25000.00,
        "Device_Type": "Tablet",
        "Location": "New York",
        "Merchant_Category": "Travel",
        "IP_Address_Flag": 0,
        "Previous_Fraudulent_Activity": 0,
        "Daily_Transaction_Count": 7,
        "Avg_Transaction_Amount_7d": 200.00,
        "Failed_Transaction_Count_7d": 2,
        "Card_Type": "Amex",
        "Card_Age": 80,
        "Transaction_Distance": 900.0,
        "Authentication_Method": "OTP",
        "Is_Weekend": 0,
        "Transaction_Hour": 21,
        "Transaction_DayOfWeek": 4,
        "Transaction_Month": 9
    }
])

test_predictions = best_model.predict(test_data)
test_probabilities = best_model.predict_proba(test_data)[:, 1]

test_data["Predicted_Fraud_Label"] = test_predictions
test_data["Predicted_Fraud_Probability"] = test_probabilities

test_data["Prediction_Text"] = test_data["Predicted_Fraud_Label"].map({
    0: "Not Fraud",
    1: "Fraud"
})

test_data["Predicted_Fraud_Probability_Percentage"] = (
    test_data["Predicted_Fraud_Probability"] * 100
).round(2)

print("\nSample Test Predictions:")
print(test_data[[
    "Transaction_Amount",
    "Transaction_Type",
    "Location",
    "Merchant_Category",
    "Prediction_Text",
    "Predicted_Fraud_Probability_Percentage"
]])