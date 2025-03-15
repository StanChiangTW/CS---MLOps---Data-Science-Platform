# preprocessing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from your environment, you might import seaborn or anything else you need
# import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

def check_missing_duplicates(df: pd.DataFrame) -> None:
    """
    Print missing values and duplicated rows info.
    """
    print("Missing Values:\n", df.isnull().sum())
    print("\nDuplicates:", df.duplicated().sum())

def encode_categorical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns as per your mappings:
      - Attrition_Flag, Gender -> LabelEncoder
      - Education_Level -> custom mapping
      - Marital_Status -> LabelEncoder
      - Income_Category -> custom mapping
      - Card_Category -> custom mapping
    Returns a new DataFrame with transformed columns.
    """
    df = df.copy()  # avoid modifying the original DataFrame in place

    # 1. Attrition_Flag
    attrition_encoder = LabelEncoder()
    df["Attrition_Flag"] = attrition_encoder.fit_transform(df["Attrition_Flag"])

    # 2. Gender
    gender_encoder = LabelEncoder()
    df["Gender"] = gender_encoder.fit_transform(df["Gender"])

    # 3. Education_Level
    education_map = {
        'Uneducated': 0,
        'High School': 1,
        'College': 2,
        'Graduate': 3,
        'Post-Graduate': 4,
        'Doctorate': 5,
        'Unknown': 6
    }
    df["Education_Level"] = df["Education_Level"].replace(education_map)

    # 4. Marital_Status
    marital_encoder = LabelEncoder()
    df["Marital_Status"] = marital_encoder.fit_transform(df["Marital_Status"])

    # 5. Income_Category
    income_map = {
        'Less than $40K': 0,
        '$40K - $60K': 1,
        '$60K - $80K': 2,
        '$80K - $120K': 3,
        '$120K +': 4,
        'Unknown': 5
    }
    df["Income_Category"] = df["Income_Category"].replace(income_map)

    # 6. Card_Category
    card_map = {
        'Blue': 0,
        'Silver': 1,
        'Gold': 2,
        'Platinum': 3
    }
    df["Card_Category"] = df["Card_Category"].replace(card_map)

    return df

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are:
      - Automatic classifier columns
      - Highly correlated columns
      - Columns that do not add useful info (e.g., 'CLIENTNUM')
    """
    df = df.copy()

    # Drop columns added by naive bayes classifier
    nb_cols = [
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ]
    for col in nb_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Drop 'Avg_Open_To_Buy' if you found it highly correlated
    if 'Avg_Open_To_Buy' in df.columns:
        df.drop(['Avg_Open_To_Buy'], axis=1, inplace=True)

    # Drop 'CLIENTNUM' if not useful
    if 'CLIENTNUM' in df.columns:
        df.drop(['CLIENTNUM'], axis=1, inplace=True)

    return df

def scale_data(df: pd.DataFrame, target_col: str = "Attrition_Flag") -> (pd.DataFrame, pd.Series):
    """
    - Split the DataFrame into features (X) and target (y).
    - Scale features using MinMaxScaler.
    - Return scaled X and original y.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled, y

def apply_pca(X: pd.DataFrame, n_components: int = 9) -> pd.DataFrame:
    """
    Apply PCA with the specified number of components on X.
    Return the transformed DataFrame.
    """
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(X)
    X_pca = pd.DataFrame(transformed)
    return X_pca

def split_and_oversample(
    X: pd.DataFrame,
    y: pd.Series,
    test_size=0.2,
    random_state=42
) -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
    """
    Train-test split + RandomOverSampler (to handle class imbalance).
    Returns X_train, X_test, y_train, y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    ros = RandomOverSampler(random_state=random_state)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test

def preprocess_data(filepath: str) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    """
    End-to-end preprocessing pipeline:
      1. Load CSV
      2. Encode columns
      3. Drop unneeded columns
      4. Scale data
      5. Apply PCA (optionally with 9 components)
      6. Train-test split + oversampling
    Returns X_train, X_test, y_train, y_test
    """
    # 1. Load CSV
    df = pd.read_csv(filepath)

    # Optionally do quick checks
    # check_missing_duplicates(df)

    # 2. Encoding
    df_encoded = encode_categorical_data(df)

    # 3. Drop columns
    df_clean = drop_unnecessary_columns(df_encoded)

    # 4. Scale data
    X_scaled, y = scale_data(df_clean, target_col="Attrition_Flag")

    # 5. PCA
    X_pca = apply_pca(X_scaled, n_components=9)

    # 6. Split + oversample
    X_train_res, X_test, y_train_res, y_test = split_and_oversample(X_pca, y)

    return X_train_res, X_test, y_train_res, y_test
