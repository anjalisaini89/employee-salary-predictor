"""
===============================================================================
model_utils.py
===============================================================================
Shared utility module for the Employee Salary Prediction project.

This module centralizes dataset loading and model training so that both the
Streamlit app (app_streamlit.py) and the Gradio app (app_gradio.py) reuse the
exact same logic instead of duplicating it.
===============================================================================
"""

import os

import kagglehub
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np


def load_dataset() -> pd.DataFrame:
    """
    Download the dataset from KaggleHub and automatically detect the CSV
    file inside the downloaded folder (no hardcoded filenames).

    Returns
    -------
    pd.DataFrame
        The raw dataset.
    """
    dataset_path = kagglehub.dataset_download(
        "rkiattisak/salaly-prediction-for-beginer"
    )

    csv_file = None
    for file_name in os.listdir(dataset_path):
        if file_name.lower().endswith(".csv"):
            csv_file = os.path.join(dataset_path, file_name)
            break

    if csv_file is None:
        raise FileNotFoundError("No CSV file found in the downloaded dataset.")

    df = pd.read_csv(csv_file)
    df = standardize_columns(df)
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names so the rest of the pipeline can reliably use
    'YearsExperience' and 'Salary', regardless of the exact spelling,
    spacing, or casing used in the source CSV (e.g. 'Years of Experience').

    Parameters
    ----------
    df : pd.DataFrame
        The raw, freshly loaded dataset.

    Returns
    -------
    pd.DataFrame
        The dataset with standardized 'YearsExperience' and 'Salary'
        column names.
    """
    df = df.rename(columns=lambda col: col.strip())

    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if "experience" in col_lower and "YearsExperience" not in rename_map.values():
            rename_map[col] = "YearsExperience"
        elif "salary" in col_lower and "Salary" not in rename_map.values():
            rename_map[col] = "Salary"

    df = df.rename(columns=rename_map)

    missing = [c for c in ("YearsExperience", "Salary") if c not in df.columns]
    if missing:
        raise KeyError(
            f"Could not find required column(s) {missing} in the dataset. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset: drop duplicates and rows missing the required
    columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    df = df.drop_duplicates()
    df = df.dropna(subset=["YearsExperience", "Salary"])
    return df


def train_and_evaluate(df: pd.DataFrame) -> dict:
    """
    Train a Linear Regression model on the given dataset and compute
    evaluation metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset containing 'YearsExperience' and 'Salary' columns.

    Returns
    -------
    dict
        A dictionary containing the trained model, train/test splits,
        predictions, and evaluation metrics. Keys:
        'model', 'X_train', 'X_test', 'y_train', 'y_test', 'y_pred',
        'mae', 'mse', 'rmse', 'r2', 'slope', 'intercept'.
    """
    X = df[["YearsExperience"]]
    y = df["Salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "slope": slope,
        "intercept": intercept,
    }


def get_data_and_model() -> tuple:
    """
    Convenience function that loads the dataset, preprocesses it, and
    trains the model in one call.

    Returns
    -------
    tuple
        (df, results_dict) where df is the cleaned DataFrame and
        results_dict is the output of train_and_evaluate().
    """
    df = load_dataset()
    df = preprocess_data(df)
    results = train_and_evaluate(df)
    return df, results


def predict_salary(model: LinearRegression, years_experience: float) -> float:
    """
    Predict a salary for a given number of years of experience.

    Parameters
    ----------
    model : LinearRegression
        The trained regression model.
    years_experience : float
        Years of professional experience.

    Returns
    -------
    float
        Predicted salary.
    """
    input_df = pd.DataFrame({"YearsExperience": [years_experience]})
    return float(model.predict(input_df)[0])
