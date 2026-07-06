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
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Columns used by the advanced, multi-feature model
MULTI_FEATURE_NUMERIC = ["Age", "YearsExperience"]
MULTI_FEATURE_CATEGORICAL = ["Education Level", "Job Title"]


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


# ==============================================================================
# ADVANCED: MULTI-FEATURE MODEL (Age, Education Level, Job Title, Experience)
# ==============================================================================
def preprocess_multi_feature_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset for the multi-feature model: drop duplicates and rows
    missing any of the required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw (or already-standardized) dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset containing all required columns.
    """
    required_cols = MULTI_FEATURE_NUMERIC + MULTI_FEATURE_CATEGORICAL + ["Salary"]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Multi-feature model requires column(s) {missing_cols}, which "
            f"were not found. Available columns: {list(df.columns)}"
        )

    df = df.drop_duplicates()
    df = df.dropna(subset=required_cols)
    return df


def train_multi_feature_model(df: pd.DataFrame) -> dict:
    """
    Train a Multiple Linear Regression model using Age, Years of
    Experience, Education Level, and Job Title as predictors of Salary.
    Categorical columns are one-hot encoded inside a scikit-learn Pipeline
    so encoding is applied consistently at both training and prediction
    time.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the required numeric and categorical columns.

    Returns
    -------
    dict
        Dictionary with keys: 'pipeline', 'X_test', 'y_test', 'y_pred',
        'mae', 'mse', 'rmse', 'r2', 'education_levels', 'job_titles'.
    """
    df = preprocess_multi_feature_data(df)

    feature_cols = MULTI_FEATURE_NUMERIC + MULTI_FEATURE_CATEGORICAL
    X = df[feature_cols]
    y = df["Salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # One-hot encode the categorical columns; pass numeric columns through
    # unchanged. handle_unknown="ignore" prevents errors if a category
    # entered at prediction time wasn't seen during training.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                MULTI_FEATURE_CATEGORICAL,
            ),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        "pipeline": pipeline,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "education_levels": sorted(df["Education Level"].dropna().unique().tolist()),
        "job_titles": sorted(df["Job Title"].dropna().unique().tolist()),
    }


def predict_salary_multi(
    pipeline: Pipeline,
    age: float,
    years_experience: float,
    education_level: str,
    job_title: str,
) -> float:
    """
    Predict a salary using the multi-feature model.

    Parameters
    ----------
    pipeline : Pipeline
        The trained multi-feature model pipeline (encoder + regressor).
    age : float
        Employee's age.
    years_experience : float
        Years of professional experience.
    education_level : str
        One of the education levels seen during training (e.g. "Bachelor's").
    job_title : str
        One of the job titles seen during training.

    Returns
    -------
    float
        Predicted salary.
    """
    input_df = pd.DataFrame(
        {
            "Age": [age],
            "YearsExperience": [years_experience],
            "Education Level": [education_level],
            "Job Title": [job_title],
        }
    )
    return float(pipeline.predict(input_df)[0])