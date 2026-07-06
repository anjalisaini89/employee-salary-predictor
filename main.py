"""
===============================================================================
Employee Salary Prediction using Linear Regression
===============================================================================
Author      : Your Name
Description : A complete, production-quality Machine Learning pipeline that
              predicts employee salaries based on years of experience using
              Simple Linear Regression. The script automatically downloads
              the dataset from KaggleHub, performs Exploratory Data Analysis
              (EDA), trains a Linear Regression model, evaluates it, and
              allows interactive salary prediction from user input.

Dataset     : rkiattisak/salaly-prediction-for-beginer (via KaggleHub)
===============================================================================
"""

# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import os
import sys

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Set a consistent, professional visual style for all plots
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# Directory where all generated graphs will be saved
IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)


def save_figure(filename):
    """
    Save the current matplotlib figure into the 'images' folder.

    Parameters
    ----------
    filename : str
        Name of the image file (e.g., 'heatmap.png').
    """
    filepath = os.path.join(IMAGES_DIR, filename)
    plt.savefig(filepath, bbox_inches="tight")
    print(f"[Saved] Plot saved to: {filepath}")
    plt.close()


# ==============================================================================
# 2. DATA LOADING
# ==============================================================================
def load_dataset():
    """
    Download the dataset from KaggleHub and automatically detect and load
    the CSV file inside the downloaded folder (no hardcoded filenames).

    Returns
    -------
    pd.DataFrame
        The loaded dataset as a pandas DataFrame.
    """
    print("\n" + "=" * 60)
    print("DATA LOADING")
    print("=" * 60)

    # Download the latest version of the dataset from KaggleHub.
    # kagglehub caches the dataset locally after the first download.
    dataset_path = kagglehub.dataset_download(
        "rkiattisak/salaly-prediction-for-beginer"
    )
    print(f"Dataset downloaded to: {dataset_path}")

    # Automatically detect the CSV file in the downloaded directory
    # instead of hardcoding a filename, in case the file name changes.
    csv_file = None
    for file_name in os.listdir(dataset_path):
        if file_name.lower().endswith(".csv"):
            csv_file = os.path.join(dataset_path, file_name)
            break

    if csv_file is None:
        raise FileNotFoundError("No CSV file found in the downloaded dataset.")

    print(f"CSV file detected: {csv_file}")

    # Load the CSV file into a pandas DataFrame
    dataframe = pd.read_csv(csv_file)
    print("Dataset loaded successfully!")

    return dataframe


# ==============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
def perform_eda(df):
    """
    Perform a complete Exploratory Data Analysis on the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze.
    """
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    # First 5 rows of the dataset
    print("\n--- First 5 Rows ---")
    print(df.head())

    # Last 5 rows of the dataset
    print("\n--- Last 5 Rows ---")
    print(df.tail())

    # Shape of the dataset (rows, columns)
    print("\n--- Dataset Shape ---")
    print(df.shape)

    # Number of rows and columns explicitly
    print(f"\nNumber of Rows    : {df.shape[0]}")
    print(f"Number of Columns : {df.shape[1]}")

    # Column names
    print("\n--- Column Names ---")
    print(list(df.columns))

    # Dataset information (data types, non-null counts, memory usage)
    print("\n--- Dataset Info ---")
    df.info()

    # Statistical summary of numeric columns
    print("\n--- Statistical Summary ---")
    print(df.describe())

    # Check for missing values in each column
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    # Check for duplicate rows
    print("\n--- Duplicate Rows ---")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")

    # Correlation matrix (numeric columns only)
    print("\n--- Correlation Matrix ---")
    numeric_df = df.select_dtypes(include=[np.number])
    print(numeric_df.corr())


# ==============================================================================
# 4. DATA VISUALIZATION
# ==============================================================================
def create_visualizations(df):
    """
    Generate and save exploratory visualizations of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to visualize.
    """
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)

    numeric_df = df.select_dtypes(include=[np.number])

    # ---- Histogram for every numeric column ----
    numeric_df.hist(figsize=(12, 8), bins=20, color="steelblue", edgecolor="black")
    plt.suptitle("Histograms of Numeric Columns", fontsize=14)
    save_figure("histogram.png")

    # ---- Correlation heatmap ----
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    save_figure("heatmap.png")

    # ---- Pairplot ----
    pairplot_fig = sns.pairplot(numeric_df)
    pairplot_fig.fig.suptitle("Pairplot of Numeric Features", y=1.02)
    pairplot_fig.savefig(os.path.join(IMAGES_DIR, "pairplot.png"), bbox_inches="tight")
    print(f"[Saved] Plot saved to: {os.path.join(IMAGES_DIR, 'pairplot.png')}")
    plt.close("all")

    # ---- Scatter plot of YearsExperience vs Salary ----
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="YearsExperience", y="Salary", data=df, color="darkorange", s=70)
    plt.title("Years of Experience vs Salary")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    save_figure("scatter.png")

    # ---- Boxplot (to detect outliers in numeric columns) ----
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=numeric_df, palette="Set2")
    plt.title("Boxplot of Numeric Columns")
    save_figure("boxplot.png")

    # ---- Distribution plot of Salary ----
    plt.figure(figsize=(8, 6))
    sns.histplot(df["Salary"], kde=True, color="mediumseagreen")
    plt.title("Distribution Plot of Salary")
    plt.xlabel("Salary")
    save_figure("distribution.png")


# ==============================================================================
# 5. DATA PREPROCESSING
# ==============================================================================
def preprocess_data(df):
    """
    Clean and prepare the dataset for model training.

    Steps performed:
    - Remove duplicate rows.
    - Drop rows with missing values in the relevant columns.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset.

    Returns
    -------
    pd.DataFrame
        The cleaned dataset.
    """
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    initial_rows = df.shape[0]

    # Remove duplicate rows, if any
    df = df.drop_duplicates()

    # Drop rows with missing values in the columns we need
    df = df.dropna(subset=["YearsExperience", "Salary"])

    final_rows = df.shape[0]
    print(f"Rows before cleaning : {initial_rows}")
    print(f"Rows after cleaning  : {final_rows}")

    return df


# ==============================================================================
# 6. MODEL TRAINING
# ==============================================================================
def train_model(df):
    """
    Split the data, train a Linear Regression model, and return the model
    along with the train/test splits.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.

    Returns
    -------
    tuple
        (model, X_train, X_test, y_train, y_test)
    """
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    # Define independent variable (X) and dependent variable (y)
    X = df[["YearsExperience"]]
    y = df["Salary"]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Model training completed successfully!")
    print(f"Training samples : {X_train.shape[0]}")
    print(f"Testing samples  : {X_test.shape[0]}")

    return model, X_train, X_test, y_train, y_test


# ==============================================================================
# 7. MODEL EVALUATION
# ==============================================================================
def evaluate_model(model, X_test, y_test):
    """
    Predict on the test set, build a comparison DataFrame, evaluate the
    model using standard regression metrics, and print the regression
    equation.

    Parameters
    ----------
    model : LinearRegression
        The trained Linear Regression model.
    X_test : pd.DataFrame
        Test set features (YearsExperience).
    y_test : pd.Series
        Test set target values (Salary).

    Returns
    -------
    pd.DataFrame
        Comparison DataFrame of actual vs predicted salaries.
    """
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    # Predict salaries on the test data
    y_pred = model.predict(X_test)

    # Build a comparison DataFrame: Years of Experience, Actual, Predicted
    comparison_df = pd.DataFrame(
        {
            "Years of Experience": X_test["YearsExperience"].values,
            "Actual Salary": y_test.values,
            "Predicted Salary": y_pred,
        }
    ).reset_index(drop=True)

    print("\n--- Actual vs Predicted Salary ---")
    print(comparison_df)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation Metrics ---")
    print(f"Mean Absolute Error (MAE)      : {mae:.2f}")
    print(f"Mean Squared Error (MSE)       : {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE) : {rmse:.2f}")
    print(f"R-squared (R2) Score           : {r2:.4f}")

    # Extract slope (coefficient) and intercept from the model
    slope = model.coef_[0]
    intercept = model.intercept_

    print("\n--- Regression Equation ---")
    print(f"Salary = {slope:.2f} * Experience + {intercept:.2f}")
    print(f"Slope (m)     : {slope:.2f}")
    print(f"Intercept (c) : {intercept:.2f}")

    return comparison_df, y_pred


# ==============================================================================
# 8. RESULT VISUALIZATION
# ==============================================================================
def visualize_results(model, X, y, X_test, y_test, y_pred):
    """
    Create professional visualizations of the model's performance.

    Parameters
    ----------
    model : LinearRegression
        The trained model.
    X : pd.DataFrame
        Full feature set (YearsExperience).
    y : pd.Series
        Full target set (Salary).
    X_test : pd.DataFrame
        Test set features.
    y_test : pd.Series
        Test set target values.
    y_pred : np.ndarray
        Predicted salary values on the test set.
    """
    print("\n" + "=" * 60)
    print("RESULT VISUALIZATION")
    print("=" * 60)

    # ---- Regression line over scatter plot (full dataset) ----
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color="royalblue", label="Actual Data", s=60)
    plt.plot(X, model.predict(X), color="red", linewidth=2, label="Regression Line")
    plt.title("Linear Regression: Experience vs Salary")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.legend()
    save_figure("regression_line.png")

    # ---- Actual vs Predicted plot ----
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color="seagreen", s=70)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Perfect Prediction Line",
    )
    plt.title("Actual vs Predicted Salary")
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.legend()
    save_figure("actual_vs_predicted.png")

    # ---- Residual plot ----
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, color="purple", s=70)
    plt.axhline(y=0, color="red", linestyle="--", linewidth=2)
    plt.title("Residual Plot")
    plt.xlabel("Predicted Salary")
    plt.ylabel("Residuals")
    save_figure("residual_plot.png")


# ==============================================================================
# 9. INTERACTIVE PREDICTION
# ==============================================================================
def predict_salary_from_input(model):
    """
    Prompt the user to enter years of experience from the keyboard and
    predict the corresponding salary using the trained model.

    Parameters
    ----------
    model : LinearRegression
        The trained Linear Regression model.
    """
    print("\n" + "=" * 60)
    print("PREDICTION")
    print("=" * 60)

    try:
        user_input = input(
            "\nEnter Years of Experience to predict salary "
            "(or press Enter to skip): "
        ).strip()

        # Allow the user to skip interactive prediction (useful for
        # automated / non-interactive runs, e.g., CI pipelines).
        if user_input == "":
            print("No input provided. Skipping interactive prediction.")
            return

        years_experience = float(user_input)

        # Build a DataFrame with the correct feature name to avoid
        # sklearn's "feature names" warning.
        input_df = pd.DataFrame({"YearsExperience": [years_experience]})
        predicted_salary = model.predict(input_df)[0]

        print(
            f"\nPredicted Salary for {years_experience} years of "
            f"experience: {predicted_salary:,.2f}"
        )

    except ValueError:
        print("Invalid input. Please enter a numeric value for years of experience.")


# ==============================================================================
# 10. MAIN PIPELINE
# ==============================================================================
def main():
    """
    Execute the complete Employee Salary Prediction pipeline:
    Data Loading -> EDA -> Visualization -> Preprocessing ->
    Model Training -> Evaluation -> Result Visualization -> Prediction.
    """
    # ---- Data Loading ----
    df = load_dataset()

    # ---- Exploratory Data Analysis ----
    perform_eda(df)

    # ---- Visualization (raw data) ----
    create_visualizations(df)

    # ---- Preprocessing ----
    df = preprocess_data(df)

    # ---- Model Training ----
    model, X_train, X_test, y_train, y_test = train_model(df)

    # ---- Evaluation ----
    comparison_df, y_pred = evaluate_model(model, X_test, y_test)

    # ---- Result Visualization ----
    X_full = df[["YearsExperience"]]
    y_full = df["Salary"]
    visualize_results(model, X_full, y_full, X_test, y_test, y_pred)

    # ---- Interactive Prediction ----
    predict_salary_from_input(model)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
