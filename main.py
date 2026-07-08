"""
===============================================================================
Employee Salary Prediction using Linear Regression
===============================================================================
"""

import os
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)


def save_figure(filename):
    filepath = os.path.join(IMAGES_DIR, filename)
    plt.savefig(filepath, bbox_inches="tight")
    print(f"Saved: {filepath}")
    plt.close()


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_dataset():

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    dataset_path = kagglehub.dataset_download(
        "rkiattisak/salaly-prediction-for-beginer"
    )

    print("Dataset Path:", dataset_path)

    csv_file = None

    for file in os.listdir(dataset_path):
        if file.endswith(".csv"):
            csv_file = os.path.join(dataset_path, file)
            break

    if csv_file is None:
        raise FileNotFoundError("CSV file not found.")

    df = pd.read_csv(csv_file)

    print("\nOriginal Columns:")
    print(df.columns.tolist())

    rename_dict = {}

    for col in df.columns:

        name = col.lower().strip()

        if "experience" in name:
            rename_dict[col] = "YearsExperience"

        elif "salary" in name:
            rename_dict[col] = "Salary"

    df.rename(columns=rename_dict, inplace=True)

    print("\nRenamed Columns:")
    print(df.columns.tolist())
    print("\nFirst Five Rows")
    print(df.head())

    return df

# ==============================================================================
# EXPLORATORY DATA ANALYSIS
# ==============================================================================
def perform_eda(df):

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\nShape")
    print(df.shape)

    print("\nColumns")
    print(df.columns.tolist())

    print("\nInformation")
    print(df.info())

    print("\nDescription")
    print(df.describe())

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())

    print("\nCorrelation Matrix")
    print(df.corr(numeric_only=True))  # FIX: avoid error on string columns

# ==============================================================================
# DATA VISUALIZATION
# ==============================================================================

def create_visualizations(df):

    print("=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    numeric_df = df.select_dtypes(include=[np.number])

    # Histogram
    numeric_df.hist(
        figsize=(10, 6),
        bins=15,
        color="steelblue",
        edgecolor="black"
    )
    plt.suptitle("Histogram of Numeric Features")
    save_figure("histogram.png")

    # Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Correlation Heatmap")
    save_figure("heatmap.png")

    # Pairplot
    pair = sns.pairplot(numeric_df)  # FIX: only numeric columns
    pair.fig.suptitle("Pairplot", y=1.02)
    pair.savefig(os.path.join(IMAGES_DIR, "pairplot.png"))
    plt.close("all")

    # Scatter Plot
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        x="YearsExperience",
        y="Salary",
        data=df,
        s=70,
        color="darkorange"
    )
    plt.title("Years of Experience vs Salary")
    save_figure("scatter_plot.png")

    # Boxplot
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=numeric_df)
    plt.title("Boxplot")
    save_figure("boxplot.png")

    # Distribution Plot
    plt.figure(figsize=(7, 5))
    sns.histplot(
        df["Salary"],
        kde=True,
        color="green"
    )
    plt.title("Salary Distribution")
    save_figure("distribution_plot.png")


# ==============================================================================
# DATA PREPROCESSING
# ==============================================================================

def preprocess_data(df):

    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    print("Rows Before Cleaning :", len(df))

    df = df.drop_duplicates()

    df = df.dropna(
        subset=["YearsExperience", "Salary"]
    )

    print("Rows After Cleaning :", len(df))

    return df


# ==============================================================================
# MODEL TRAINING
# ==============================================================================

def train_model(df):

    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    X = df[["YearsExperience"]]
    y = df["Salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    print("Training Completed Successfully")

    print("Training Samples :", len(X_train))
    print("Testing Samples  :", len(X_test))

    return (
        model,
        X_train,
        X_test,
        y_train,
        y_test
    )
# ==============================================================================
# MODEL EVALUATION
# ==============================================================================

def evaluate_model(model, X_test, y_test):

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    y_pred = model.predict(X_test)

    comparison_df = pd.DataFrame({
        "YearsExperience": X_test["YearsExperience"].values,
        "Actual Salary": y_test.values,
        "Predicted Salary": y_pred
    })

    print("\nActual vs Predicted")
    print(comparison_df)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nEvaluation Metrics")
    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    print("\nRegression Equation")
    print(
        f"Salary = {model.coef_[0]:.2f} × Experience + {model.intercept_:.2f}"
    )

    return comparison_df, y_pred


# ==============================================================================
# RESULT VISUALIZATION
# ==============================================================================

def visualize_results(model, df, X_test, y_test, y_pred):

    X = df[["YearsExperience"]]
    y = df["Salary"]

    # Regression Line
    plt.figure(figsize=(8,6))

    plt.scatter(
        X,
        y,
        color="royalblue",
        s=60,
        label="Actual Data"
    )

    plt.plot(
        X,
        model.predict(X),
        color="red",
        linewidth=2,
        label="Regression Line"
    )

    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.title("Linear Regression Fit")
    plt.legend()

    save_figure("regression_line.png")

    # Actual vs Predicted

    plt.figure(figsize=(7,6))

    plt.scatter(
        y_test,
        y_pred,
        color="green",
        s=70
    )

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        linewidth=2
    )

    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title("Actual vs Predicted Salary")

    save_figure("actual_vs_predicted.png")

    # Residual Plot

    residuals = y_test - y_pred

    plt.figure(figsize=(7,6))

    plt.scatter(
        y_pred,
        residuals,
        color="purple",
        s=70
    )

    plt.axhline(
        y=0,
        color="red",
        linestyle="--"
    )

    plt.xlabel("Predicted Salary")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    save_figure("residual_plot.png")


# ==============================================================================
# PREDICTION
# ==============================================================================

def predict_salary(model):

    print("=" * 60)
    print("SALARY PREDICTION")
    print("=" * 60)

    try:

        years = input(
            "\nEnter Years of Experience (Press Enter to Skip): "
        )

        if years.strip() == "":
            print("Prediction Skipped")
            return

        years = float(years)

        prediction = model.predict(
            pd.DataFrame({
                "YearsExperience": [years]
            })
        )[0]

        print(
            f"\nPredicted Salary : ₹{prediction:,.2f}"
        )

    except ValueError:
        print("Please enter a valid number.")


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():

    df = load_dataset()

    perform_eda(df)

    create_visualizations(df)

    df = preprocess_data(df)

    model, X_train, X_test, y_train, y_test = train_model(df)

    comparison_df, y_pred = evaluate_model(
        model,
        X_test,
        y_test
    )

    visualize_results(
        model,
        df,
        X_test,
        y_test,
        y_pred
    )

    predict_salary(model)

    print("\n" + "="*60)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    main()