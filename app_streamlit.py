"""
===============================================================================
app_streamlit.py
===============================================================================
Streamlit web application for the Employee Salary Prediction project.

Run locally with:
    streamlit run app_streamlit.py

Or deploy for free on Streamlit Community Cloud:
    1. Push this repo to GitHub.
    2. Go to https://share.streamlit.io and connect your repository.
    3. Set the main file path to "app_streamlit.py".
    4. Add your Kaggle API credentials as Streamlit "Secrets" (see README).
===============================================================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from model_utils import get_data_and_model, predict_salary, train_multi_feature_model, predict_salary_multi

# ------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="centered",
)


# ------------------------------------------------------------------------
# DATA & MODEL LOADING (cached so it only runs once per session)
# ------------------------------------------------------------------------
@st.cache_resource(show_spinner="Downloading dataset and training models...")
def load_model_and_data():
    """
    Load the dataset and train both the simple (Experience-only) model and
    the advanced multi-feature (Age, Experience, Education, Job Title)
    model once, then cache the results for the session.
    """
    df, simple_results = get_data_and_model()
    multi_results = train_multi_feature_model(df)
    return df, simple_results, multi_results


# ------------------------------------------------------------------------
# APP HEADER
# ------------------------------------------------------------------------
st.title("💼 Employee Salary Prediction")
st.markdown(
    """
    Predict an employee's salary based on **years of experience** using a
    Simple Linear Regression model trained on the
    [Kaggle Salary Prediction dataset](https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer).
    """
)

# Load data and models (with a spinner shown on first run)
try:
    df, results, multi_results = load_model_and_data()
except Exception as error:  # noqa: BLE001 - surface any loading error to the user
    st.error(
        "Failed to load the dataset or train the model. "
        "Make sure your Kaggle API credentials are configured correctly."
    )
    st.exception(error)
    st.stop()

model = results["model"]
multi_pipeline = multi_results["pipeline"]

# ------------------------------------------------------------------------
# SIDEBAR: MODEL INFO
# ------------------------------------------------------------------------
with st.sidebar:
    st.header("📊 Simple Model (Experience Only)")
    st.metric("R² Score", f"{results['r2']:.4f}")
    st.metric("MAE", f"{results['mae']:,.2f}")
    st.metric("RMSE", f"{results['rmse']:,.2f}")
    st.markdown(
        f"**Regression Equation**  \n"
        f"Salary = {results['slope']:.2f} × Experience + {results['intercept']:.2f}"
    )

    st.markdown("---")

    st.header("🧬 Advanced Model (Multi-Feature)")
    st.metric("R² Score", f"{multi_results['r2']:.4f}")
    st.metric("MAE", f"{multi_results['mae']:,.2f}")
    st.metric("RMSE", f"{multi_results['rmse']:,.2f}")
    st.caption("Features: Age, Experience, Education Level, Job Title")

    st.markdown("---")
    st.caption(f"Dataset size: {df.shape[0]} rows")

# ------------------------------------------------------------------------
# SECTION 1: INTERACTIVE PREDICTION (Quick vs. Advanced)
# ------------------------------------------------------------------------
st.header("🔮 Predict a Salary")

quick_tab, advanced_tab = st.tabs(["⚡ Quick Prediction", "🧬 Advanced Prediction"])

# ---- Quick Prediction: Years of Experience only ----
with quick_tab:
    st.caption("Uses the Simple Linear Regression model (Experience only).")

    years_experience = st.slider(
        "Years of Experience",
        min_value=0.0,
        max_value=float(max(20.0, df["YearsExperience"].max())),
        value=5.0,
        step=0.1,
        key="quick_years",
    )

    if st.button("Predict Salary", type="primary", key="quick_predict_btn"):
        predicted_salary = predict_salary(model, years_experience)
        st.success(f"### 💰 Predicted Salary: {predicted_salary:,.2f}")

# ---- Advanced Prediction: Age, Experience, Education Level, Job Title ----
with advanced_tab:
    st.caption(
        "Uses the Multi-Feature model — Age, Experience, Education Level, "
        "and Job Title all contribute to the prediction."
    )

    col1, col2 = st.columns(2)
    with col1:
        adv_age = st.slider(
            "Age",
            min_value=int(df["Age"].min()) if "Age" in df.columns else 18,
            max_value=int(df["Age"].max()) if "Age" in df.columns else 70,
            value=30,
            key="adv_age",
        )
        adv_education = st.selectbox(
            "Education Level",
            options=multi_results["education_levels"],
            key="adv_education",
        )
    with col2:
        adv_years = st.slider(
            "Years of Experience",
            min_value=0.0,
            max_value=float(max(20.0, df["YearsExperience"].max())),
            value=5.0,
            step=0.1,
            key="adv_years",
        )
        adv_job_title = st.selectbox(
            "Job Title",
            options=multi_results["job_titles"],
            key="adv_job_title",
        )

    if st.button("Predict Salary", type="primary", key="adv_predict_btn"):
        predicted_salary_multi = predict_salary_multi(
            multi_pipeline,
            age=adv_age,
            years_experience=adv_years,
            education_level=adv_education,
            job_title=adv_job_title,
        )
        st.success(f"### 💰 Predicted Salary: {predicted_salary_multi:,.2f}")

# ------------------------------------------------------------------------
# SECTION 2: REGRESSION LINE VISUALIZATION
# ------------------------------------------------------------------------
st.header("📈 Regression Line")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(
    df["YearsExperience"], df["Salary"], color="royalblue", label="Actual Data", s=50
)
X_sorted = df[["YearsExperience"]].sort_values("YearsExperience")
ax.plot(
    X_sorted,
    model.predict(X_sorted),
    color="red",
    linewidth=2,
    label="Regression Line",
)
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Salary")
ax.legend()
st.pyplot(fig)

# ------------------------------------------------------------------------
# SECTION 3: RAW DATA & CORRELATION
# ------------------------------------------------------------------------
with st.expander("🔍 View Raw Dataset"):
    st.dataframe(df)

with st.expander("📊 Correlation Heatmap"):
    numeric_df = df.select_dtypes(include=[np.number])
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    st.pyplot(fig2)

st.markdown("---")
st.caption("Built with Streamlit • Linear Regression • scikit-learn")