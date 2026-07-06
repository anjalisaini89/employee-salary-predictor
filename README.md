# 💼 Employee Salary Prediction using Linear Regression

🔗 **[Try the Live App](https://employee-salary-predictor-6x4xrdfy3waxafhbaruclv.streamlit.app/)**


A complete, production-quality Machine Learning project that predicts employee
salaries based on years of experience using **Simple Linear Regression**. The
project covers the full ML lifecycle — automated dataset acquisition,
exploratory data analysis (EDA), data visualization, model training,
evaluation, and interactive prediction — following clean coding practices and
PEP 8 standards.

---

## 📌 Project Description

This project builds a Simple Linear Regression model to predict an employee's
salary based on their years of professional experience. The dataset is
automatically downloaded from **KaggleHub**, explored through a thorough EDA
process, visualized with multiple professional charts, and used to train a
regression model that is evaluated with standard error metrics. The script
also allows users to interactively input years of experience and receive a
predicted salary in real time.

The project is designed to be **beginner-friendly yet professional**, making
it suitable for GitHub portfolios, resumes, and internship/job applications.

---

## ✨ Features

- 📥 **Automatic dataset download** from KaggleHub — no manual file handling.
- 🔍 **Automatic CSV detection** using `os.listdir()` — no hardcoded filenames.
- 📊 **Complete EDA**: head/tail preview, shape, info, statistical summary,
  missing values, duplicate checks, and correlation matrix.
- 📈 **Rich visualizations**: histograms, correlation heatmap, pairplot,
  scatter plot, boxplot, and distribution plot.
- 🤖 **Linear Regression model** trained with scikit-learn.
- 🧮 **Full evaluation suite**: MAE, MSE, RMSE, and R² Score.
- 📐 **Regression equation** printed in the form `Salary = m × Experience + c`.
- 🖼️ **All graphs automatically saved** to the `images/` folder.
- ⌨️ **Interactive prediction** — enter years of experience and get an instant
  salary prediction.
- 🧬 **Advanced multi-feature prediction** — a second model that also factors
  in Age, Education Level, and Job Title for a more accurate estimate,
  available in both the console script's data and the Streamlit app.
- 🧹 Clean, well-commented, PEP 8-compliant code organized into clear sections.
- 🌐 **Deployable web apps** — interactive Streamlit and Gradio interfaces for
  live salary prediction (see [Deployment](#-deployment-streamlit--gradio-web-apps)).

---

## 🛠️ Technologies Used

| Technology         | Purpose                                   |
|---------------------|--------------------------------------------|
| Python 3.9+          | Core programming language                 |
| pandas               | Data loading and manipulation              |
| numpy                | Numerical computations                     |
| matplotlib           | Data visualization                         |
| seaborn              | Statistical data visualization             |
| scikit-learn         | Machine Learning (Linear Regression)       |
| kagglehub            | Automated dataset download from Kaggle     |

---

## 📂 Dataset Information

- **Source:** [Kaggle - Salary Prediction for Beginners](https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer)
- **Handle:** `rkiattisak/salaly-prediction-for-beginer`
- **Description:** A dataset containing employee attributes such as Age,
  Gender, Education Level, Job Title, Years of Experience, and Salary. This
  project uses `YearsExperience` as the input feature and `Salary` as the
  target variable.
- **Access Method:** Automatically downloaded at runtime via the `kagglehub`
  Python library — no manual download required.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/employee-salary-prediction.git
cd employee-salary-prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Kaggle API credentials

`kagglehub` requires Kaggle API authentication to download datasets:

1. Go to your [Kaggle Account Settings](https://www.kaggle.com/settings) and
   click **"Create New Token"** to download `kaggle.json`.
2. Place the file at:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
3. Alternatively, `kagglehub` will prompt for credentials interactively on
   first run.

---

## ▶️ How to Run

Simply run the main script:

```bash
python main.py
```

The script will:
1. Download the dataset automatically.
2. Perform EDA and print results to the console.
3. Generate and save all visualizations to the `images/` folder.
4. Train and evaluate the Linear Regression model.
5. Prompt you to enter years of experience for a live salary prediction.

**Example prompt:**

```
Enter Years of Experience to predict salary (or press Enter to skip): 5
Predicted Salary for 5.0 years of experience: 70,245.31
```

---

## 📁 Folder Structure

```
employee-salary-prediction/
│
├── images/
│   ├── histogram.png
│   ├── heatmap.png
│   ├── pairplot.png
│   ├── scatter.png
│   ├── boxplot.png
│   ├── distribution.png
│   ├── regression_line.png
│   ├── actual_vs_predicted.png
│   └── residual_plot.png
│
├── main.py              # Full EDA + training pipeline (console script)
├── model_utils.py        # Shared data-loading & training logic for the web apps
├── app_streamlit.py       # Streamlit web app
├── app_gradio.py          # Gradio web app
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 📊 Sample Output

**Console Output (Excerpt):**

```
--- Model Evaluation Metrics ---
Mean Absolute Error (MAE)      : 3013.48
Mean Squared Error (MSE)       : 13913936.42
Root Mean Squared Error (RMSE) : 3730.14
R-squared (R2) Score           : 0.9936

--- Regression Equation ---
Salary = 8879.92 * Experience + 25049.17
Slope (m)     : 8879.92
Intercept (c) : 25049.17
```

**Comparison Table (Excerpt):**

| Years of Experience | Actual Salary | Predicted Salary |
|----------------------|----------------|--------------------|
| 18.5                 | 193,641.66     | 189,327.64         |
| 13.6                 | 151,904.13     | 145,816.04         |
| 5.3                  | 78,397.00      | 72,112.73          |

---

## 🖼️ Screenshots

All generated charts are available in the `images/` folder after running the
script. Key visualizations include:

| Visualization              | File                          |
|------------------------------|---------------------------------|
| Feature Histograms           | `images/histogram.png`          |
| Correlation Heatmap          | `images/heatmap.png`            |
| Pairplot                     | `images/pairplot.png`           |
| Experience vs Salary Scatter | `images/scatter.png`            |
| Boxplot                      | `images/boxplot.png`            |
| Salary Distribution          | `images/distribution.png`       |
| Regression Line              | `images/regression_line.png`    |
| Actual vs Predicted          | `images/actual_vs_predicted.png`|
| Residual Plot                | `images/residual_plot.png`      |

---

## 📈 Model Evaluation

The Linear Regression model was evaluated using four standard regression
metrics:

| Metric | Description |
|--------|-------------|
| **MAE**  | Average absolute difference between actual and predicted salaries. |
| **MSE**  | Average squared difference — penalizes larger errors more heavily. |
| **RMSE** | Square root of MSE, interpretable in the same units as salary. |
| **R² Score** | Proportion of variance in salary explained by experience (closer to 1.0 is better). |

Given the strong linear relationship between experience and salary in this
dataset, the model typically achieves an **R² score above 0.90**, indicating
excellent predictive performance for a simple linear model.

---

## 🚀 Future Improvements

- Incorporate additional features (Age, Education Level, Job Title) using
  Multiple Linear Regression or advanced models (Random Forest, XGBoost).
- Perform feature encoding for categorical variables (Gender, Job Title).
- Add cross-validation for more robust performance estimation.
- Deploy the model as a web application using Flask/Streamlit.
- Add hyperparameter tuning and model comparison (Ridge, Lasso, Polynomial
  Regression).
- Containerize the project with Docker for easier reproducibility.

---

## 🚀 Deployment (Streamlit & Gradio Web Apps)

In addition to the console script (`main.py`), this project includes two
interactive web apps built on the shared `model_utils.py` module so both
apps stay in sync with the same training logic.

### Option A — Streamlit App

**Run locally:**

```bash
streamlit run app_streamlit.py
```

**Deploy for free on [Streamlit Community Cloud](https://share.streamlit.io):**
1. Push this repository to GitHub.
2. Go to share.streamlit.io → **"New app"** → select your repo and branch.
3. Set the **main file path** to `app_streamlit.py`.
4. Under **Advanced settings → Secrets**, add your Kaggle credentials so
   `kagglehub` can authenticate at runtime:
   ```toml
   KAGGLE_USERNAME = "your_kaggle_username"
   KAGGLE_KEY = "your_kaggle_api_key"
   ```
5. Click **Deploy**. Streamlit Cloud will install `requirements.txt`
   automatically.

### Option B — Gradio App

**Run locally:**

```bash
python app_gradio.py
```

This launches a local server (by default at `http://127.0.0.1:7860`).

**Deploy for free on [Hugging Face Spaces](https://huggingface.co/new-space):**
1. Create a new Space and choose **Gradio** as the SDK.
2. Upload/push all project files, including `requirements.txt` and
   `model_utils.py`.
3. Rename `app_gradio.py` to `app.py` (Hugging Face Spaces looks for
   `app.py` by default), or set `app_file: app_gradio.py` in the Space's
   README metadata block.
4. In the Space **Settings → Repository secrets**, add:
   ```
   KAGGLE_USERNAME = your_kaggle_username
   KAGGLE_KEY = your_kaggle_api_key
   ```
5. The Space will build automatically and expose a public URL.

> **Note:** Both apps download the dataset via `kagglehub` on first load and
> cache the trained model in memory for the session (Streamlit uses
> `@st.cache_resource`; Gradio trains once at startup), so predictions are
> instantaneous after the initial load.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE)
file for details.

---

## 🙌 Acknowledgements

- Dataset provided by [rkiattisak on Kaggle](https://www.kaggle.com/rkiattisak).
- Built with ❤️ using Python, scikit-learn, and the open-source data science
  ecosystem.