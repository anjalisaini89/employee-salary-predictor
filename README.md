# 💼 Employee Salary Prediction using Linear Regression

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
- 🧹 Clean, well-commented, PEP 8-compliant code organized into clear sections.

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
├── main.py
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

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE)
file for details.

---

## 🙌 Acknowledgements

- Dataset provided by [rkiattisak on Kaggle](https://www.kaggle.com/rkiattisak).
- Built with ❤️ using Python, scikit-learn, and the open-source data science
  ecosystem.
