
# 🚢 Internship Task 1: Data Cleaning & Preprocessing – Titanic Dataset

## 🎯 Objective

The objective of this task is to learn and demonstrate **data cleaning and preprocessing** techniques on a real-world dataset. We used the Titanic dataset to perform operations like null value handling, outlier treatment, categorical encoding, and feature scaling to prepare the data for machine learning models.

---

## 🧰 Tools & Libraries Used

- **Python**
- **Pandas** – for data manipulation
- **NumPy** – for numerical operations
- **Matplotlib / Seaborn** – for data visualization
- **Scikit-learn** – for feature scaling

---

## 📥 Dataset Description

- Dataset Name: `Titanic-Dataset.csv`
- Shape: `891 rows × 12 columns`
- Source: Loaded locally using `pandas.read_csv()`

---

## 🧾 Variable Interpretation

| Column       | Description |
|--------------|-------------|
| `PassengerId` | Unique ID assigned to each passenger |
| `Survived` | Survival status (0 = No, 1 = Yes) |
| `Pclass` | Passenger class (1 = Upper, 2 = Middle, 3 = Lower) |
| `Name` | Full name of the passenger |
| `Sex` | Gender |
| `Age` | Age in years (can be fractional for children) |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Ticket` | Ticket number |
| `Fare` | Ticket fare (in GBP) |
| `Cabin` | Cabin number (often missing) |
| `Embarked` | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## 📊 Step-by-Step Process

### 🔍 Step 1: Data Overview

- Loaded the dataset using `pd.read_csv()`
- Explored structure using `.head()`, `.tail()`, `.info()`, `.describe()`
- Checked for nulls using `.isnull().sum()`

> **Observations:**
> - 891 total entries, 12 columns
> - Null values in `Age`, `Cabin`, and `Embarked`
> - Categorical columns: `Sex`, `Embarked`
> - Numerical columns: `Age`, `Fare`, `SibSp`, `Parch`, etc.

---

### 🧼 Step 2: Data Preprocessing

#### a) Data Cleaning
- Dropped irrelevant columns: `PassengerId`, `Name`, `Ticket`
- Renaming not needed as column names were already meaningful

#### b) Null Value Handling

##### 🧹 Manufacturing Rules:
1. **< 15% nulls** → drop rows
2. **20%–70% nulls** → impute
3. **> 75% nulls** → drop column

##### 🧠 Imputation:
- Numerical (`Age`) → filled with **mean**
- Categorical (`Embarked`) → filled with **mode**
- `Cabin` column → dropped (77% missing)

#### c) Outlier Treatment
- Visualized with boxplots (`Age`, `Fare`)
- Detected outliers treated using **IQR method**
- Outliers replaced with column mean

---

### 🧾 Step 3: Encoding Categorical Variables

- One-hot encoded `Sex` and `Embarked` using `pd.get_dummies()`
- Converted `True/False` to `1/0` using `.astype(int)`

---

### 📐 Step 4: Feature Scaling

- Applied **StandardScaler** to normalize `Fare`
- Standardization not done on `Age` since some models (like trees) don’t need it, but can be applied optionally

---

## 📈 Exploratory Data Analysis (EDA)

### 🔗 Feature Correlation

Generated a heatmap of feature correlations:
- `Fare` has a positive correlation (`0.28`) with `Survived`
- `Pclass` and `Fare` are negatively correlated (`-0.64`)
- `SibSp` and `Parch` moderately correlated (`0.41`)
- Weak or no correlation found in some features

---

## ✅ Final Cleaned Dataset

| Column | Description |
|--------|-------------|
| `Survived` | Target variable |
| `Pclass`, `Age`, `SibSp`, `Parch`, `Fare` | Scaled/cleaned numeric features |
| `Sex_male`, `Embarked_Q`, `Embarked_S` | One-hot encoded features |

All columns are now:
- Numeric
- Free of null values
- Scaled or encoded
- Ready for modeling

---

## 📊 Class Balance Check

Used `.value_counts(normalize=True)` on `Survived`:
- Class 0 (Did not survive): ~62%
- Class 1 (Survived): ~38%
> ✅ Not highly imbalanced – safe for basic modeling without resampling

---

## 💾 Output Files

- `Titanic_Cleaned.csv`: Cleaned dataset (optional to export)
- `Internship_task_1.ipynb`: Jupyter notebook with all code and steps
- `README.md`: This documentation file

---

## 📚 Learnings

- How to explore and clean real-world data
- Null value handling strategies with practical rules
- Outlier visualization and treatment
- Categorical encoding and feature scaling
- Interpreting data distribution and correlation for ML readiness

---

## 📌 Next Steps (Optional)

- Train classification models (Logistic Regression, Decision Trees, etc.)
- Perform univariate, bivariate, and multivariate EDA
- Feature selection using importance or RFE
- Build a prediction system with interpretability

---

## 🧠 Author Note

This task was completed as part of a **data science internship assignment**. The focus was on building a strong foundation in data preprocessing, which is the most critical stage before any ML model development.

---
