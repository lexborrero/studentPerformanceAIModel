# 🎓 Student Performance Prediction Using Random Forest (UCI Dataset)

This project demonstrates how to build a basic AI model in Python to predict student performance using data from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/). It uses a **Random Forest Classifier** and serves as a beginner-friendly introduction to machine learning (ML) model development and evaluation.

---

## 📘 Project Overview

**Goal:**  
Predict the performance level of students (e.g., low, medium, high) based on features like study time, family background, and school-related attributes.

**Dataset Source:**  
[UCI Student Performance Dataset (ID: 320)](https://archive.ics.uci.edu/ml/datasets/student+performance)  
Loaded using the `ucimlrepo` Python package.

---

## 🧰 Technologies Used

- Python 3.11+
- `pandas`, `numpy` – Data manipulation
- `sklearn` – Machine learning tools
- `ucimlrepo` – For loading UCI datasets
- `matplotlib` (optional) – Visualization

---

## 📦 Installation & Setup

1. **Install Dependencies:**

```bash
pip install pandas numpy scikit-learn ucimlrepo matplotlib

```
2. **Run the script: **
   
```bash
python student_model.py
```
## 🧠 How It Works

### 1. **Data Loading**

We fetch the dataset using:

```python
from ucimlrepo import fetch_ucirepo  
student_data = fetch_ucirepo(id=320)
```
### 2. Data Preprocessing

- We combine features and targets.

- Categorical features (like school, gender, etc.) are encoded using LabelEncoder.

- The dataset is then split into training and testing sets using train_test_split.

## 3. Model Training
A Random Forest Classifier is trained using:
```python
from sklearn.ensemble import RandomForestClassifier  
model = RandomForestClassifier(n_estimators=100, random_state=42)
```
### 📈 **Linear Regression Model**

Used for predicting **exact grade scores** (regression + rounded classification).

```python
from sklearn.linear_model import LinearRegression

# Train the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
```

# Predict and post-process

```python
y_pred_continuous = lr_model.predict(X_test)
y_pred_class = np.clip(np.round(y_pred_continuous), min_grade, max_grade)
```

## 4. Model Evaluation
We evaluate performance using:
- Accuracy Score
- Classification Report (Precision, Recall, F1-score)
___

## 👩‍💻 Author

**Alexander Borrero**  
AI student & developer @ Purdue| Project-based Learner  
_This is an educational project designed to build intuition and skills around supervised ML._

---

## 📚 References

- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- [ML Crash Course by Google](https://developers.google.com/machine-learning/crash-course)
