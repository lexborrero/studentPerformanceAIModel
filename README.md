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

## 4. Model Evaluation
We evaluate performance using:
- Accuracy Score
- Classification Report (Precision, Recall, F1-score)
