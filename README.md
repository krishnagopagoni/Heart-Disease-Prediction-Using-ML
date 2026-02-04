# Heart-Disease-Prediction-Using-ML
📌 Project Overview

Heart disease is one of the leading causes of death worldwide. Early detection using data-driven systems can significantly improve prevention and treatment outcomes.

This project builds an end-to-end Machine Learning pipeline to predict whether a patient has heart disease based on clinical and physiological attributes. The final trained model is deployed using a Streamlit web application, allowing users to upload patient data and view predictions interactively.

🎯 Problem Statement

Given patient health data such as age, blood pressure, cholesterol levels, ECG results, and exercise-related indicators, the goal is to predict the binary outcome:

0 → No heart disease

1 → Heart disease present

This is treated as a binary classification problem in the healthcare domain, where minimizing false negatives (missing disease cases) is critical.

📂 Dataset Description

The dataset consists of clinical features including:

Age

Resting blood pressure

Serum cholesterol

Chest pain type

Thallium stress test results (thal)

ECG-based measurements

Exercise-induced angina

Number of major vessels

A unique patient_id is included for traceability but excluded from model training to avoid data leakage.

🛠️ Technologies Used

Python

Pandas, NumPy – Data processing

Scikit-learn – Machine learning models

Matplotlib, Seaborn – Data visualization

Streamlit – Web application deployment

Pickle – Model serialization

🧠 Machine Learning Workflow

Exploratory Data Analysis (EDA)

Distribution analysis

Feature–target relationships

Correlation analysis

Preprocessing

One-hot encoding for categorical features

Feature scaling using StandardScaler

Train–test split with stratification

Model Building

Logistic Regression (baseline)

Random Forest Classifier

Gradient Boosting Classifier (final model)

Model Evaluation

Accuracy

ROC-AUC

Confusion Matrix

Recall for heart disease class

Model Selection

Gradient Boosting selected due to:

Highest ROC-AUC

Zero false negatives

100% recall for heart disease cases

🚀 Streamlit Web Application

The project includes a Streamlit-based web interface that allows users to:

Upload a CSV file containing patient data

Automatically preprocess inputs

View predictions and risk probabilities

See patient-wise medical risk summaries

Example Features of the App

CSV upload support

Patient ID–based result display

High-risk vs low-risk visual indicators
