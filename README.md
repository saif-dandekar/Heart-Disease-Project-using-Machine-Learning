❤️ Heart Disease Prediction using Machine Learning
📌 Project Overview

This project focuses on predicting the presence of heart disease using machine learning classification algorithms. The dataset was analyzed using exploratory data analysis (EDA), cleaned, preprocessed, and multiple models were trained and evaluated to identify the best-performing algorithm.

The goal of this project is to demonstrate end-to-end machine learning workflow including data preprocessing, visualization, model building, and performance comparison.

📊 Dataset

Source: Heart Disease Dataset

Target Variable: HeartDisease (0 = No, 1 = Yes)

Features include: Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Max Heart Rate, and more.

🛠️ Tools & Technologies

Python

Pandas, NumPy

Matplotlib, Seaborn

scikit-learn

🔍 Exploratory Data Analysis (EDA)

Checked missing and duplicate values

Replaced zero values in Cholesterol and RestingBP with mean values

Visualized data using:

Histograms

Count plots

Box plots

Violin plots

Correlation heatmap

⚙️ Data Preprocessing

Handled missing and zero values

Encoded categorical features using:

One-Hot Encoding

Label Encoding

Applied StandardScaler for feature scaling

Split data into training and testing sets (80/20)

🤖 Machine Learning Models Used

Logistic Regression

K-Nearest Neighbors (KNN)

Naive Bayes

Decision Tree

Support Vector Machine (SVM)

📈 Model Performance Comparison
| Model               | Accuracy | F1-Score |
| ------------------- | -------- | -------- |
| Logistic Regression | 86.96%   | 0.8857   |
| KNN                 | 86.41%   | 0.8815   |
| Naive Bayes         | 85.33%   | 0.8683   |
| Decision Tree       | 79.35%   | 0.8155   |
| SVM                 | 84.78%   | 0.8679   |

✅ Logistic Regression achieved the best overall performance.
