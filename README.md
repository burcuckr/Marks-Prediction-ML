# 🏠 Housing Price Prediction (Linear Regression)

This project was developed as a Machine Learning homework assignment.

The goal is to predict housing prices using **Linear Regression**, while handling missing data using **SimpleImputer**.

---

## 🎯 Objective

To practice:

- Linear Regression
- Handling missing values
- Train-test splitting
- Model evaluation

---

## 🧰 Libraries Used

- pandas
- numpy
- scikit-learn

---

## 📁 Dataset

The dataset contains housing-related features and a target variable (house price).

Some values are missing (NaN), which are handled using mean imputation.

---

## ⚙️ Methodology

1. Load dataset  
2. Split into input (X) and output (y)  
3. Divide into training (80%) and testing (20%) sets  
4. Handle missing values using **SimpleImputer (mean strategy)**  
5. Train Linear Regression model  
6. Make predictions  
7. Evaluate using:
   - MAE (Mean Absolute Error)
   - MSE (Mean Squared Error)
