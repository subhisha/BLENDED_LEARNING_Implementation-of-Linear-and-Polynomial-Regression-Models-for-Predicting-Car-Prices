# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Collect and preprocess the car price dataset (clean data and split into train/test sets).

2.Train a Linear Regression model and evaluate its performance.

3.Transform features to polynomial form, train a Polynomial Regression model, and evaluate it.

4.Compare both models and select the one with better prediction accuracy.
```
## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import matplotlib.pyplot as plt
df = pd.read_csv('encoded_car_data (1).csv')
print(df.head)
X = df[['enginesize', 'horsepower', 'citympg', 'highwaympg']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
lr= Pipeline([
    ('scaler', StandardScaler()),
    ('model',LinearRegression())
])
lr.fit(X_train, y_train)
y_pred_linear = lr.predict(X_test)
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)
print('Name: SUBHISHA P')
print('Reg. No: 212225040143')
print("linear Regression:")
mae=mean_absolute_error(y_test, y_pred_poly)
mse=mean_squared_error(y_test,y_pred_linear)
print('MSE=',mean_squared_error(y_test,y_pred_linear))
print('MSE=',mean_absolute_error(y_test,y_pred_linear))
r2score=r2_score(y_test,y_pred_linear)
print('R2 Score=',r2score)
print("\nPolynomial Regression:")
print(f"MSE: {mean_squared_error(y_test, y_pred_poly):.2f}")
print(f"R^2: {r2_score(y_test, y_pred_poly):.2f}")plt.figure(figsize=(10,5))
print(f"MAE: {mean_squared_error(y_test, y_pred_poly):.2f}")
plt.scatter(y_test, y_pred_linear, label='Linear', alpha=0.5)
plt.scatter(y_test, y_pred_poly, label='Polynomial(degree=2)', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--' ,label='Perfect Prediction')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()
```

## Output:
<img width="797" height="777" alt="image" src="https://github.com/user-attachments/assets/510dc3ba-c101-4f46-8207-ea9837f858e4" />

<img width="722" height="543" alt="image" src="https://github.com/user-attachments/assets/87d1bb3f-3d45-4aa1-8fd5-43cf20c96d5f" />

<img width="1173" height="771" alt="image" src="https://github.com/user-attachments/assets/7ebfd643-d574-4f58-b061-bdee0388c7ee" />

## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
