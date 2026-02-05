import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
#nan missing values
from sklearn.impute import SimpleImputer
import numpy as np

df = pd.read_csv("Day2-3/LinearRegressionModel/HousingData.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X_train)

X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

model = LinearRegression()

model.fit(X_train, y_train)

prediction = model.predict(X_test)

mae = mean_absolute_error(y_test, prediction)
print("MAE: ", mae)

mse = mean_squared_error(y_test, prediction)
print("MSE: ", mse)

print("Predictions: " , prediction)
print("Real values: ", y_test)



