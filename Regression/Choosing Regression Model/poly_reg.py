import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

dataset = pd.read_csv('data.csv') 
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


poly_reg = PolynomialFeatures(degree = 4)

x_poly = poly_reg.fit_transform(x_train)
regressor = LinearRegression()
regressor.fit(x_poly , y_train)

y_pred = regressor.predict(poly_reg.transform(x_test))
np.printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred) , 1) , y_test.reshape(len(y_test) , 1)) , 1))

# Evaluating Model
print(r2_score(y_test ,y_pred))
# 0.9442738891819027