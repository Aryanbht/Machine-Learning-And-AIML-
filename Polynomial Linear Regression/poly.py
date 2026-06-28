import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[ : , 1 : -1].values
y = dataset.iloc[ : , -1].values 

# Training the Linear Regression model on the whole dataset

lin_regressor = LinearRegression()
lin_regressor.fit(x, y)

pol_regressor = PolynomialFeatures(degree = 4)
x_poly = pol_regressor.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y) 

plt.subplot(1, 2, 1)
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_regressor.predict(x), color = 'blue')
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')


plt.subplot(1, 2, 2)
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(x_poly), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predincting Salary
print(lin_regressor.predict([[6.5]]))

# predicting Salary Accurately
print(lin_reg_2.predict(pol_regressor.fit_transform([[6.5]])))



