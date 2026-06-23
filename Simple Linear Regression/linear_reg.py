import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1].values

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=0)

# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)


# Traing Linear Regression model on the training set 

regressor = LinearRegression()
regressor.fit(x_train ,y_train)


# Predicting the test set result 

y_pred = regressor.predict(x_test)

# Visualization Of Train Set 
plt.subplot(1, 2, 1)  
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')


# Visualization Of Test Set

plt.subplot(1, 2, 2)  
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary Vs Experience (Test Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')

plt.tight_layout()

# plt.show()

# Making a single prediction
print(regressor.predict([[12]]))

# Getting the final linear regression equation with the values of the coefficients

print(regressor.coef_)
print(regressor.intercept_)

# Salary = 9345.94 × YearsExperience + 26816.19

