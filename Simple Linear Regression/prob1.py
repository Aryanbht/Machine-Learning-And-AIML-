import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('exercise.csv')

ind_data = dataset.iloc[: , :-1].values
dep_data = dataset.iloc[: , -1].values

ind_data_train , ind_data_test , dep_data_train , dep_data_test = train_test_split(ind_data , dep_data , test_size=0.2 , random_state=0)

regressor = LinearRegression()
regressor.fit(ind_data_train , dep_data_train)

dep_data_pred = regressor.predict(ind_data_test)

plt.subplot(1,2,1)
plt.scatter(ind_data_train,dep_data_train , color = 'red')
plt.plot(ind_data_train , regressor.predict(ind_data_train) , color ='blue')
plt.title('Workout Duration vs Calories')
plt.xlabel('Duration')
plt.ylabel('Calories')

plt.subplot(1,2,2)
plt.scatter(ind_data_test,dep_data_test , color='red')
plt.plot(ind_data_train , regressor.predict(ind_data_train) , color = 'blue')
plt.title('Workout Duration vs Calories')
plt.xlabel('Duration')
plt.ylabel('Calories')

plt.tight_layout()
plt.show()

