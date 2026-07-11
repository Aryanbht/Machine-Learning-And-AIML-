import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[: , 1:-1].values
y = dataset.iloc[: , -1].values

regressor = RandomForestRegressor(n_estimators=10 , random_state=0)
regressor.fit(x , y)
print(regressor.predict([[6.5]])) 


x_grid = np.arange(x.min() , x.max() , 0.1)
x_grid = x_grid.reshape((len(x_grid) , 1))
plt.scatter(x , y , color = 'red')
plt.plot(x_grid , regressor.predict(x_grid) , color ='blue')
plt.show()


