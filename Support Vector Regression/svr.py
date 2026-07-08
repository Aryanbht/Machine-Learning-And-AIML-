import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler #It demands 2d array 
from sklearn.svm import SVR 


dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[ : ,1:-1].values
y = dataset.iloc[ : , -1].values

y = np.reshape(y , (len(y) , 1) ) 

# Feature Scaling

sc = StandardScaler()
sc2 = StandardScaler()
x = sc.fit_transform(x)
y = sc2.fit_transform(y)

regressor = SVR(kernel = 'rbf')
regressor.fit(x , y)

y_pred = sc2.inverse_transform(regressor.predict(sc.transform([[6.5]])).reshape(-1,1))
# print(y_pred)

plt.scatter(sc.inverse_transform(x) , sc2.inverse_transform(y) , color = 'red')
plt.plot(sc.inverse_transform(x) , sc2.inverse_transform(regressor.predict(x).reshape(-1,1)) , color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()



# print(x)
# print(y)
