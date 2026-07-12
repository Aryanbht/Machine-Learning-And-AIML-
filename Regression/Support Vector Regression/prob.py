import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


dataset = pd.read_csv('data.csv') 
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = np.reshape(y, (len(y), 1))


sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)


regressor = SVR(kernel='rbf')
regressor.fit(x, y.ravel()) 


y_pred_scaled = regressor.predict(x).reshape(-1, 1)


x_original = sc_x.inverse_transform(x)
y_original = sc_y.inverse_transform(y)
y_pred_original = sc_y.inverse_transform(y_pred_scaled)


plt.scatter(x_original, y_original, color='red', label='Actual Data')
plt.plot(x_original, y_pred_original, color='blue', label='SVR Prediction')
plt.title('SVR Regression Result')
plt.xlabel('Features')
plt.ylabel('Target')
plt.legend()
plt.show()