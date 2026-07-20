# Importing Libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.impute import SimpleImputer #(For Missing Data)
from sklearn.compose import ColumnTransformer #(For Column Transforming)
from sklearn.preprocessing import OneHotEncoder #(For Column Transforming)
from sklearn.preprocessing import LabelEncoder #(For Dependent Variable Transforming)
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

# Importing DataSets

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values #(Independent Variable)
y = dataset.iloc[:,-1].values #(Dependent Variable)

# Taking Care of missing Data 

imputer = SimpleImputer(missing_values=np.nan , strategy='mean')
imputer.fit(x[: , 1:3])
x[: , 1:3] = imputer.transform(x[: , 1:3])
# print(x)

# Encoding Categorical Data 
ct = ColumnTransformer(transformers= [('encoder', OneHotEncoder() , [0] )], remainder= 'passthrough')
x = np.array(ct.fit_transform(x))
# print(x)


# Encoding For Dependent Variable 

le = LabelEncoder()
y = le.fit_transform(y)
# print(y)

# Splitting DataSet into Training and Test Set 

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size= 0.2 , random_state=1)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

# Feature Scaling 

sc = StandardScaler()
x_train[: , 3:] = sc.fit_transform(x_train[: , 3:])
x_test[: , 3:] = sc.transform(x_test[: , 3:])

print(x_train)
print(x_test)