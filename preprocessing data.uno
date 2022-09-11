import numpy as py
import pandas as pd
dataset = pd.read_csv('dataset.csv')
dataset
dataset.shape
X=dataset[["develop","age","salary"]].values
X
y=dataset[["married"]].values
y
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values="NaN",strategy="mean")
from sklearn.preprocessing import LabelEncoder
label_encode_x=LabelEncoder()
x[:,0]=label_encode_x.fit_transform(x[:,0])
x
from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder()
onehotencoder.fit_transform(dataset.develop.values.reshape(-1,1)).toarray()
label_encode_y=LabelEncoder()
y=label_encode_y.fit_transform(y)
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train
y_train
x_test
y_test
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
x_train
x_test
