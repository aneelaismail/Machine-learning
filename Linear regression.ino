import numpy as py
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('my_data.csv')
dataset
y=dataset.iloc[:,1].values
y
x=dataset.iloc[:,:-1].values
x

#Training and Testing Data (divide the data into two part)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.3, random_state=0)

X_train
X_test
from sklearn.linear_model import LinearRegression
reg= LinearRegression()
reg.fit(X_train,y_train)
y_predict=reg.predict(X_test)
y_predict

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,reg.predict(X_train),color='blue')
plt.title('Linear Regression Salary Vs Experience')
plt.xlabel('Year of Employee')
plt.ylabel('Salary of Employee')
plt.show()
  
plt.scatter( X_test, y_test,color='red')
plt.plot(X_train,reg.predict(X_train),color='blue')
plt.title('Linear Regression Salary Vs Experience')
plt.xlabel('Year of Employee')
plt.ylabel('Salary of Employee')
plt.show()
  
