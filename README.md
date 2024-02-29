# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: kamaleshwar kv
RegisterNumber: 212223240063 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

## Output:
Dataset:
![image](https://github.com/Kamaleshwa/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144980199/7357464e-0c4c-4023-af7f-75cb8877c7e2)


Head values:
![image](https://github.com/Kamaleshwa/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144980199/00bf3582-843e-42f4-a9a7-bc7580ba720f)


Tail values:
![image](https://github.com/Kamaleshwa/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144980199/f15d94c6-3518-420f-8e57-f74ad9505b39)


X and Y values:
![image](https://github.com/Kamaleshwa/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144980199/0f9a489b-5b83-466d-a9a9-32da20f85f6a)


Predication values of X and Y:
![image](https://github.com/Kamaleshwa/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144980199/1fc9decc-4949-42dd-9955-e7a858616c94)


MSE,MAE and RMSE:
![image](https://github.com/Kamaleshwa/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144980199/45303e37-77a1-4866-8164-4c5ab4e4afc6)


Training Set:
![image](https://github.com/Kamaleshwa/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144980199/5ed3511f-d5ca-4a64-b586-030bb9a7fa27)


Testing Set:
![image](https://github.com/Kamaleshwa/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144980199/99b1bc0a-e7ca-4f25-aa0e-1846381ffb71)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
