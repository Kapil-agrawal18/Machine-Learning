import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Salary_Data.csv")

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Training the simple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the test result
Y_Pred = regressor.predict(X_test)
plt.plot(X_test,Y_Pred)
plt.scatter(X_test,Y_test, color='red')
plt.show()

# Predicting training set result
y_Pred = regressor.predict(X_train)
plt.plot(X_train,y_Pred)
plt.scatter(X_train,Y_train, color='red')
plt.show()

print(regressor.predict([[5]]))
print(regressor.predict([[8]]))
print(regressor.predict([[12]]))
print(regressor.predict([[15]]))

