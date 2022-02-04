import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1:].values

plt.scatter(X, Y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

theta = np.array([[0],[0]])
print(theta)
m = np.size(Y)
one = np.ones((m,1), dtype=int)
X = np.concatenate((one,X), axis=1)
X_T = X.transpose()

print(np.std(Y))

# applying cost function 
alpha = 0.01
for i in range(50000):
    h_x = ((X @ theta) - Y)
    
    theta = theta - (alpha/m)*(X_T @ h_x)
    
new = np.zeros((m,1), dtype=int)
for n in range(m):
    new[n,0] = theta[0,0] + theta[1,0] * X[n,1]
    
plt.plot(X[:, 1:],new)
plt.show()

