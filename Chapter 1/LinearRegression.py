import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import sklearn
import html

data = pd.read_csv(
    "D:/OneDrive/Documents/101 Online Learning/101 Udemy/machine-learning-101-with-scikit-learn-and-stats-models/S3_L16/1.01. Simple linear regression.csv")
print(data.head(10))
print(data.describe())

Y = data['GPA']
X = data['SAT']
print(Y.head())
print(X.head())
plt.scatter(X, Y)
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
# plt.show()

x1 = sm.add_constant(X)
results = sm.OLS(Y, x1).fit()
print(results.summary())
yhat = 0.2750 + 0.0017 * X
yhat1 = 0 + 0.0017 * 1850

print(yhat1)

# plt.scatter(X,yhat, c='Orange',lw=4)
fig = plt.plot(X, yhat, c='Red', lw=9, label='Linear Regression')
plt.show()
