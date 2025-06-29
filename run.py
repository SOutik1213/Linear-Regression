import pandas as pd
import numpy as np
from LinearModel import LinearRegression

train=pd.read_csv("./Linear-Regression/train.csv")
test=pd.read_csv("./Linear-Regression/test.csv")
m,n=train.shape
a,b=test.shape
X_train=np.array(train.iloc[:,:-1])
y_train=np.array(train.iloc[:,-1])
X_test=np.array(test.iloc[:,:-1])
y_test=np.array(test.iloc[:,-1])

lr=LinearRegression()
lr.fit(X_train,y_train)

y=lr.predict(X_test)
from sklearn.metrics import r2_score
print("R² Score:", r2_score(y_test, y))

import matplotlib.pyplot as plt

# Scatter plot: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Price (Linear Regression)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("linear_regression_plot.png")