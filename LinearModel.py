import pandas as pd
import numpy as np

class LinearRegression():
    def __init__(self):
        self.learning_rate=0.001
        self.theta=None

    def add_intercept(self,X):
        new_x = np.zeros((X.shape[0], X.shape[1] + 1), dtype=X.dtype)
        new_x[:, 0] = 1
        new_x[:, 1:] = X

        return new_x
    


    def fit(self,X,y):
        X=self.add_intercept(X)
        m,n=X.shape
        self.theta= np.linalg.inv(X.T @ X) @ X.T @ y


    def predict(self,X):
        X=self.add_intercept(X)
        y=X @ self.theta

        return y