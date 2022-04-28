from turtle import filling
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import plotting as pt

iris_dataset = load_iris()
#split 75% and 25% dataset into training set and testing set 
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)

# Test the split dataset features 
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

# visualize the train set
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

grr = pt.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), 
    marker='o', hist_kwds={'bins':20}, s=60, alpha=.8)  

plt.show()