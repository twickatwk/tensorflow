# KNN algorithm example

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# assign column names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names)

dataset.head()

# preprocessing
X = dataset.iloc[:, :-1].values # select everything except the last column
y = dataset.iloc[:, 4].values # only the fourth column is selected

from sklearn.model_selection import train_test_split
# this splits the dataset into 80% train data and 20% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# training and predictions
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

# make the prediction
y_pred = classifier.predict(X_test)

# evaluating the algorithm
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# comparing error rate with the K value
error = []

# calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12,6))
plt.plot(range(1,40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
