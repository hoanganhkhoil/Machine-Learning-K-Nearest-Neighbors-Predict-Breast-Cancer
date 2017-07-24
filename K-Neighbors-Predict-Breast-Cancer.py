# Author: Khoi Hoang
# K-Neighbors-Using-SKlearn
# Predict-Breast-Cancer
# Data-taken-from-Wisconsin

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

# Load the dataset
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-999999,inplace=True)

# Drop the id column from the dataset
df.drop(['id'], 1, inplace=True)

# Distribute X and y
X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

# Distribute X and y into trainning set and testing set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

# Create a classifier - using K-Neighbors classifier
clf = neighbors.KNeighborsClassifier()

# Train the data
clf.fit(X_train, y_train)

# Test the data
accuracy = clf.score(X_test, y_test)

# Predict with some samples
X_sample = np.array([[4,2,1,1,1,2,3,1,1], [8,10,10,8,7,10,9,7,1]], dtype=np.float64)
prediction = clf.predict(X_sample)


print (accuracy)
print (prediction)
