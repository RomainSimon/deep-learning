# Artificial Neural Network
# Churn prediction

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('./datasets/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Building the ANN

# Import Keras libs
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing
classifier = Sequential()

# Input layer and one hidden layer
classifier.add(Dense(units=6, activation='relu',
                     kernel_initializer='uniform', input_dim=11))

# Second hidden layer
classifier.add(Dense(units=6, activation='relu',
                     kernel_initializer='uniform'))

# Output layer
classifier.add(Dense(units=1, activation='sigmoid',
                     kernel_initializer='uniform'))

# Compiling ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy'])

# Training ANN
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
