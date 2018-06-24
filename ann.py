# Artificial Neural Network
# Churn prediction

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('./datasets/Churn_Modelling.csv')
X = dataset.iloc[:, [3, 13]].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_Geo = LabelEncoder()
X[:, 1] = labelencoder_X_Geo.fit_transform(X[:, 1])
labelencoder_X_Gender = LabelEncoder()
X[:, 2] = labelencoder_X_Gender.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Splitting dataset into training and test set
from sklearn.mode_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

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