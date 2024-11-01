import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.metrics import Accuracy

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# 1st step: Load the dataset:
data = load_breast_cancer()
X = data.data
y = data.target

# 2nd step: Split the data into trainning and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)

# 3rd step: Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4th step: Build the model with TensorFlow2 & Keras
N, D = X_train.shape # it can be the number of features from the test set too
model = Sequential(
    [
        Input(shape=(D,)),
        Dense(1, activation='sigmoid')        
    ]
)

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 5th step: Train the model & plot the metrics
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()