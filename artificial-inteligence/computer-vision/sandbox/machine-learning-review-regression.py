import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.metrics import Accuracy

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# 1st step: Load the dataset:
data = fetch_california_housing()
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
        Dense(1, activation='relu')
    ]
)

model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae']
)
# MSE is Mean Square Error

# 5th step: Train the model & plot the metrics
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Model Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.title('Model MAE')

plt.show()