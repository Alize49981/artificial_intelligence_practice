import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

#create dataset
X = np.array([1,2,3,4,5,6,7,8,9,10], dtype=float)
y = np.array([0,0,0,0,1,0,1,1,1,1], dtype=float)

#build neural network
# Sequential model
model = Sequential()

# Input layer + hidden layer
model.add(Dense(units=4, activation='relu', input_shape=(1,)))

# Output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary
model.summary()

#train neural network
history = model.fit(X, y, epochs=100, verbose=0)

# Evaluate
loss, accuracy = model.evaluate(X, y)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

#make predictions
# Predict probability
input_data = np.array([[6]])
pred = model.predict(input_data)
print("Probability of passing if studying 6 hours:", pred[0][0])

# Convert probability to class
pred_class = int(pred[0][0] > 0.5)
print("Predicted class:", pred_class)

#visualize training
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()