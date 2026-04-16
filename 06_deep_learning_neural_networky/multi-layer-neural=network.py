import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

#create dataset
X = np.array([1,2,3,4,5,6,7,8,9,10], dtype=float)
y = np.array([0,0,0,0,1,0,1,1,1,1], dtype=float)

#build multilayer neural network
model = Sequential()

# Input layer + first hidden layer
model.add(Dense(units=8, activation='relu', input_shape=(1,)))

# Second hidden layer
model.add(Dense(units=4, activation='relu'))

# Output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary
model.summary()
#train model
history = model.fit(X, y, epochs=150, verbose=0)

# Evaluate
loss, accuracy = model.evaluate(X, y)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

#make prediction
# Predict probability
pred = model.predict([[6, 2, 8]])
for i, val in enumerate([[6,2,8]]):
    prob = pred[i][0]
    pred_class = int(prob > 0.5)
    print(f"Hours studied: {val}, Probability of passing: {prob:.2f}, Predicted class: {pred_class}")


#visualize training
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Accuracy & Loss')
plt.legend()
plt.show()
