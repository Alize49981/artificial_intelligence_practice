import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

#Create Sequential Dataset
X = np.array([[0,1,2],[1,2,3],[2,3,4],[3,4,5]], dtype=float)
y = np.array([3,4,5,6], dtype=float)

# Reshape to 3D [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))
print("Input shape:", X.shape)

#build RNN model
model = Sequential()

# Simple RNN layer
model.add(SimpleRNN(10, activation='relu', input_shape=(X.shape[1], X.shape[2])))

# Output layer
model.add(Dense(1))

# Compile
model.compile(optimizer='adam', loss='mse')

model.summary()

#train RNN
history = model.fit(X, y, epochs=200, verbose=0)

# Evaluate
loss = model.evaluate(X, y)
print(f"Training Loss: {loss:.4f}")

#make prediction
# Predict next number after [4,5,6]
input_seq = np.array([4,5,6]).reshape((1,3,1))
pred = model.predict(input_seq)
print("Predicted next number:", pred[0][0])

