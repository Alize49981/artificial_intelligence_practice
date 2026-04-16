import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Data: hours studied vs score
hours = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)
scores = np.array([35,45,55,65,75,85,95], dtype=float)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train model
model.fit(hours, scores, epochs=500)

# Predict
prediction = model.predict([10])
print("Predicted score for 6 hours:", prediction)

#visualization
plt.scatter(hours, scores)
plt.plot(hours, model.predict(hours), color='red')
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.show()

