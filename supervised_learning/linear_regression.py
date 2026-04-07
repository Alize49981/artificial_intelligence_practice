import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

#Sample dataset
data = {
    'Size': [50, 60, 70, 80, 90],
    'Price': [150, 180, 210, 240, 270]
}
df = pd.DataFrame(data)

# Check data
print(df)

#Split Data
X = df[['Size']]  # feature
y = df['Price']   # target

# Split into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train First ML Model (Linear Regression)
# Initialize model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("Predicted Prices:", y_pred)

#Visualize Results
# Plot
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, model.predict(X), color='red', label='Predicted')
plt.xlabel('Size (sq meters)')
plt.ylabel('Price (in $1000s)')
plt.title('House Price Prediction')
plt.legend()
plt.show()