import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#create dataset
data = {
    "Hours_Studied": [1,2,3,4,5,6,7,8,9,10],
    "Score": [10,20,30,40,50,60,70,80,90,100]
}

df = pd.DataFrame(data)
#split data
X = df[["Hours_Studied"]]
y = df["Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Train Model
model = LinearRegression()
model.fit(X_train, y_train)
#make prediction
y_pred = model.predict(X_test)
print(y_pred)
#test my ai
# Predict score for 6.5 hours
prediction = model.predict([[6.5]])
print("Predicted Score:", prediction[0])
#import matrics
from sklearn.metrics import mean_squared_error, r2_score
#Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
#R² Score
r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)
#Visualize Predictions
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, label="Actual")
plt.plot(X_test, y_pred, label="Predicted")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Actual vs Predicted")
plt.legend()
plt.show()