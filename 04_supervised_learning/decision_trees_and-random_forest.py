import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Create Dataset
data = {
    'Hours_Study': [1,2,3,4,5,6,7,8,9,10],
    'Hours_Sleep': [8,7,7,6,5,6,7,6,5,4],
    'Pass': [0,0,0,0,1,0,1,1,1,1]
}

df = pd.DataFrame(data)
print(df)

#Split Data
X = df[['Hours_Study','Hours_Sleep']]
y = df['Pass']

#split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train Decision Tree Classifier
# Initialize model
dt_model = DecisionTreeClassifier(random_state=42)

# Train
dt_model.fit(X_train, y_train)

# Predict
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree Predictions:", y_pred_dt)

# Accuracy
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

#visualize decision tree
plt.figure(figsize=(10,6))
plot_tree(dt_model, feature_names=['Hours_Study','Hours_Sleep'], class_names=['Fail','Pass'], filled=True)
plt.show()

#Train Random Forest Classifier
# Initialize Random Forest
rf_model = RandomForestClassifier(n_estimators=5, random_state=42)  

# Train
rf_model.fit(X_train, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Predictions:", y_pred_rf)

# Accuracy
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))