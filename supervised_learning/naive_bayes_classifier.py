import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#create dataset
data = {
    'Hours_Study': [1,2,3,4,5,6,7,8,9,10],
    'Hours_Sleep': [8,7,7,6,5,6,7,6,5,4],
    'Pass': [0,0,0,0,1,0,1,1,1,1]
}

df = pd.DataFrame(data)
X = df[['Hours_Study','Hours_Sleep']]
y = df['Pass']

#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train naive classifier
# Initialize Gaussian Naive Bayes
nb_model = GaussianNB()

# Train
nb_model.fit(X_train, y_train)

# Predict
y_pred_nb = nb_model.predict(X_test)
print("Naive Bayes Predictions:", y_pred_nb)

# Accuracy
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

#probability of class
# Probability prediction for a student studying 6 hours, sleeping 5 hours
prob = nb_model.predict_proba([[6,5]])
print("Probability of Fail/Pass:", prob)
