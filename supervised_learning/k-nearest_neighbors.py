import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
X = df[['Hours_Study','Hours_Sleep']]
y = df['Pass']

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train KNN classifier
# Initialize KNN with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train
knn.fit(X_train, y_train)

# Predict
y_pred_knn = knn.predict(X_test)
print("KNN Predictions:", y_pred_knn)

# Accuracy
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

#viasualization
import matplotlib.pyplot as plt
import numpy as np

# Only for 2D features
def plot_knn(model, X, y):
    h = 0.1
    x_min, x_max = X.iloc[:,0].min()-1, X.iloc[:,0].max()+1
    y_min, y_max = X.iloc[:,1].min()-1, X.iloc[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y, s=50, edgecolors='k')
    plt.xlabel('Hours_Study')
    plt.ylabel('Hours_Sleep')
    plt.title('KNN Decision Boundary')
    plt.show()

plot_knn(knn, X, y)

