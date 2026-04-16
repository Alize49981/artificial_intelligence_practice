import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

#create dataset
data = {
    'X': [1, 1.5, 3, 5, 3.5, 4.5, 3.5, 6, 7, 7.5],
    'Y': [1, 2, 4, 7, 5, 5, 4.5, 8, 8, 7.5]
}

df = pd.DataFrame(data)
X_values = df.values

#apply k-means
# Initialize KMeans with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)

# Fit and predict clusters
clusters = kmeans.fit_predict(X_values)
df['Cluster'] = clusters

print(df)

#visualize clusters
plt.scatter(df['X'], df['Y'], c=df['Cluster'], cmap='viridis', s=100)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='red', marker='X', s=200, label='Centroids')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-Means Clustering')
plt.legend()
plt.show()