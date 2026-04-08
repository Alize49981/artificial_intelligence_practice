import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

#create data
data = {
    'Feature1': [2, 3, 5, 6, 8, 9],
    'Feature2': [1, 4, 6, 7, 9, 10],
    'Feature3': [5, 3, 4, 6, 8, 7]
}

df = pd.DataFrame(data)
X = df.values

#standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#apply pca
# Reduce to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Transformed Data:\n", X_pca)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

#visualize PCA
plt.scatter(X_pca[:,0], X_pca[:,1], color='blue')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - 2D Projection')
plt.show()
