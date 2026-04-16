import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

#create dataset
data = {
    'X': [1, 1.5, 3, 5, 3.5, 4.5, 3.5, 6, 7, 7.5],
    'Y': [1, 2, 4, 7, 5, 5, 4.5, 8, 8, 7.5]
}

df = pd.DataFrame(data)
X_values = df.values

#compute linkage matrix
# 'ward' minimizes variance within clusters
Z = linkage(X_values, method='ward')

#plot dendrogram
plt.figure(figsize=(10,5))
dendrogram(Z, labels=range(1, len(X_values)+1))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()

#form flst clusters
# Cut dendrogram at distance threshold 5 to form clusters
clusters = fcluster(Z, t=5, criterion='distance')
df['Cluster'] = clusters
print(df)